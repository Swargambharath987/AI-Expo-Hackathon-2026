"""
Duty Line — FastAPI Backend

Serves the ReAct agent over HTTP so the frontend can call it.
Runs on http://localhost:8000

Start:
    python3 -m uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dutyline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent
OUTPUT_DIR     = ROOT / "output"
PROFILE_FILE   = ROOT / "profile.json"
GSA_CACHE      = ROOT / "data" / "gsa_cache.json"
VECTORSTORE    = ROOT / "vectorstore"
FRONTEND_DIST  = ROOT / "frontend" / "dist"
OUTPUT_DIR.mkdir(exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "claude-sonnet-4-6")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "")

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Duty Line API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # fallback
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ReAct agent — lazy singleton, one instance reused across requests
# ---------------------------------------------------------------------------
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        from agents.react_agent import ReActAgent
        _agent = ReActAgent()
        log.info("ReAct agent initialised")
    return _agent

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    tab: str = "travel"
    profile: dict | None = None   # accepted but ignored — backend uses profile.json

    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("message cannot be empty")
        if len(v) > 2000:
            raise ValueError("message too long (max 2000 chars)")
        return v

    @field_validator("tab")
    @classmethod
    def valid_tab(cls, v: str) -> str:
        allowed = {"travel", "leave", "regulation", "eval"}
        return v if v in allowed else "travel"


class ToolCallItem(BaseModel):
    """Structured tool trace — matches the frontend ToolTrace type."""
    tool:           str
    label:          str
    result_summary: str


class ChatResponse(BaseModel):
    response:    str
    tool_calls:  list[ToolCallItem]   # frontend reads data.tool_calls
    form_output: dict | None
    error:       str | None


class ProfileModel(BaseModel):
    name_last_first: str = ""
    rank:            str = ""
    grade:           str = ""
    ssn_last4:       str = ""
    dod_id:          str = ""
    unit:            str = ""
    installation:    str = ""
    uic:             str = ""
    supervisor_name: str = ""
    supervisor_title: str = ""

    @field_validator("ssn_last4")
    @classmethod
    def ssn_digits_only(cls, v: str) -> str:
        if v and not re.match(r"^\d{4}$", v):
            raise ValueError("ssn_last4 must be exactly 4 digits")
        return v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_LABEL_TO_NAME = {
    "Searching regulations":   "search_regulations",
    "Getting per diem":        "get_per_diem",
    "Calculating travel cost": "calculate_travel_cost",
    "Filling":                 "fill_form",
}


def _parse_agent_output(raw: str) -> tuple[str, list[dict], dict | None]:
    """
    Parse raw agent output into:
      - answer:      final response text
      - tool_calls:  list of {tool, label, result_summary} matching frontend ToolTrace
      - form_output: {form_name, filled_fields, missing_fields, pdf_path, summary} or None
    """
    lines        = raw.split("\n")
    answer_lines = []
    tool_calls   = []
    form_output  = None
    fill_used    = False

    # Pair ⚙ header lines with their following → result lines
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("⚙"):
            label = stripped.lstrip("⚙").strip()
            # Determine canonical tool name
            tool_name = "search_regulations"
            for marker, name in TOOL_LABEL_TO_NAME.items():
                if marker in label:
                    tool_name = name
                    break
            if tool_name == "fill_form":
                fill_used = True

            # Grab the → result line that usually follows
            result_summary = ""
            if i + 1 < len(lines) and lines[i + 1].strip().startswith("→"):
                result_summary = lines[i + 1].strip().lstrip("→").strip()
                i += 1  # skip the result line so it doesn't appear in answer

            tool_calls.append({
                "tool":           tool_name,
                "label":          label,
                "result_summary": result_summary,
            })

        elif stripped.startswith("→") or stripped.startswith("⚠"):
            pass  # observation/warning — already consumed above or skip

        elif stripped.startswith("[Error"):
            answer_lines.append(stripped)

        else:
            answer_lines.append(lines[i])

        i += 1

    # Detect form output when fill_form was called
    if fill_used:
        pdfs = sorted(OUTPUT_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        txts = sorted(OUTPUT_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        recent_pdf = pdfs[0] if pdfs and (time.time() - pdfs[0].stat().st_mtime) < 30 else None
        recent_txt = txts[0] if txts and (time.time() - txts[0].stat().st_mtime) < 30 else None

        txt_content = recent_txt.read_text() if recent_txt else ""
        form_name   = recent_pdf.stem if recent_pdf else (recent_txt.stem if recent_txt else "form")

        # Parse txt_content into filled_fields for the frontend table
        filled_fields:  dict[str, str] = {}
        missing_fields: list[str]      = []
        for line in txt_content.splitlines():
            if ": " in line and not line.startswith("Missing"):
                k, _, v = line.partition(": ")
                filled_fields[k.strip()] = v.strip()
            elif line.startswith("Missing") or line.startswith("- "):
                field = line.lstrip("- ").strip()
                if field:
                    missing_fields.append(field)

        form_output = {
            "form_name":      form_name,
            "filled_fields":  filled_fields,
            "missing_fields": missing_fields,
            "pdf_available":  recent_pdf is not None,
            "pdf_url":        f"/api/forms/{recent_pdf.name}" if recent_pdf else None,
            "txt_summary":    txt_content,
        }

    answer = "\n".join(answer_lines).strip()
    if not answer:
        answer = "I was unable to generate a response. Please try rephrasing."

    return answer, tool_calls, form_output


def _detect_provider(base_url: str) -> str:
    """Detect the LLM provider from the base URL."""
    if "anthropic" in base_url:
        return "claude"
    if "openrouter" in base_url:
        return "openrouter"
    if "localhost" in base_url or "11434" in base_url:
        return "ollama"
    return "custom"


def _check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _check_cloud_llm(provider: str) -> bool:
    """Validate cloud API key by format — fast, no network call, no tokens burned."""
    key = LLM_API_KEY or ""
    if not key or "your-key" in key or len(key) < 20:
        return False
    if provider == "claude":
        return key.startswith("sk-ant-")
    if provider == "openrouter":
        return key.startswith("sk-or-")
    # Custom/unknown provider — key is present and not a placeholder
    return True


def _count_vector_chunks() -> int:
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(VECTORSTORE),
            settings=Settings(anonymized_telemetry=False),
        )
        col = client.get_or_create_collection("dutyline")
        return col.count()
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    """Check backend, LLM provider, vector store, and GSA cache status."""
    provider    = _detect_provider(LLM_BASE_URL)
    chunk_count = _count_vector_chunks()
    gsa_ok      = GSA_CACHE.exists() and GSA_CACHE.stat().st_size > 1000

    # For Ollama: actively ping localhost to confirm it is running.
    # For cloud providers (Claude, OpenRouter, custom): assume ready — we
    # cannot ping the API without burning tokens; missing key is caught at
    # first chat request.
    if provider == "ollama":
        ollama_ok = _check_ollama()
        llm_ready = ollama_ok
    else:
        ollama_ok = False
        llm_ready = _check_cloud_llm(provider)

    offline_ready = provider == "ollama" and llm_ready and chunk_count > 0 and gsa_ok

    return {
        "status":              "ok",
        "llm_provider":        provider,
        "llm_ready":           llm_ready,
        "model":               LLM_MODEL,
        "vector_store_chunks": chunk_count,
        "vector_store_ready":  chunk_count > 0,
        "gsa_cache_loaded":    gsa_ok,
        "offline_ready":       offline_ready,
        # kept for backwards compatibility with frontend health status dots
        "ollama":              ollama_ok,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint. Runs the ReAct agent and returns structured response.
    """
    log.info(f"chat | tab={req.tab} | msg={req.message[:80]!r}")

    try:
        agent = get_agent()
    except Exception as e:
        log.error(f"Agent init failed: {e}")
        return ChatResponse(
            response="The AI assistant failed to initialise. Make sure your LLM API key is set and the vector store exists (run ingest.py).",
            tool_calls=[],
            form_output=None,
            error=str(e),
        )

    try:
        raw_parts: list[str] = []
        for token in agent.chat(req.message):
            raw_parts.append(token)

        raw = "".join(raw_parts)
        answer, tool_calls, form_output = _parse_agent_output(raw)

        log.info(f"chat done | tools={[t['tool'] for t in tool_calls]} | answer_len={len(answer)}")

        return ChatResponse(
            response=answer,
            tool_calls=[ToolCallItem(**t) for t in tool_calls],
            form_output=form_output,
            error=None,
        )

    except Exception as e:
        log.error(f"Agent chat error: {e}", exc_info=True)
        return ChatResponse(
            response="An error occurred while processing your request. Please try again.",
            tool_calls=[],
            form_output=None,
            error=str(e),
        )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat via Server-Sent Events.
    Each event: data: {"type": "reasoning"|"token"|"done", "text": "..."}
    Frontend: use EventSource or fetch with ReadableStream.
    """
    import asyncio

    def _run_agent(message: str):
        agent = get_agent()
        return list(agent.chat(message))

    async def generate():
        try:
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(None, _run_agent, req.message)
            for token in tokens:
                stripped = token.strip()
                if stripped.startswith("⚙") or stripped.startswith("→") or stripped.startswith("⚠"):
                    event = json.dumps({"type": "reasoning", "text": token})
                else:
                    event = json.dumps({"type": "token", "text": token})
                yield f"data: {event}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/forms/{filename}")
async def get_form(filename: str):
    """
    Download a generated form PDF or TXT.
    Validates path to prevent directory traversal.
    """
    # Security: reject any path traversal attempts
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    # Only allow known extensions
    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are served")

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Form '{filename}' not found")

    media_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


@app.get("/api/profile")
async def get_profile():
    """Load the saved soldier profile."""
    if not PROFILE_FILE.exists():
        return {}
    try:
        return json.loads(PROFILE_FILE.read_text())
    except Exception as e:
        log.error(f"Profile read error: {e}")
        raise HTTPException(status_code=500, detail="Failed to read profile")


@app.post("/api/profile")
async def save_profile(profile: ProfileModel):
    """Save soldier profile to local file. Reloads the agent's profile."""
    try:
        data = profile.model_dump()
        PROFILE_FILE.write_text(json.dumps(data, indent=2))

        # Reload profile in running agent without reinitialising
        if _agent is not None:
            _agent._profile = data

        log.info(f"Profile saved: {data.get('name_last_first')} {data.get('rank')}")
        return {"status": "saved"}
    except Exception as e:
        log.error(f"Profile save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history")
async def clear_history():
    """Clear the agent's conversation history."""
    if _agent is not None:
        _agent.reset()
    return {"status": "cleared"}


if FRONTEND_DIST.is_dir():
    # Production: serve the built React app from the same process.
    # /api/* is registered above and always wins; everything else falls
    # through to index.html so client-side routing works.
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
else:
    @app.get("/")
    async def root():
        return {"service": "Duty Line API", "version": "1.0.0", "docs": "/docs"}
