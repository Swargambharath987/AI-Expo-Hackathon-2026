"""
Duty Line — FastAPI Backend

Serves the ReAct agent over HTTP so the frontend can call it.
Runs on http://localhost:8000

Start:
    python3 -m uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
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
# Sessions — one ReAct agent (and so one conversation history) per browser
# session per tab. The session id lives in an httponly cookie. State is
# in-memory, which is fine with Cloud Run's --max-instances 1; an instance
# restart just starts everyone on a fresh conversation.
# ---------------------------------------------------------------------------
SESSION_COOKIE    = "dl_session"
SESSION_IDLE_SECS = 60 * 60   # drop conversations idle for an hour
MAX_AGENTS        = 500       # hard cap on live (session, tab) agents
CHAT_TIMEOUT_SECS = float(os.getenv("CHAT_TIMEOUT", "120"))
_SID_RE           = re.compile(r"^[0-9a-f]{32}$")


class _AgentSlot:
    def __init__(self, agent):
        self.agent     = agent
        self.lock      = threading.Lock()   # one in-flight turn per conversation
        self.last_used = time.time()


_agents: OrderedDict[tuple[str, str], _AgentSlot] = OrderedDict()
_agents_lock = threading.Lock()


def _get_slot(session_id: str, tab: str) -> _AgentSlot:
    """Return the agent for this session+tab, creating it on first use."""
    key = (session_id, tab)
    with _agents_lock:
        now  = time.time()
        slot = _agents.pop(key, None)
        # Least-recently-used first: evict idle conversations and enforce the cap
        while _agents:
            oldest_key, oldest = next(iter(_agents.items()))
            if now - oldest.last_used > SESSION_IDLE_SECS or len(_agents) >= MAX_AGENTS:
                del _agents[oldest_key]
            else:
                break
        if slot is None:
            from agents.react_agent import ReActAgent
            slot = _AgentSlot(ReActAgent())
            log.info(f"agent created | tab={tab} | live={len(_agents) + 1}")
        slot.last_used = now
        _agents[key] = slot
        return slot


def _session_slots(session_id: str) -> list[tuple[str, _AgentSlot]]:
    with _agents_lock:
        return [(tab, slot) for (sid, tab), slot in _agents.items() if sid == session_id]


@app.middleware("http")
async def session_cookie(request: Request, call_next):
    sid = request.cookies.get(SESSION_COOKIE, "")
    is_new = not _SID_RE.match(sid)
    if is_new:
        sid = uuid.uuid4().hex
    request.state.session_id = sid
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            SESSION_COOKIE, sid,
            max_age=7 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=request.headers.get("x-forwarded-proto") == "https",
        )
    return response

# ---------------------------------------------------------------------------
# Rate limiting — /api/chat spends real Claude API credit on a public URL.
# Sliding windows per client IP, plus a global daily cap that bounds total
# spend even if per-IP limits are evaded.
# ---------------------------------------------------------------------------
RATE_PER_IP_MINUTE = int(os.getenv("RATE_PER_IP_MINUTE", "10"))
RATE_PER_IP_DAY    = int(os.getenv("RATE_PER_IP_DAY",    "100"))
RATE_GLOBAL_DAY    = int(os.getenv("RATE_GLOBAL_DAY",    "500"))

_rate_hits: defaultdict[str, deque[float]] = defaultdict(deque)
_rate_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Cloud Run's front end appends the real client IP to X-Forwarded-For;
    # earlier entries are client-supplied and can be spoofed.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _rate_limit(ip: str) -> int | None:
    """Record a hit and return None, or return retry-after seconds if over a limit."""
    now   = time.time()
    rules = [
        (f"ip-min:{ip}", RATE_PER_IP_MINUTE, 60),
        (f"ip-day:{ip}", RATE_PER_IP_DAY,    86400),
        ("global-day",   RATE_GLOBAL_DAY,    86400),
    ]
    with _rate_lock:
        for key, limit, window in rules:
            hits = _rate_hits[key]
            while hits and now - hits[0] >= window:
                hits.popleft()
            if len(hits) >= limit:
                return int(window - (now - hits[0])) + 1
        for key, _, _ in rules:
            _rate_hits[key].append(now)
        if len(_rate_hits) > 10_000:
            for key in [k for k, v in _rate_hits.items() if not v or now - v[-1] >= 86400]:
                del _rate_hits[key]
    return None

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


def _parse_agent_output(raw: str, form: dict | None) -> tuple[str, list[dict], dict | None]:
    """
    Parse raw agent output into:
      - answer:      final response text
      - tool_calls:  list of {tool, label, result_summary} matching frontend ToolTrace
      - form_output: built from this conversation's own fill_form result, or None
    """
    lines        = raw.split("\n")
    answer_lines = []
    tool_calls   = []
    form_output  = None

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

    if form:
        pdf_path = Path(form["pdf_path"]) if form.get("pdf_path") else None
        pdf_ok   = pdf_path is not None and pdf_path.exists()
        form_output = {
            "form_name":      pdf_path.stem if pdf_path else form.get("form_name", "form"),
            "filled_fields":  form.get("filled_fields", {}),
            "missing_fields": form.get("missing_fields", []),
            "pdf_available":  pdf_ok,
            "pdf_url":        f"/api/forms/{pdf_path.name}" if pdf_ok else None,
            "txt_summary":    form.get("summary", ""),
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


def _run_turn(slot: _AgentSlot, message: str) -> tuple[str, dict | None]:
    """Run one agent turn. Blocking — call via the threadpool, never on the event loop."""
    with slot.lock:
        raw = "".join(slot.agent.chat(message))
        return raw, slot.agent.last_form


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    Main chat endpoint. Runs this session's agent for the given tab and
    returns a structured response.
    """
    ip = _client_ip(request)
    retry_after = _rate_limit(ip)
    if retry_after is not None:
        log.warning(f"rate limited | ip={ip} | retry_after={retry_after}s")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — please wait a moment and try again."},
            headers={"Retry-After": str(retry_after)},
        )

    log.info(f"chat | tab={req.tab} | ip={ip} | msg={req.message[:80]!r}")

    try:
        slot = await run_in_threadpool(_get_slot, request.state.session_id, req.tab)
    except Exception as e:
        log.error(f"Agent init failed: {e}")
        return ChatResponse(
            response="The AI assistant failed to initialise. Make sure your LLM API key is set and the vector store exists (run ingest.py).",
            tool_calls=[],
            form_output=None,
            error=str(e),
        )

    if slot.lock.locked():
        return ChatResponse(
            response="Still working on your previous message in this tab — please wait for it to finish.",
            tool_calls=[],
            form_output=None,
            error=None,
        )

    try:
        raw, form = await asyncio.wait_for(
            run_in_threadpool(_run_turn, slot, req.message),
            timeout=CHAT_TIMEOUT_SECS,
        )
        answer, tool_calls, form_output = _parse_agent_output(raw, form)

        log.info(f"chat done | tools={[t['tool'] for t in tool_calls]} | answer_len={len(answer)}")

        return ChatResponse(
            response=answer,
            tool_calls=[ToolCallItem(**t) for t in tool_calls],
            form_output=form_output,
            error=None,
        )

    except asyncio.TimeoutError:
        log.error(f"chat timed out after {CHAT_TIMEOUT_SECS}s | tab={req.tab}")
        return ChatResponse(
            response="This request took too long and was stopped. Please try again.",
            tool_calls=[],
            form_output=None,
            error="timeout",
        )

    except Exception as e:
        log.error(f"Agent chat error: {e}", exc_info=True)
        return ChatResponse(
            response="An error occurred while processing your request. Please try again.",
            tool_calls=[],
            form_output=None,
            error=str(e),
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

        # Reload profile in running agents without resetting their conversations
        with _agents_lock:
            for slot in _agents.values():
                slot.agent.update_profile(data)

        log.info(f"Profile saved: {data.get('name_last_first')} {data.get('rank')}")
        return {"status": "saved"}
    except Exception as e:
        log.error(f"Profile save error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/chat/history")
async def clear_history(request: Request, tab: str | None = None):
    """Clear this session's conversation history — one tab, or all tabs if none given."""
    for slot_tab, slot in _session_slots(request.state.session_id):
        if tab is None or slot_tab == tab:
            slot.agent.reset()
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
