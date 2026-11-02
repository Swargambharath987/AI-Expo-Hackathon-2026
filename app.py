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

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator, model_validator

from agents.react_agent import (UnsafeURLError, check_custom_base_url,
                                custom_url_host, llm_http_client)

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dutyline")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT           = Path(__file__).parent
OUTPUT_DIR     = ROOT / "output"
GSA_CACHE      = ROOT / "data" / "gsa_cache.json"
VECTORSTORE    = ROOT / "vectorstore"
FRONTEND_DIST  = ROOT / "frontend" / "dist"
OUTPUT_DIR.mkdir(exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "claude-sonnet-4-6")
LLM_API_KEY  = os.getenv("LLM_API_KEY",  "")

# Providers a visitor may use with their own key. Preset base URLs are fixed
# here — never taken from the request — so a preset key can only go to its
# provider. All speak the OpenAI chat-completions API. An empty
# default_model means "pick from the key check's model list" (the frontend
# does this) — set only where a current tool-capable id is known.
# "custom" is any OpenAI-compatible endpoint the visitor names; its URL is
# SSRF-checked (see agents/react_agent.py check_custom_base_url).
BYOK_PROVIDERS = {
    "anthropic":  {"label": "Anthropic (Claude)", "base_url": "https://api.anthropic.com/v1",
                   "default_model": "claude-sonnet-4-6", "key_placeholder": "sk-ant-..."},
    "openai":     {"label": "OpenAI", "base_url": "https://api.openai.com/v1",
                   "default_model": "gpt-4.1", "key_placeholder": "sk-..."},
    # One key for many vendors' models (Llama, Gemini, Mistral, Claude, GPT…)
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
                   "default_model": "anthropic/claude-sonnet-4.6", "key_placeholder": "sk-or-v1-..."},
    "groq":       {"label": "Groq", "base_url": "https://api.groq.com/openai/v1",
                   "default_model": "", "key_placeholder": "gsk_..."},
    "mistral":    {"label": "Mistral", "base_url": "https://api.mistral.ai/v1",
                   "default_model": "mistral-large-latest", "key_placeholder": "Mistral API key"},
    "gemini":     {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "default_model": "", "key_placeholder": "AIza..."},
    "deepseek":   {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                   "default_model": "deepseek-chat", "key_placeholder": "sk-..."},
    "xai":        {"label": "xAI (Grok)", "base_url": "https://api.x.ai/v1",
                   "default_model": "", "key_placeholder": "xai-..."},
    "together":   {"label": "Together AI", "base_url": "https://api.together.xyz/v1",
                   "default_model": "", "key_placeholder": "Together API key"},
    "custom":     {"label": "Custom (OpenAI-compatible)", "base_url": "",
                   "default_model": "", "key_placeholder": "API key"},
}

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


# Generated form files each session may download — forms carry SSN-4 and
# DoD ID, so a filename alone isn't enough to fetch one.
# LRU-capped; the files themselves are deleted after an hour anyway.
_session_files: OrderedDict[str, set[str]] = OrderedDict()
MAX_FILE_SESSIONS = 2000


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


def _rate_limit(ip: str, byok: bool = False) -> int | None:
    """Record a hit and return None, or return retry-after seconds if over a limit."""
    now   = time.time()
    rules = [(f"ip-min:{ip}", RATE_PER_IP_MINUTE, 60)]
    if not byok:
        rules += [
            (f"ip-day:{ip}", RATE_PER_IP_DAY, 86400),
            ("global-day",   RATE_GLOBAL_DAY, 86400),
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

    # Lenient on purpose: the profile rides along with every chat request,
    # so a bad field should be dropped, not turn every message into a 422.
    @field_validator("*")
    @classmethod
    def cap_length(cls, v: str) -> str:
        return v.strip()[:100]

    @field_validator("ssn_last4")
    @classmethod
    def ssn_digits_only(cls, v: str) -> str:
        return v if re.match(r"^\d{4}$", v) else ""

class KeyCheckRequest(BaseModel):
    provider: str
    api_key:  str
    # Only used (and required) for provider "custom"; ignored for presets,
    # whose URL always comes from BYOK_PROVIDERS.
    base_url: str = ""

    @field_validator("provider")
    @classmethod
    def known_provider(cls, v: str) -> str:
        if v not in BYOK_PROVIDERS:
            raise ValueError("unknown provider")
        return v

    @model_validator(mode="after")
    def custom_url(self):
        if self.provider == "custom":
            # Syntax only here (runs on the event loop); the DNS check runs in
            # a worker thread via _resolved_base_url().
            try:
                self.base_url = check_custom_base_url(self.base_url, resolve=False)
            except UnsafeURLError as e:
                raise ValueError(str(e)) from None
        else:
            self.base_url = ""
        return self

    def preset_or_custom_url(self) -> str:
        return self.base_url if self.provider == "custom" else BYOK_PROVIDERS[self.provider]["base_url"]

    @field_validator("api_key")
    @classmethod
    def key_shape(cls, v: str) -> str:
        v = v.strip()
        if not 20 <= len(v) <= 300 or not re.match(r"^[\w\-.]+$", v):
            raise ValueError("API key doesn't look valid")
        return v


class LLMOverride(KeyCheckRequest):
    model: str = ""

    @field_validator("model")
    @classmethod
    def model_shape(cls, v: str) -> str:
        v = v.strip()
        if v and not re.match(r"^[\w\-.:/]{1,100}$", v):
            raise ValueError("invalid model name")
        return v

    @model_validator(mode="after")
    def model_known(self):
        if not (self.model or BYOK_PROVIDERS[self.provider]["default_model"]):
            raise ValueError("choose a model for this provider")
        return self

    def resolved(self) -> dict:
        p = BYOK_PROVIDERS[self.provider]
        return {"base_url": self.preset_or_custom_url(), "api_key": self.api_key,
                "model": self.model or p["default_model"], "provider": self.provider}


class ChatRequest(BaseModel):
    message: str
    tab: str = "travel"
    # The soldier profile lives in the user's browser and comes with every
    # request, so each user's forms use their own details.
    profile: ProfileModel | None = None
    # Optional "bring your own key" — used for this request instead of the
    # server's key. Never logged or stored server-side beyond the session.
    llm: LLMOverride | None = None

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


class Verification(BaseModel):
    """Grounding check of the answer against this turn's tool results."""
    checked:              bool
    numbers_ok:           bool
    unverified_numbers:   list[str]
    citations_ok:         bool
    unverified_citations: list[str]


class ChatResponse(BaseModel):
    response:     str
    tool_calls:   list[ToolCallItem]   # frontend reads data.tool_calls
    form_output:  dict | None
    error:        str | None
    # None when the answer had nothing to check (no $ figures, no citations)
    verification: Verification | None = None


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


_NON_CHAT_MODEL = re.compile(r"embed|tts|whisper|dall-e|audio|realtime|image|transcribe|moderation|search|babbage|davinci")


def _probe_llm_key(base_url: str, api_key: str, custom: bool = False) -> tuple[bool | None, list[str]]:
    """
    Ask the provider whether a key works by listing models — free, no tokens.
    Returns (True, model_ids) if accepted, (False, []) if rejected,
    (None, []) if the provider couldn't be reached.
    custom=True: a visitor-supplied URL — fetched only through the
    public-only, no-redirect client, and never matched to a preset below.
    """
    if not api_key:
        return False, []
    base = base_url.rstrip("/")
    try:
        if custom:
            with llm_http_client(custom=True, timeout=8) as c:
                r = c.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"})
        elif "anthropic" in base:
            r = requests.get("https://api.anthropic.com/v1/models",
                             headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                             params={"limit": 100}, timeout=8, allow_redirects=False)
        elif "openrouter.ai" in base:
            # OpenRouter's model list is public (accepts any key), so check
            # the key on /key, then list only models that support tool calls.
            auth = requests.get(f"{base}/key", headers={"Authorization": f"Bearer {api_key}"},
                                timeout=8, allow_redirects=False)
            if auth.status_code in (401, 403):
                return False, []
            if not auth.ok or auth.is_redirect:
                return None, []
            r = requests.get(f"{base}/models", timeout=8, allow_redirects=False)
            if not r.ok or r.is_redirect:
                return None, []
            ids = sorted(m.get("id", "") for m in r.json().get("data", [])
                         if "tools" in (m.get("supported_parameters") or []))
            return True, [i for i in ids if i and not i.endswith(":batch")]
        else:
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {api_key}"},
                             timeout=8, allow_redirects=False)
    except (requests.RequestException, httpx.HTTPError):
        return None, []
    if r.status_code in (401, 403):
        return False, []
    # Redirects are never followed — treat one as "couldn't check"
    if not 200 <= r.status_code < 300:
        return None, []
    try:
        body = r.json()
    except ValueError:
        return None, []
    # Most providers return {"data": [...]}; Together returns a bare list
    items = body.get("data", []) if isinstance(body, dict) else body if isinstance(body, list) else []
    ids = sorted(str(m.get("id", "")) for m in items if isinstance(m, dict))
    return True, [i for i in ids if i and not _NON_CHAT_MODEL.search(i)]


_key_status: dict[str, Any] = {"ok": None, "checked": 0.0}


def _server_key_ready() -> bool:
    """Real check of the server's own key, cached (10 min if good, 1 min if not)."""
    ttl = 600 if _key_status["ok"] else 60
    if time.time() - _key_status["checked"] > ttl:
        ok, _ = _probe_llm_key(LLM_BASE_URL, LLM_API_KEY)
        if ok is not None or _key_status["ok"] is None:
            _key_status["ok"] = bool(ok)
        _key_status["checked"] = time.time()
    return bool(_key_status["ok"])


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

    # Ollama: ping localhost. Cloud providers: ask the provider whether the
    # key is accepted (model listing — free, no tokens; cached).
    if provider == "ollama":
        ollama_ok = await run_in_threadpool(_check_ollama)
        llm_ready = ollama_ok
    else:
        ollama_ok = False
        llm_ready = await run_in_threadpool(_server_key_ready)

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


def _run_turn(slot: _AgentSlot, message: str, profile: dict,
              llm: dict | None) -> tuple[str, dict | None, dict | None]:
    """Run one agent turn. Blocking — call via the threadpool, never on the event loop."""
    with slot.lock:
        raw = "".join(slot.agent.chat(message, profile, llm))
        return raw, slot.agent.last_form, getattr(slot.agent, "last_verification", None)


async def _check_custom_dns(req: KeyCheckRequest) -> JSONResponse | None:
    """DNS half of the custom-URL check (blocking, so off the event loop).
    Returns a 422 response if the endpoint isn't allowed, else None."""
    if req.provider != "custom":
        return None
    try:
        await run_in_threadpool(check_custom_base_url, req.base_url)
    except UnsafeURLError as e:
        log.warning(f"custom endpoint rejected | host={custom_url_host(req.base_url)} | {e}")
        return JSONResponse(status_code=422, content={"detail": f"Custom endpoint not allowed: {e}"})
    return None


@app.get("/api/llm/providers")
async def llm_providers():
    """Providers a visitor can use with their own key (custom last)."""
    return [
        {"id": pid, "label": p["label"], "key_placeholder": p["key_placeholder"],
         "default_model": p["default_model"], "needs_base_url": pid == "custom"}
        for pid, p in sorted(BYOK_PROVIDERS.items(), key=lambda kv: kv[0] == "custom")
    ]


@app.post("/api/llm/check")
async def check_llm_key(req: KeyCheckRequest, request: Request):
    """Validate a visitor's own key and list the chat models it can use."""
    retry_after = _rate_limit(_client_ip(request), byok=True)
    if retry_after is not None:
        return JSONResponse(status_code=429, content={"detail": "Too many requests — please wait a moment."},
                            headers={"Retry-After": str(retry_after)})
    if (rejected := await _check_custom_dns(req)) is not None:
        return rejected
    if req.provider == "custom":
        ok, models = await run_in_threadpool(_probe_llm_key, req.base_url, req.api_key, True)
    else:
        ok, models = await run_in_threadpool(_probe_llm_key, req.preset_or_custom_url(), req.api_key)
    return {
        "valid":         ok,
        "models":        models,
        "default_model": BYOK_PROVIDERS[req.provider]["default_model"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """
    Main chat endpoint. Runs this session's agent for the given tab and
    returns a structured response.
    """
    ip = _client_ip(request)
    # Own-key requests don't spend the server's API credit, so they skip the
    # daily caps but still get the per-minute limit.
    retry_after = _rate_limit(ip, byok=req.llm is not None)
    if retry_after is not None:
        log.warning(f"rate limited | ip={ip} | retry_after={retry_after}s")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests — please wait a moment and try again."},
            headers={"Retry-After": str(retry_after)},
        )

    own = "no"
    if req.llm:
        own = req.llm.provider
        if req.llm.provider == "custom":
            own += f"@{custom_url_host(req.llm.base_url)}"   # host only, never the full URL
    log.info(f"chat | tab={req.tab} | ip={ip} | own_key={own} | msg={req.message[:80]!r}")

    # Re-validated every turn (DNS may have changed since the key check);
    # the agent's transport also re-checks each connection.
    if req.llm and (rejected := await _check_custom_dns(req.llm)) is not None:
        return rejected

    try:
        slot = await run_in_threadpool(_get_slot, request.state.session_id, req.tab)
    except Exception:
        log.exception("Agent init failed")
        return ChatResponse(
            response="The AI assistant failed to initialise. Make sure your LLM API key is set and the vector store exists (run ingest.py).",
            tool_calls=[],
            form_output=None,
            error="agent_init_failed",
        )

    if slot.lock.locked():
        return ChatResponse(
            response="Still working on your previous message in this tab — please wait for it to finish.",
            tool_calls=[],
            form_output=None,
            error=None,
        )

    try:
        raw, form, verification = await asyncio.wait_for(
            run_in_threadpool(_run_turn, slot, req.message,
                              req.profile.model_dump() if req.profile else {},
                              req.llm.resolved() if req.llm else None),
            timeout=CHAT_TIMEOUT_SECS,
        )
        answer, tool_calls, form_output = _parse_agent_output(raw, form)
        if form:
            owned = _session_files.setdefault(request.state.session_id, set())
            _session_files.move_to_end(request.state.session_id)
            while len(_session_files) > MAX_FILE_SESSIONS:
                _session_files.popitem(last=False)
            for key in ("pdf_path", "txt_path", "acroform_pdf_path"):
                if form.get(key):
                    owned.add(Path(form[key]).name)

        log.info(f"chat done | tools={[t['tool'] for t in tool_calls]} | answer_len={len(answer)}"
                 f" | verified={'n/a' if not verification else verification['numbers_ok'] and verification['citations_ok']}")

        return ChatResponse(
            response=answer,
            tool_calls=[ToolCallItem(**t) for t in tool_calls],
            form_output=form_output,
            error=None,
            verification=verification,
        )

    except asyncio.TimeoutError:
        log.error(f"chat timed out after {CHAT_TIMEOUT_SECS}s | tab={req.tab}")
        return ChatResponse(
            response="This request took too long and was stopped. Please try again.",
            tool_calls=[],
            form_output=None,
            error="timeout",
        )

    except Exception:
        # Full details stay in the server log; the browser gets a generic code.
        log.exception("Agent chat error")
        return ChatResponse(
            response="An error occurred while processing your request. Please try again.",
            tool_calls=[],
            form_output=None,
            error="internal_error",
        )


@app.get("/api/forms/{filename}")
async def get_form(filename: str, request: Request):
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
    if filename not in _session_files.get(request.state.session_id, set()) or not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Form '{filename}' not found")

    media_type = "application/pdf" if filename.endswith(".pdf") else "text/plain"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


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
