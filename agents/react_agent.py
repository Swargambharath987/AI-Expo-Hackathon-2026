"""
Duty Line — ReAct Agent

Single reasoning loop:
  Thought  → what do I need to answer this?
  Action   → call a tool (or answer directly)
  Observe  → receive tool result
  Repeat   → until sufficient, then stream final answer

One model (claude-sonnet-4-6 via Claude API by default; Ollama for offline), all tools available every turn.
No separate classifier. No routing failures.

Usage:
    from agents.react_agent import ReActAgent
    agent = ReActAgent()
    for token in agent.chat("I need TDY to Fort Moore next month"):
        print(token, end="", flush=True)
"""

from __future__ import annotations

import ipaddress
import json
from datetime import datetime
import logging
import os
import re
import socket
import time
import warnings
from typing import Any, Generator
from urllib.parse import urlsplit

import httpcore
import httpx
from dotenv import load_dotenv
from openai import AuthenticationError, DefaultHttpxClient, OpenAI

from agents.tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
log = logging.getLogger("dutyline.agent")
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*existing embedding ID.*")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",  "https://api.anthropic.com/v1")
LLM_MODEL      = os.getenv("LLM_MODEL",     "claude-sonnet-4-6")
LLM_API_KEY    = os.getenv("LLM_API_KEY",   "")
LLM_TIMEOUT    = float(os.getenv("LLM_TIMEOUT", "30"))   # seconds per LLM call
MAX_ITERATIONS = 5      # max tool calls per turn before forcing a final answer
HISTORY_WINDOW = 8      # max messages kept in context (sliding window)
RAG_TOP_K      = 2      # chunks per search — keeps context under 4096 tokens

# ---------------------------------------------------------------------------
# System prompt — tells the model what it is, what docs exist, how to reason
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Duty Line, an AI assistant for US military personnel.
Mission: reduce bureaucratic burden — help soldiers navigate regulations, plan TDY travel, manage leave, and complete forms.

## Documents searchable via search_regulations (nothing else is in the library):
- JTR: all TDY travel — per diem, lodging, mileage, GTC, entitlements
- AR 600-8-10: Army leave — accrual, DA 31, leave types, deployment leave
- AFI 36-2406 / BUPERSINST 1610.10F / MCO 1610.7: Air Force / Navy / Marine evaluations
- AFI 36-3003: Air Force leave (MILPERSMAN 1050, Navy leave, is only partially covered)
- DoD FMR Vol 7A: military pay
Army evaluations (AR 623-3, NCOER/OER) are NOT in the library.

## Tools:
- search_regulations — exact regulation text with citations
- get_per_diem — offline GSA rate lookup
- calculate_travel_cost — full JTR cost breakdown (lodging + meals + mileage)
- fill_form — populate DD_1610, DA_31, DA_4856, DA_4187 — outputs filled PDF

## MANDATORY tool use (no exceptions):
- ANY question about policy, rules, entitlements, eligibility, procedures → call search_regulations FIRST
- TDY cost questions → call calculate_travel_cost (never estimate dollar amounts)
- Form fill requests → call fill_form at the end

## Response rules:
1. Regulation answers: call search_regulations → read the result → answer from it. Never answer regulation questions from memory.
   If the results don't contain the answer (or the regulation isn't in the library), say so plainly and suggest the official source — do not fill the gap from memory.
2. YES/NO questions: first call the relevant tool, then answer "Yes" or "No" on line 1. One sentence of citation.
3. TDY cost: show a brief dollar breakdown (lodging / meals / mileage / total). Never estimate.
4. Forms: only call fill_form when the user asks for a form. Then say "Your form is ready for download." and list any fields still missing.
   Form content must come ONLY from what the user said, their profile, or tool results. Never invent or embellish: no added titles/positions, no added consequences or UCMJ warnings, no praise the user didn't give. If something the form needs is missing (e.g. counseling key points), ask for it before filling.
   If the form is about someone other than the user (e.g. sending a soldier on TDY), pass that person as subject_name / subject_rank.
   When filling DD_1610 after a cost estimate, pass every known trip detail: destination, travel_days, departure/return dates, travel_mode, cost_estimate.
5. Citations: end every regulation answer with "— [document name], [section]", using only section numbers that appear in this turn's search results. Never cite a section from memory.
6. NEVER mention file paths, file names, output directories, file formats, or internal system details.

## Response format -- follow these examples exactly:

TDY cost example:
"**TDY Cost -- Fort Moore, GA (5 days, POV)**
- Lodging: $416 (4 nights x $104)
- Meals: $288 (3 full days x $64, 2 travel days x $48)
- Mileage: $518 (370 mi x $0.70 x 2)
- **Total: $1,222**"

Regulation example:
"Yes, the GTC is mandatory for all TDY travel expenses unless a specific exemption applies. -- JTR, [section number from the search result]"

Leave example:
"You accrue 2.5 days per month (30 days/year). Max carry-over is 60 days at end of FY. -- AR 600-8-10, [paragraph from the search result]"

Rules:
- Never repeat the question. Never add disclaimers or please note.
- Never mention file paths, file formats, or system internals.
- Always include the dollar amounts from tool results for TDY queries.
- Always use full form titles: "DD Form 1610 (Request and Authorization for TDY Travel)", "DA Form 31 (Request and Authority for Leave)", "DA Form 4856 (Developmental Counseling Form)", "DA Form 4187 (Personnel Action)".
- Keep total response under 8 lines."""

# ---------------------------------------------------------------------------
# Custom endpoint guard — a visitor may point the agent at any
# OpenAI-compatible base URL, which makes this server fetch a URL of their
# choosing (SSRF). On Cloud Run the metadata server (169.254.169.254) and
# other internal addresses must stay unreachable, so custom URLs are:
#   - https only, no userinfo/query/fragment, hostname required
#   - rejected if the hostname resolves to ANY non-public address
#   - re-checked at connect time: the transport below resolves the host
#     itself, validates every address, and connects to the validated IP
#     (TLS still verifies the certificate against the hostname). So a DNS
#     record that flips to a private address after the first check
#     (rebinding) is caught on the actual connection, not just up front.
#   - never redirected (follow_redirects=False everywhere)
# ALLOW_PRIVATE_LLM_URLS=1 relaxes the public-address rule (and allows
# plain http) for local development, e.g. Ollama on localhost. Default off.
# ---------------------------------------------------------------------------

class UnsafeURLError(ValueError):
    """A custom LLM base URL that this server refuses to call."""


def _allow_private_urls() -> bool:
    return os.getenv("ALLOW_PRIVATE_LLM_URLS", "") == "1"


def _is_public_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])   # drop IPv6 zone id
    if ip.version == 6 and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global


def resolve_public_addresses(host: str, port: int) -> list[str]:
    """Resolve host; raise UnsafeURLError unless every address is public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError):
        raise UnsafeURLError("the endpoint's hostname does not resolve") from None
    addrs = list(dict.fromkeys(info[4][0] for info in infos))
    if not addrs:
        raise UnsafeURLError("the endpoint's hostname does not resolve")
    if not _allow_private_urls() and not all(_is_public_ip(a) for a in addrs):
        raise UnsafeURLError("the endpoint must be on the public internet")
    return addrs


def check_custom_base_url(url: str, resolve: bool = True) -> str:
    """
    Validate a visitor-supplied base URL; return it normalized.
    resolve=False does the syntax checks only (safe on the event loop);
    resolve=True also does the DNS check (blocking).
    """
    url = (url or "").strip()
    if not url:
        raise UnsafeURLError("a base URL is required for a custom endpoint")
    if len(url) > 300:
        raise UnsafeURLError("base URL too long")
    parts = urlsplit(url)
    schemes = {"https", "http"} if _allow_private_urls() else {"https"}
    if parts.scheme not in schemes:
        raise UnsafeURLError("the endpoint must use https")
    if "@" in parts.netloc:
        raise UnsafeURLError("credentials in the URL are not allowed")
    if not parts.hostname:
        raise UnsafeURLError("the URL needs a hostname")
    if parts.query or parts.fragment:
        raise UnsafeURLError("the URL must not have a query string or fragment")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        raise UnsafeURLError("invalid port") from None
    if resolve:
        resolve_public_addresses(parts.hostname, port)
    return url.rstrip("/")


def custom_url_host(url: str) -> str:
    """Hostname only — safe to log (paths can carry tokens)."""
    try:
        return urlsplit(url).hostname or "?"
    except ValueError:
        return "?"


class _PublicOnlyBackend(httpcore.NetworkBackend):
    """Connects only to public addresses, checked at connect time."""

    def __init__(self):
        self._inner = httpcore.SyncBackend()

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        try:
            addrs = resolve_public_addresses(host, port)
        except UnsafeURLError as e:
            raise httpcore.ConnectError(str(e)) from None
        last_exc: Exception = httpcore.ConnectError("no address to connect to")
        for addr in addrs:
            try:
                return self._inner.connect_tcp(addr, port, timeout=timeout,
                                               local_address=local_address,
                                               socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                last_exc = e
        raise last_exc

    def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def public_only_transport() -> httpx.BaseTransport:
    transport = httpx.HTTPTransport()
    # httpx has no public hook for the network backend; this swaps the one
    # on its connection pool (pinned httpx 0.28 / httpcore 1.x — covered by
    # tests/test_grounding_and_byok.py, which fails loudly if this breaks).
    transport._pool._network_backend = _PublicOnlyBackend()
    return transport


def llm_http_client(custom: bool, timeout: float) -> httpx.Client:
    """httpx client for LLM calls: never follows redirects; custom URLs are
    public-only and ignore proxy env vars (a proxy would bypass the check)."""
    if custom:
        return DefaultHttpxClient(transport=public_only_transport(), trust_env=False,
                                  follow_redirects=False, timeout=timeout)
    return DefaultHttpxClient(follow_redirects=False, timeout=timeout)


# ---------------------------------------------------------------------------
# Grounding guardrails — after the model's final answer, check every dollar
# figure and every regulation citation against this turn's tool results.
# Deterministic, no extra LLM call unless the numbers fail (then ONE retry).
# ---------------------------------------------------------------------------

# $1,222  $0.70  $416.00  $ 64
_DOLLAR_RE = re.compile(r"\$\s?((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)")
# JTR sections are 6 digits with a leading 0 (010204, 020210, 032301)
_JTR_SECTION_RE = re.compile(r"(?<![\d$.,/-])(0[1-9]\d{4})(?![\d,.]?\d)")
_PARA_RE    = re.compile(r"\bpara(?:graph)?s?\.?\s*(\d{1,2}[-\u2013]\d{1,3})(?:[a-z]\b)?", re.I)
_TABLE_RE   = re.compile(r"\btable\s+(\d{1,2}[-\u2013]\d{1,3})\b", re.I)
_CHAPTER_RE = re.compile(r"\b(?:chapter|ch\.)\s*(\d{1,2})\b", re.I)


def extract_dollar_amounts(text: str) -> list[tuple[str, float]]:
    """[(as written, value)] for every $ amount in text."""
    return [(m.group(0).replace(" ", ""), float(m.group(1).replace(",", "")))
            for m in _DOLLAR_RE.finditer(text or "")]


def _collect_numbers(obj: Any, out: list[float]):
    """Every numeric value in parsed tool JSON, plus $ amounts and bare
    numbers inside strings (e.g. summaries, regulation text)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        out.extend(v for _, v in extract_dollar_amounts(obj))
        bare = obj.strip().lstrip("$").replace(",", "")
        if re.fullmatch(r"\d+(?:\.\d+)?", bare):
            out.append(float(bare))
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numbers(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numbers(v, out)


def tool_numbers(tool_results: list[dict]) -> list[float]:
    nums: list[float] = []
    for t in tool_results:
        try:
            _collect_numbers(json.loads(t["result"]), nums)
        except (json.JSONDecodeError, TypeError):
            nums.extend(v for _, v in extract_dollar_amounts(str(t.get("result", ""))))
    return nums


def _amount_matches(written: str, value: float, known: list[float]) -> bool:
    # Cent tolerance; a figure written without cents may also be the tool's
    # value rounded to whole dollars ($416 for 415.75 — common in prose).
    whole = "." not in written
    for k in known:
        if abs(k - value) < 0.005:
            return True
        if whole and int(k + 0.5) == int(value):
            return True
    return False


def extract_citations(text: str) -> list[tuple[str, str, str]]:
    """[(as written, kind, identifier)] for section/para/table/chapter refs."""
    found: list[tuple[str, str, str]] = []
    for m in _JTR_SECTION_RE.finditer(text or ""):
        found.append((m.group(1), "section", m.group(1)))
    for m in _PARA_RE.finditer(text or ""):
        found.append((m.group(0).strip(), "para", m.group(1).replace("\u2013", "-")))
    for m in _TABLE_RE.finditer(text or ""):
        found.append((m.group(0).strip(), "table", m.group(1).replace("\u2013", "-")))
    for m in _CHAPTER_RE.finditer(text or ""):
        found.append((m.group(0).strip(), "chapter", m.group(1)))
    seen, unique = set(), []
    for c in found:
        if c[0].lower() not in seen:
            seen.add(c[0].lower())
            unique.append(c)
    return unique


def regulation_haystack(tool_results: list[dict]) -> str:
    """Text, section titles and sources of this turn's regulation searches."""
    parts = []
    for t in tool_results:
        if t.get("name") != "search_regulations":
            continue
        try:
            r = json.loads(t["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        for c in r.get("results", []) or []:
            parts += [str(c.get("text", "")), str(c.get("section", "")), str(c.get("source", ""))]
    return "\n".join(parts).replace("\u2013", "-")


def _citation_found(kind: str, ident: str, hay: str) -> bool:
    if kind == "section":
        return re.search(rf"(?<!\d){ident}(?!\d)", hay) is not None
    if kind in ("para", "table"):
        return re.search(rf"(?<![\d-]){re.escape(ident)}(?!\d)", hay) is not None
    # chapter N: named outright, or a para N-x / JTR 0N xxxx section from it
    n = int(ident)
    return bool(re.search(rf"\b(?:chapter|ch\.)\s*{n}\b", hay, re.I)
                or re.search(rf"(?<![\d-]){n}-\d", hay)
                or re.search(rf"(?<!\d){n:02d}\d{{4}}(?!\d)", hay))


def verify_answer(answer: str, tool_results: list[dict], user_message: str = "") -> dict | None:
    """
    Check an answer against this turn's tool results.
    Rule: returns None when the answer has nothing checkable (no dollar
    amounts, no citations — e.g. a greeting or a plain yes/no without a
    section number). Otherwise returns the report with checked=True.
    checked=False is never produced here; it stays in the contract for a
    check that couldn't run.
    Dollar figures the user typed themselves count as known.
    """
    amounts   = extract_dollar_amounts(answer)
    citations = extract_citations(answer)
    if not amounts and not citations:
        return None

    known = tool_numbers(tool_results) + [v for _, v in extract_dollar_amounts(user_message)]
    bad_numbers = list(dict.fromkeys(w for w, v in amounts if not _amount_matches(w, v, known)))

    hay = regulation_haystack(tool_results)
    bad_cites = [w for w, kind, ident in citations if not _citation_found(kind, ident, hay)]

    return {
        "checked":              True,
        "numbers_ok":           not bad_numbers,
        "unverified_numbers":   bad_numbers,
        "citations_ok":         not bad_cites,
        "unverified_citations": bad_cites,
    }


class ReActAgent:
    """
    Stateful ReAct agent. One instance per user session.
    Call .chat(message) → returns a streaming generator of tokens.
    Call .reset() to start a new conversation.
    """

    def __init__(self):
        self._clients: dict[tuple[str, str, bool], OpenAI] = {}
        self._use_llm(None)
        self._history: list[dict] = []   # conversation turns
        self._profile: dict       = {}   # set per turn by chat()
        self._user_message: str   = ""   # this turn's raw message
        self.last_form: dict | None = None   # fill_form result from the latest turn
        # Every tool call of the latest turn: {name, args, result (raw string)}
        self.last_tool_results: list[dict] = []
        # Grounding report for the latest answer (see verify_answer), or None
        self.last_verification: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, profile: dict | None = None,
             llm: dict | None = None) -> Generator[str, None, None]:
        """
        Process a user message. Yields response tokens as they stream.
        Yields tool observation lines (prefixed with ⚙) during reasoning.
        profile is the requesting soldier's profile, used for context and form fills.
        llm is an optional {base_url, api_key, model, provider} for the
        visitor's own key; None uses the server's configured provider.
        """
        self._profile  = profile or {}
        self._use_llm(llm)
        self.last_form = None
        self.last_tool_results = []
        self.last_verification = None
        self._user_message = user_message
        # Enrich message with extracted structured fields before LLM sees it
        enriched = self._enrich_message(user_message)
        self._add_message("user", enriched)

        try:
            yield from self._react_loop()
        except AuthenticationError:
            log.warning("LLM provider rejected the API key")
            self.last_verification = None
            error_msg = "\n[Error: the AI provider rejected the API key. If you're using your own key, check it in AI settings.]"
            self._add_message("assistant", error_msg)
            yield error_msg
        except Exception as exc:
            # Details go to the server log only — the raw exception can carry
            # API error bodies or internals that shouldn't reach the browser.
            # A custom endpoint's URL can carry secrets in its path, so for
            # those only the exception type and host are logged.
            self.last_verification = None
            if self._custom:
                log.warning(f"Error in reasoning loop | custom host={custom_url_host(self._base_url)} | {type(exc).__name__}")
            else:
                log.exception("Error in reasoning loop")
            error_msg = "\n[Error: the assistant hit a problem answering this. Please try again.]"
            self._add_message("assistant", error_msg)
            yield error_msg

    def _use_llm(self, llm: dict | None):
        """Select the client and model for this turn."""
        cfg = llm or {"base_url": LLM_BASE_URL, "api_key": LLM_API_KEY, "model": LLM_MODEL}
        # Visitor-supplied endpoints go through the public-only transport.
        # app.py has already validated the URL (incl. DNS) for this turn; the
        # transport re-checks every connection, so a later DNS change can't
        # redirect a cached client to an internal address.
        custom = cfg.get("provider") == "custom"
        key = (cfg["base_url"], cfg["api_key"], custom)
        if key not in self._clients:
            if len(self._clients) >= 3:
                for old in self._clients.values():
                    old.close()
                self._clients.clear()
            # Bounded timeout + a single retry so a hung API call fails visibly
            # instead of holding the request open indefinitely. Redirects are
            # never followed (the SDK's default client would).
            self._clients[key] = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"],
                                        timeout=LLM_TIMEOUT, max_retries=1,
                                        http_client=llm_http_client(custom, LLM_TIMEOUT))
        self._client   = self._clients[key]
        self._model    = cfg["model"]
        self._custom   = custom
        self._base_url = cfg["base_url"]
        # OpenAI's newer models take max_completion_tokens and only the
        # default temperature; reasoning models also need a bigger budget.
        if custom_url_host(cfg["base_url"]) == "api.openai.com":
            self._gen_params = {"max_completion_tokens": 2048}
        else:
            self._gen_params = {"temperature": 0.1, "max_tokens": 512}

    def _enrich_message(self, message: str) -> str:
        """
        Extract structured fields from natural language before the LLM reasons.
        Injects them as hints so the model passes correct args to tools.
        """
        import re
        hints = []

        # Extract explicit mileage: "370 miles", "~370 mi"
        miles_match = re.search(r"[~≈]?\s*\b(\d{2,4})\s*[-\s]?mi(?:les?)?\b", message, re.IGNORECASE)
        if miles_match:
            hints.append(f"one_way_miles={miles_match.group(1)}")
        else:
            # Look up known installation pair distances
            from agents.tools import KNOWN_DISTANCES, canonical_installation_text
            msg_lower = canonical_installation_text(message.lower())
            for (orig, dest), miles in KNOWN_DISTANCES.items():
                if orig in msg_lower and dest in msg_lower:
                    hints.append(f"one_way_miles={miles} (looked up: {orig}→{dest})")
                    break

        # Extract number of days
        days_match = re.search(r"(\d{1,2})\s*[-\s]?days?", message, re.IGNORECASE)
        if days_match:
            hints.append(f"travel_days={days_match.group(1)}")

        # Extract travel mode
        if re.search(r"\bPOV\b|personal\s+vehicle|own\s+car|driving\s+her|driving\s+his|driving\s+my", message, re.IGNORECASE):
            hints.append("travel_mode=POV")
        elif re.search(r"\bflight\b|\bfly\b|\bplane\b|\bairfare\b", message, re.IGNORECASE):
            hints.append("travel_mode=PLANE")
        elif re.search(r"\brental\b|\brent\s+a\s+car\b", message, re.IGNORECASE):
            hints.append("travel_mode=RENTAL")

        if not hints:
            return message

        return f"{message}\n[Extracted context: {', '.join(hints)}]"

    def reset(self):
        """Clear conversation history."""
        self._history = []
        self.last_form = None
        self.last_tool_results = []
        self.last_verification = None

    def get_history(self) -> list[dict]:
        return list(self._history)

    # ------------------------------------------------------------------
    # ReAct loop
    # ------------------------------------------------------------------

    def _react_loop(self) -> Generator[str, None, None]:
        messages = self._build_messages()
        iterations = 0

        while iterations < MAX_ITERATIONS:
            iterations += 1

            # Call LLM — non-streaming for tool decision phase
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                **self._gen_params,
            )

            choice      = response.choices[0]
            finish      = choice.finish_reason
            msg         = choice.message

            # --- Case 1: LLM wants to call tools ---
            if finish == "tool_calls" and msg.tool_calls:
                # Add assistant's tool-call intent to message history
                messages.append({
                    "role":       "assistant",
                    "content":    msg.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                })

                # Execute each tool call, yield observation to UI
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    try:
                        tool_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    # Yield a reasoning step visible in the UI
                    observation_header = f"\n⚙ {self._tool_label(tool_name, tool_args)}\n"
                    yield observation_header

                    # Execute tool
                    result_str = execute_tool(tool_name, tool_args, self._profile)
                    self.last_tool_results.append(
                        {"name": tool_name, "args": tool_args, "result": result_str})
                    if tool_name == "fill_form":
                        try:
                            form = json.loads(result_str)
                            if form.get("success"):
                                self.last_form = form
                        except json.JSONDecodeError:
                            pass

                    # Yield a short summary of the result
                    yield self._summarize_tool_result(tool_name, result_str) + "\n"

                    # Add tool result as observation for LLM
                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result_str,
                    })

                # Continue loop — LLM will now reason with the observations
                continue

            # --- Case 2: LLM is ready to give a final answer ---
            final_text = msg.content or ""
            if not final_text.strip():
                final_text = "I was unable to generate a response. Please try rephrasing your question."
            else:
                final_text = self._ground(messages, final_text)

            # Save to history (trimmed to avoid context bloat)
            self._add_message("assistant", final_text)
            self._trim_history()

            # Stream the final answer token by token
            yield "\n"
            yield from self._stream_text(final_text)
            return

        # Safety: exceeded max iterations — force a final answer
        yield "\n⚠ Reached reasoning limit. Generating best available answer...\n\n"
        messages.append({
            "role":    "user",
            "content": "Based on everything gathered so far, provide your best answer now.",
        })
        # Non-streaming so the answer can be grounding-checked before it's shown
        fallback = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            **self._gen_params,
        )
        full = fallback.choices[0].message.content or ""
        if full.strip():
            full = self._ground(messages, full)
        self._add_message("assistant", full)
        yield full

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------

    def _ground(self, messages: list[dict], answer: str) -> str:
        """
        Verify the answer against this turn's tool results. If a dollar figure
        isn't backed by a tool, ask the model ONCE to restate using only tool
        figures, then re-check. Returns the answer to show; the report lands
        on self.last_verification (unverified figures stay flagged there).
        """
        report = verify_answer(answer, self.last_tool_results, self._user_message)
        if report is None or report["numbers_ok"]:
            self.last_verification = report
            return answer

        log.info(f"grounding: unverified figures {report['unverified_numbers']} — retrying once")
        if self.last_tool_results:
            hint = "Restate your answer using only figures that appear in this turn's tool results."
        else:
            hint = ("No tool was called this turn, so these figures are unsupported. "
                    "Restate your answer without them (or say a calculation is needed).")
        retry_messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                f"[Verification] These dollar figures in your answer do not match any tool output: "
                f"{', '.join(report['unverified_numbers'])}. {hint} Keep the same format; do not mention this check.")},
        ]
        try:
            # No tools offered: the retry must be a text answer
            resp = self._client.chat.completions.create(
                model=self._model, messages=retry_messages, **self._gen_params)
            retried = (resp.choices[0].message.content or "").strip()
        except AuthenticationError:
            raise
        except Exception as exc:
            log.warning(f"grounding retry failed: {type(exc).__name__}")
            retried = ""

        if retried:
            report2 = verify_answer(retried, self.last_tool_results, self._user_message)
            # Keep the retry unless it made the numbers worse
            if report2 is None or len(report2["unverified_numbers"]) <= len(report["unverified_numbers"]):
                answer, report = retried, report2
        if report is not None and not report["numbers_ok"]:
            log.warning(f"grounding: answer still has unverified figures {report['unverified_numbers']}")
        self.last_verification = report
        return answer

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Build the full message list: system + profile context + history."""
        # Dates the model infers ("next month", "June 3") must land in the
        # right year — without this it guessed past years.
        system_content = SYSTEM_PROMPT + f"\nToday's date: {datetime.now():%d %B %Y}"
        if self._profile:
            # Compact profile — only key fields to save tokens
            compact = {k: v for k, v in self._profile.items()
                       if k in ("name_last_first", "rank", "unit", "installation")}
            if compact:
                system_content += f"\nSoldier: {json.dumps(compact)}"

        messages = [{"role": "system", "content": system_content}]
        window = self._history[-HISTORY_WINDOW:]
        # After an error or timeout the window can start mid-pair; the
        # conversation sent to the model should open with a user turn.
        while window and window[0]["role"] != "user":
            window = window[1:]
        messages.extend(window)
        return messages

    def _add_message(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def _trim_history(self):
        """Keep history within window to avoid context overflow."""
        if len(self._history) > HISTORY_WINDOW:
            # Always keep the first message for context, trim the middle
            self._history = self._history[-HISTORY_WINDOW:]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stream_text(self, text: str) -> Generator[str, None, None]:
        """Simulate streaming by yielding the full text at once.
        Replace with actual streaming call if latency becomes a concern."""
        yield text

    def _tool_label(self, name: str, args: dict) -> str:
        """Human-readable label for a tool call shown in the UI."""
        if name == "search_regulations":
            domain = f" [{args.get('domain', 'all')}]" if args.get("domain") else ""
            return f"Searching regulations{domain}: \"{args.get('query', '')[:60]}\""
        if name == "get_per_diem":
            return f"Getting per diem rates for {args.get('city', '')}, {args.get('state', '')}"
        if name == "calculate_travel_cost":
            return (
                f"Calculating travel cost: {args.get('destination_city', '')} "
                f"({args.get('travel_days', '?')} days, {args.get('travel_mode', 'POV')})"
            )
        if name == "fill_form":
            return f"Filling {args.get('form_name', 'form')}..."
        return f"Calling {name}"

    def _summarize_tool_result(self, name: str, result_str: str) -> str:
        """One-line summary of tool result for the UI reasoning trace."""
        try:
            r = json.loads(result_str)
        except Exception:
            return "  → result received"

        if not r.get("success"):
            return f"  → Error: {r.get('error', 'unknown')}"

        if name == "search_regulations":
            results = r.get("results", [])
            if results:
                top = results[0]
                return f"  → Found: {top['source']} | {top['section'][:50]} (score {top['score']})"
            return "  → No results found"

        if name == "get_per_diem":
            return (
                f"  → {r.get('city', '')}, {r.get('state', '')}: "
                f"Lodging ${r.get('lodging', '?')}/night | M&IE ${r.get('mie', '?')}/day"
            )

        if name == "calculate_travel_cost":
            return f"  → {r.get('summary', 'calculated')}"

        if name == "fill_form":
            n_filled  = len(r.get("filled_fields", {}))
            n_missing = len(r.get("missing_fields", []))
            pdf_note  = "PDF ready" if r.get("pdf_generated") else "text summary saved"
            return f"  → {r.get('form_name', '')}: {n_filled} fields filled, {n_missing} missing — {pdf_note}"

        return "  → done"


# ---------------------------------------------------------------------------
# Standalone test — python3 agents/react_agent.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    agent = ReActAgent()

    tests = [
        (
            "GTC question",
            "Is the government travel charge card mandatory for my TDY trip?",
        ),
        (
            "TDY cost — SPC Rivera scenario",
            "I need to send SPC Rivera to Fort Moore, Georgia for 5 days starting July 10. She's driving her POV from Fort Liberty.",
        ),
        (
            "Leave accrual",
            "How many days of leave do I accrue per month on active duty?",
        ),
    ]

    for label, query in tests:
        print(f"\n{'='*60}")
        print(f"TEST: {label}")
        print(f"Q: {query}")
        print(f"{'='*60}")
        print("A: ", end="")
        for token in agent.chat(query):
            print(token, end="", flush=True)
        print()
        agent.reset()   # fresh conversation for each test
