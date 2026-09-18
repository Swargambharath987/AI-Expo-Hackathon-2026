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

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv
from openai import OpenAI

from agents.tools import TOOL_SCHEMAS, execute_tool

load_dotenv()

os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
logging.getLogger("chromadb").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*existing embedding ID.*")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_BASE_URL   = os.getenv("LLM_BASE_URL",  "https://api.anthropic.com/v1")
LLM_MODEL      = os.getenv("LLM_MODEL",     "claude-sonnet-4-6")
LLM_API_KEY    = os.getenv("LLM_API_KEY",   "")
MAX_ITERATIONS = 5      # max tool calls per turn before forcing a final answer
HISTORY_WINDOW = 8      # max messages kept in context (sliding window)
RAG_TOP_K      = 2      # chunks per search — keeps context under 4096 tokens
PROFILE_FILE   = Path(__file__).parent.parent / "profile.json"

# ---------------------------------------------------------------------------
# System prompt — tells the model what it is, what docs exist, how to reason
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are Duty Line, an AI assistant for US military personnel.
Mission: reduce bureaucratic burden — help soldiers navigate regulations, plan TDY travel, manage leave, and complete forms.

## Documents searchable via search_regulations:
- JTR: all TDY travel — per diem, lodging, mileage, GTC, entitlements
- AR 600-8-10: Army leave — accrual, DA 31, leave types, deployment leave
- AR 623-3 / AFI 36-2406 / BUPERSINST 1610: evaluation reports by branch
- MILPERSMAN 1050 / AFI 36-3003: Navy / Air Force leave
- DoD FMR Vol 7A: military pay

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
2. YES/NO questions: first call the relevant tool, then answer "Yes" or "No" on line 1. One sentence of citation.
3. TDY cost: show a brief dollar breakdown (lodging / meals / mileage / total). Never estimate.
4. Forms: call fill_form. Say "Your form is ready for download." Nothing else about the form.
5. Citations: end every regulation answer with "— [document name], [section]"
6. NEVER mention file paths, file names, output directories, file formats, or internal system details.

## Response format -- follow these examples exactly:

TDY cost example:
"**TDY Cost -- Fort Moore, GA (5 days, POV)**
- Lodging: $416 (4 nights x $104)
- Meals: $288 (3 full days x $64, 2 travel days x $48)
- Mileage: $518 (370 mi x $0.70 x 2)
- **Total: $1,222**
Your DD Form 1610 (Request and Authorization for TDY Travel) is ready for download."

Regulation example:
"Yes, the GTC is mandatory for all TDY travel expenses unless a specific exemption applies. -- JTR 010204"

Leave example:
"You accrue 2.5 days per month (30 days/year). Max carry-over is 60 days at end of FY. -- AR 600-8-10, para 3-1"

Rules:
- Never repeat the question. Never add disclaimers or please note.
- Never mention file paths, file formats, or system internals.
- Always include the dollar amounts from tool results for TDY queries.
- Always use full form titles: "DD Form 1610 (Request and Authorization for TDY Travel)", "DA Form 31 (Request and Authority for Leave)", "DA Form 4856 (Developmental Counseling Form)", "DA Form 4187 (Personnel Action)".
- Keep total response under 8 lines."""


class ReActAgent:
    """
    Stateful ReAct agent. One instance per user session.
    Call .chat(message) → returns a streaming generator of tokens.
    Call .reset() to start a new conversation.
    """

    def __init__(self):
        self._client  = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self._history: list[dict] = []   # conversation turns
        self._profile: dict       = self._load_profile()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Process a user message. Yields response tokens as they stream.
        Yields tool observation lines (prefixed with ⚙) during reasoning.
        """
        # Enrich message with extracted structured fields before LLM sees it
        enriched = self._enrich_message(user_message)
        self._add_message("user", enriched)

        try:
            yield from self._react_loop()
        except Exception as e:
            error_msg = f"\n[Error in reasoning loop: {e}]"
            self._add_message("assistant", error_msg)
            yield error_msg

    def _enrich_message(self, message: str) -> str:
        """
        Extract structured fields from natural language before the LLM reasons.
        Injects them as hints so the model passes correct args to tools.
        """
        import re
        hints = []

        # Extract explicit mileage: "370 miles", "~370 mi"
        miles_match = re.search(r"[~≈]?\s*(\d{2,4})\s*[-\s]?mi(?:les?)?", message, re.IGNORECASE)
        if miles_match:
            hints.append(f"one_way_miles={miles_match.group(1)}")
        else:
            # Look up known installation pair distances
            from agents.tools import KNOWN_DISTANCES
            msg_lower = message.lower()
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
        """Clear conversation history, keep profile."""
        self._history = []

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
                model=LLM_MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=512,
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
                    result_str = execute_tool(tool_name, tool_args)

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
        fallback = self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
            stream=True,
        )
        full = ""
        for chunk in fallback:
            token = chunk.choices[0].delta.content or ""
            full += token
            yield token
        self._add_message("assistant", full)

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self) -> list[dict]:
        """Build the full message list: system + profile context + history."""
        system_content = SYSTEM_PROMPT
        if self._profile:
            # Compact profile — only key fields to save tokens
            compact = {k: v for k, v in self._profile.items()
                       if k in ("name_last_first", "rank", "unit", "installation")}
            if compact:
                system_content += f"\nSoldier: {json.dumps(compact)}"

        messages = [{"role": "system", "content": system_content}]
        window = self._history[-HISTORY_WINDOW:] if len(self._history) > HISTORY_WINDOW else self._history
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

    def _load_profile(self) -> dict:
        if PROFILE_FILE.exists():
            try:
                return json.loads(PROFILE_FILE.read_text())
            except Exception:
                pass
        return {}

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
