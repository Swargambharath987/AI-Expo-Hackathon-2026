"""
Optional LLM-as-judge (--judge). MODEL-JUDGED, not deterministic: use it as a
secondary signal next to the deterministic grounding checks, never instead.

Uses the same OpenAI-compatible endpoint/model the app is configured with
(LLM_BASE_URL / LLM_MODEL / LLM_API_KEY), so by default Claude grades Claude —
a known self-preference bias; point JUDGE_MODEL at a different model to reduce it.
"""

from __future__ import annotations

import json
import os
import re

from evals.common import RUBRIC_FILE

MAX_EVIDENCE_CHARS = 14_000


def _client():
    from openai import OpenAI
    return OpenAI(base_url=os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1"),
                  api_key=os.getenv("LLM_API_KEY", ""), timeout=60, max_retries=1)


def judge_turn(message: str, tool_calls: list[dict], answer: str, client=None) -> dict:
    client = client or _client()
    rubric = RUBRIC_FILE.read_text()
    evidence = []
    for c in tool_calls:
        r = c.get("result") or {}
        if c["name"] == "search_regulations":
            r = {"results": [{"source": x.get("source"), "text": x.get("text", "")[:3000]}
                             for x in r.get("results", [])]}
        evidence.append({"tool": c["name"], "args": c["args"], "result": r})
    ev = json.dumps(evidence, indent=1, default=str)[:MAX_EVIDENCE_CHARS]
    prompt = (f"{rubric}\n\n=== USER MESSAGE ===\n{message}\n\n=== TOOL RESULTS ===\n{ev or '(none)'}"
              f"\n\n=== ASSISTANT ANSWER ===\n{answer}\n")
    resp = client.chat.completions.create(
        model=os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=400,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    usage = getattr(resp, "usage", None)
    return {
        "faithfulness": out.get("faithfulness"), "relevance": out.get("relevance"),
        "unsupported_claims": out.get("unsupported_claims", []), "rationale": out.get("rationale", text[:300]),
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
    }
