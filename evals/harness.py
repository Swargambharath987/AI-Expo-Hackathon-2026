"""
In-process driver for the full stack: FastAPI app -> session/tab agent ->
ReAct loop -> real LLM -> real tools.

Instrumentation (no app code is modified; everything is wrapped at runtime):
  * agents.react_agent.execute_tool is wrapped (that is the name the ReAct loop
    resolves at call time) to record every raw call: name, args, parsed result.
  * openai Completions.create is wrapped to count LLM calls and token usage.
Recording uses a ContextVar so concurrent cases don't mix (Starlette's
threadpool / anyio propagate contextvars into worker threads; the harness also
cross-checks recorded tool calls against the API's own tool_calls list).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from dataclasses import dataclass, field

_REC: contextvars.ContextVar["Recorder | None"] = contextvars.ContextVar("dl_eval_rec", default=None)


@dataclass
class Recorder:
    tool_calls: list[dict] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)


_installed = False


def install_instrumentation():
    """Wrap execute_tool and the OpenAI client once per process."""
    global _installed
    if _installed:
        return
    import agents.react_agent as ra
    from openai.resources.chat.completions import Completions

    real_execute = ra.execute_tool

    def recording_execute(name, arguments, profile=None):
        t0 = time.time()
        out = real_execute(name, arguments, profile)
        rec = _REC.get()
        if rec is not None:
            try:
                parsed = json.loads(out)
            except Exception:
                parsed = {"_unparsed": out}
            rec.tool_calls.append({"name": name, "args": arguments, "result": parsed,
                                   "ms": int((time.time() - t0) * 1000)})
        return out

    ra.execute_tool = recording_execute

    real_create = Completions.create

    def counting_create(self, *args, **kwargs):
        t0 = time.time()
        resp = real_create(self, *args, **kwargs)
        rec = _REC.get()
        if rec is not None:
            usage = getattr(resp, "usage", None)
            rec.llm_calls.append({
                "model": kwargs.get("model"),
                "stream": bool(kwargs.get("stream")),
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "ms": int((time.time() - t0) * 1000),
            })
        return resp

    Completions.create = counting_create
    _installed = True


def load_app():
    """Import the app with eval-friendly limits (rate limits would throttle a run)."""
    import app as appmod
    appmod.RATE_PER_IP_MINUTE = 10_000
    appmod.RATE_PER_IP_DAY = 100_000
    appmod.RATE_GLOBAL_DAY = 100_000
    install_instrumentation()
    return appmod


async def run_conversation(appmod, case: dict, profile: dict) -> dict:
    """Run every turn of one case in one cookie-scoped session. Returns the trace."""
    import httpx

    rec = Recorder()
    token = _REC.set(rec)
    turns_out = []
    try:
        transport = httpx.ASGITransport(app=appmod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://evals.local",
                                     timeout=300) as client:
            for i, turn in enumerate(case["turns"]):
                n_tools_before = len(rec.tool_calls)
                n_llm_before = len(rec.llm_calls)
                t0 = time.time()
                body = {"message": turn["message"], "tab": case.get("tab", "travel"),
                        "profile": case.get("profile", profile)}
                try:
                    r = await client.post("/api/chat", json=body)
                    status = r.status_code
                    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                except Exception as e:  # transport-level failure
                    status, data = -1, {"error": f"harness_exception: {e!r}"}
                turns_out.append({
                    "index": i,
                    "message": turn["message"],
                    "http_status": status,
                    "response": data.get("response", ""),
                    "api_tool_calls": data.get("tool_calls", []),
                    "form_output": data.get("form_output"),
                    "error": data.get("error") or (None if status == 200 else f"http_{status}"),
                    "tool_calls": rec.tool_calls[n_tools_before:],
                    "llm_calls": rec.llm_calls[n_llm_before:],
                    "seconds": round(time.time() - t0, 1),
                })
    finally:
        _REC.reset(token)
    return {"turns": turns_out}


async def run_many(appmod, jobs: list[tuple[dict, int]], profile: dict, concurrency: int,
                   on_done=None) -> list[tuple[dict, int, dict]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case, rep):
        async with sem:
            trace = await run_conversation(appmod, case, profile)
            if on_done:
                on_done(case, rep, trace)
            return case, rep, trace

    return await asyncio.gather(*(one(c, r) for c, r in jobs))


def llm_env_summary() -> dict:
    return {"base_url": os.getenv("LLM_BASE_URL", "https://api.anthropic.com/v1"),
            "model": os.getenv("LLM_MODEL", "claude-sonnet-4-6")}
