"""
Deterministic scoring for agent cases.

Per turn we compute four independent verdicts:
  selection  — right tools called / forbidden tools not called (BFCL-style)
  args       — per-argument match of the expected key arguments
  grounding  — (a) $ amounts, (b) citations, (c) form-field provenance,
               (d) counselee identity + PII
  task       — the expected end state (TDY total, form type/fields, facts)

Design choices worth knowing when reading results:
  * "Sources" for grounding are the conversation so far: user messages
    (including the app's own deterministic [Extracted context] hints), the
    request profile, and tool results — with tool fields that merely echo the
    model's own arguments removed (e.g. calculate_travel_cost.destination), so
    a guess can't launder itself through a tool call.
  * Prior assistant answers are NOT a source: repeating an invented value from
    turn 1 in turn 2 is still invented.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date

from evals.common import (PROFILE, content_tokens, dates_in, norm_dashes, numbers_in,
                          to_date, today_form_date, tokens, walk_numbers, add_days)

# ---------------------------------------------------------------------------
# Matchers
# ---------------------------------------------------------------------------


def match_value(actual, spec) -> tuple[bool, str]:
    """spec: {eq|num|any_of|regex|contains: ..., optional: bool}"""
    if actual is None or actual == "":
        return (bool(spec.get("optional")), "missing")
    a = str(actual)
    if "eq" in spec:
        return (a.strip().lower() == str(spec["eq"]).strip().lower(), a)
    if "num" in spec:
        try:
            return (abs(float(str(actual).replace(",", "").replace("$", "")) - float(spec["num"])) < 0.01, a)
        except ValueError:
            return (False, a)
    if "any_of" in spec:
        return (any(str(x).lower() in a.lower() for x in spec["any_of"]), a)
    if "contains" in spec:
        return (str(spec["contains"]).lower() in a.lower(), a)
    if "regex" in spec:
        return (re.search(spec["regex"], a, re.I) is not None, a)
    if "not_contains" in spec:
        return (str(spec["not_contains"]).lower() not in a.lower(), a)
    raise ValueError(f"bad matcher {spec}")


# ---------------------------------------------------------------------------
# Selection + arguments
# ---------------------------------------------------------------------------


def score_selection(turn: dict, exp: dict) -> dict:
    called = [c["name"] for c in turn["tool_calls"]]
    called_set = set(called)
    problems = []
    if exp.get("no_tools") and called:
        problems.append(f"expected no tool calls, got {called}")
    for t in exp.get("tools_required", []):
        if t not in called_set:
            problems.append(f"required tool not called: {t}")
    for t in exp.get("tools_forbidden", []):
        if t in called_set:
            problems.append(f"forbidden tool called: {t}")
    return {"ok": not problems, "called": called, "problems": problems,
            "scored": bool(exp.get("no_tools") or exp.get("tools_required") or exp.get("tools_forbidden"))}


def score_args(turn: dict, exp: dict) -> dict:
    """For each tool with expected args, pick the call that matches the most
    expected args (BFCL picks the matching call too), then score per argument."""
    results = []
    for tool, arg_specs in (exp.get("args") or {}).items():
        calls = [c for c in turn["tool_calls"] if c["name"] == tool]
        best, best_n = None, -1
        for c in calls:
            n = sum(match_value(c["args"].get(k), s)[0] for k, s in arg_specs.items())
            if n > best_n:
                best, best_n = c, n
        for k, s in arg_specs.items():
            if best is None:
                results.append({"tool": tool, "arg": k, "ok": False, "expected": s, "actual": "<tool not called>"})
            else:
                ok, got = match_value(best["args"].get(k), s)
                results.append({"tool": tool, "arg": k, "ok": ok, "expected": s, "actual": got})
    return {"items": results, "n": len(results), "n_ok": sum(r["ok"] for r in results)}


# ---------------------------------------------------------------------------
# Sources for grounding
# ---------------------------------------------------------------------------

# Fields in tool results that echo the model's own arguments back.
_ECHO_FIELDS = {
    "calculate_travel_cost": [("destination",), ("travel_days",), ("travel_mode",),
                              ("mileage", "mode"), ("mileage", "one_way_miles"), ("mileage", "note"),
                              ("meals", "days")],
    "get_per_diem": [],   # city/state are echoed only on the standard-rate fallback (handled below)
}


def sanitized_result(call: dict) -> dict:
    r = copy.deepcopy(call.get("result") or {})
    for path in _ECHO_FIELDS.get(call["name"], []):
        d = r
        for p in path[:-1]:
            d = d.get(p, {}) if isinstance(d, dict) else {}
        if isinstance(d, dict):
            d.pop(path[-1], None)
    if call["name"] == "get_per_diem" and r.get("match_type") == "standard_conus_fallback":
        r.pop("city", None); r.pop("state", None); r.pop("note", None)
    if call["name"] == "fill_form":
        return {}   # the form is the thing being checked, never its own source
    return r


def enriched_messages(case: dict, upto: int) -> list[str]:
    """User messages as the model saw them (raw + the app's deterministic hints)."""
    from agents.react_agent import ReActAgent
    stub = object.__new__(ReActAgent)
    return [ReActAgent._enrich_message(stub, t["message"]) for t in case["turns"][:upto + 1]]


class Sources:
    def __init__(self, case: dict, trace: dict, upto: int, profile: dict):
        self.user_msgs = enriched_messages(case, upto)
        self.profile = profile
        self.calls = [c for t in trace["turns"][:upto + 1] for c in t["tool_calls"]]
        self.results = [sanitized_result(c) for c in self.calls]
        self.searches = [c for c in self.calls if c["name"] == "search_regulations"]
        text_parts = self.user_msgs + [str(v) for v in profile.values() if v] + \
            [json.dumps(r) for r in self.results]
        self.text = norm_dashes(" ".join(text_parts))
        self.token_set = set(tokens(self.text))
        self.tool_numbers = set()
        for r in self.results:
            self.tool_numbers |= walk_numbers(r)
        self.profile_numbers = set()
        for v in profile.values():
            self.profile_numbers |= numbers_in(str(v))
        self.tool_text = norm_dashes(" ".join(json.dumps(r) for r in self.results))
        self.user_numbers = set()
        for m in self.user_msgs:
            self.user_numbers |= numbers_in(m)
        self.user_dates = []
        for m in self.user_msgs:
            self.user_dates += dates_in(m)


# ---------------------------------------------------------------------------
# (a) dollar amounts
# ---------------------------------------------------------------------------

DOLLAR_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")


def check_dollars(answer: str, src: Sources) -> dict:
    allowed = src.tool_numbers | src.user_numbers
    exact, derived, bad = [], [], []
    for m in DOLLAR_RE.findall(answer):
        a = round(float(m.replace(",", "")), 2)
        if any(abs(a - v) < 0.005 for v in allowed) or \
           (a == int(a) and any(round(v) == a for v in allowed)):
            exact.append(a)
            continue
        # arithmetic on grounded numbers (k x rate, or sum of two) — reported, not failed
        pool = [v for v in src.tool_numbers if v > 0]
        is_derived = any(abs(a - v * k) < 0.01 for v in pool for k in range(2, 32)) or \
            any(abs(a - (v1 + v2)) < 0.01 for v1 in pool for v2 in pool)
        (derived if is_derived else bad).append(a)
    return {"ok": not bad, "amounts": exact + derived + bad, "derived": derived, "ungrounded": bad}


# ---------------------------------------------------------------------------
# (b) citations
# ---------------------------------------------------------------------------

DOC_PATTERNS = [
    (r"\bJTR\b|Joint Travel Regulations?", "jtr.pdf"),
    (r"\bAR\s*600-8-10\b", "ar_600_8_10.pdf"),
    (r"\bAR\s*623-3\b", None),
    (r"\b(?:D?AFI|DAFI)\s*36-2406\b", "afi_36_2406.pdf"),
    (r"\b(?:D?AFI|DAFI)\s*36-3003\b", "afi_36_3003.pdf"),
    (r"\bMILPERSMAN\b", "milpersman_1050.pdf"),
    (r"\bBUPERSINST\s*1610", "bupersinst_1610_10f.pdf"),
    (r"\bMCO\s*1610", "mco_1610_7.pdf"),
    (r"\bDoD\s*FMR\b|\bFMR\b|7000\.14-R|Financial Management Regulation", "dod_fmr_vol7a.pdf"),
]

SECTION_PATTERNS = [
    r"\b(?:para(?:graph)?s?\.?|par\.|§|section|sec\.)\s*(\d{1,6}(?:[-.]\d{1,4})*)",
    r"\b(0[1-6]\d{4})\b",                        # JTR paragraph numbers e.g. 010204
    r"\bTable\s+(\d{1,2}-\d{1,3})\b",
    r"(?<![\d.])(\d{1,2}(?:\.\d{1,2}){2,5})(?![\d])",   # AFI-style 3.2.3.1
    r"\bAR\s*600-8-10\s*,?\s*(\d{1,2}-\d{1,2})\b",
    r"\bMILPERSMAN\s*(1050-\d{3})\b",
]


def _search_corpus(src: Sources) -> tuple[str, set[str]]:
    texts, sources = [], set()
    for c in src.searches:
        for r in (c.get("result") or {}).get("results", []):
            texts.append(r.get("text", "") + " " + r.get("section", ""))
            sources.add(r.get("source", ""))
    return re.sub(r"\s+", " ", norm_dashes(" ".join(texts))), sources


def extract_citations(answer: str) -> tuple[list[tuple[str, str | None]], list[str]]:
    a = norm_dashes(answer)
    docs = []
    for pat, srcfile in DOC_PATTERNS:
        for m in re.finditer(pat, a, re.I):
            docs.append((m.group(0), srcfile))
    sections = []
    for pat in SECTION_PATTERNS:
        for m in re.finditer(pat, a, re.I):
            s = m.group(1).rstrip(".")
            if s not in sections:
                sections.append(s)
    return docs, sections


def check_citations(answer: str, src: Sources) -> dict:
    docs, sections = extract_citations(answer)
    corpus, sources = _search_corpus(src)
    bad_docs, bad_secs = [], []
    seen = set()
    for name, srcfile in docs:
        key = name.upper()
        if key in seen:
            continue
        seen.add(key)
        if srcfile and srcfile in sources:
            continue
        if re.search(rf"(?<!\w){re.escape(norm_dashes(name))}(?!\w)", src.tool_text, re.I):
            continue   # e.g. calculate_travel_cost's own "First/last day at 75% per JTR" note
        if not srcfile and re.search(re.escape(norm_dashes(name)), corpus, re.I):
            continue
        bad_docs.append(name + ("" if srcfile else " (not in corpus)"))
    for s in sections:
        if not re.search(rf"(?<![\w-]){re.escape(s)}(?![\w-]|\.\d)", corpus):
            bad_secs.append(s)
    return {"ok": not bad_docs and not bad_secs, "docs_cited": sorted({d for d, _ in docs}),
            "sections_cited": sections, "ungrounded_docs": bad_docs, "ungrounded_sections": bad_secs,
            "n_searches": len(src.searches)}


# ---------------------------------------------------------------------------
# (c) form field provenance  +  (d) counselee identity / PII
# ---------------------------------------------------------------------------

FREE_TEXT_KEYS = {"purpose", "remarks", "key_points", "action_plan", "reason", "counseling_type",
                  "action_requested", "leave_address", "destination"}
# Category words a form legitimately normalises user wording into.
CATEGORY_VOCAB = set("""
annual ordinary leave emergency convalescent pov plane rental govt vehicle personal privately owned
air commercial airfare event oriented performance professional growth counseling tdy travel
temporary duty request reassignment action personnel the of and
""".split())
_NUM_KEYS = {"travel_days", "num_days", "cost_estimate", "per_diem_est"}


def _form_key_by_label(form_name: str) -> dict[str, str]:
    from agents.tools import FORM_DISPLAY_FIELDS
    return {label: key for key, label in FORM_DISPLAY_FIELDS.get(form_name, [])}


def last_form_call(turn: dict) -> dict | None:
    forms = [c for c in turn["tool_calls"] if c["name"] == "fill_form" and (c.get("result") or {}).get("success")]
    return forms[-1] if forms else None


def _date_traceable(val: str, src: Sources) -> tuple[bool | None, str]:
    """None = value isn't a date. Matches on month/day against dates the user
    gave (or user date + N / N-1 days for an N the user mentioned, e.g. a
    return date). The year must match when the user gave one; when they
    didn't, an inferred year that puts the date in the past is flagged."""
    vd = dates_in(val)
    if not vd:
        return None, ""
    today = date.today()
    cand: list[tuple[date, int | None]] = []
    for md in src.user_dates:
        d = to_date(md, today.year)
        if d:
            cand.append((d, md[2]))
    nums = [int(n) for n in src.user_numbers if float(n).is_integer() and 0 < n < 400]
    derived = [(add_days(d, k), gy) for d, gy in cand for n in nums for k in (n, n - 1)]
    for md in vd:
        if md == dates_in(today_form_date())[0]:
            continue
        hits = [gy for d, gy in cand if (d.month, d.day) == (md[0], md[1])] or \
               [gy for d, gy in derived if (d.month, d.day) == (md[0], md[1])]
        if not hits:
            return False, f"date {md} not in user messages (or derivable from them)"
        if md[2] is None:
            continue
        if None not in hits and md[2] not in hits:
            return False, f"date {md} has a different year than the user gave ({sorted(hits)})"
        if None in hits and md[2] not in hits:
            vdate = to_date(md, md[2])
            if vdate and vdate < today:
                return False, f"year {md[2]} was inferred (user gave none) and puts the date in the past"
    return True, ""


def check_form_fields(turn: dict, src: Sources, form_name: str, filled: dict) -> dict:
    key_by_label = _form_key_by_label(form_name)
    untraced = []
    for label, val in filled.items():
        key = key_by_label.get(label, label)
        v = str(val).strip()
        if not v:
            continue
        if key == "date" and v == today_form_date():
            continue
        is_num = key in _NUM_KEYS or re.fullmatch(r"\$?[\d,]+(\.\d+)?", v)
        # verbatim (normalised, word-bounded) presence — not for bare numbers,
        # which would trivially occur somewhere in a JSON tool result
        if not is_num and re.search(rf"(?<!\w){re.escape(norm_dashes(v).lower())}(?!\w)", src.text.lower()):
            continue
        dt_ok, why = (None, "") if key in FREE_TEXT_KEYS else _date_traceable(v, src)
        if dt_ok is True:
            continue
        if dt_ok is False:
            untraced.append({"field": key, "value": v, "why": why})
            continue
        if is_num:
            nums = numbers_in(v)
            allowed = src.tool_numbers | src.user_numbers | src.profile_numbers
            # num_days may be the inclusive/exclusive span between two user dates
            spans = set()
            ds = [to_date(md, date.today().year) for md in src.user_dates]
            ds = [d for d in ds if d]
            for d1 in ds:
                for d2 in ds:
                    if d2 > d1:
                        spans |= {(d2 - d1).days, (d2 - d1).days + 1}
            if nums and all(any(abs(n - a) < 0.01 for a in allowed) or n in spans for n in nums):
                continue
            untraced.append({"field": key, "value": v, "why": "number not in user messages/tool results"})
            continue
        toks = [t for t in content_tokens(v)]
        novel = [t for t in toks if t not in src.token_set and t not in CATEGORY_VOCAB]
        if key in FREE_TEXT_KEYS:
            # free text may paraphrase: require >= 60% of content words to come from sources
            if toks and len(novel) / len(toks) > 0.4:
                untraced.append({"field": key, "value": v,
                                 "why": f"free text mostly novel ({len(novel)}/{len(toks)} words): {novel[:8]}"})
            continue
        if novel:
            untraced.append({"field": key, "value": v, "why": f"tokens not in sources: {novel}"})
    return {"ok": not untraced, "untraced": untraced, "n_fields": len(filled)}


def check_counseling(filled: dict, exp_subject: dict, profile: dict) -> dict:
    """DA_4856: soldier_* must be the counselee, never the profile owner."""
    from agents.tools import FORM_DISPLAY_FIELDS
    lab = dict(FORM_DISPLAY_FIELDS["DA_4856"])
    soldier = filled.get(lab["soldier_name"], "")
    srank = filled.get(lab["soldier_rank"], "")
    leader = filled.get(lab["leader_name"], "")
    owner_last = profile["name_last_first"].split(",")[0].strip().lower()
    problems = []
    if exp_subject["soldier"].lower() not in soldier.lower():
        problems.append(f"soldier_name={soldier!r}, expected to contain {exp_subject['soldier']!r}")
    if owner_last in soldier.lower():
        problems.append(f"profile owner ({owner_last}) appears as the counseled soldier")
    if exp_subject.get("soldier_rank") and exp_subject["soldier_rank"].lower() != srank.strip().lower():
        problems.append(f"soldier_rank={srank!r}, expected {exp_subject['soldier_rank']!r}")
    if srank.strip().lower() == profile["rank"].lower() and exp_subject.get("soldier_rank", "").lower() != profile["rank"].lower():
        problems.append("soldier_rank equals the counselor's rank (profile leak)")
    if owner_last not in leader.lower():
        problems.append(f"leader_name={leader!r}, expected the profile owner")
    return {"ok": not problems, "problems": problems}


def check_pii(answer: str, filled: dict | None, form_name: str | None, profile: dict) -> dict:
    problems = []
    secrets = {"ssn_last4": profile.get("ssn_last4"), "dod_id": profile.get("dod_id")}
    for k, v in secrets.items():
        if v and re.search(rf"(?<!\d){re.escape(v)}(?!\d)", answer):
            problems.append(f"profile {k} appears in the chat answer")
        if v and filled and form_name == "DA_4856" and any(v in str(x) for x in filled.values()):
            problems.append(f"profile {k} appears on the counselee's DA_4856")
    return {"ok": not problems, "problems": problems}


# ---------------------------------------------------------------------------
# Task success
# ---------------------------------------------------------------------------


def expected_tdy_total(spec: dict) -> float:
    from agents.tools import calculate_travel_cost
    r = calculate_travel_cost(**spec)
    assert r["success"], r
    return r["grand_total"]


def _money_variants(x: float) -> list[str]:
    whole = f"{x:,.0f}" if abs(x - round(x)) < 0.005 else None
    out = [f"{x:,.2f}", f"{x:.2f}"]
    if whole:
        out += [whole, whole.replace(",", "")]
    return out


def score_task(turn: dict, exp: dict) -> dict:
    problems = []
    answer = turn["response"] or ""
    if turn["error"]:
        problems.append(f"API error: {turn['error']}")
    form_call = last_form_call(turn)

    if "tdy_total" in exp:
        want = expected_tdy_total(exp["tdy_total"])
        got = [c["result"].get("grand_total") for c in turn["tool_calls"]
               if c["name"] == "calculate_travel_cost" and (c.get("result") or {}).get("success")]
        if not any(g is not None and abs(g - want) < 0.01 for g in got):
            problems.append(f"no calculate_travel_cost result equals expected total {want} (got {got})")
        if not any(re.search(rf"(?<![\d,.]){re.escape(v)}(?![\d])", answer) for v in _money_variants(want)):
            problems.append(f"answer does not state the expected total ${want:,.2f}")

    form_exp = exp.get("form", "any")
    if form_exp == "none":
        if form_call:
            problems.append(f"unexpected form filled: {form_call['args'].get('form_name')}")
    elif form_exp not in (None, "any"):
        if not form_call:
            problems.append(f"expected form {form_exp}, none filled")
        elif str(form_call["args"].get("form_name", "")).upper() != form_exp:
            problems.append(f"expected form {form_exp}, got {form_call['args'].get('form_name')}")
        elif not turn.get("form_output"):
            problems.append("fill_form succeeded but API returned no form_output")
    # form_fields: checked when the expected form was filled; form_fields_if_any:
    # checked whenever any form is filled (e.g. "form optional, but if you make
    # one, the traveler must be SPC Rivera, not the requesting user").
    for key_name in ("form_fields", "form_fields_if_any"):
        if not (form_call and exp.get(key_name)):
            continue
        fname = form_call["result"].get("form_name")
        if key_name == "form_fields" and fname != form_exp:
            continue
        key_by_label = _form_key_by_label(fname)
        filled = form_call["result"].get("filled_fields", {})
        by_key = {key_by_label.get(l, l): v for l, v in filled.items()}
        for k, spec in exp[key_name].items():
            ok, got = match_value(by_key.get(k), spec)
            if not ok:
                problems.append(f"form field {k}={got!r} fails {spec}")

    for rx in exp.get("answer_regex", []):
        if not re.search(rx, answer, re.I | re.S):
            problems.append(f"answer missing /{rx}/")
    for rx in exp.get("answer_not_regex", []):
        if re.search(rx, answer, re.I | re.S):
            problems.append(f"answer contains forbidden /{rx}/")
    return {"ok": not problems, "problems": problems}


# ---------------------------------------------------------------------------
# Whole case
# ---------------------------------------------------------------------------


def score_case(case: dict, trace: dict, profile: dict = PROFILE) -> dict:
    turns = []
    for i, (tdef, t) in enumerate(zip(case["turns"], trace["turns"])):
        exp = tdef.get("expect", {}) or {}
        src = Sources(case, trace, i, profile)
        sel = score_selection(t, exp)
        args = score_args(t, exp)
        dollars = check_dollars(t["response"] or "", src)
        cites = check_citations(t["response"] or "", src)
        form_call = last_form_call(t)
        filled = form_call["result"].get("filled_fields", {}) if form_call else None
        form_name = form_call["result"].get("form_name") if form_call else None
        fields = check_form_fields(t, src, form_name, filled) if filled else {"ok": True, "untraced": [], "n_fields": 0}
        counsel = {"ok": True, "problems": [], "applied": False}
        if form_name == "DA_4856" and exp.get("counseling_subject"):
            counsel = {**check_counseling(filled, exp["counseling_subject"], profile), "applied": True}
        pii = check_pii(t["response"] or "", filled, form_name, profile)
        task = score_task(t, exp)
        grounding = {
            "dollars": dollars, "citations": cites, "form_fields": fields,
            "counselee": counsel, "pii": pii,
            "ok": dollars["ok"] and cites["ok"] and fields["ok"] and counsel["ok"] and pii["ok"],
        }
        # fill_form silently ignores keys that aren't on the form — record them
        dropped = []
        from agents.tools import FORM_DISPLAY_FIELDS
        for c in t["tool_calls"]:
            if c["name"] == "fill_form":
                fn = str(c["args"].get("form_name", "")).upper()
                known = {k for k, _ in FORM_DISPLAY_FIELDS.get(fn, [])}
                fv = c["args"].get("field_values") or {}
                dropped += [f"{fn}.{k}={str(v)[:40]!r}" for k, v in fv.items() if known and k not in known]
        harness_warn = None
        if len(t["api_tool_calls"]) != len(t["tool_calls"]):
            harness_warn = (f"recorded {len(t['tool_calls'])} tool calls but API reported "
                            f"{len(t['api_tool_calls'])}")
        turns.append({
            "index": i, "message": t["message"], "response": t["response"], "error": t["error"],
            "selection": sel, "args": args, "grounding": grounding, "task": task,
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in t["tool_calls"]],
            "form": {"form_name": form_name, "filled_fields": filled} if filled else None,
            "form_dropped_keys": dropped,
            "llm_calls": len(t["llm_calls"]),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in t["llm_calls"]),
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in t["llm_calls"]),
            "seconds": t["seconds"], "harness_warning": harness_warn,
            "ok": sel["ok"] and args["n_ok"] == args["n"] and grounding["ok"] and task["ok"],
        })
    return {
        "id": case["id"], "category": case["category"], "tags": case.get("tags", []),
        "turns": turns,
        "selection_ok": all(t["selection"]["ok"] for t in turns),
        "args_n": sum(t["args"]["n"] for t in turns), "args_ok": sum(t["args"]["n_ok"] for t in turns),
        "grounding_ok": all(t["grounding"]["ok"] for t in turns),
        "task_ok": all(t["task"]["ok"] for t in turns),
        "ok": all(t["ok"] for t in turns),
    }
