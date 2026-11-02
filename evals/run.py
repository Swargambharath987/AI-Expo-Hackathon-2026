"""
Duty Line eval CLI.

  python -m evals.run                         # retrieval + agent, every case once
  python -m evals.run --only retrieval        # free, deterministic
  python -m evals.run --only agent --tags tdy,form --repeats 3
  python -m evals.run --case reg_leave_accrual --judge
  python -m evals.run --estimate              # print planned turns / cost, run nothing
  python -m evals.run --rescore evals/reports/traces/A.json evals/reports/traces/B.json

Outputs evals/reports/<UTC stamp>.md + .json (scored results), and for agent
runs evals/reports/traces/<stamp>.json (raw traces incl. full tool results;
gitignored) so checks can be changed and re-scored without new API spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict

from evals.common import PROFILE, REPORTS_DIR, load_cases, utc_stamp

# Claude Sonnet list price (USD per million tokens); override via env.
PRICE_IN = float(os.getenv("EVAL_PRICE_IN_PER_M", "3"))
PRICE_OUT = float(os.getenv("EVAL_PRICE_OUT_PER_M", "15"))
# Planning estimates per turn (calibrated on the first baseline run).
EST_CALLS_PER_TURN = 1.9
EST_IN_TOKENS_PER_CALL = 3650
EST_OUT_TOKENS_PER_CALL = 145


def select_cases(args) -> list[dict]:
    cases = load_cases()
    if args.case:
        want = {c.strip() for c in args.case.split(",")}
        missing = want - {c["id"] for c in cases}
        if missing:
            sys.exit(f"unknown case id(s): {sorted(missing)}")
        cases = [c for c in cases if c["id"] in want]
    if args.tags:
        tags = {t.strip() for t in args.tags.split(",")}
        cases = [c for c in cases if tags & set(c.get("tags", []))]
    return cases


def estimate(cases, repeats) -> dict:
    turns = sum(len(c["turns"]) for c in cases) * repeats
    calls = turns * EST_CALLS_PER_TURN
    cost = calls * (EST_IN_TOKENS_PER_CALL * PRICE_IN + EST_OUT_TOKENS_PER_CALL * PRICE_OUT) / 1e6
    return {"cases": len(cases), "repeats": repeats, "turns": turns,
            "est_llm_calls": round(calls), "est_cost_usd": round(cost, 2)}


# ---------------------------------------------------------------------------
# Agent layer
# ---------------------------------------------------------------------------


def run_agent(cases, repeats, concurrency) -> dict:
    from evals.harness import llm_env_summary, load_app, run_many

    appmod = load_app()
    jobs = [(c, r) for r in range(repeats) for c in cases]
    done = {"n": 0}
    t0 = time.time()

    def progress(case, rep, trace):
        done["n"] += 1
        calls = sum(len(t["llm_calls"]) for t in trace["turns"])
        errs = [t["error"] for t in trace["turns"] if t["error"]]
        print(f"  [{done['n']}/{len(jobs)}] {case['id']} rep{rep} — {calls} LLM calls"
              f"{' ERRORS ' + str(errs) if errs else ''}", flush=True)

    results = asyncio.run(run_many(appmod, jobs, PROFILE, concurrency, progress))
    runs = [{"case_id": c["id"], "repeat": r, "trace": tr} for c, r, tr in results]
    return {"llm": llm_env_summary(), "repeats": repeats, "seconds": round(time.time() - t0, 1),
            "runs": runs}


def score_runs(trace_files: list[dict], cases_by_id: dict, judge: bool) -> dict:
    from evals.checks import score_case

    scored = []
    judge_client = None
    for tf_i, tf in enumerate(trace_files):
        for run in tf["runs"]:
            case = cases_by_id.get(run["case_id"])
            if case is None:
                continue
            sc = score_case(case, run["trace"], PROFILE)
            sc["repeat"] = run["repeat"]
            sc["batch"] = tf_i
            sc["batch_repeats"] = tf["repeats"]
            if judge:
                from evals.judge import _client, judge_turn
                judge_client = judge_client or _client()
                for st, raw in zip(sc["turns"], run["trace"]["turns"]):
                    st["judge"] = judge_turn(raw["message"], raw["tool_calls"], raw["response"], judge_client)
            scored.append(sc)
    return aggregate(scored)


def _rate(num, den):
    return round(num / den, 3) if den else None


def aggregate(scored: list[dict]) -> dict:
    turns = [t for s in scored for t in s["turns"]]
    sel_turns = [t for t in turns if t["selection"]["scored"]]
    no_tool_turns = [t for s in scored if s["category"] == "no_tool" for t in s["turns"]]
    g = lambda k: [t["grounding"][k]["ok"] for t in turns]
    applic = {
        "dollars": [t for t in turns if t["grounding"]["dollars"]["amounts"]],
        "citations": [t for t in turns if t["grounding"]["citations"]["docs_cited"] or t["grounding"]["citations"]["sections_cited"]],
        "form_fields": [t for t in turns if t["form"]],
        "counselee": [t for t in turns if t["grounding"]["counselee"].get("applied")],
    }
    llm_calls = sum(t["llm_calls"] for t in turns)
    p_in = sum(t["prompt_tokens"] for t in turns)
    p_out = sum(t["completion_tokens"] for t in turns)
    jt = [t["judge"] for t in turns if t.get("judge") and t["judge"].get("faithfulness")]
    j_in = sum(t["judge"]["prompt_tokens"] for t in turns if t.get("judge"))
    j_out = sum(t["judge"]["completion_tokens"] for t in turns if t.get("judge"))

    summary = {
        "case_runs": len(scored), "unique_cases": len({s["id"] for s in scored}), "turns": len(turns),
        "tool_selection_acc": _rate(sum(t["selection"]["ok"] for t in sel_turns), len(sel_turns)),
        "irrelevance_acc": _rate(sum(t["selection"]["ok"] for t in no_tool_turns), len(no_tool_turns)),
        "arg_acc": _rate(sum(s["args_ok"] for s in scored), sum(s["args_n"] for s in scored)),
        "grounding": {
            "dollars": {"pass": _rate(sum(t["grounding"]["dollars"]["ok"] for t in applic["dollars"]), len(applic["dollars"])), "applicable_turns": len(applic["dollars"])},
            "citations": {"pass": _rate(sum(t["grounding"]["citations"]["ok"] for t in applic["citations"]), len(applic["citations"])), "applicable_turns": len(applic["citations"])},
            "form_fields": {"pass": _rate(sum(t["grounding"]["form_fields"]["ok"] for t in applic["form_fields"]), len(applic["form_fields"])), "applicable_turns": len(applic["form_fields"])},
            "counselee": {"pass": _rate(sum(t["grounding"]["counselee"]["ok"] for t in applic["counselee"]), len(applic["counselee"])), "applicable_turns": len(applic["counselee"])},
            "pii": {"pass": _rate(sum(g("pii")), len(turns)), "applicable_turns": len(turns)},
            "all_turns": _rate(sum(t["grounding"]["ok"] for t in turns), len(turns)),
        },
        "task_success": _rate(sum(s["task_ok"] for s in scored), len(scored)),
        "overall_pass": _rate(sum(s["ok"] for s in scored), len(scored)),
        "llm_calls": llm_calls, "prompt_tokens": p_in, "completion_tokens": p_out,
        "agent_cost_usd": round((p_in * PRICE_IN + p_out * PRICE_OUT) / 1e6, 3),
        "llm_calls_per_turn": _rate(llm_calls, len(turns)),
        "fill_form_dropped_keys": [f"{s['id']} r{s['batch']}.{s['repeat']} t{t['index']}: {t['form_dropped_keys']}"
                                   for s in scored for t in s["turns"] if t.get("form_dropped_keys")],
        "harness_warnings": [f"{s['id']} r{s['repeat']} t{t['index']}: {t['harness_warning']}"
                             for s in scored for t in s["turns"] if t["harness_warning"]],
    }
    if jt:
        summary["judge"] = {
            "MODEL_JUDGED": True, "turns": len(jt),
            "faithfulness_mean": round(sum(j["faithfulness"] for j in jt) / len(jt), 2),
            "relevance_mean": round(sum(j["relevance"] for j in jt) / len(jt), 2),
            "cost_usd": round((j_in * PRICE_IN + j_out * PRICE_OUT) / 1e6, 3),
        }

    by_cat = defaultdict(list)
    for s in scored:
        by_cat[s["category"]].append(s)
    categories = {}
    for cat, ss in sorted(by_cat.items()):
        ts = [t for s in ss for t in s["turns"]]
        categories[cat] = {
            "case_runs": len(ss),
            "selection": _rate(sum(s["selection_ok"] for s in ss), len(ss)),
            "args": _rate(sum(s["args_ok"] for s in ss), sum(s["args_n"] for s in ss)),
            "grounding": _rate(sum(s["grounding_ok"] for s in ss), len(ss)),
            "task": _rate(sum(s["task_ok"] for s in ss), len(ss)),
            "pass": _rate(sum(s["ok"] for s in ss), len(ss)),
            "llm_calls_per_turn": _rate(sum(t["llm_calls"] for t in ts), len(ts)),
        }

    # Consistency (tau-bench pass^k): computed within each batch that was run
    # with --repeats >= 2, so k is exactly the requested repeat count.
    runs_by_case = defaultdict(list)
    for s in scored:
        if s["batch_repeats"] >= 2:
            runs_by_case[(s["batch"], s["id"])].append(s["ok"])
    cons = {cid: v for (_, cid), v in runs_by_case.items()}
    consistency = None
    if cons:
        per_case = {cid: {"k": len(v), "passes": sum(v), "pass^k": all(v), "mean": round(sum(v) / len(v), 3)}
                    for cid, v in sorted(cons.items())}
        ks = sorted({v["k"] for v in per_case.values()})
        consistency = {
            "k": ks[0] if len(ks) == 1 else ks,
            "cases": len(per_case),
            "pass^k": _rate(sum(v["pass^k"] for v in per_case.values()), len(per_case)),
            "mean_pass": _rate(sum(v["mean"] for v in per_case.values()), len(per_case)),
            "pass^1_any": _rate(sum(v["passes"] > 0 for v in per_case.values()), len(per_case)),
            "per_case": per_case,
        }
    return {"summary": summary, "categories": categories, "consistency": consistency, "case_runs": scored}


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def render_md(report: dict) -> str:
    L = [f"# Duty Line eval report — {report['stamp']}", ""]
    L += [f"- Command: `{report['command']}`", f"- Git: `{report.get('git', '?')}`"]
    if report.get("agent"):
        L.append(f"- Model: `{report['agent']['llm']['model']}` via `{report['agent']['llm']['base_url']}`")
    L.append("")

    r = report.get("retrieval")
    if r:
        L += ["## Layer 1 — Retrieval (search_regulations, no LLM)", "",
              f"{r['summary']['no_filter']['n']} gold queries · n={r['settings']['n']} · "
              f"embed `{r['settings']['embed_model']}` · {r['settings']['collection_count']} chunks", "",
              "| Condition | Recall@1 | Recall@2 | Recall@4 | MRR |", "|---|---|---|---|---|"]
        for cond in ("no_filter", "domain_filter"):
            s = r["summary"][cond]
            L.append(f"| {cond} | {_pct(s['Recall@1'])} | {_pct(s['Recall@2'])} | {_pct(s['Recall@4'])} | {s['MRR']:.3f} |")
        L += ["", "| Group (no filter) | n | Recall@1 | Recall@2 | Recall@4 | MRR |", "|---|---|---|---|---|---|"]
        for gname, gv in r["by_group"].items():
            s = gv["no_filter"]
            L.append(f"| {gname} | {s['n']} | {_pct(s['Recall@1'])} | {_pct(s['Recall@2'])} | {_pct(s['Recall@4'])} | {s['MRR']:.3f} |")
        ss = r["score_separation"]
        L += ["", f"Top-1 cosine similarity: answerable mean {ss['answerable_top1_mean']} (min {ss['answerable_top1_min']}) "
              f"vs out-of-corpus mean {ss['out_of_corpus_top1_mean']} (max {ss['out_of_corpus_top1_max']}).", ""]
        L += ["**Misses at k=4 (no filter):**", ""]
        for q in r["per_query"]:
            if not q["no_filter"]["hit@4"]:
                tops = "; ".join(f"{t['source']} {t['score']}: \"{t['snippet'][:70]}\"" for t in q["no_filter"]["top"][:2])
                fr = q["domain_filter"]["first_relevant_rank"]
                L.append(f"- `{q['id']}` ({q['difficulty']}) \"{q['query']}\" — with domain filter: "
                         f"{'rank ' + str(fr) if fr else 'still missed'}. Top: {tops}")
        L += ["", "**Out-of-corpus queries** (no right answer exists):", ""]
        for o in r["out_of_corpus"]:
            L.append(f"- `{o['id']}` \"{o['query']}\" — top score {o['top_score']} from {o['top_sources'][:2]}")
        if r["warnings"]:
            L += ["", "Label warnings:", ""] + [f"- {w}" for w in r["warnings"]]
        L.append("")

    a = report.get("agent_scores")
    if a:
        s = a["summary"]
        g = s["grounding"]
        L += ["## Layers 2–5 — Agent (real model, full API stack)", "",
              f"{s['unique_cases']} cases · {s['case_runs']} case-runs · {s['turns']} turns · "
              f"{s['llm_calls']} LLM calls ({s['llm_calls_per_turn']}/turn) · "
              f"{s['prompt_tokens']:,} in / {s['completion_tokens']:,} out tokens · ≈${s['agent_cost_usd']}", "",
              "| Metric | Value | Denominator |", "|---|---|---|",
              f"| Tool selection accuracy | {_pct(s['tool_selection_acc'])} | turns with tool expectations |",
              f"| Irrelevance (no-tool) accuracy | {_pct(s['irrelevance_acc'])} | no-tool turns |",
              f"| Argument accuracy | {_pct(s['arg_acc'])} | expected key args (missing call = all wrong) |",
              f"| Grounding (a) $ amounts | {_pct(g['dollars']['pass'])} | {g['dollars']['applicable_turns']} turns with $ |",
              f"| Grounding (b) citations | {_pct(g['citations']['pass'])} | {g['citations']['applicable_turns']} turns citing |",
              f"| Grounding (c) form fields | {_pct(g['form_fields']['pass'])} | {g['form_fields']['applicable_turns']} turns with a form |",
              f"| Grounding (d) counselee identity | {_pct(g['counselee']['pass'])} | {g['counselee']['applicable_turns']} DA 4856 turns |",
              f"| Grounding (d) PII (SSN/DoD ID) | {_pct(g['pii']['pass'])} | all turns |",
              f"| Grounding — all checks | {_pct(g['all_turns'])} | all turns |",
              f"| Task success | {_pct(s['task_success'])} | case-runs |",
              f"| **Overall pass** (selection ∧ args ∧ grounding ∧ task) | **{_pct(s['overall_pass'])}** | case-runs |"]
        if s.get("judge"):
            j = s["judge"]
            L.append(f"| LLM judge — faithfulness (1–5, MODEL-JUDGED) | {j['faithfulness_mean']} | {j['turns']} turns |")
            L.append(f"| LLM judge — relevance (1–5, MODEL-JUDGED) | {j['relevance_mean']} | {j['turns']} turns |")
        L += ["", "| Category | Runs | Selection | Args | Grounding | Task | Pass | LLM calls/turn |", "|---|---|---|---|---|---|---|---|"]
        for cat, c in a["categories"].items():
            L.append(f"| {cat} | {c['case_runs']} | {_pct(c['selection'])} | {_pct(c['args'])} | {_pct(c['grounding'])} | "
                     f"{_pct(c['task'])} | {_pct(c['pass'])} | {c['llm_calls_per_turn']} |")
        c = a.get("consistency")
        if c:
            L += ["", f"### Consistency (τ-bench pass^k, k={c['k']}, {c['cases']} cases)", "",
                  f"pass^k **{_pct(c['pass^k'])}** · mean pass rate {_pct(c['mean_pass'])} · passed at least once {_pct(c['pass^1_any'])}", "",
                  "| Case | Passes | pass^k |", "|---|---|---|"]
            for cid, v in c["per_case"].items():
                L.append(f"| {cid} | {v['passes']}/{v['k']} | {'yes' if v['pass^k'] else 'no'} |")
        if s["fill_form_dropped_keys"]:
            L += ["", "fill_form arguments silently dropped by the tool (key not on the form — informational):", ""]
            L += [f"- {w}" for w in s["fill_form_dropped_keys"]]
        if s["harness_warnings"]:
            L += ["", "Harness warnings:", ""] + [f"- {w}" for w in s["harness_warnings"]]

        L += ["", "### Failure details", ""]
        for sc in a["case_runs"]:
            if sc["ok"]:
                continue
            L.append(f"#### `{sc['id']}` (run {sc['batch']}.{sc['repeat']}) — "
                     f"selection {'ok' if sc['selection_ok'] else 'FAIL'}, args {sc['args_ok']}/{sc['args_n']}, "
                     f"grounding {'ok' if sc['grounding_ok'] else 'FAIL'}, task {'ok' if sc['task_ok'] else 'FAIL'}")
            for t in sc["turns"]:
                if t["ok"]:
                    continue
                L.append(f"- **Turn {t['index'] + 1}**: \"{t['message'][:110]}\"")
                L.append(f"  - tools: {[(c['name'], c['args']) for c in t['tool_calls']] or 'none'}")
                for p in t["selection"]["problems"]:
                    L.append(f"  - selection: {p}")
                for it in t["args"]["items"]:
                    if not it["ok"]:
                        L.append(f"  - arg `{it['tool']}.{it['arg']}`: expected {it['expected']}, got `{it['actual']}`")
                gr = t["grounding"]
                if gr["dollars"]["ungrounded"]:
                    L.append(f"  - ungrounded $ amounts: {gr['dollars']['ungrounded']}")
                if gr["citations"]["ungrounded_docs"] or gr["citations"]["ungrounded_sections"]:
                    L.append(f"  - ungrounded citations: docs {gr['citations']['ungrounded_docs']}, "
                             f"sections {gr['citations']['ungrounded_sections']} (searches this conversation: {gr['citations']['n_searches']})")
                for u in gr["form_fields"]["untraced"]:
                    L.append(f"  - untraceable form field `{u['field']}` = \"{u['value'][:80]}\" — {u['why']}")
                for p in gr["counselee"]["problems"] + gr["pii"]["problems"]:
                    L.append(f"  - identity/PII: {p}")
                for p in t["task"]["problems"]:
                    L.append(f"  - task: {p}")
                ans = (t["response"] or "").replace("\n", " ⏎ ")
                L.append(f"  - answer: \"{ans[:400]}{'…' if len(ans) > 400 else ''}\"")
            L.append("")
    return "\n".join(L) + "\n"


def _git_rev() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["retrieval", "agent"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--judge", action="store_true", help="also score each turn with an LLM judge (costs API calls)")
    ap.add_argument("--case", help="comma-separated case ids")
    ap.add_argument("--tags", help="comma-separated tags, e.g. tdy,form")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--estimate", action="store_true", help="print the planned run size and cost; run nothing")
    ap.add_argument("--rescore", nargs="+", metavar="TRACES_JSON",
                    help="re-score saved traces (no agent API calls); retrieval re-runs unless --only agent")
    args = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()

    cases = select_cases(args)
    if args.estimate:
        print(json.dumps(estimate(cases, args.repeats), indent=2))
        return

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "traces").mkdir(exist_ok=True)
    stamp = utc_stamp()
    report = {"stamp": stamp, "command": "python -m evals.run " + " ".join(argv if argv is not None else sys.argv[1:]),
              "git": _git_rev()}

    if args.only in (None, "retrieval"):
        from evals.retrieval import run_retrieval
        print("Retrieval eval…", flush=True)
        report["retrieval"] = run_retrieval()
        s = report["retrieval"]["summary"]["no_filter"]
        print(f"  Recall@1 {s['Recall@1']}  @2 {s['Recall@2']}  @4 {s['Recall@4']}  MRR {s['MRR']}")

    if args.only in (None, "agent"):
        cases_by_id = {c["id"]: c for c in load_cases()}
        if args.rescore:
            trace_files = [json.load(open(p)) for p in args.rescore]
            report["agent"] = {"llm": trace_files[0]["llm"], "rescored_from": args.rescore,
                               "repeats": [tf["repeats"] for tf in trace_files]}
        else:
            est = estimate(cases, args.repeats)
            print(f"Agent eval: {est['cases']} cases × {args.repeats} = {est['turns']} turns "
                  f"(est. {est['est_llm_calls']} LLM calls, ≈${est['est_cost_usd']})", flush=True)
            agent = run_agent(cases, args.repeats, args.concurrency)
            tpath = REPORTS_DIR / "traces" / f"{stamp}.json"
            tpath.write_text(json.dumps(agent, default=str))
            print(f"  traces → {tpath}")
            trace_files = [agent]
            report["agent"] = {"llm": agent["llm"], "repeats": args.repeats, "seconds": agent["seconds"],
                               "traces_file": str(tpath.relative_to(REPORTS_DIR.parent.parent))}
        report["agent_scores"] = score_runs(trace_files, cases_by_id, args.judge)
        s = report["agent_scores"]["summary"]
        print(f"  selection {s['tool_selection_acc']}  args {s['arg_acc']}  grounding {s['grounding']['all_turns']}  "
              f"task {s['task_success']}  pass {s['overall_pass']}  | {s['llm_calls']} LLM calls ≈${s['agent_cost_usd']}")
        if report["agent_scores"].get("consistency"):
            c = report["agent_scores"]["consistency"]
            print(f"  pass^{c['k']} {c['pass^k']}  mean {c['mean_pass']}")

    (REPORTS_DIR / f"{stamp}.json").write_text(json.dumps(report, indent=1, default=str))
    (REPORTS_DIR / f"{stamp}.md").write_text(render_md(report))
    print(f"Report → evals/reports/{stamp}.md (+ .json)")


if __name__ == "__main__":
    main()
