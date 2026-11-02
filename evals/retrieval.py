"""
Layer 1 — retrieval eval. No LLM, deterministic, free.

Calls the app's real `agents.tools.search_regulations` (same embedding model,
same ChromaDB collection, same `n`) for every gold query and scores the ranked
list against the gold relevance rules in retrieval_gold.yaml.

Settings that matter (as the app actually runs):
  * search_regulations(n=4) is the Python default; the tool schema tells the
    model "default 2", and react_agent.RAG_TOP_K = 2 is defined but never used.
    So the model gets 4 chunks unless it passes n. We retrieve n=4 once and
    report Recall@1/@2/@4 off the same ranking.
  * Two conditions: no domain filter (model omitted `domain`) and the gold
    domain filter (model passed the right one).
"""

from __future__ import annotations

import re
import time

from evals.common import GOLD_FILE, load_yaml, norm_dashes

KS = (1, 2, 4)
N_RETRIEVE = 4


def _is_relevant(chunk_source: str, chunk_text: str, rules: list[dict]) -> bool:
    text = re.sub(r"\s+", " ", norm_dashes(chunk_text))
    for r in rules:
        if chunk_source == r["source"] and re.search(norm_dashes(r["text_regex"]), text, re.I):
            return True
    return False


def _relevant_ids(collection, rules: list[dict]) -> set[str]:
    ids: set[str] = set()
    for src in {r["source"] for r in rules}:
        got = collection.get(where={"source_file": src}, include=["documents"])
        for cid, doc in zip(got["ids"], got["documents"]):
            if _is_relevant(src, doc, rules):
                ids.add(cid)
    return ids


def run_retrieval(only_ids: set[str] | None = None) -> dict:
    from agents import tools

    gold = load_yaml(GOLD_FILE)
    _, collection = tools._get_embed_and_collection()

    per_query = []
    warnings: list[str] = []
    t0 = time.time()
    for q in gold["queries"]:
        if only_ids and q["id"] not in only_ids:
            continue
        rel_ids = _relevant_ids(collection, q["relevant"])
        if not rel_ids:
            warnings.append(f"{q['id']}: gold rules match NO chunk in the collection (label broken)")
        elif set(q.get("verified_chunks", [])) != rel_ids:
            warnings.append(
                f"{q['id']}: relevant set drifted from verified_chunks "
                f"(verified={sorted(q.get('verified_chunks', []))}, now={sorted(rel_ids)})"
            )
        row = {"id": q["id"], "category": q["category"], "difficulty": q["difficulty"],
               "query": q["query"], "domain": q["domain"], "split": q.get("split", "dev"),
               "n_relevant_in_corpus": len(rel_ids)}
        for cond, domain in (("no_filter", None), ("domain_filter", q["domain"])):
            res = tools.search_regulations(q["query"], domain=domain, n=N_RETRIEVE)
            ranked = res.get("results", [])
            hits = [_is_relevant(c["source"], c["text"], q["relevant"]) for c in ranked]
            first = next((i + 1 for i, h in enumerate(hits) if h), None)
            row[cond] = {
                "first_relevant_rank": first,
                "rr": (1.0 / first) if first else 0.0,
                **{f"hit@{k}": any(hits[:k]) for k in KS},
                "top": [
                    {"source": c["source"], "score": c["score"], "relevant": h,
                     "snippet": norm_dashes(c["text"][:110]).replace("\n", " ")}
                    for c, h in zip(ranked, hits)
                ],
            }
        per_query.append(row)

    ooc = []
    for q in gold.get("out_of_corpus", []):
        if only_ids and q["id"] not in only_ids:
            continue
        res = tools.search_regulations(q["query"], domain=q.get("domain"), n=N_RETRIEVE)
        top = res.get("results", [])
        ooc.append({
            "id": q["id"], "query": q["query"], "note": q.get("note", ""),
            "top_score": top[0]["score"] if top else None,
            "top_sources": [c["source"] for c in top],
        })

    def agg(rows, cond):
        n = len(rows)
        if not n:
            return {}
        out = {"n": n, "MRR": round(sum(r[cond]["rr"] for r in rows) / n, 3)}
        for k in KS:
            out[f"Recall@{k}"] = round(sum(r[cond][f"hit@{k}"] for r in rows) / n, 3)
        return out

    summary = {cond: agg(per_query, cond) for cond in ("no_filter", "domain_filter")}
    by_cat: dict[str, dict] = {}
    groups: dict[str, list] = {}
    for r in per_query:
        groups.setdefault(r["category"].split("_")[0], []).append(r)
        groups.setdefault("difficulty:" + r["difficulty"], []).append(r)
        groups.setdefault("split:" + r["split"], []).append(r)
    for g, rows in sorted(groups.items()):
        by_cat[g] = {cond: agg(rows, cond) for cond in ("no_filter", "domain_filter")}

    # Mean similarity of the top hit for answerable vs unanswerable queries —
    # shows whether a score threshold could separate "no answer in corpus".
    ans_top = [r["no_filter"]["top"][0]["score"] for r in per_query if r["no_filter"]["top"]]
    ooc_top = [o["top_score"] for o in ooc if o["top_score"] is not None]
    return {
        "settings": {"n": N_RETRIEVE, "ks": list(KS), "embed_model": tools.EMBED_MODEL,
                     "collection_count": collection.count()},
        "summary": summary,
        "by_group": by_cat,
        "per_query": per_query,
        "out_of_corpus": ooc,
        "score_separation": {
            "answerable_top1_mean": round(sum(ans_top) / len(ans_top), 3) if ans_top else None,
            "answerable_top1_min": min(ans_top) if ans_top else None,
            "out_of_corpus_top1_mean": round(sum(ooc_top) / len(ooc_top), 3) if ooc_top else None,
            "out_of_corpus_top1_max": max(ooc_top) if ooc_top else None,
        },
        "warnings": warnings,
        "seconds": round(time.time() - t0, 1),
    }
