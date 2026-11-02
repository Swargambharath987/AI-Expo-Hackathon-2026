# Duty Line evals

An evaluation harness for the Duty Line agent. The 43 unit tests under
`tests/` fake the LLM. This harness calls the real model through the real
FastAPI app and measures what the system actually does. It does not modify
any app code. Everything is wrapped at runtime.

```
python -m evals.run --only retrieval          # free, deterministic, ~30 s
python -m evals.run --estimate                # planned turns, LLM calls and cost; runs nothing
python -m evals.run                           # retrieval + every agent case once
python -m evals.run --only agent --tags tdy,form --repeats 3 --concurrency 3
python -m evals.run --case reg_leave_accrual,chit_greeting
python -m evals.run --only agent --rescore evals/reports/traces/A.json evals/reports/traces/B.json
python -m evals.run --case reg_gtcc_mandatory --judge   # adds the LLM judge (extra API calls)
```

Run from the repo root with the app's venv. Agent runs need `LLM_API_KEY` in
`.env`; they use the same `LLM_BASE_URL` and `LLM_MODEL` as the app.

## What each layer measures, and why

Tool-using RAG agents fail in separable places: the retriever misses, the
model picks the wrong tool or arguments, the model invents numbers that no
tool returned, or the end result is wrong. One end-to-end score hides which
of these happened. The layers keep them apart.

| Layer | What it checks | How it scores | LLM? |
|---|---|---|---|
| 1. Retrieval | `search_regulations` returns the passage that answers the question | Recall@1/@2/@4, MRR over 48 gold queries (32 dev + 16 held-out), with and without the domain filter; 3 out-of-corpus probes | No |
| 2. Tool-call accuracy | The agent calls the right tools with the right key arguments | BFCL-style: tool-selection accuracy (required ⊆ called, forbidden ∩ called = ∅), plus per-argument match for destination city/state, travel_days, travel_mode, one_way_miles, form_name and domain | Yes |
| 3. Grounding ("no invented data") | (a) every `$` amount in the answer appears in the conversation's tool results; (b) every cited document and section appears in the search results; (c) every filled form field traces to the user's messages, the profile or tool output; (d) on DA 4856 the soldier fields hold the counselee, not the profile owner, and the profile SSN and DoD ID never show up in an answer or on the counselee's form | Deterministic | Yes (scores the agent's output) |
| 4. Task success | The expected end state: TDY total equals `calculate_travel_cost(expected args)`, the right form or no form, required form fields, required facts in the answer (e.g. "2.5") | Deterministic | Yes |
| 5. Consistency | Whether the agent passes reliably across repeated runs | τ-bench pass^k (a case passes only if all k runs pass), shown with the mean pass rate | Yes |
| 6. LLM judge (optional) | Faithfulness and relevance from 1 to 5, using `judge_rubric.md` | **Model-judged.** Uses the app's own model unless `JUDGE_MODEL` is set, so expect self-preference bias | Yes, extra calls |

A case-run **passes overall** only when, on every turn, selection,
arguments, grounding and task all pass.

### How it hooks into the app
- `evals/harness.py` builds a fresh `httpx.AsyncClient(transport=ASGITransport(app))`
  for each case-run. One cookie gives one session, so multi-turn cases reuse
  the server-side conversation exactly as a browser would. The harness raises
  the app's rate limits in memory and leaves them alone on disk.
- `agents.react_agent.execute_tool` is wrapped to record each raw call (name,
  args and the parsed result). The API itself only returns summaries.
  `openai...Completions.create` is wrapped to count LLM calls and tokens.
  The recordings sit in a `ContextVar`, which keeps concurrent cases apart. On
  every turn the harness checks the recorded call count against the API's
  `tool_calls` and lists any mismatch under "Harness warnings".

### Grounding sources (important for reading layer 3)
- Sources are the conversation so far: the user's messages, including the
  app's own deterministic `[Extracted context: …]` hints; the request
  profile; and the tool results.
- Fields in a tool result that only echo the model's own arguments are
  removed first, so a guess can't be laundered through a tool call and come
  back looking grounded. Examples: `calculate_travel_cost.destination`,
  `.mileage.one_way_miles`, and the city on the standard-rate fallback.
- Earlier assistant answers are **not** a source.
- `$` amounts that equal k × (a tool number) or the sum of two tool numbers
  pass, but the report flags them as `derived`. Any other amount fails.
- A form date must match a date the user gave, or one derivable from it
  (start + N or start + N−1 days, where N is a number the user mentioned). It
  must keep the user's year. When the user gave no year, an inferred year
  that puts the date in the past is flagged.
- Free-text form fields such as purpose, key points, plan of action and
  reason may paraphrase. They fail only when more than 40% of their content
  words appear in no source.

## Cost

Priced for claude-sonnet-5-5 at $2/M input and $10/M output (the baseline
runs used claude-sonnet-4-6 at $3/$15); override with `EVAL_PRICE_IN_PER_M` / `EVAL_PRICE_OUT_PER_M`:

- About 1.9 LLM calls per turn, averaging about 3.6k input and 140 output
  tokens per call. A turn that searches carries 4 regulation chunks of about
  3k characters each, so input tokens dominate the cost.
- The full suite at `--repeats 1` (23 cases, 27 turns, 51 calls) cost
  **$0.70**.
- `--tags tdy,form --repeats 3` (12 cases, 48 turns, 91 calls) cost
  **$1.15**.
- Retrieval and `--rescore` are free.
- `--estimate` prints the plan and its cost using these calibrated averages.

## Adding a case

Add an entry to any `evals/cases/*.yaml`. Each case has an `id`, a
`category`, optional `tags` (used by `--tags`), a `tab` (travel, leave,
regulation or eval) and a list of `turns`, each with a `message` and an
`expect` block:

```yaml
- id: tdy_example
  category: tdy_cost
  tags: [tdy]
  tab: travel
  turns:
    - message: "TDY from Fort Riley to Fort Leavenworth, 3 days, POV"
      expect:
        tools_required: [calculate_travel_cost]   # must be called this turn
        tools_forbidden: [fill_form]              # must not be called
        # no_tools: true                          # irrelevance cases
        args:                                     # per-argument matchers
          calculate_travel_cost:
            destination_city: {any_of: ["leavenworth"]}   # substring, case-insensitive
            travel_days: {num: 3}                         # numeric ±0.01
            travel_mode: {eq: POV}                        # case-insensitive equality
            one_way_miles: {num: 70}
          search_regulations:
            domain: {eq: travel, optional: true}          # omitted = ok, wrong = fail
        tdy_total: {destination_city: Junction City, destination_state: KS,   # expected total is
                    travel_days: 3, travel_mode: POV, one_way_miles: 70}     # computed by the real tool
        form: none            # none | any | DD_1610 | DA_31 | DA_4856 | DA_4187
        form_fields:          # checked when the expected form was filled (semantic keys)
          name_last_first: {contains: Carter}
        form_fields_if_any:   # checked whenever any form is filled
          name_last_first: {contains: Rivera}
        counseling_subject: {soldier: Ortiz, soldier_rank: PFC}   # DA 4856 identity check
        answer_regex: ["\\b60\\b"]
        answer_not_regex: ["\\$\\d"]
```

Matchers: `eq`, `num`, `any_of`, `contains`, `regex`, `not_contains`, each
with an optional `optional: true`. Form field keys are the semantic keys in
`agents/tools.py` `FORM_DISPLAY_FIELDS`. The requesting user is always the
`PROFILE` in `evals/common.py` (SSG James R. Carter, SSN-4 4821). Keep case
subjects distinct from that profile so identity mix-ups stay detectable.

**Retrieval gold** (`retrieval_gold.yaml`): a chunk counts as relevant when
its `source_file` matches and its text matches `text_regex` once whitespace
and dashes are normalised. Write the label by reading the collection, not by
looking at what the retriever returns. Put the chunk ids you checked in
`verified_chunks`. Every run re-derives the relevant set and warns when it
drifts, for example after a re-ingest.

Queries marked `split: holdout` are for checking that a retrieval change
generalises: tune on the dev queries, then confirm the held-out numbers
moved too. Their labels were completed by pooling (the top results of the
old and new retrievers were read and judged before scoring). A phrase
search alone misses relevant passages and makes every retriever look worse.

## Reading the report

`evals/reports/<UTC stamp>.md` contains:

1. **Retrieval**: Recall@k and MRR, unfiltered and with the gold domain
   filter, broken down by source group, by difficulty (direct vs
   paraphrase) and by split (dev vs held-out). It then lists every query missed at k=4, along with the
   wrong chunks that won and the out-of-corpus probes with their top scores.
   If the out-of-corpus top scores sit in the same range as the answerable
   ones, a score threshold can't tell the agent "nothing found".
2. **Agent summary table**: each metric with its denominator and a 95%
   Wilson interval. Grounding
   rates are over *applicable* turns (turns with a `$`, a citation or a form).
3. **Per-category table**, then **consistency** (pass^k per case).
4. **Failure details** for every failed case-run: the raw tool calls with
   their arguments, each failing check with expected vs actual, the offending
   numbers, citations and fields, and the start of the answer.
5. `fill_form` keys the tool **silently dropped** because the key isn't on
   the form. This is informational, and often the root cause of a wrong form.

The `.json` next to the report holds the same scored data, and
`reports/traces/<stamp>.json` (gitignored) holds the raw traces, including
full tool results. Use `--rescore` to re-score those traces after you change
a check, with no new API spend.

## Known limitations of the method
- **Sample size.** 23 cases and k=3 repeats give wide confidence intervals.
  Treat ±10 points as noise.
- **Lenient grounding check (a).** Search results contain many numbers, so
  a `$` amount that happens to appear in any retrieved chunk passes.
- **Lenient grounding check (b).** A section id that appears anywhere in the
  retrieved text counts as grounded, even when the claim attached to it is
  wrong.
- **Free-text form fields** are judged by word overlap. That can't catch a
  paraphrase that changes the meaning.
- **Retrieval relevance is lexical.** A relevant chunk worded differently
  from the label regex counts as a miss. Labels were written to cover the
  passages found by searching the corpus, but that search wasn't exhaustive.
- **Ambiguous expectations.** Some are policy choices; for example, whether
  an unrequested DD 1610 on a cost question is acceptable (it is `form: any`
  except in `tdy_estimate_only_no_form`).
- **Single model.** Results describe one model on the date of the run. The
  committed reports so far are claude-sonnet-4-6 (temperature 0.1); Sonnet 5.5
  thinks before answering and takes no temperature, so re-baseline before
  comparing numbers across the switch.

## Baseline

`reports/20261102T190733Z.md` (+ `.json`) is the committed baseline: every case once, plus `--tags tdy,form --repeats 3`, both re-scored together with `--rescore`, at commit 228acb6 with claude-sonnet-4-6.
