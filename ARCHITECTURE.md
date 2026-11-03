# Duty Line — System Architecture

**GenAI.mil Track | SCSP Hackathon 2026**

> "The US military runs on paperwork. Build the AI assistant that makes the rank-and-file faster, smarter, and less buried in bureaucracy — and does it offline."

---

## Core Design Principle

A soldier does not come to this system to ask questions.
They come to **complete a task** — submit a form, plan a trip, check eligibility.

The architecture is built around **task completion**, not question answering.
The output is always a **filled artifact** (form, cost breakdown, checklist) plus the reasoning that produced it.

---

## Why Not Simple RAG

| Simple RAG | Duty Line |
|------------|-----------|
| Answers one question at a time | Completes multi-step tasks |
| Returns text | Returns filled forms + text |
| No memory between turns | Tracks task state across turns |
| Retrieves from one source | Combines regulations + live rates + forms |
| Model is a text formatter | Model reasons and decides what it needs |
| Demo: "it answers questions" | Demo: soldier describes trip → filled DD 1610 in 15 seconds |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  Chat UI  (streaming)                        │
│         Shows reasoning steps as they happen                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              CONVERSATION + TASK STATE                       │
│                                                              │
│  message_history  — last N turns (sliding window)           │
│  current_task     — TRAVEL | LEAVE | REGULATION | EVAL       │
│  collected_fields — what we know so far from conversation    │
│  missing_fields   — what we still need to complete the task  │
│  soldier_profile  — sent by the browser with each request    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│       REACT AGENT  (claude-sonnet-5-5 by default;            │
│                     Ollama + gemma4:e4b for offline)         │
│                                                              │
│  Principle: reason first, act only when needed              │
│                                                              │
│  Loop:                                                       │
│    Thought   → what do I need to answer this completely?     │
│    Action    → call a tool | ask user | answer directly      │
│    Observe   → receive tool result                           │
│    Thought   → is this enough? what's still missing?        │
│    (repeat until sufficient, then generate final response)   │
│                                                              │
│  The model decides IF and WHEN to call tools.               │
│  It does NOT call tools by default on every message.        │
└───┬────────────┬──────────────┬──────────────┬──────────────┘
    │            │              │              │
    ▼            ▼              ▼              ▼
 TOOL 1       TOOL 2         TOOL 3         TOOL 4
search_reg  get_per_diem  calc_cost      fill_form
(hybrid)    (local JSON)  (pure math)    (reportlab)
 offline      offline       offline        offline
```

---

## The Four Tools

The 4 "agents" in the original plan are tools — functions the single reasoning model can call.
Adding a new domain = adding one tool function. No new agents, no new routing logic.

### Tool 1 — `search_regulations(query, domain=None, n=4)`
- Hybrid search over the pre-built ChromaDB index: semantic similarity plus keyword (BM25) ranking, merged with reciprocal rank fusion (see *Retrieval Architecture*)
- Optional domain filter: `travel | leave | regs | eval`
- Returns top-N passages with source file and section
- Used for: any regulatory question, rule lookup, cross-reference resolution

### Tool 2 — `get_per_diem(city, state, year)`
- Reads from local `data/gsa_cache.json` — pre-fetched offline
- Returns: `{lodging, mie, match_type}`; installation names and their old and new names (Fort Bragg/Liberty, Fort Sam Houston/JBSA, …) resolve to the right city
- Falls back to the standard CONUS rate if the city isn't in the cache
- Used for: TDY cost calculation, travel planning

### Tool 3 — `calculate_travel_cost(destination_city, destination_state, travel_days, travel_mode, one_way_miles, origin)`
- Pure math — no LLM, no network
- Applies JTR rules: 75% M&IE on first and last day, lodging nights = days − 1, POV mileage at the TDY rate
- Returns a structured breakdown: lodging, meals, mileage, grand total
- Used for: TDY planning, pre-trip cost estimates

### Tool 4 — `fill_form(form_name, field_values)`
- Forms: `DD_1610`, `DA_31`, `DA_4856`, `DA_4187`; always produces a reportlab PDF, plus an AcroForm fill for DD 1610
- The requesting soldier's profile is injected by the server (the model never supplies it) and fills identity fields
- Forms about someone else (a TDY for a subordinate, a counseling) take the subject from the conversation, and the user's own name, rank, SSN-4 and DoD ID never land on that person's form. On DA 4856 the user becomes the counselor
- Before the PDF is made, identity and role values (name, rank, unit, position, …) that appear nowhere in the user's messages, their profile or a tool result are left blank and listed as missing. DD 1610 cost boxes are copied from the cost tool's estimate, not retyped by the model
- Keys that aren't on the form are reported back to the model (`ignored_fields`)
- Output goes to `output/` with unguessable names, is downloadable only by the session that made it, and is deleted after about an hour

---

## Soldier Profile System

Soldiers fill their profile **once**. Every subsequent form is pre-populated.
The system only asks for task-specific information (destination, dates, purpose).

**Soldier profile (stored in the browser's localStorage, sent with each chat request, never saved on the server):**
```json
{
  "name_last_first": "Rivera, Maria J.",
  "rank": "SPC",
  "grade": "E-4",
  "ssn_last4": "1234",
  "dod_id": "1234567890",
  "unit": "1-503 Infantry, 82nd Airborne Division",
  "installation": "Fort Liberty",
  "uic": "W4XXXX",
  "supervisor_name": "SGT Johnson",
  "supervisor_title": "Team Leader",
  "duty_position": "Team Leader"
}
```

**How information flows into forms:**
```
1. Profile fields  → auto-filled (name, rank, unit, SSN-4)
2. Conversation    → LLM extracts (dates, destination, purpose, travel mode)
3. Clarifying Q    → only for what's still missing after 1 and 2
4. fill_form()     → merges all three → outputs PDF
```

---

## Task State Machine

When a message arrives, the system first classifies the task, then tracks what it has and what it needs.

```
Task: TDY_PLANNING
  Required fields: {destination, travel_dates, origin, travel_mode}
  From profile:    {origin (installation)}
  From convo:      extract {destination, dates, travel_mode}
  Still missing:   ask one question at a time

Task: LEAVE_REQUEST
  Required fields: {leave_type, start_date, end_date, leave_address}
  From profile:    {rank, unit, name}
  From convo:      extract {dates, type, destination}

Task: REGULATION_QA
  Required fields: {question}
  Execution:       search_regulations() → answer with citation
  No form output

Task: EVAL_ASSIST
  Required fields: {soldier_name, period, accomplishments}
  Output:          NCOER bullet suggestions + DA 4856 pre-fill
```

---

## What's in the Library

Regulations (2,063 searchable passages in `vectorstore/`, built by `ingest.py` from the PDFs in `data/`):

| Document | Branch | Domain |
|----------|--------|--------|
| Joint Travel Regulations (JTR) | DoD | travel |
| AR 600-8-10 (Leaves and Passes) | Army | leave |
| DAFI 36-3003 (Military Leave Program) | Air Force | leave |
| MILPERSMAN 1050 | Navy | leave |
| AFI 36-2406 (Evaluation Systems) | Air Force | eval |
| BUPERSINST 1610.10F | Navy | eval |
| MCO 1610.7 | Marine Corps | eval |
| DoD FMR Volume 7A | DoD | regs (pay) |

Also: fillable templates for DD 1610, DA 31, DA 4856, DA 4187 (and DA 2062) in `data/forms/`, and the GSA FY2026 per diem cache `data/gsa_cache.json`.

**Not in the library:** AR 623-3 and DA PAM 623-3 (Army evaluations) and eCFR Title 32. The agent is told this and says so instead of answering from memory. Adding a document means putting its PDF in `data/`, listing it in `ingest.py`, and rebuilding.

---

## Retrieval Architecture

### Why not just semantic search

Military regulations are full of exact terms ("400 miles", "020301", "GTCC") that an embedding can blur, and many passages never name the document or chapter they belong to. Duty Line combines both kinds of matching and labels each passage with where it came from.

```
At build time (ingest.py, ~3 min, reads the PDFs in data/):
  PDF pages → sections (split at 020101. / 2-1. / A2.1. headings) → passages
  → header added to each passage: document title, chapter, headings inside it
  → embedded with BAAI/bge-small-en-v1.5 → ChromaDB (cosine) in vectorstore/
  → Chroma's unused full-text index is dropped (54 MB → 23 MB, same results)

At question time (search_regulations):
  domain filter (optional)
  → semantic search: question embedded with the bge query instruction, top 20
  → keyword search: BM25 over passage text + headers, held in memory, top 20
  → reciprocal rank fusion (keyword weighted half) → top n returned
  → the model reads them, and searches again with a refined query if they don't answer
```

Measured on 48 hand-labelled questions (16 of them held out from tuning), no domain filter: embeddings alone MRR 0.60 → hybrid 0.65 → plus headers and query instruction 0.72 on the dev set, and 0.47 → 0.54 on the held-out set. Everyday wording remains the weakest case. See `evals/` for the method.

### Known limits
- Passage size limits in `ingest.py` count words, so most passages are longer than the embedding model's 512-token window and only their first part is embedded. Keyword search and the LLM still see all of it. Re-splitting to fit the window was tested and scored worse, so it stays.
- Passage text is stored once in Chroma's database; the keyword index is built in memory at first use (about 70 MB).

---

## LLM Provider Architecture

Claude API is the **default** provider — higher answer quality, no local GPU required.
Ollama is the **offline fallback** — air-gap / SCIF environments with no internet.

| Deployment | Provider | Config |
|------------|----------|--------|
| Default (cloud-connected) | Claude API (`claude-sonnet-5-5`) | `LLM_BASE_URL=https://api.anthropic.com/v1` |
| Offline / air-gap | Ollama (`gemma4:e4b`) | `LLM_BASE_URL=http://localhost:11434/v1` |
| Alternative cloud | llama.cpp server | `LLM_BASE_URL=http://localhost:8080/v1` |
| Alternative cloud | LM Studio | `LLM_BASE_URL=http://localhost:1234/v1` |

**Switching inference backends:** one environment variable change — zero code changes.
The abstraction is already in place via the OpenAI-compatible client interface.

**Visitors can bring their own key** (Anthropic, OpenAI, OpenRouter, Groq, Mistral, Gemini, DeepSeek, xAI, Together, or any OpenAI-compatible HTTPS endpoint). The key stays in the visitor's browser and is sent with their requests only. Custom endpoints are checked so they can't reach internal services (public addresses only, re-checked at connection time, no redirects).

**Model settings.** Claude models get no `temperature` (Claude 4.7 and later reject values below 1) and `max_tokens` 4096, because Claude 5 models always think before answering and that counts against the limit. Other providers keep `temperature` 0.1 and 512 tokens.

### Offline components (Ollama mode)

| Component | Offline mechanism |
|-----------|------------------|
| LLM inference | Ollama + local model weights |
| Regulation retrieval | ChromaDB persistent local store |
| Per diem rates | Pre-fetched `gsa_cache.json` (build once) |
| Embeddings | BAAI/bge-small-en-v1.5 cached by sentence-transformers |
| Form filling | reportlab + pypdf + local PDF templates |
| Soldier profile | Browser `localStorage`, sent with each request |

---

## Answer Verification

After the model answers, the server checks the answer against this turn's own tool output:
- every `$` figure must match a value a tool returned (or the user's message)
- every citation (JTR section number, paragraph, table, chapter) must appear in the regulation text retrieved this turn

A mismatched figure gets one corrective retry. Whatever is still unmatched is returned in `verification` so the UI can show a badge. Forms get their own check before the PDF is made (see Tool 4). These checks catch fabricated numbers and citations; they can't tell whether a correctly quoted rule actually applies to the question.

---

## Deployment

One Docker image serves the React build and the API. It is deployed to Google Cloud Run (`gcloud run deploy --source .`): 2 GiB of memory, a single instance (sessions and rate limits live in memory), the model key supplied as an environment variable. Memory was raised from 1 GiB after the first regulation search on the live server was killed for running out of it; the search model, the vector index and the keyword index all load at that point. With no traffic the instance shuts down, so the first request after idle takes about a minute.

The embedding model is baked into the image, and the pre-built `vectorstore/` is copied in as-is. The index is not built during deployment (it took longer than Cloud Build's time limit).

## Tests and Evals

- `tests/` — 104 pytest cases with the LLM faked: JTR math, form filling and provenance, API sessions and limits, own-key handling, retrieval headers, and a guard on the committed index.
- `evals/` — measures the real system layer by layer: retrieval, tool choice and arguments, grounding, task success, consistency across repeated runs. See `evals/README.md`.
- GitHub Actions runs the tests, the retrieval eval (with a minimum score) and the frontend and Docker builds on every push. Nothing in CI calls a paid model.

---

## Streaming

The ReAct reasoning trace streams to the UI token by token.
The soldier sees the system thinking in real time:

```
Searching regulations for POV travel entitlements...
  → Found: JTR | 020210. Privately Owned Vehicles (POV)

Getting per diem for Columbus, GA FY2026...
  → Lodging: $104/night | M&IE: $64/day

Calculating trip cost (10 days, POV, 370 miles each way)...
  → Lodging: $1,040 | Meals: $620 | Mileage: $495.80 | Total: $2,155.80

Generating DD 1610...
  → Form ready for download

Based on JTR Section 020402, SPC Rivera is entitled to...
```

In the UI these steps are collapsed to a one-line summary by default and open on click. This is the demo moment. Not "it answered a question." "Watch it work."

---

## What This Is Not

- Not a general-purpose chatbot
- Not a web search tool
- Not connected to any live DoD system
- Not storing soldier data on the server: the profile stays in the user's browser, and generated forms are kept for about an hour, tied to the session that made them

The system is a **local reasoning engine** over a curated set of offline-cached military regulations and forms.
It does not replace legal advice or official S1 guidance.
It reduces the time to get to the right answer from hours to seconds.
