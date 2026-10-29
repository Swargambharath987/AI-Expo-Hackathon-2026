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
│       REACT AGENT  (claude-sonnet-4-6 by default;            │
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
(ChromaDB)  (local JSON)  (pure math)    (pypdf)
 offline      offline       offline        offline
```

---

## The Four Tools

The 4 "agents" in the original plan are tools — functions the single reasoning model can call.
Adding a new domain = adding one tool function. No new agents, no new routing logic.

### Tool 1 — `search_regulations(query, domain=None, n=4)`
- Queries ChromaDB vector store
- Optional domain filter: `travel | leave | regs | eval`
- Returns top-N chunks with source citation (doc, section, paragraph)
- Used for: any regulatory question, rule lookup, cross-reference resolution

### Tool 2 — `get_per_diem(city, state, year)`
- Reads from local `data/gsa_cache.json` — pre-fetched offline
- Returns: `{lodging_rate, mie_rate, first_last_day_rate, total_daily}`
- Falls back to standard rate if city not in cache
- Used for: TDY cost calculation, travel planning

### Tool 3 — `calculate_travel_cost(origin, dest, days, mode, rank)`
- Pure math — no LLM, no network
- Applies JTR rules: 75% M&IE first/last day, mileage at current GSA rate
- Returns structured cost breakdown: lodging, meals, mileage, total
- Used for: TDY planning, pre-trip cost estimates

### Tool 4 — `fill_form(form_name, field_values)`
- Maps field values to PDF form fields
- Merges soldier profile (pre-loaded) with task-specific values
- Returns filled PDF as bytes, saved to output/
- Forms: `DD_1610`, `DA_31`, `DA_4856`, `DA_4187`
- Used for: any form generation output

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
  "supervisor_title": "Team Leader"
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

## Data Sources — Priority Tiers

### Tier 1 — Non-Negotiable (demo fails without these)

| Document | Domain | Where |
|----------|--------|-------|
| JTR (Joint Travel Regulations) | travel | `data/jtr/jtr.pdf` ✓ |
| AR 600-8-10 (Leaves and Passes) | leave | `data/army_regs/ar_600_8_10.pdf` ✓ |
| DD Form 1610 (TDY Request) | travel | `data/forms/dd1610.pdf` |
| DA Form 31 (Leave Request) | leave | `data/forms/da31.pdf` |
| GSA Per Diem cache (JSON) | travel | `data/gsa_cache.json` (build from API) |

### Tier 2 — Strong Nice-to-Have (build after Tier 1 works)

| Document | Domain | Why |
|----------|--------|-----|
| DA Form 4856 (Counseling) | eval | Most downloaded Army form |
| DA Form 4187 (Personnel Action) | hr | Very high volume use |
| AR 623-3 (Evaluations) | eval | 4th major use case |
| DA PAM 623-3 | eval | Companion guide to AR 623-3 |

### Tier 3 — Extensibility Story (show the architecture scales, don't block on it)

| Document | Domain | Why |
|----------|--------|-----|
| MILPERSMAN 1050 | leave/navy | Multi-branch capability |
| AFI 36-3003 | leave/airforce | Multi-branch capability |
| eCFR Title 32 | regs | Broader regulatory coverage |

---

## Retrieval Architecture

### Why not just semantic search

Military regulations have known cross-reference patterns. JTR Chapter 2 always covers per diem. AR 600-8-10 Chapter 2 always covers accrual. Pure semantic search treats every chunk equally — a bad chunk about something adjacent will score higher than the right chunk phrased differently.

### Three-layer retrieval

```
Layer 1: Domain routing
  → based on task classification, filter to relevant domain
  → reduces search space, improves precision

Layer 2: Semantic search (ChromaDB)
  → cosine similarity on BAAI/bge-small-en-v1.5 embeddings
  → returns top-K candidates within domain

Layer 3: LLM reranking (within ReAct Thought step)
  → model reads chunks, decides if they answer the question
  → if not sufficient: calls search again with refined query
  → this is the multi-hop capability — follows cross-references
```

### Semantic chunking (already implemented)

Chunks split at regulation section boundaries (not fixed token count):
- JTR: `020101.` format
- Army Regs: `2-1.` format
- Min 150 tokens, max 600 tokens
- Section header carried into every chunk

---

## LLM Provider Architecture

Claude API is the **default** provider — higher answer quality, no local GPU required.
Ollama is the **offline fallback** — air-gap / SCIF environments with no internet.

| Deployment | Provider | Config |
|------------|----------|--------|
| Default (cloud-connected) | Claude API (`claude-sonnet-4-6`) | `LLM_BASE_URL=https://api.anthropic.com/v1` |
| Offline / air-gap | Ollama (`gemma4:e4b`) | `LLM_BASE_URL=http://localhost:11434/v1` |
| Alternative cloud | llama.cpp server | `LLM_BASE_URL=http://localhost:8080/v1` |
| Alternative cloud | LM Studio | `LLM_BASE_URL=http://localhost:1234/v1` |

**Switching inference backends:** one environment variable change — zero code changes.
The abstraction is already in place via the OpenAI-compatible client interface.

### Offline components (Ollama mode)

| Component | Offline mechanism |
|-----------|------------------|
| LLM inference | Ollama + local model weights |
| Regulation retrieval | ChromaDB persistent local store |
| Per diem rates | Pre-fetched `gsa_cache.json` (build once) |
| Embeddings | BAAI/bge-small-en-v1.5 cached by sentence-transformers |
| Form filling | pypdf + local PDF templates |
| Soldier profile | Local `profile.json` |

---

## Streaming

The ReAct reasoning trace streams to the UI token by token.
The soldier sees the system thinking in real time:

```
Searching regulations for POV travel entitlements...
  → Found: JTR 020402. Private Vehicle Use

Getting per diem for Columbus, GA FY2026...
  → Lodging: $104/night | M&IE: $64/day

Calculating trip cost (10 days, POV, 370 miles each way)...
  → Lodging: $1,040 | Meals: $620 | Mileage: $495.80 | Total: $2,155.80

Generating DD 1610...
  → Form ready for download

Based on JTR Section 020402, SPC Rivera is entitled to...
```

This is the demo moment. Not "it answered a question." "Watch it work."

---

## What This Is Not

- Not a general-purpose chatbot
- Not a web search tool
- Not connected to any live DoD system
- Not storing soldier data anywhere except the local device

The system is a **local reasoning engine** over a curated set of offline-cached military regulations and forms.
It does not replace legal advice or official S1 guidance.
It reduces the time to get to the right answer from hours to seconds.
