# Duty Line

**SCSP AI Expo Hackathon 2026 — GenAI.mil Track**
*Phase 1: April 25–26, 2026*

---

## Team

**Team Name:** Duty Line

| Name |
|------|
| Seth Poling |
| Yingquan Li |
| Bharath Kumar Swargam |
| Ingrid Carlina Caceres Paredes |

**Track:** GenAI.mil

---

## What We Built

An AI-powered military admin assistant that takes a natural language request, navigates the Joint Travel Regulations and service branch regulations, calculates TDY costs to the penny, and generates compliant travel authorizations and leave forms — in seconds instead of hours.

Duty Line is not a chatbot that answers questions. It is a **task completion engine** — the output is always a filled artifact (cost breakdown, signed form, cited regulation) plus the reasoning chain that produced it.

### The Problem

A junior NCO planning a TDY trip today must:
1. Navigate the Joint Travel Regulations (1,000+ page PDF) for entitlement rules
2. Look up GSA per diem rates for the destination
3. Calculate lodging, meals, and mileage by hand using JTR rules
4. Fill DA Form 1610 or DD Form 1610 manually
5. Route it through the chain of command for signatures

**This takes 2–4 hours per trip and is error-prone.** Mistakes mean delayed reimbursement or out-of-pocket costs for soldiers. With 2.1M active duty and 800K reserve service members, the bureaucratic tail drains mission readiness across every branch.

### The Solution

A single conversation replaces the entire workflow:

> **"I need to send SPC Rivera to Fort Moore, Georgia for 5 days starting July 10. She's driving her POV from Fort Liberty."**
>
> Duty Line's ReAct agent calls three tools in sequence:
> 1. Looks up GSA per diem: Columbus, GA — $104 lodging / $64 M&IE
> 2. Calculates: $416 lodging + $288 meals (75% first/last day per JTR) + $518 mileage (370 mi x $0.70 x 2) = **$1,222 total**
> 3. Generates a filled DD Form 1610 PDF, ready for signature
>
> **60 seconds. Zero manual math. Zero PDF hunting.**

---

## Datasets & APIs Used

All data sourced from SCSP-recommended public sources:

| Source | Type | What We Use It For |
|--------|------|--------------------|
| [Joint Travel Regulations (JTR)](https://www.travel.dod.mil/Policy-Regulations/Joint-Travel-Regulations/) | PDF (1,000+ pages) | TDY entitlement rules — chunked and embedded into vector store |
| [GSA Per Diem Rates FY2026](https://www.gsa.gov/travel/plan-book/per-diem-rates) | Spreadsheet → JSON cache | Pre-cached per diem lookup: 649 locations, 42,358 ZIP codes |
| [Army Publishing Directorate](https://armypubs.army.mil/) | PDFs | AR 600-8-10 (leave), AR 623-3 (evals), DA forms (31, 1610, 4856, 4187) |
| [Air Force e-Publishing](https://www.e-publishing.af.mil) | PDFs | AFI 36-3003 (leave), AFI 36-2406 (evals) |
| [Navy HR](https://www.mynavyhr.navy.mil/) | PDFs | MILPERSMAN 1050 (leave), BUPERSINST 1610.10F (evals) |
| [Marines Publications](https://www.marines.mil/News/Publications/MCPEL/) | PDFs | MCO 1610.7 (evals) |
| [DoD Comptroller](https://comptroller.defense.gov/) | PDF | DoD FMR Vol 7A (pay during leave) |
| [eCFR API](https://www.ecfr.gov/developers/documentation/api/v1) | API (no auth) | Title 32 (National Defense) regulatory text |
| [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1) | API (no auth) | DoD policy updates and notices |

### Regulation Coverage

| Document | Branch | Domain |
|----------|--------|--------|
| Joint Travel Regulations (JTR) | DoD (all branches) | Travel / TDY |
| AR 600-8-10 | Army | Leave |
| AR 623-3 | Army | Evaluations |
| AFI 36-3003 | Air Force | Leave |
| AFI 36-2406 | Air Force | Evaluations |
| MILPERSMAN 1050 | Navy | Leave |
| BUPERSINST 1610.10F | Navy | Evaluations |
| MCO 1610.7 | Marine Corps | Evaluations |
| DoD FMR Vol 7A | DoD (all branches) | Pay |

---

## Architecture

```
User (chat)
     |
     v
React Frontend (Vite + TypeScript + Tailwind)
     |  POST /api/chat
     v
FastAPI Backend (Python)
     |
     v
ReAct Agent (Thought -> Action -> Observation loop)
  |-- Tool 1: search_regulations  <- ChromaDB semantic search over 2,063 reg chunks
  |-- Tool 2: get_per_diem        <- GSA FY2026 rates, 649 locations pre-cached
  |-- Tool 3: calculate_travel_cost <- JTR-compliant math (mileage, M&IE, lodging)
  '-- Tool 4: fill_form           <- PDF generation (reportlab + AcroForm fill)
```

### Why ReAct, Not Simple RAG

Simple RAG answers questions. Duty Line **completes tasks**. A soldier doesn't want to know what the JTR says — they want a filled DD 1610 with correct math. ReAct lets the agent chain tool calls (look up rates → calculate cost → generate form) to produce an actionable artifact, not just text.

### Model-Agnostic LLM Layer

The agent connects to any OpenAI-compatible API through a single environment variable. No code changes, no redeployment — just swap the endpoint:

| Provider | Config | Cost (per 1M tokens) |
|----------|--------|----------------------|
| **Claude API (default)** | `LLM_BASE_URL=https://api.anthropic.com/v1` | ~$3–15 input / $15–75 output |
| OpenRouter (Llama 3.1 70B) | `LLM_BASE_URL=https://openrouter.ai/api/v1` | ~$0.40 input / $0.40 output |
| Ollama (optional) | `LLM_BASE_URL=http://localhost:11434/v1` | $0 — requires local GPU |

This means Duty Line isn't locked into a single vendor. If a better model comes out tomorrow, or if procurement requires a specific provider, or if policy changes which APIs are authorized on a given network — the switch is one line in a config file. The rest of the system (retrieval, cost calculation, form generation) is completely independent of which LLM is behind the endpoint.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Vite, TypeScript, Tailwind CSS |
| Backend | FastAPI, Uvicorn, Python 3.10+ |
| Agent | ReAct loop — single model, 4 tools, max 5 iterations |
| LLM | Model-agnostic via OpenAI-compatible API (Claude API default) |
| Vector Store | ChromaDB (local, persistent, ~2,063 chunks) |
| Embeddings | BAAI/bge-small-en-v1.5 (sentence-transformers, runs locally) |
| Chunking | Semantic — splits at JTR section boundaries (020101. format) |
| PDF Output | reportlab (all forms) + PyPDF AcroForm fill (DD 1610) |
| Per Diem Data | GSA FY2026 rates pre-cached in gsa_cache.json |

---

## How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- An LLM API key (Claude API recommended, or OpenRouter)

### Setup

```bash
# 1. Clone and set up Python environment
git clone https://github.com/Swargambharath987/AI-Expo-Hackathon-2026.git
cd AI-Expo-Hackathon-2026
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure LLM provider
cp .env.example .env
# Edit .env — add your Claude API key (or OpenRouter key)

# 3. (Optional) Rebuild the vector store — only needed if you're changing
#    the source regulations. A pre-built vectorstore/ ships in the repo,
#    so most setups can skip straight to step 4.
#    Needs pdfplumber, which isn't in the runtime requirements:
pip install -r requirements-ingest.txt
python3 ingest.py

# 4. Start the backend (Terminal 1)
python3 -m uvicorn app:app --reload --port 8000

# 5. Start the frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

### Verify

```bash
curl http://localhost:8000/api/health
```

---

## Deploying (Render)

The `Dockerfile` builds the React frontend and bundles it into the same
FastAPI process (backend serves the built UI under `/`, API under `/api/*`)
— one service, one URL.

1. Push to GitHub (already done if you're reading this from the repo).
2. On [Render](https://render.com), New → Web Service → connect this repo.
   Render auto-detects `render.yaml` and the Docker runtime.
3. Set the `LLM_API_KEY` environment variable in the Render dashboard
   (kept out of `render.yaml`/git on purpose).
4. Deploy. First build takes a few minutes (installs `sentence-transformers`
   / `chromadb` / torch). The free instance (512MB RAM) may be tight for the
   embedding model at load — bump to Starter if it OOMs on boot.

---

## Demo Scenarios (5 minutes)

**1. TDY Travel Planning (2 min)**
```
"I need to send SPC Rivera to Fort Moore, Georgia for 5 days starting July 10.
She's driving her POV from Fort Liberty."
```
Agent calls: `get_per_diem` → `calculate_travel_cost` → `fill_form DD_1610`
Output: Cost breakdown ($1,222) + filled DD 1610 PDF download

**2. Regulation Lookup (1 min)**
```
"Is the GTC mandatory for TDY travel?"
```
Agent calls: `search_regulations [travel]`
Output: Cited answer from JTR 010204 with paragraph text

**3. Leave Request (1 min)**
```
"I need 10 days annual leave starting June 3 to visit family in Texas."
```
Agent calls: `search_regulations [leave]` → `fill_form DA_31`
Output: Eligibility check + filled DA 31 with 8 fields from soldier profile

**4. Cross-Domain Question (1 min)**
```
"My soldier is going TDY but also needs leave the week before — what paperwork?"
```
Agent calls: `search_regulations [travel]` → `search_regulations [leave]`
Output: Regulation citations from both JTR and AR 600-8-10, forms needed

---

## Project Structure

```
AI-Expo-Hackathon-2026/
├── app.py                      # FastAPI backend — all API routes
├── ingest.py                   # PDF -> ChromaDB ingestion pipeline
├── requirements.txt
├── .env.example                # LLM provider configuration template
├── agents/
│   ├── react_agent.py          # ReAct agent — reasoning loop + tool orchestration
│   └── tools.py                # 4 tools: search, per diem, cost calc, form fill
├── data/
│   ├── gsa_cache.json          # GSA FY2026 per diem (649 locations, 42K zips)
│   ├── forms/                  # Fillable PDF templates (DD 1610, DA 31, DA 4856, DA 4187)
│   ├── jtr/                    # Joint Travel Regulations
│   ├── army_regs/              # AR 600-8-10, AR 623-3
│   ├── navy_regs/              # MILPERSMAN 1050, BUPERSINST 1610.10F
│   ├── af_regs/                # AFI 36-3003, AFI 36-2406
│   ├── marine_regs/            # MCO 1610.7
│   └── dod_regs/               # DoD FMR Vol 7A
├── frontend/
│   ├── src/App.tsx             # React UI — chat, profile, tool traces, form download
│   ├── src/AnimatedBackground.tsx
│   ├── src/index.css           # Military color scheme + classification banner
│   └── vite.config.ts          # Proxies /api/* -> localhost:8000
├── vectorstore/                # ChromaDB persistent store (gitignored)
├── output/                     # Generated form PDFs (gitignored)
├── ARCHITECTURE.md             # Detailed design decisions
└── FRONTEND_API_SPEC.md        # Full API contract for frontend/backend
```

---

## Why This Matters

- **3 million** service members navigate military bureaucracy daily
- **5–10 hours/week** spent by NCOs on administrative tasks that could be automated
- **Cross-branch coverage** — JTR is DoD-wide, plus Army, Navy, Air Force, and Marine Corps regulations
- **Verifiable** — judges can check the per diem rates, the JTR math, and the form fields. Every answer cites a specific regulation paragraph
- **Extensible** — adding a new regulation domain = ingesting one PDF + zero code changes

---

## Key JTR Constants (FY2026)

| Rule | Value |
|------|-------|
| POV mileage rate | $0.70/mile |
| First/last day M&IE | 75% of daily rate |
| Lodging nights | travel days - 1 |
| Standard CONUS lodging fallback | $110/night |
| Standard CONUS M&IE fallback | $68/day |
