# Duty Line — Frontend ↔ Backend API Spec

**For the frontend team.**
Backend runs FastAPI on `http://localhost:8000`.
Frontend dev server proxies `/api/*` → `http://localhost:8000`.

---

## Endpoints

### POST `/api/chat`
Main chat endpoint. Runs the ReAct agent and returns a structured response.

**Request:**
```json
{
  "message": "I need TDY to Fort Moore for 5 days starting July 10",
  "tab": "travel",
  "profile": { ... }
}
```
`tab` values: `"travel"` | `"leave"` | `"regulation"` | `"eval"`
`profile` is the soldier profile (same fields as `SoldierProfile`). It lives in the browser's
localStorage and is sent with every request; the backend uses it for that request's context and
form fills and never stores it. Invalid fields are dropped, not rejected (e.g. a non-4-digit
`ssn_last4` becomes `""`).

`llm` (optional) — visitor's own key: `{"provider": "anthropic" | "openai", "api_key": "...", "model": "..."}`.
Used for this request instead of the server key; `model` defaults per provider. Base URLs are fixed
server-side. Own-key requests skip the daily rate caps.

### POST `/api/llm/check`
Validate a visitor's own key before saving it. Body: `{"provider": "anthropic" | "openai", "api_key": "..."}`.
Response: `{"valid": true | false | null, "models": [...], "default_model": "..."}` — `null` means the
provider couldn't be reached. Uses the provider's model-list endpoint (no tokens spent).

`error` in the response is a short code (`internal_error`, `agent_init_failed`, `timeout`) —
never raw exception text. `response` always holds a user-facing message.

**Response:**
```json
{
  "response": "The total cost for SPC Rivera's TDY to Fort Moore will be $1,222.00...",
  "tool_calls": [
    {
      "tool": "calculate_travel_cost",
      "label": "Calculating travel cost: Fort Moore (5 days, POV)",
      "result_summary": "Lodging: $416.00 | Meals: $288.00 | Mileage: $518.00 | TOTAL: $1222.00"
    }
  ],
  "form_output": null,
  "error": null
}
```

`tool` values: `"search_regulations"` | `"get_per_diem"` | `"calculate_travel_cost"` | `"fill_form"`

`form_output` when a form was filled — **pdf_path is never null now** (reportlab ensures a PDF is always generated):
```json
{
  "form_name": "DA_31",
  "filled_fields": {
    "Name (Last, First, MI)": "Rivera, Maria J.",
    "Rank": "SPC",
    "Organization/Unit": "1-503 INF, 82nd ABN",
    "Type of Leave": "Annual Leave",
    "Date Leave Begins": "03 JUN 2026",
    "Number of Days": "10"
  },
  "missing_fields": ["Date Leave Ends", "Contact Phone", "Approving Official"],
  "pdf_path": "/api/forms/DA_31_20260425_222438.pdf",
  "summary": "==================================================\nFORM: DA_31\n..."
}
```

---

### GET `/api/health`
Check backend + model status. Call on mount to populate sidebar status dots.

**Response:**
```json
{
  "status": "ok",
  "llm_provider": "claude",
  "llm_ready": true,
  "model": "claude-sonnet-4-6",
  "vector_store_chunks": 2063,
  "vector_store_ready": true,
  "gsa_cache_loaded": true,
  "offline_ready": false,
  "ollama": false
}
```

`llm_provider` values: `"claude"` | `"openrouter"` | `"ollama"` | `"custom"`
`llm_ready`: true for all cloud providers (Claude, OpenRouter); true for Ollama only when localhost:11434 is reachable.
`offline_ready`: true only when provider is `"ollama"` and Ollama + vector store + GSA cache are all available.
`ollama`: kept for backwards compatibility — true only when provider is `"ollama"` and Ollama is reachable; false otherwise.

---

### GET `/api/forms/{filename}`
Download a generated form PDF or TXT.

**Example:** `GET /api/forms/DA_31_20260425_222438.pdf`
**Response:** PDF binary (`Content-Type: application/pdf`)
**Security:** path traversal and extension checks — only `.pdf` and `.txt` served.

---

### DELETE `/api/chat/history?tab=<tab>`
Clear this session's conversation memory for one tab (omit `tab` to clear all tabs). Call when user clicks "New Conversation".
Conversations are per browser session (httponly `dl_session` cookie) and per tab.

**Response:** `{"status": "cleared"}`

---

## Integration Status — as of 2026-04-25 evening

### Backend ✅ ALL DONE
- [x] `POST /api/chat` — ReAct agent, structured `tool_calls` + `form_output`
- [x] `POST /api/chat` returns `429` + `Retry-After` when rate limited (10/min, 100/day per IP; 500/day total)
- [x] `GET /api/health` — provider-agnostic LLM + ChromaDB + GSA status (`llm_provider`, `llm_ready`, `ollama` for backwards compat)
- [x] `GET /api/forms/{filename}` — PDF download with path traversal protection
- [x] Soldier profile sent with each `POST /api/chat` (stored in browser localStorage; `/api/profile` endpoints removed)
- [x] `DELETE /api/chat/history` — clears conversation
- [x] Response shape matches frontend `ToolCallItem` / `FormOutput` types exactly
- [x] `profile` field accepted in chat request (no 422 errors)
- [x] **PDF always generated** — reportlab fallback for all forms (DA_31, DA_4856, DA_4187)
- [x] **DD_1610 AcroForm filling** — uses actual PDF field names (`name`, `ssn`, `org_elem`, `pds`, `proc_date`, etc.) + reportlab summary PDF
- [x] 2063 regulation chunks in ChromaDB (JTR, AR 600-8-10, AFI 36-3003, MILPERSMAN 1050, AR 623-3, AFI 36-2406, BUPERSINST 1610, DoD FMR Vol 7A, MCO 1610.7)
- [x] 40 installation distance pairs (Fort Liberty↔Moore, Alaska, Hawaii, Texas, DC area, West Coast)
- [x] `search_regulations` always called for policy/reg questions — no hallucinated answers

### Frontend ✅ DONE (team shipped new App.tsx)
- [x] Chat UI with tab system (travel / leave / regulation / eval)
- [x] `tool_calls` traces rendered above response
- [x] Status states: thinking / searching / calculating / done / error
- [x] Example queries per tab
- [x] Health status dots wired to `GET /api/health` on mount
- [x] `SoldierProfile` type matches backend `ProfileModel` exactly
- [x] `FormOutput.pdf_path` used for download button
- [x] `ToolCallItem` type matches backend response exactly

### Remaining ❌ (frontend wiring — check with team)
- [ ] "New conversation" / clear button → `DELETE /api/chat/history`
- [ ] `error` field handling: show error state when `data.error !== null`

---

## Demo Scenarios — verified working

| Scenario | Tools called | Output |
|----------|-------------|--------|
| "SPC Rivera TDY to Fort Moore, 5 days, POV from Fort Liberty" | `calculate_travel_cost` | $1,222 breakdown (lodging $416 + meals $288 + mileage $518) |
| "10 days annual leave starting June 3" | `fill_form DA_31` | PDF download, 8 fields auto-filled from profile |
| "Is the GTC mandatory for TDY?" | `search_regulations [travel]` | Answer from JTR 010204, no hallucination |
| "What does JTR say about POV mileage?" | `search_regulations [travel]` | Exact JTR text with section citation |
| "Fill a DD 1610 for Fort Moore trip" | `fill_form DD_1610` | 52KB real AcroForm PDF + formatted summary |

---

## Error Handling

All errors return:
```json
{"response": "...", "tool_calls": [], "form_output": null, "error": "description"}
```
Frontend should show error state when `error !== null`.

---

## CORS

Backend allows: `http://localhost:5173`, `http://localhost:3000`, `http://127.0.0.1:5173`, `http://127.0.0.1:3000`
No auth required — local only.
