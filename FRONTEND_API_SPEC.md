# Duty Line — Frontend ↔ Backend API Spec

Backend: FastAPI. Locally it runs on `http://localhost:8000` and the Vite dev
server proxies `/api/*` to it. In production one container serves both the
built React app (`/`) and the API (`/api/*`).

Types the frontend uses live in `frontend/src/types.ts`; keep them in sync with
`app.py`.

---

## Sessions

The first request sets an httponly `dl_session` cookie. Conversations are kept
per session **and per tab** in server memory (reset on restart), and generated
forms can only be downloaded by the session that made them. Nothing about the
soldier is stored server-side: the profile and any own API key live in the
browser's `localStorage` and travel with each request.

---

## Endpoints

### POST `/api/chat`
Runs the ReAct agent for one message.

**Request**
```json
{
  "message": "I need TDY to Fort Moore for 5 days starting July 10",
  "tab": "travel",
  "profile": { "name_last_first": "Carter, James R.", "rank": "SSG" },
  "llm": { "provider": "anthropic", "api_key": "sk-ant-…", "model": "claude-sonnet-5-5" }
}
```
- `message` — required, 1–2000 characters.
- `tab` — `"travel"` | `"leave"` | `"regulation"` | `"eval"` (anything else becomes `"travel"`).
- `profile` — optional. Fields: `name_last_first`, `rank`, `grade`, `ssn_last4`
  (exactly 4 digits or it is dropped), `dod_id`, `unit`, `installation`, `uic`,
  `supervisor_name`, `supervisor_title`, `duty_position`. Each is trimmed to 100
  characters; invalid fields are dropped, never rejected. Used for this request
  only.
- `llm` — optional "use your own key". `provider` is one of `anthropic`,
  `openai`, `openrouter`, `groq`, `mistral`, `gemini`, `deepseek`, `xai`,
  `together`, or `custom` (an OpenAI-compatible HTTPS endpoint, which also
  needs `base_url`). For presets the base URL is fixed server-side and `model`
  defaults per provider. Requests with their own key skip the daily rate caps
  but keep the per-minute limit. See *Custom endpoints* below.

**Response** (always HTTP 200 unless rate limited or the request is invalid)
```json
{
  "response": "**TDY Cost — Fort Moore, GA** …",
  "tool_calls": [
    {
      "tool": "calculate_travel_cost",
      "label": "Calculating travel cost: Fort Moore (5 days, POV)",
      "result_summary": "Lodging: $416.00 | Meals: $288.00 | Mileage: $518.00 | TOTAL: $1222.00"
    }
  ],
  "form_output": null,
  "error": null,
  "verification": {
    "checked": true,
    "numbers_ok": true,
    "unverified_numbers": [],
    "citations_ok": true,
    "unverified_citations": []
  }
}
```
- `tool` is `"search_regulations"` | `"get_per_diem"` | `"calculate_travel_cost"` | `"fill_form"`.
- `error` is `null` or a short code: `internal_error`, `agent_init_failed`,
  `timeout`. It is never raw exception text. `response` always holds a message
  that is safe to show the user, including when `error` is set.
- `verification` is `null` when the answer had nothing to check. Otherwise
  every `$` amount was matched to this turn's tool output and every citation
  (JTR section number, paragraph, table, chapter) to the regulation text
  retrieved this turn. A mismatched amount triggers one corrective retry
  before the answer is returned; anything still unmatched is listed in
  `unverified_*` so the UI can flag it.

**`form_output`** (when a form was filled this turn)
```json
{
  "form_name": "DA_31_20261103_101500_a1b2c3d4e5f60718",
  "filled_fields": { "Name (Last, First, MI)": "Carter, James R.", "Rank": "SSG" },
  "missing_fields": ["Date Leave Ends", "Contact Phone"],
  "pdf_available": true,
  "pdf_url": "/api/forms/DA_31_20261103_101500_a1b2c3d4e5f60718.pdf",
  "txt_summary": "FORM: DA_31 …"
}
```
`missing_fields` also lists identity or role values the user never gave (the
server leaves a guessed name, rank, unit or position blank instead of printing
it). `pdf_url` is `null` if PDF generation failed.

**Status codes**
- `429` + `Retry-After` — rate limited: 10 requests/min and 100/day per IP,
  500/day across all users (own-key requests skip the daily caps).
- `422` — invalid body (empty/too-long message, malformed `llm`, or a custom
  endpoint that isn't allowed).

---

### GET `/api/llm/providers`
Providers available for "use your own key" (`custom` is last).
```json
[{ "id": "anthropic", "label": "Anthropic (Claude)", "key_placeholder": "sk-ant-...",
   "default_model": "claude-sonnet-5-5", "needs_base_url": false }]
```

### POST `/api/llm/check`
Validate a key before saving it. Body: `{ "provider": "...", "api_key": "...", "base_url": "" }`
(`base_url` only for `custom`). Uses the provider's model-list call, so it
spends no tokens.
```json
{ "valid": true, "models": ["claude-sonnet-5-5", "…"], "default_model": "claude-sonnet-5-5" }
```
`valid` is `true`, `false` (key rejected) or `null` (provider unreachable).
Shares the per-minute rate limit; returns `429` or `422` like `/api/chat`.

**Custom endpoints.** For `provider: "custom"` the URL must be `https`, with no
credentials, query or fragment; every address it resolves to must be public
(checked again at connection time, so it can't be pointed at internal
services); redirects are never followed. Set `ALLOW_PRIVATE_LLM_URLS=1` only
for local development.

---

### GET `/api/health`
Called on page load to drive the status lights.
```json
{
  "status": "ok",
  "llm_provider": "claude",
  "llm_ready": true,
  "model": "claude-sonnet-5-5",
  "vector_store_chunks": 2063,
  "vector_store_ready": true,
  "gsa_cache_loaded": true,
  "offline_ready": false,
  "ollama": false
}
```
- `llm_provider`: `"claude"` | `"openrouter"` | `"ollama"` | `"custom"`.
- `llm_ready` is a real check, not a format check. For cloud providers the
  server asks the provider whether its key is accepted (a free model-list
  call, cached 10 min when good, 1 min when bad), and it also turns `false`
  for 5 minutes after a chat is refused for billing or quota reasons. For
  Ollama it is `true` only when `localhost:11434` answers.
- `offline_ready`: `true` only for Ollama with the vector store and GSA cache present.
- `ollama`: kept for older clients.

---

### GET `/api/forms/{filename}`
Download a generated form (`.pdf` or `.txt`). Filenames contain a random
suffix, are only served to the session that generated them, and are deleted
after about an hour. Anything else is `404`; path traversal or other
extensions are `400`.

### DELETE `/api/chat/history?tab=<tab>`
Clear this session's conversation for one tab (omit `tab` for all tabs).
Returns `{"status": "cleared"}`.

---

## Errors the UI should handle

| Situation | What arrives | UI |
|---|---|---|
| Rate limited | HTTP 429, `detail`, `Retry-After` | "Please wait" message |
| Agent failure or timeout | HTTP 200, `error` set, friendly `response` | Show `response` as an error |
| Provider out of credit | HTTP 200, friendly `response`, health turns red | Suggest using their own key |
| Bad own key | HTTP 200, `response` says the provider rejected it | Open AI settings |
| Server down | network error / non-JSON | "Couldn't reach the server" |

---

## CORS
Allowed origins: `http://localhost:5173`, `http://localhost:3000`,
`http://127.0.0.1:5173`, `http://127.0.0.1:3000`. In production the app and API
share one origin. No login is required.
