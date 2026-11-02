import type { LlmProvider, SoldierProfile, Tab } from "@/types";

export const MODULES: { key: Tab; label: string; short: string; subtitle: string; placeholder: string }[] = [
  { key: "travel",     label: "TDY Travel",  short: "Travel",  subtitle: "Per diem, cost estimates, DD 1610", placeholder: "Ask about TDY travel…" },
  { key: "leave",      label: "Leave / HR",  short: "Leave",   subtitle: "DA 31 leave requests and HR actions", placeholder: "Ask about leave or HR actions…" },
  { key: "regulation", label: "Regulations", short: "Regs",    subtitle: "JTR, ARs and AFIs with citations", placeholder: "Ask about a regulation…" },
  { key: "eval",       label: "Evaluations", short: "Evals",   subtitle: "NCOERs, OERs and counseling", placeholder: "Ask about evaluations or counseling…" },
];

export const STARTERS: Record<Tab, string[]> = {
  travel: [
    "Send SPC Rivera TDY to Fort Moore, GA for 5 days starting July 10, driving POV from Fort Liberty",
    "What's the per diem for San Diego, CA?",
    "Do I need a GTC for a 2-day TDY?",
  ],
  leave: [
    "I need 10 days ordinary leave starting June 3 to visit family in Texas",
    "How many days of leave do I accrue per month?",
    "Can I take leave while on a temporary profile?",
  ],
  regulation: [
    "What does JTR say about POV mileage reimbursement?",
    "What's the receipt threshold for travel expenses?",
    "When must I file a travel voucher after returning from TDY?",
  ],
  eval: [
    "Help me write NCOER bullets for a squad leader",
    "What are the counseling requirements per AR 623-3?",
    "When is initial counseling due for a new soldier?",
  ],
};

export const EMPTY_PROFILE: SoldierProfile = {
  name_last_first: "", rank: "", grade: "", ssn_last4: "",
  dod_id: "", unit: "", installation: "", uic: "",
  supervisor_name: "", supervisor_title: "", duty_position: "",
};

const FORM_TITLES: [prefix: string, title: string][] = [
  ["DD_1610", "DD Form 1610"],
  ["DA_4856", "DA Form 4856"],
  ["DA_4187", "DA Form 4187"],
  ["DA_31",   "DA Form 31"],
];

const FORM_SUBTITLES: Record<string, string> = {
  "DD Form 1610": "TDY travel request",
  "DA Form 31":   "Leave request",
  "DA Form 4856": "Developmental counseling",
  "DA Form 4187": "Personnel action",
};

/** "DA_31_20260425_222438" → "DA Form 31". */
export function formTitle(formName: string): string {
  const upper = formName.toUpperCase();
  const hit = FORM_TITLES.find(([prefix]) => upper.startsWith(prefix));
  return hit ? hit[1] : formName.replace(/_\d{8}.*$/, "").replace(/_/g, " ");
}

export function formSubtitle(title: string): string | undefined {
  return FORM_SUBTITLES[title];
}

/** Used when the backend doesn't expose GET /api/llm/providers. */
export const FALLBACK_PROVIDERS: LlmProvider[] = [
  { id: "anthropic",  label: "Anthropic (Claude)", key_placeholder: "sk-ant-…", default_model: "claude-sonnet-4-6", needs_base_url: false },
  { id: "openai",     label: "OpenAI (GPT)",       key_placeholder: "sk-…",     default_model: "gpt-4.1",           needs_base_url: false },
  { id: "openrouter", label: "OpenRouter (Llama, Gemini, Mistral, …)", key_placeholder: "sk-or-…", default_model: "anthropic/claude-sonnet-4.6", needs_base_url: false },
];
