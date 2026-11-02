// Shapes shared with the backend (see FRONTEND_API_SPEC.md). Keep in sync with app.py.

export type Tab = "travel" | "leave" | "regulation" | "eval";

export type AgentStatus = "idle" | "thinking" | "done" | "error";

export interface ToolCallItem {
  tool: string;
  label: string;
  result_summary: string;
}

export interface FormOutput {
  form_name: string;
  filled_fields?: Record<string, string>;
  missing_fields?: string[];
  pdf_available: boolean;
  pdf_url: string | null;
  txt_summary: string;
}

export interface Verification {
  checked: boolean;
  numbers_ok: boolean | null;
  unverified_numbers: string[];
  citations_ok: boolean | null;
  unverified_citations: string[];
}

export interface ChatResponse {
  response: string;
  tool_calls: ToolCallItem[];
  form_output: FormOutput | null;
  error: string | null;
  verification?: Verification | null;
}

export interface HealthResponse {
  status: string;
  llm_provider?: string;
  llm_ready?: boolean;
  ollama?: boolean;
  vector_store_chunks: number;
  gsa_cache_loaded: boolean;
  model: string;
  offline_ready?: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isError?: boolean;
  tool_calls?: ToolCallItem[];
  form_output?: FormOutput;
  verification?: Verification | null;
}

export interface SoldierProfile {
  name_last_first: string;
  rank: string;
  grade: string;
  ssn_last4: string;
  dod_id: string;
  unit: string;
  installation: string;
  uic: string;
  supervisor_name: string;
  supervisor_title: string;
  duty_position: string;
}

/** Visitor's own LLM key — lives only in this browser. */
export interface OwnKey {
  provider: string;
  api_key: string;
  model: string;
  base_url?: string;
}

export interface LlmProvider {
  id: string;
  label: string;
  key_placeholder: string;
  default_model: string;
  needs_base_url: boolean;
}

export interface KeyCheckResult {
  valid: boolean | null;
  models: string[];
  default_model: string;
}

/** A form generated during this session, shown in the sidebar. */
export interface GeneratedForm {
  id: string;
  tab: Tab;
  form_name: string;
  title: string;
  pdf_url: string | null;
  missing_fields: string[];
  created_at: Date;
}
