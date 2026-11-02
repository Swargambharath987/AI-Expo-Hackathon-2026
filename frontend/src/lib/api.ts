// Thin wrappers around the Duty Line backend. Vite proxies /api → :8000 in dev.
import { FALLBACK_PROVIDERS } from "@/lib/constants";
import type {
  ChatResponse, HealthResponse, KeyCheckResult, LlmProvider, OwnKey, SoldierProfile, Tab,
} from "@/types";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function detailOf(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => null);
  return typeof body?.detail === "string" ? body.detail : fallback;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}

export type ChatResult =
  | { ok: true; data: ChatResponse }
  /** `message` is always user-facing. */
  | { ok: false; message: string };

export async function sendChat(args: {
  message: string;
  tab: Tab;
  profile: SoldierProfile;
  llm: OwnKey | null;
}): Promise<ChatResult> {
  let res: Response;
  try {
    res = await fetch("/api/chat", {
      method: "POST",
      headers: JSON_HEADERS,
      body: JSON.stringify({
        message: args.message,
        tab: args.tab,
        profile: args.profile,
        ...(args.llm ? { llm: args.llm } : {}),
      }),
    });
  } catch {
    return { ok: false, message: "Couldn't reach the server — check your connection and try again." };
  }

  if (res.status === 429) {
    return { ok: false, message: await detailOf(res, "Too many requests — please wait a moment and try again.") };
  }
  if (!res.ok) {
    return { ok: false, message: `The server returned an error (${res.status}). Please try again.` };
  }
  const data: ChatResponse = await res.json();
  // On error the backend still puts a user-facing message in `response`.
  if (data.error) return { ok: false, message: data.response || "Something went wrong — please try again." };
  return { ok: true, data };
}

export async function clearChatHistory(tab: Tab): Promise<void> {
  await fetch(`/api/chat/history?tab=${encodeURIComponent(tab)}`, { method: "DELETE" });
}

/** Provider list from the backend; falls back to the built-in three if the endpoint is missing. */
export async function fetchProviders(): Promise<LlmProvider[]> {
  try {
    const res = await fetch("/api/llm/providers");
    if (!res.ok) return FALLBACK_PROVIDERS;
    const list: LlmProvider[] = await res.json();
    return Array.isArray(list) && list.length > 0 ? list : FALLBACK_PROVIDERS;
  } catch {
    return FALLBACK_PROVIDERS;
  }
}

export type KeyCheckOutcome =
  | { kind: "valid"; models: string[]; default_model: string }
  | { kind: "rejected" }        // provider said the key is wrong
  | { kind: "unreachable" }     // provider couldn't be reached
  | { kind: "refused" }         // 422: the backend refused the URL/key itself
  | { kind: "error"; message: string };

export async function checkKey(body: { provider: string; api_key: string; base_url?: string }): Promise<KeyCheckOutcome> {
  let res: Response;
  try {
    res = await fetch("/api/llm/check", { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body) });
  } catch {
    return { kind: "error", message: "Couldn't reach the Duty Line server — check your connection." };
  }
  if (res.status === 422) return { kind: "refused" };
  if (res.status === 429) return { kind: "error", message: await detailOf(res, "Too many checks — wait a moment and try again.") };
  if (!res.ok) return { kind: "error", message: `Key check failed (server error ${res.status}).` };

  const data: KeyCheckResult = await res.json();
  if (data.valid === null) return { kind: "unreachable" };
  if (!data.valid) return { kind: "rejected" };
  return { kind: "valid", models: data.models ?? [], default_model: data.default_model };
}
