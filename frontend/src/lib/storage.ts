// Browser-only persistence. Storage can be unavailable (private windows,
// blocked site data) — the app still works, it just won't remember things
// across reloads. Every access is wrapped in try/catch for that reason.
import { EMPTY_PROFILE } from "@/lib/constants";
import type { OwnKey, SoldierProfile } from "@/types";

const PROFILE_KEY = "dutyline_profile";
const OWN_KEY_KEY = "dutyline_own_key";

function read<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}

function write(key: string, value: unknown | null) {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable — keep going in memory */
  }
}

/** The soldier profile, sent with every chat request so forms carry the user's own details. */
export function loadProfile(): SoldierProfile | null {
  const saved = read<Partial<SoldierProfile>>(PROFILE_KEY);
  return saved ? { ...EMPTY_PROFILE, ...saved } : null;
}

export function storeProfile(profile: SoldierProfile) {
  write(PROFILE_KEY, profile);
}

/** Optional "use your own key" — never leaves this browser except with the user's own requests. */
export function loadOwnKey(): OwnKey | null {
  const k = read<OwnKey>(OWN_KEY_KEY);
  return k && typeof k.api_key === "string" && typeof k.provider === "string" ? k : null;
}

export function storeOwnKey(key: OwnKey | null) {
  write(OWN_KEY_KEY, key);
}
