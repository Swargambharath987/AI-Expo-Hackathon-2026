import { useEffect, useState } from "react";
import { fetchHealth } from "@/lib/api";
import type { HealthResponse } from "@/types";

/**
 * Polls GET /api/health: on mount, again after 4s (the backend may still be
 * warming up), then every 2 minutes so a key that breaks mid-session shows up.
 */
export function useHealth(): HealthResponse | null {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    let alive = true;
    const load = () => fetchHealth().then(h => { if (alive) setHealth(h); }).catch(() => {});
    load();
    const retry = setTimeout(load, 4000);
    const poll = setInterval(load, 120_000);
    return () => { alive = false; clearTimeout(retry); clearInterval(poll); };
  }, []);

  return health;
}
