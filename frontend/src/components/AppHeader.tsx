import { MenuIcon, UserRoundIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { AgentStatus, SoldierProfile } from "@/types";

type Pill = { label: string; tone: "ok" | "busy" | "bad" | "idle" };

export function statusPill(status: AgentStatus, llmReady: boolean | null): Pill {
  if (llmReady === false) return { label: "AI unavailable", tone: "bad" };
  if (status === "error")    return { label: "Error", tone: "bad" };
  if (status === "thinking") return { label: "Reasoning…", tone: "busy" };
  if (status === "done")     return { label: "Complete", tone: "ok" };
  if (llmReady === null)     return { label: "Connecting…", tone: "idle" };
  return { label: "Ready", tone: "ok" };
}

const TONE = {
  ok:   "border-green-300 bg-green-50 text-green-800 [&>span]:bg-green-500",
  busy: "border-amber-200 bg-amber-50 text-amber-800 [&>span]:bg-amber-500 [&>span]:animate-pulse",
  bad:  "border-red-300 bg-red-50 text-red-700 [&>span]:bg-red-500",
  idle: "border-slate-200 bg-slate-50 text-slate-500 [&>span]:bg-slate-300",
};

export function AppHeader({
  profile,
  pill,
  onOpenProfile,
  onOpenMenu,
}: {
  profile: SoldierProfile;
  pill: Pill;
  onOpenProfile: () => void;
  onOpenMenu: () => void;
}) {
  const hasProfile = !!profile.name_last_first;

  return (
    <header className="flex shrink-0 items-center gap-2 border-b bg-card px-4 py-2 shadow-xs sm:gap-3 sm:px-6">
      <Button variant="ghost" size="icon-lg" className="-ml-2 md:hidden" onClick={onOpenMenu} aria-label="Open forms and status">
        <MenuIcon className="size-5" />
      </Button>

      <div className="flex min-w-0 items-center gap-3">
        <img src="/logo-cropped.png" alt="Duty Line" className="h-9 w-auto object-contain sm:h-11" />
        <div className="hidden h-8 w-px bg-border lg:block" />
        <span className="hidden text-[11px] font-bold tracking-[0.12em] text-slate-400 uppercase lg:inline">AI Military Assistant</span>
      </div>

      <div className="ml-auto flex items-center gap-2 sm:gap-3">
        <div role="status" aria-live="polite"
          className={cn("flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-semibold whitespace-nowrap sm:px-3", TONE[pill.tone])}>
          <span className="size-2 rounded-full" />
          {pill.label}
        </div>

        <Button variant="outline" size="lg" onClick={onOpenProfile}
          className={cn("h-10 gap-2 px-2.5 sm:px-3", !hasProfile && "border-crimson/40 text-crimson")}>
          <UserRoundIcon className="size-4" />
          {hasProfile ? (
            <span className="hidden max-w-48 flex-col items-start leading-tight sm:flex">
              <span className="truncate text-[13px] font-semibold">{profile.rank} {profile.name_last_first}</span>
              {(profile.unit || profile.installation) && (
                <span className="truncate text-[11px] font-normal text-muted-foreground">
                  {[profile.unit, profile.installation].filter(Boolean).join(" · ")}
                </span>
              )}
            </span>
          ) : (
            <span className="hidden sm:inline">Set up profile</span>
          )}
          <span className="sm:hidden">Profile</span>
        </Button>
      </div>
    </header>
  );
}
