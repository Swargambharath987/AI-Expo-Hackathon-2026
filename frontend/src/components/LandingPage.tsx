import { useState } from "react";
import AnimatedBackground from "@/components/AnimatedBackground";
import { ProfileForm } from "@/components/ProfileForm";
import { Button } from "@/components/ui/button";
import { EMPTY_PROFILE } from "@/lib/constants";
import type { SoldierProfile } from "@/types";

const HIGHLIGHTS = [
  { label: "TDY Travel",  desc: "Per diem & DD 1610" },
  { label: "Leave / HR",  desc: "DA 31 & policy" },
  { label: "Regulations", desc: "JTR & ARs" },
  { label: "Evaluations", desc: "NCOERs & OERs" },
];

export function LandingPage({ onComplete }: { onComplete: (profile: SoldierProfile) => void }) {
  const [step, setStep] = useState<"welcome" | "profile">("welcome");

  return (
    <div className="relative min-h-dvh overflow-x-hidden bg-ink text-white">
      <AnimatedBackground />

      <div className="relative z-10 mx-auto flex min-h-dvh w-full flex-col items-center justify-center px-4 py-10">
        {step === "welcome" ? (
          <div className="flex max-w-md flex-col items-center text-center">
            <img src="/logo-cropped.png" alt="Duty Line" className="mb-8 h-32 w-auto brightness-110 sm:mb-10 sm:h-44" />

            <p className="mb-8 text-sm leading-relaxed text-white/60">
              Travel, leave, regulations and evaluations — answered with citations, with the forms filled in for you.
            </p>

            <ul className="mb-10 grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-4 sm:gap-6">
              {HIGHLIGHTS.map(m => (
                <li key={m.label}>
                  <div className="text-xs font-semibold text-white/85">{m.label}</div>
                  <div className="mt-0.5 text-[11px] text-white/40">{m.desc}</div>
                </li>
              ))}
            </ul>

            <Button size="lg" onClick={() => setStep("profile")}
              className="h-11 bg-crimson px-12 text-[15px] font-semibold text-white hover:bg-crimson-700">
              Get started
            </Button>
            <button type="button" onClick={() => onComplete(EMPTY_PROFILE)}
              className="mt-4 text-xs text-white/40 underline-offset-4 hover:text-white/70 hover:underline">
              Skip setup
            </button>

            <div className="mt-12 text-[10px] tracking-[0.1em] text-white/25 uppercase">
              GenAI.mil Track | SCSP AI Expo Hackathon 2026
            </div>
          </div>
        ) : (
          <div className="flex w-full max-w-xl flex-col items-center">
            <img src="/logo-cropped.png" alt="Duty Line" className="mb-6 h-14 w-auto brightness-110" />
            <div className="w-full rounded-2xl bg-card p-5 text-card-foreground shadow-2xl sm:p-7">
              <h1 className="text-lg font-semibold">Soldier profile</h1>
              <p className="mt-1 mb-5 text-sm text-muted-foreground">
                Auto-fills your forms. Stored in this browser only — you can change it any time from the Profile button.
              </p>
              <ProfileForm
                defaultValues={EMPTY_PROFILE}
                onSubmit={onComplete}
                submitLabel="Continue"
                onCancel={() => setStep("welcome")}
                cancelLabel="Back"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
