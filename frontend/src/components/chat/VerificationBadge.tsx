import { CircleCheckIcon, TriangleAlertIcon } from "lucide-react";
import type { Verification } from "@/types";

/** Result of the backend's post-answer check of figures and citations. Renders nothing when absent. */
export function VerificationBadge({ verification }: { verification?: Verification | null }) {
  if (!verification || !verification.checked) return null;

  const badNumbers   = verification.unverified_numbers ?? [];
  const badCitations = verification.unverified_citations ?? [];
  const passed: string[] = [];
  if (verification.numbers_ok === true && badNumbers.length === 0) passed.push("Figures verified against the calculation");
  if (verification.citations_ok === true && badCitations.length === 0) passed.push("Citations found in the regulations");

  if (passed.length === 0 && badNumbers.length === 0 && badCitations.length === 0) return null;

  return (
    <div className="flex flex-col gap-1.5">
      {passed.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {passed.map(text => (
            <span key={text} className="inline-flex items-center gap-1 rounded-full border border-olive/25 bg-olive/5 px-2 py-0.5 text-[11px] font-medium text-olive">
              <CircleCheckIcon className="size-3" />
              {text}
            </span>
          ))}
        </div>
      )}
      {(badNumbers.length > 0 || badCitations.length > 0) && (
        <div role="note" className="flex gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900">
          <TriangleAlertIcon className="mt-0.5 size-3.5 shrink-0 text-amber-600" />
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="font-semibold">Double-check before you rely on this answer</span>
            {badNumbers.length > 0 && (
              <span className="break-words">
                Figures not found in the calculation: {badNumbers.join(", ")}
              </span>
            )}
            {badCitations.length > 0 && (
              <span className="break-words">
                Citations not found in the regulations: {badCitations.join(", ")}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
