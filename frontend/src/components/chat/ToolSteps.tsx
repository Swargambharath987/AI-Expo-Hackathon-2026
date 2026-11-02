import { useState } from "react";
import { BookOpenIcon, CalculatorIcon, ChevronRightIcon, FileTextIcon, MapPinIcon, WrenchIcon } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ToolCallItem } from "@/types";

const ICONS: Record<string, typeof WrenchIcon> = {
  search_regulations:    BookOpenIcon,
  get_per_diem:          MapPinIcon,
  calculate_travel_cost: CalculatorIcon,
  fill_form:             FileTextIcon,
};

const clean = (s: string) => s.replace(/^\s*→\s*/, "").trim();

/** Short past-tense phrase for one tool call, e.g. "searched JTR 020210". */
function stepPhrase(tc: ToolCallItem): string {
  const summary = clean(tc.result_summary);
  switch (tc.tool) {
    case "search_regulations": {
      // "Found: JTR | 020210 Per diem … (score 0.81)"
      const m = summary.match(/^Found:\s*([^|]+?)\s*\|\s*([^\s(|]+)/);
      return m ? `searched ${m[1]} ${m[2]}` : "searched regulations";
    }
    case "get_per_diem": {
      const m = tc.label.match(/for\s+([^,]+)/);
      return m ? `per diem for ${m[1].trim()}` : "looked up per diem";
    }
    case "calculate_travel_cost":
      return "calculated cost";
    case "fill_form": {
      const m = tc.label.match(/Filling\s+([A-Z]{2}_\d+)/i);
      return m ? `filled ${m[1].replace("_", " ")}` : "filled form";
    }
    default:
      return tc.tool.replace(/_/g, " ");
  }
}

export function summarizeSteps(calls: ToolCallItem[]): string {
  const phrases = [...new Set(calls.map(stepPhrase))];
  const shown = phrases.slice(0, 3).join(" · ");
  const more = phrases.length > 3 ? ` · +${phrases.length - 3} more` : "";
  return `${calls.length} step${calls.length === 1 ? "" : "s"} · ${shown}${more}`;
}

/** The agent's tool calls, collapsed to a one-line summary until clicked. */
export function ToolSteps({ calls }: { calls: ToolCallItem[] }) {
  const [open, setOpen] = useState(false);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="group flex w-full min-w-0 items-center gap-1.5 rounded-md py-1 text-left text-xs font-medium text-muted-foreground hover:text-navy">
        <ChevronRightIcon className={cn("size-3.5 shrink-0 transition-transform", open && "rotate-90")} />
        <span className="truncate">{summarizeSteps(calls)}</span>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ol className="mt-2 flex flex-col gap-1.5">
          {calls.map((tc, i) => {
            const Icon = ICONS[tc.tool] ?? WrenchIcon;
            return (
              <li key={i} className="flex gap-2.5 rounded-lg border bg-slate-50 px-3 py-2">
                <Icon className="mt-0.5 size-3.5 shrink-0 text-navy/70" />
                <div className="min-w-0">
                  <div className="text-xs font-semibold break-words text-navy">{tc.label}</div>
                  {tc.result_summary && (
                    <div className="mt-0.5 text-xs break-words text-muted-foreground">{clean(tc.result_summary)}</div>
                  )}
                </div>
              </li>
            );
          })}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
}
