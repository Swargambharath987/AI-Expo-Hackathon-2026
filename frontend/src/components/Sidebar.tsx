import { DownloadIcon, FileTextIcon, SettingsIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MODULES, formSubtitle } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { GeneratedForm, HealthResponse, OwnKey, Tab } from "@/types";

const time = (d: Date) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export function Sidebar({
  forms,
  onOpenForm,
  health,
  ownKey,
  llmReady,
  onOpenAiSettings,
  className,
}: {
  forms: GeneratedForm[];
  onOpenForm: (tab: Tab) => void;
  health: HealthResponse | null;
  ownKey: OwnKey | null;
  llmReady: boolean | null;
  onOpenAiSettings: () => void;
  className?: string;
}) {
  return (
    <div className={cn("flex h-full flex-col bg-card", className)}>
      <div className="flex-1 overflow-y-auto p-4">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-[11px] font-bold tracking-[0.12em] text-slate-400 uppercase">Your forms</h2>
          {forms.length > 0 && <span className="text-[11px] text-slate-400">{forms.length} this session</span>}
        </div>
        {forms.length === 0 ? <FormsEmptyState /> : (
          <ul className="flex flex-col gap-2">
            {[...forms].reverse().map(f => <FormRow key={f.id} form={f} onOpen={() => onOpenForm(f.tab)} />)}
          </ul>
        )}
      </div>

      <StatusPanel health={health} ownKey={ownKey} llmReady={llmReady} onOpenAiSettings={onOpenAiSettings} />
    </div>
  );
}

function FormsEmptyState() {
  return (
    <div className="rounded-xl border border-dashed px-4 py-6 text-center">
      <FileTextIcon className="mx-auto mb-2 size-5 text-slate-300" />
      <p className="text-xs font-medium text-slate-600">No forms yet</p>
      <p className="mt-1 text-xs leading-relaxed text-slate-400">
        Ask for a TDY request, a leave form or a counseling record. Filled PDFs show up here to download.
      </p>
    </div>
  );
}

function FormRow({ form, onOpen }: { form: GeneratedForm; onOpen: () => void }) {
  const tabLabel = MODULES.find(m => m.key === form.tab)?.short;
  const missing = form.missing_fields.length;

  return (
    <li className="rounded-xl border bg-card p-3 transition-colors hover:border-slate-300">
      <div className="flex items-start gap-2.5">
        <button type="button" onClick={onOpen} className="min-w-0 flex-1 text-left" title={`Go to the ${tabLabel} chat`}>
          <div className="truncate text-[13px] font-semibold text-foreground">{form.title}</div>
          <div className="truncate text-[11px] text-muted-foreground">
            {formSubtitle(form.title) ?? tabLabel} · {time(form.created_at)}
          </div>
        </button>
        {form.pdf_url && (
          <Button asChild variant="outline" size="icon-sm" className="shrink-0" aria-label={`Download ${form.title}`}>
            <a href={form.pdf_url} download><DownloadIcon /></a>
          </Button>
        )}
      </div>
      <div className="mt-2">
        {missing > 0 ? (
          <Badge variant="outline" className="border-amber-300 bg-amber-50 text-[10px] text-amber-800">
            {missing} field{missing === 1 ? "" : "s"} to fill by hand
          </Badge>
        ) : (
          <Badge variant="outline" className="border-olive/25 bg-olive/5 text-[10px] text-olive">Complete</Badge>
        )}
      </div>
    </li>
  );
}

function StatusPanel({
  health, ownKey, llmReady, onOpenAiSettings,
}: {
  health: HealthResponse | null;
  ownKey: OwnKey | null;
  llmReady: boolean | null;
  onOpenAiSettings: () => void;
}) {
  const aiLabel = ownKey
    ? `AI Model · your ${ownKey.provider} key`
    : health?.llm_provider === "ollama" ? "AI Model · Ollama (local)" : "AI Model";

  const rows: { label: string; ok: boolean | null }[] = [
    { label: aiLabel, ok: llmReady },
    { label: `Regulation Search${health ? ` · ${health.vector_store_chunks.toLocaleString()} sections` : ""}`, ok: health ? health.vector_store_chunks > 0 : null },
    { label: "Per Diem Rates", ok: health ? health.gsa_cache_loaded : null },
  ];

  return (
    <div className="border-t px-4 py-3">
      <ul className="mb-2 flex flex-col gap-1.5">
        {rows.map(({ label, ok }) => (
          <li key={label} className="flex items-center gap-2 text-[11px] text-slate-500">
            <span className={cn("size-2 shrink-0 rounded-full", ok === null ? "bg-slate-300" : ok ? "bg-green-500" : "bg-red-500")}
              aria-label={ok === null ? "checking" : ok ? "ready" : "unavailable"} />
            <span className="truncate">{label}</span>
          </li>
        ))}
      </ul>
      <Button variant="outline" size="sm" onClick={onOpenAiSettings}
        className={cn("w-full justify-start", llmReady === false && "border-crimson/30 text-crimson")}>
        <SettingsIcon />
        {llmReady === false ? "AI settings — use your own key" : "AI settings"}
      </Button>
    </div>
  );
}
