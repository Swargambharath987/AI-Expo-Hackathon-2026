import { useEffect, useMemo, useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";
import { CircleAlertIcon, CircleCheckIcon, Loader2Icon, ShieldCheckIcon } from "lucide-react";
import { ModelCombobox } from "@/components/ModelCombobox";
import { Button } from "@/components/ui/button";
import {
  Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { checkKey, fetchProviders, type KeyCheckOutcome } from "@/lib/api";
import { FALLBACK_PROVIDERS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { LlmProvider, OwnKey } from "@/types";

type Values = { provider: string; api_key: string; base_url: string };

function makeSchema(providers: LlmProvider[]) {
  return z.object({
    provider: z.string().min(1, "Choose a provider."),
    api_key:  z.string().trim().min(1, "Paste your API key."),
    base_url: z.string().trim(),
  }).superRefine((v, ctx) => {
    const p = providers.find(x => x.id === v.provider);
    if (p?.needs_base_url && !/^https?:\/\/[^\s/]+/i.test(v.base_url)) {
      ctx.addIssue({ code: "custom", path: ["base_url"], message: "Enter the full URL, starting with https://" });
    }
  });
}

type CheckState = { kind: "idle" } | { kind: "checking" } | KeyCheckOutcome;

export function AiSettingsDialog({
  open,
  onOpenChange,
  current,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  current: OwnKey | null;
  onSave: (key: OwnKey | null) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>AI settings</DialogTitle>
          <DialogDescription>
            Optionally use your own API key instead of the app's.
          </DialogDescription>
        </DialogHeader>
        <AiSettingsForm current={current} onSave={onSave} onClose={() => onOpenChange(false)} />
      </DialogContent>
    </Dialog>
  );
}

function AiSettingsForm({
  current,
  onSave,
  onClose,
}: {
  current: OwnKey | null;
  onSave: (key: OwnKey | null) => void;
  onClose: () => void;
}) {
  const [providers, setProviders] = useState<LlmProvider[]>(FALLBACK_PROVIDERS);
  useEffect(() => { fetchProviders().then(setProviders); }, []);

  const schema = useMemo(() => makeSchema(providers), [providers]);
  const form = useForm<Values>({
    resolver: zodResolver(schema),
    defaultValues: {
      provider: current?.provider ?? FALLBACK_PROVIDERS[0].id,
      api_key:  current?.api_key ?? "",
      base_url: current?.base_url ?? "",
    },
  });

  // A saved key was already verified; start from it.
  const [check, setCheck] = useState<CheckState>(
    current ? { kind: "valid", models: [current.model], default_model: current.model } : { kind: "idle" },
  );
  const [model, setModel] = useState(current?.model ?? "");

  // Any edit to provider / key / URL invalidates the previous check.
  useEffect(() => {
    const sub = form.watch((_, { name }) => { if (name) setCheck({ kind: "idle" }); });
    return () => sub.unsubscribe();
  }, [form]);

  const providerId = form.watch("provider");
  const provider = providers.find(p => p.id === providerId) ?? providers[0];
  const isCustom = provider?.needs_base_url ?? false;

  async function runCheck(v: Values) {
    setCheck({ kind: "checking" });
    const outcome = await checkKey({
      provider: v.provider,
      api_key: v.api_key.trim(),
      ...(isCustom ? { base_url: v.base_url.trim() } : {}),
    });
    setCheck(outcome);
    if (outcome.kind === "valid") {
      const { models, default_model } = outcome;
      setModel(models.includes(default_model) ? default_model : (models[0] ?? default_model));
    }
  }

  function onSubmit(v: Values) {
    if (check.kind !== "valid") { void runCheck(v); return; }
    onSave({
      provider: v.provider,
      api_key: v.api_key.trim(),
      model: model || provider.default_model,
      ...(isCustom ? { base_url: v.base_url.trim() } : {}),
    });
    onClose();
  }

  const models = check.kind === "valid"
    ? (check.models.length > 0 ? check.models : [check.default_model].filter(Boolean))
    : [];

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} noValidate className="flex flex-col gap-5">
      <FieldGroup className="gap-4">
        <Controller
          name="provider"
          control={form.control}
          render={({ field }) => (
            <Field>
              <FieldLabel htmlFor="ai-provider">Provider</FieldLabel>
              <Select value={field.value} onValueChange={field.onChange}>
                <SelectTrigger id="ai-provider" className="w-full">
                  <SelectValue placeholder="Choose a provider" />
                </SelectTrigger>
                <SelectContent position="popper">
                  {providers.map(p => <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>)}
                </SelectContent>
              </Select>
            </Field>
          )}
        />

        {isCustom && (
          <Controller
            name="base_url"
            control={form.control}
            render={({ field, fieldState }) => (
              <Field data-invalid={fieldState.invalid}>
                <FieldLabel htmlFor="ai-base-url">Base URL</FieldLabel>
                <Input {...field} id="ai-base-url" type="url" inputMode="url" autoComplete="off"
                  placeholder="https://api.example.com/v1" aria-invalid={fieldState.invalid} />
                {fieldState.invalid
                  ? <FieldError errors={[fieldState.error]} />
                  : <FieldDescription>An OpenAI-compatible endpoint (the part before <code>/chat/completions</code>).</FieldDescription>}
              </Field>
            )}
          />
        )}

        <Controller
          name="api_key"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="ai-key">API key</FieldLabel>
              <Input {...field} id="ai-key" type="password" autoComplete="off" spellCheck={false}
                placeholder={provider?.key_placeholder ?? "Your API key"} aria-invalid={fieldState.invalid} />
              {fieldState.invalid && <FieldError errors={[fieldState.error]} />}
            </Field>
          )}
        />

        {check.kind === "valid" && (
          <Field>
            <FieldLabel htmlFor="ai-model">Model</FieldLabel>
            <ModelCombobox id="ai-model" models={models} value={model} onChange={setModel} />
          </Field>
        )}
      </FieldGroup>

      <CheckMessage check={check} isCustom={isCustom} />

      <p className="flex gap-2 rounded-lg bg-muted px-3 py-2.5 text-xs leading-relaxed text-muted-foreground">
        <ShieldCheckIcon className="mt-0.5 size-3.5 shrink-0 text-olive" />
        Your key is stored only in this browser and sent with your own requests. It is never saved on the server.
      </p>

      <DialogFooter className="flex-col gap-2 sm:flex-row sm:justify-between">
        {current ? (
          <Button type="button" variant="ghost" size="lg" className="text-muted-foreground"
            onClick={() => { onSave(null); onClose(); }}>
            Stop using my key
          </Button>
        ) : <span />}
        <div className="flex flex-col-reverse gap-2 sm:flex-row">
          <Button type="button" variant="outline" size="lg" onClick={onClose}>Cancel</Button>
          <Button type="submit" size="lg" disabled={check.kind === "checking"}
            className={cn("px-4", check.kind === "valid" && "bg-olive hover:bg-olive-700")}>
            {check.kind === "checking" && <Loader2Icon className="animate-spin" />}
            {check.kind === "valid" ? "Use this key" : check.kind === "checking" ? "Checking…" : "Check key"}
          </Button>
        </div>
      </DialogFooter>
    </form>
  );
}

function CheckMessage({ check, isCustom }: { check: CheckState; isCustom: boolean }) {
  const msg = (() => {
    switch (check.kind) {
      case "valid":
        return { ok: true, text: "Key accepted. Pick a model, then choose “Use this key”." };
      case "rejected":
        return { ok: false, text: "The provider rejected this key. Check it was copied in full and is still active." };
      case "unreachable":
        return {
          ok: false,
          text: isCustom
            ? "Couldn't reach that URL. Check the address and that the server is up."
            : "Couldn't reach the provider right now. Try again in a moment.",
        };
      case "refused":
        return {
          ok: false,
          text: isCustom
            ? "That base URL isn't allowed. Use a public https:// address (local and private addresses are blocked)."
            : "The server refused this key. Check its format and try again.",
        };
      case "error":
        return { ok: false, text: check.message };
      default:
        return null;
    }
  })();
  if (!msg) return null;

  return (
    <div role="status" className={cn(
      "flex items-start gap-2 rounded-lg border px-3 py-2.5 text-sm",
      msg.ok ? "border-olive/25 bg-olive/5 text-olive" : "border-crimson/25 bg-crimson/5 text-crimson",
    )}>
      {msg.ok ? <CircleCheckIcon className="mt-0.5 size-4 shrink-0" /> : <CircleAlertIcon className="mt-0.5 size-4 shrink-0" />}
      <span>{msg.text}</span>
    </div>
  );
}
