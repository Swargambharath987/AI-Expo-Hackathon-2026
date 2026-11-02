import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUpIcon, BookOpenIcon, ClipboardListIcon, PlaneIcon, RotateCcwIcon, StarIcon } from "lucide-react";
import { MessageItem, TypingIndicator } from "@/components/chat/MessageItem";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { MODULES, STARTERS } from "@/lib/constants";
import type { Message, Tab } from "@/types";

export const MODULE_ICONS: Record<Tab, typeof PlaneIcon> = {
  travel: PlaneIcon,
  leave: ClipboardListIcon,
  regulation: BookOpenIcon,
  eval: StarIcon,
};

export function ChatPanel({
  tab,
  onTabChange,
  messages,
  busy,
  onSend,
  onClear,
}: {
  tab: Tab;
  onTabChange: (tab: Tab) => void;
  messages: Message[];
  busy: boolean;
  onSend: (text: string) => void;
  onClear: () => void;
}) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const mod = MODULES.find(m => m.key === tab)!;
  const ModIcon = MODULE_ICONS[tab];

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [messages, busy, tab]);

  function submit(e?: FormEvent) {
    e?.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    onSend(text);
    setInput("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
  }

  return (
    <main className="flex min-w-0 flex-1 flex-col">
      {/* Module tabs — the only module switcher in the app */}
      <div className="flex shrink-0 items-center gap-2 border-b bg-card px-4 py-2 sm:px-5">
        <Tabs value={tab} onValueChange={v => onTabChange(v as Tab)} className="min-w-0 flex-1">
          <TabsList className="h-9 w-full bg-transparent p-0 sm:w-fit">
            {MODULES.map(m => {
              const Icon = MODULE_ICONS[m.key];
              return (
                <TabsTrigger key={m.key} value={m.key}
                  className="h-9 flex-1 gap-1.5 px-2 text-[13px] text-slate-500 data-active:bg-navy! data-active:text-white! data-active:shadow-none sm:flex-none sm:px-3.5">
                  <Icon className="hidden size-3.5 sm:block" />
                  <span className="sm:hidden">{m.short}</span>
                  <span className="hidden sm:inline">{m.label}</span>
                </TabsTrigger>
              );
            })}
          </TabsList>
        </Tabs>
        {messages.length > 0 && (
          <Button variant="ghost" size="sm" onClick={onClear} disabled={busy} className="shrink-0 text-muted-foreground"
            aria-label="New conversation in this tab">
            <RotateCcwIcon /> <span className="hidden sm:inline">New chat</span>
          </Button>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-6">
        {messages.length === 0 ? (
          <div className="mx-auto flex min-h-full w-full max-w-xl flex-col justify-center">
            <div className="mb-1 flex items-center gap-2.5 text-2xl font-bold text-foreground">
              <span className="flex size-9 items-center justify-center rounded-lg bg-navy text-white">
                <ModIcon className="size-4.5" />
              </span>
              {mod.label}
            </div>
            <p className="mb-6 text-sm text-muted-foreground">{mod.subtitle}. Describe your task below, or try one of these:</p>
            <div className="flex flex-col gap-2">
              {STARTERS[tab].map(s => (
                <button key={s} type="button"
                  onClick={() => { setInput(s); inputRef.current?.focus(); }}
                  className="rounded-xl border bg-card px-4 py-3 text-left text-sm text-slate-600 shadow-xs transition-colors hover:border-navy hover:text-navy">
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="mx-auto flex w-full max-w-2xl flex-col gap-5">
            {messages.map(m => <MessageItem key={m.id} message={m} />)}
            {busy && <TypingIndicator />}
            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="shrink-0 border-t bg-card px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-6">
        <form onSubmit={submit} className="mx-auto flex max-w-2xl items-end gap-2">
          <label htmlFor="chat-input" className="sr-only">Message</label>
          <Textarea
            id="chat-input"
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={mod.placeholder}
            rows={1}
            disabled={busy}
            className="max-h-36 min-h-10 resize-none rounded-xl bg-background px-3.5 py-2.5 text-base md:text-sm"
          />
          <Button type="submit" size="icon-lg" disabled={!input.trim() || busy} aria-label="Send"
            className="size-10 shrink-0 rounded-xl">
            <ArrowUpIcon className="size-4.5" />
          </Button>
        </form>
        <p className="mt-2 text-center text-[11px] text-slate-400">
          AI may make mistakes — verify critical information before acting on it.
        </p>
      </div>
    </main>
  );
}
