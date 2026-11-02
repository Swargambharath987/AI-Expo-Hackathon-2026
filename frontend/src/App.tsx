import { useRef, useState } from "react";
import { toast } from "sonner";
import { AiSettingsDialog } from "@/components/AiSettingsDialog";
import { AppHeader, statusPill } from "@/components/AppHeader";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { LandingPage } from "@/components/LandingPage";
import { ProfileDialog } from "@/components/ProfileDialog";
import { Sidebar } from "@/components/Sidebar";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Toaster } from "@/components/ui/sonner";
import { useHealth } from "@/hooks/useHealth";
import { clearChatHistory, sendChat } from "@/lib/api";
import { EMPTY_PROFILE, formTitle } from "@/lib/constants";
import { loadOwnKey, loadProfile, storeOwnKey, storeProfile } from "@/lib/storage";
import type { AgentStatus, GeneratedForm, Message, OwnKey, SoldierProfile, Tab } from "@/types";

const EMPTY_THREADS: Record<Tab, Message[]> = { travel: [], leave: [], regulation: [], eval: [] };
const newId = () => crypto.randomUUID();

export default function App() {
  const [savedProfile] = useState(loadProfile);
  const [page, setPage]       = useState<"landing" | "main">(savedProfile?.name_last_first ? "main" : "landing");
  const [profile, setProfile] = useState<SoldierProfile>(savedProfile ?? EMPTY_PROFILE);
  const [ownKey, setOwnKey]   = useState<OwnKey | null>(loadOwnKey);
  const [tab, setTab]         = useState<Tab>("travel");
  const [threads, setThreads] = useState<Record<Tab, Message[]>>(EMPTY_THREADS);
  const [forms, setForms]     = useState<GeneratedForm[]>([]);
  const [status, setStatus]   = useState<AgentStatus>("idle");
  const [profileOpen, setProfileOpen] = useState(false);
  const [aiOpen, setAiOpen]           = useState(false);
  const [menuOpen, setMenuOpen]       = useState(false);
  const statusTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  const health = useHealth();
  // With their own key the visitor doesn't depend on the server key's status
  // (their key was verified when they saved it).
  const llmReady = ownKey ? true : health ? (health.llm_ready ?? health.ollama ?? false) : null;
  const busy = status === "thinking";

  function settle(next: AgentStatus, ms: number) {
    setStatus(next);
    clearTimeout(statusTimer.current);
    statusTimer.current = setTimeout(() => setStatus("idle"), ms);
  }

  const append = (t: Tab, m: Message) => setThreads(p => ({ ...p, [t]: [...p[t], m] }));

  async function send(text: string) {
    const t = tab; // the reply belongs to the tab it was asked in, even if the user switches
    append(t, { id: newId(), role: "user", content: text, timestamp: new Date() });
    clearTimeout(statusTimer.current);
    setStatus("thinking");

    const result = await sendChat({ message: text, tab: t, profile, llm: ownKey });
    if (!result.ok) {
      append(t, { id: newId(), role: "assistant", content: result.message, timestamp: new Date(), isError: true });
      settle("error", 3000);
      return;
    }

    const { data } = result;
    const now = new Date();
    append(t, {
      id: newId(), role: "assistant", content: data.response || "—", timestamp: now,
      tool_calls: data.tool_calls ?? [], form_output: data.form_output ?? undefined, verification: data.verification ?? null,
    });
    if (data.form_output) {
      const f = data.form_output;
      setForms(p => [...p, {
        id: newId(), tab: t, form_name: f.form_name, title: formTitle(f.form_name),
        pdf_url: f.pdf_available ? f.pdf_url : null, missing_fields: f.missing_fields ?? [], created_at: now,
      }]);
    }
    settle("done", 1500);
  }

  async function clearTab() {
    try { await clearChatHistory(tab); } catch { /* local reset still helps */ }
    setThreads(p => ({ ...p, [tab]: [] }));
    setStatus("idle");
  }

  function saveProfile(p: SoldierProfile) {
    setProfile(p);
    storeProfile(p);
  }

  function saveOwnKey(k: OwnKey | null) {
    setOwnKey(k);
    storeOwnKey(k);
    toast.success(k ? `Using your ${k.provider} key · ${k.model}` : "Switched back to the app's AI");
  }

  if (page === "landing") {
    return <LandingPage onComplete={p => { saveProfile(p); setPage("main"); }} />;
  }

  const sidebar = (
    <Sidebar
      forms={forms}
      onOpenForm={t => { setTab(t); setMenuOpen(false); }}
      health={health}
      ownKey={ownKey}
      llmReady={llmReady}
      onOpenAiSettings={() => { setMenuOpen(false); setAiOpen(true); }}
    />
  );

  return (
    <>
      <div className="flex h-dvh flex-col overflow-hidden">
        <AppHeader
          profile={profile}
          pill={statusPill(status, llmReady)}
          onOpenProfile={() => setProfileOpen(true)}
          onOpenMenu={() => setMenuOpen(true)}
        />

        <div className="flex min-h-0 flex-1">
          <aside className="hidden w-64 shrink-0 border-r md:block">{sidebar}</aside>
          <ChatPanel
            tab={tab}
            onTabChange={setTab}
            messages={threads[tab]}
            busy={busy}
            onSend={send}
            onClear={clearTab}
          />
        </div>
      </div>

      {/* < md: the sidebar lives in a sheet opened from the header */}
      <Sheet open={menuOpen} onOpenChange={setMenuOpen}>
        <SheetContent side="left" className="w-[85vw] max-w-xs gap-0 p-0">
          <SheetHeader className="border-b">
            <SheetTitle>Duty Line</SheetTitle>
            <SheetDescription>Forms from this session and system status.</SheetDescription>
          </SheetHeader>
          <div className="min-h-0 flex-1">{sidebar}</div>
        </SheetContent>
      </Sheet>

      <ProfileDialog open={profileOpen} onOpenChange={setProfileOpen} profile={profile}
        onSave={p => { saveProfile(p); toast.success("Profile saved"); }} />
      <AiSettingsDialog open={aiOpen} onOpenChange={setAiOpen} current={ownKey} onSave={saveOwnKey} />
      <Toaster position="top-center" />
    </>
  );
}
