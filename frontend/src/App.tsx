import { useState, useRef, useEffect, useCallback, type FormEvent, type KeyboardEvent } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

type Tab = "travel" | "leave" | "regulation" | "eval";
type AgentStatus = "idle" | "thinking" | "streaming" | "done" | "error";

interface ToolCallItem {
  tool:           string;
  label:          string;
  result_summary: string;
}

interface FormOutput {
  form_name:    string;
  pdf_available: boolean;
  pdf_url:      string | null;
  txt_summary:  string;
}

interface ChatResponse {
  response:    string;
  tool_calls:  ToolCallItem[];
  form_output: FormOutput | null;
  error:       string | null;
}

interface HealthResponse {
  status:               string;
  llm_provider?:        string;
  llm_ready?:           boolean;
  ollama?:              boolean;
  vector_store_chunks:  number;
  gsa_cache_loaded:     boolean;
  model:                string;
  offline_ready?:       boolean;
}

interface Message {
  id:          string;
  role:        "user" | "assistant";
  content:     string;
  timestamp:   Date;
  tool_calls?: ToolCallItem[];
  form_output?: FormOutput;
}

// ─── Markdown renderer ────────────────────────────────────────────────────────

function renderMarkdown(text: string): React.ReactNode {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let i = 0;

  const inlineFormat = (str: string): React.ReactNode => {
    const parts = str.split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g);
    return parts.map((p, idx) => {
      if (p.startsWith("**") && p.endsWith("**"))
        return <strong key={idx}>{p.slice(2, -2)}</strong>;
      if (p.startsWith("*") && p.endsWith("*"))
        return <em key={idx}>{p.slice(1, -1)}</em>;
      if (p.startsWith("`") && p.endsWith("`"))
        return <code key={idx} style={{ background: "#f1f5f9", padding: "1px 5px", borderRadius: 3, fontSize: 12, fontFamily: "monospace" }}>{p.slice(1, -1)}</code>;
      return p;
    });
  };

  while (i < lines.length) {
    const line = lines[i];

    // Horizontal rule
    if (/^---+$/.test(line.trim())) {
      elements.push(<hr key={i} style={{ border: "none", borderTop: "1px solid #e2e8f0", margin: "10px 0" }} />);
      i++; continue;
    }

    // Table — collect all | lines
    if (line.trim().startsWith("|")) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i++;
      }
      const rows = tableLines.filter(l => !/^\|[-:\s|]+\|$/.test(l.trim()));
      const parseRow = (l: string) => l.split("|").filter((_, ci) => ci > 0 && ci < l.split("|").length - 1).map(c => c.trim());
      if (rows.length > 0) {
        const [header, ...body] = rows;
        elements.push(
          <table key={i} style={{ borderCollapse: "collapse", width: "100%", fontSize: 13, margin: "8px 0" }}>
            <thead>
              <tr>{parseRow(header).map((h, ci) => <th key={ci} style={{ textAlign: "left", padding: "6px 10px", background: "#f8fafc", borderBottom: "2px solid #e2e8f0", fontWeight: 600, color: "#374151" }}>{inlineFormat(h)}</th>)}</tr>
            </thead>
            <tbody>
              {body.map((row, ri) => (
                <tr key={ri} style={{ borderBottom: "1px solid #f1f5f9" }}>
                  {parseRow(row).map((cell, ci) => <td key={ci} style={{ padding: "6px 10px", color: "#4b5563" }}>{inlineFormat(cell)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        );
      }
      continue;
    }

    // Headings
    const h2 = line.match(/^##\s+(.+)/);
    const h3 = line.match(/^###\s+(.+)/);
    if (h2) { elements.push(<h3 key={i} style={{ fontSize: 15, fontWeight: 700, margin: "12px 0 4px", color: "#1e293b" }}>{inlineFormat(h2[1])}</h3>); i++; continue; }
    if (h3) { elements.push(<h4 key={i} style={{ fontSize: 13, fontWeight: 700, margin: "10px 0 4px", color: "#1e293b" }}>{inlineFormat(h3[1])}</h4>); i++; continue; }

    // Blockquote
    if (line.startsWith("> ")) {
      elements.push(<blockquote key={i} style={{ borderLeft: "3px solid #e2e8f0", paddingLeft: 12, margin: "6px 0", color: "#6b7280", fontStyle: "italic" }}>{inlineFormat(line.slice(2))}</blockquote>);
      i++; continue;
    }

    // Bullet list — collect consecutive bullets
    if (/^[-•*]\s/.test(line) || /^\d+\.\s/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && (/^[-•*]\s/.test(lines[i]) || /^\d+\.\s/.test(lines[i]))) {
        items.push(lines[i].replace(/^[-•*]\s/, "").replace(/^\d+\.\s/, ""));
        i++;
      }
      elements.push(
        <ul key={i} style={{ margin: "4px 0", paddingLeft: 18, display: "flex", flexDirection: "column", gap: 2 }}>
          {items.map((it, ii) => <li key={ii} style={{ fontSize: 14, color: "#374151" }}>{inlineFormat(it)}</li>)}
        </ul>
      );
      continue;
    }

    // Warning line (⚠)
    if (line.startsWith("⚠") || line.includes("⚠")) {
      elements.push(<p key={i} style={{ margin: "6px 0", color: "#b91c1c", fontSize: 13 }}>{inlineFormat(line)}</p>);
      i++; continue;
    }

    // Empty line → spacer
    if (line.trim() === "") {
      elements.push(<div key={i} style={{ height: 6 }} />);
      i++; continue;
    }

    // Regular paragraph
    elements.push(<p key={i} style={{ margin: "2px 0", fontSize: 14, color: "#1e293b", lineHeight: 1.7 }}>{inlineFormat(line)}</p>);
    i++;
  }

  return <>{elements}</>;
}

interface SoldierProfile {
  name_last_first:  string;
  rank:             string;
  grade:            string;
  ssn_last4:        string;
  dod_id:           string;
  unit:             string;
  installation:     string;
  uic:              string;
  supervisor_name:  string;
  supervisor_title: string;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const NAVY    = "#1e2540";
const CRIMSON = "#b91c33";
const OLIVE   = "#445233";

const MODULES: { key: Tab; label: string; subtitle: string; icon: string }[] = [
  { key: "travel",     label: "TDY Travel",  subtitle: "Per diem, DD 1610",  icon: "✈" },
  { key: "leave",      label: "Leave / HR",  subtitle: "DA 31, actions",     icon: "📋" },
  { key: "regulation", label: "Regulations", subtitle: "JTR, ARs, AFIs",     icon: "📖" },
  { key: "eval",       label: "Evaluations", subtitle: "NCOERs, OERs",       icon: "⭐" },
];

const STARTERS: Record<Tab, string[]> = {
  travel: [
    "Send SPC Rivera TDY to Fort Moore, GA for 5 days starting July 10, driving POV from Fort Liberty",
    "What's the per diem for San Diego, CA?",
    "Do I need a GTC for a 2-day TDY?",
  ],
  leave: [
    "I need 10 days ordinary leave starting June 3 to visit family in Texas",
    "How many days of leave do I accrue per month?",
    "Can I take leave while on a temporary profile?",
  ],
  regulation: [
    "What does JTR say about POV mileage reimbursement?",
    "What's the receipt threshold for travel expenses?",
    "When must I file a travel voucher after returning from TDY?",
  ],
  eval: [
    "Help me write NCOER bullets for a squad leader",
    "What are the counseling requirements per AR 623-3?",
    "When is initial counseling due for a new soldier?",
  ],
};

const EMPTY_PROFILE: SoldierProfile = {
  name_last_first: "", rank: "", grade: "", ssn_last4: "",
  dod_id: "", unit: "", installation: "", uic: "",
  supervisor_name: "", supervisor_title: "",
};

// ─── Landing Page ────────────────────────────────────────────────────────────

function LandingPage({ onComplete }: { onComplete: (profile: SoldierProfile) => void }) {
  const [draft, setDraft] = useState<SoldierProfile>(EMPTY_PROFILE);
  const [step, setStep] = useState<"welcome" | "profile">("welcome");

  async function handleStart() {
    if (!draft.name_last_first || !draft.rank) return;
    try {
      await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(draft),
      });
    } catch {}
    onComplete(draft);
  }

  const canStart = draft.name_last_first.trim() && draft.rank.trim();

  return (
    <div style={{ minHeight: "100vh", background: NAVY, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", position: "relative", overflow: "hidden" }}>
      <div style={{ position: "absolute", inset: 0, opacity: 0.03, backgroundImage: "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)", backgroundSize: "40px 40px" }} />

      <div style={{ position: "relative", zIndex: 1, maxWidth: 440, width: "100%", padding: "0 24px" }}>
        {step === "welcome" ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", textAlign: "center" }}>
            <img src="/logo-cropped.png" alt="Duty Line" style={{ height: 180, marginBottom: 40, filter: "brightness(1.1)" }} />

            <div style={{ display: "flex", gap: 24, marginBottom: 40 }}>
              {[
                { label: "TDY Travel", desc: "Per diem & forms" },
                { label: "Leave / HR", desc: "DA 31 & policy" },
                { label: "Regulations", desc: "JTR & ARs" },
                { label: "Evaluations", desc: "NCOERs & OERs" },
              ].map(m => (
                <div key={m.label} style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "rgba(255,255,255,0.8)" }}>{m.label}</div>
                  <div style={{ fontSize: 10, color: "rgba(255,255,255,0.35)", marginTop: 2 }}>{m.desc}</div>
                </div>
              ))}
            </div>

            <button onClick={() => setStep("profile")}
              style={{ padding: "12px 48px", background: CRIMSON, color: "#fff", border: "none", borderRadius: 8, fontSize: 15, fontWeight: 600, cursor: "pointer", letterSpacing: "0.02em", transition: "opacity 0.2s" }}
              onMouseEnter={e => (e.currentTarget as HTMLElement).style.opacity = "0.9"}
              onMouseLeave={e => (e.currentTarget as HTMLElement).style.opacity = "1"}
            >
              Get Started
            </button>

            <button onClick={() => onComplete(EMPTY_PROFILE)}
              style={{ background: "none", border: "none", color: "rgba(255,255,255,0.3)", fontSize: 12, cursor: "pointer", marginTop: 16 }}>
              Skip setup
            </button>

            <div style={{ marginTop: 48, fontSize: 10, color: "rgba(255,255,255,0.2)", letterSpacing: "0.1em", textTransform: "uppercase" }}>
              GenAI.mil Track | SCSP AI Expo Hackathon 2026
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
            <img src="/logo-cropped.png" alt="Duty Line" style={{ height: 60, marginBottom: 24, filter: "brightness(1.1)" }} />

            <div style={{ width: "100%", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 12, padding: 28 }}>
              <p style={{ fontSize: 15, fontWeight: 600, color: "#fff", marginTop: 0, marginBottom: 4 }}>Soldier Profile</p>
              <p style={{ fontSize: 12, color: "rgba(255,255,255,0.4)", marginBottom: 20 }}>Auto-fills your forms. Stored on this device only.</p>

              {([
                ["name_last_first",  "Name (Last, First, MI)"],
                ["rank",             "Rank (e.g. SPC, SSG)"],
                ["grade",            "Grade (e.g. E-4)"],
                ["unit",             "Unit (e.g. 1-503 INF, 82nd ABN)"],
                ["installation",     "Installation (e.g. Fort Liberty)"],
                ["supervisor_name",  "Supervisor Name"],
              ] as [keyof SoldierProfile, string][]).map(([f, label]) => (
                <input key={f} placeholder={label}
                  style={{ display: "block", width: "100%", marginBottom: 10, border: "1px solid rgba(255,255,255,0.1)", borderRadius: 6, padding: "10px 14px", fontSize: 13, color: "#fff", background: "rgba(255,255,255,0.06)", outline: "none", boxSizing: "border-box", transition: "border-color 0.2s" }}
                  value={draft[f]}
                  onChange={e => setDraft({ ...draft, [f]: e.target.value })}
                  onFocus={e => { (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.3)"; (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.1)"; }}
                  onBlur={e => { (e.currentTarget as HTMLElement).style.borderColor = "rgba(255,255,255,0.1)"; (e.currentTarget as HTMLElement).style.background = "rgba(255,255,255,0.06)"; }}
                />
              ))}

              <button onClick={handleStart} disabled={!canStart}
                style={{ width: "100%", padding: "12px 0", background: canStart ? CRIMSON : "rgba(255,255,255,0.1)", color: canStart ? "#fff" : "rgba(255,255,255,0.3)", border: "none", borderRadius: 8, fontSize: 14, fontWeight: 600, cursor: canStart ? "pointer" : "not-allowed", marginTop: 4, transition: "background 0.2s" }}>
                Continue
              </button>

              <button onClick={() => setStep("welcome")}
                style={{ width: "100%", padding: "8px 0", background: "none", border: "none", color: "rgba(255,255,255,0.3)", fontSize: 12, cursor: "pointer", marginTop: 8 }}>
                Back
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage]         = useState<"landing" | "main">("landing");
  const [tab, setTab]           = useState<Tab>("travel");
  const [status, setStatus]     = useState<AgentStatus>("idle");
  const [messages, setMessages] = useState<Record<Tab, Message[]>>({ travel: [], leave: [], regulation: [], eval: [] });
  const [input, setInput]       = useState("");
  const [profile, setProfile]   = useState<SoldierProfile>(EMPTY_PROFILE);
  const [health, setHealth]     = useState<HealthResponse | null>(null);
  const [showReasoning, setShowReasoning]   = useState<Record<string, boolean>>({});
  const [editingProfile, setEditingProfile] = useState(false);
  const [profileDraft, setProfileDraft]     = useState<SoldierProfile>(EMPTY_PROFILE);

  const endRef   = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const busy = status === "thinking" || status === "streaming";

  useEffect(() => {
    const fetchHealth = () => fetch("/api/health").then(r => r.json()).then(setHealth).catch(() => {});
    fetchHealth();
    const t = setTimeout(fetchHealth, 4000);
    fetch("/api/profile").then(r => r.json()).then((p) => {
      if (p?.name_last_first) { setProfile(p); setProfileDraft(p); setPage("main"); }
    }).catch(() => {});
    return () => clearTimeout(t);
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, tab]);

  const toggleReasoning = useCallback((id: string) => {
    setShowReasoning(prev => ({ ...prev, [id]: !prev[id] }));
  }, []);

  async function saveProfile() {
    setProfile(profileDraft);
    setEditingProfile(false);
    try {
      await fetch("/api/profile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(profileDraft),
      });
    } catch {}
  }

  async function clearHistory() {
    try { await fetch("/api/chat/history", { method: "DELETE" }); } catch {}
    setMessages(p => ({ ...p, [tab]: [] }));
    setStatus("idle");
  }

  async function send(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setMessages(p => ({ ...p, [tab]: [...p[tab], { id: crypto.randomUUID(), role: "user", content: text, timestamp: new Date() }] }));
    setInput("");
    setStatus("thinking");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, tab }),
      });
      if (!res.ok) throw new Error(`${res.status}`);
      const data: ChatResponse = await res.json();
      if (data.error) {
        setMessages(p => ({ ...p, [tab]: [...p[tab], { id: crypto.randomUUID(), role: "assistant", content: `Error: ${data.error}`, timestamp: new Date() }] }));
        setStatus("error");
      } else {
        const msgId = crypto.randomUUID();
        setMessages(p => ({ ...p, [tab]: [...p[tab], {
          id: msgId, role: "assistant", content: data.response || "—", timestamp: new Date(),
          tool_calls: data.tool_calls, form_output: data.form_output ?? undefined,
        }] }));
        // Auto-expand reasoning if tools were called
        if (data.tool_calls && data.tool_calls.length > 0) {
          setShowReasoning(prev => ({ ...prev, [msgId]: true }));
        }
        setStatus("done");
      }
      setTimeout(() => setStatus("idle"), 1500);
    } catch {
      setMessages(p => ({ ...p, [tab]: [...p[tab], {
        id: crypto.randomUUID(), role: "assistant", timestamp: new Date(),
        content: "Backend not reachable. Run:\n  python -m uvicorn app:app --port 8000",
      }] }));
      setStatus("error");
      setTimeout(() => setStatus("idle"), 3000);
    }
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(e as unknown as FormEvent); }
  }

  const hasProfile    = !!profile.name_last_first;
  const currentMod    = MODULES.find(m => m.key === tab)!;
  const tabMessages   = messages[tab];
  const userMessages  = tabMessages.filter(m => m.role === "user");
  const llmReady     = health ? (health.llm_ready ?? health.ollama ?? false) : null;
  const llmDown      = llmReady === false;

  // Status pill
  const pillBg    = llmDown || status === "error" ? "#fee2e2" : busy ? "#fef3c7" : "#dcfce7";
  const pillText  = llmDown || status === "error" ? "#b91c1c" : busy ? "#92400e" : "#166534";
  const pillBorder= llmDown || status === "error" ? "#fca5a5" : busy ? "#fde68a" : "#86efac";
  const pillLabel = llmDown ? "LLM Offline" : status === "error" ? "Error" :
                    status === "thinking" ? "Reasoning…" : status === "streaming" ? "Responding…" :
                    status === "done" ? "Complete" : "Ready";
  const dotColor  = llmDown || status === "error" ? "#ef4444" : busy ? "#f59e0b" : "#22c55e";
  const dotAnim   = busy ? "pulse 1.5s infinite" : "none";

  if (page === "landing") {
    return <LandingPage onComplete={(p) => { setProfile(p); setProfileDraft(p); setPage("main"); }} />;
  }

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column", background: "#f8fafc", color: "#1e293b" }}>

      {/* ── Header ── */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 24px", background: "#fff", borderBottom: "1px solid #e2e8f0", flexShrink: 0, boxShadow: "0 1px 3px rgba(0,0,0,0.06)" }}>

        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <img src="/logo-cropped.png" alt="Duty Line" style={{ height: 44, width: "auto", objectFit: "contain" }} />
          <div style={{ width: 1, height: 36, background: "#e2e8f0" }} />
          <span style={{ fontSize: 11, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.12em", textTransform: "uppercase" }}>AI Military Assistant</span>
        </div>

        {/* Status pill */}
        <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "5px 14px", borderRadius: 999, border: `1px solid ${pillBorder}`, background: pillBg, color: pillText, fontSize: 12, fontWeight: 600 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: dotColor, animation: dotAnim, display: "inline-block" }} />
          {pillLabel}
        </div>

        {/* Soldier info */}
        <div style={{ textAlign: "right" }}>
          {hasProfile ? (
            <>
              <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "flex-end" }}>
                <span style={{ fontSize: 14, fontWeight: 600, color: "#1e293b" }}>{profile.rank} {profile.name_last_first}</span>
                <button onClick={() => { setProfileDraft(profile); setEditingProfile(true); }} style={{ background: "none", border: "none", cursor: "pointer", color: "#94a3b8", fontSize: 13, padding: 0 }}>✏</button>
              </div>
              <div style={{ fontSize: 12, color: "#94a3b8" }}>{profile.unit} · {profile.installation}</div>
            </>
          ) : (
            <button onClick={() => setEditingProfile(true)} style={{ background: "none", border: "none", cursor: "pointer", color: CRIMSON, fontSize: 12, fontWeight: 500 }}>
              Set up soldier profile →
            </button>
          )}
        </div>
      </header>

      {/* ── Profile modal ── */}
      {editingProfile && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div style={{ background: "#fff", borderRadius: 12, boxShadow: "0 20px 60px rgba(0,0,0,0.15)", width: 320, padding: 24 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <span style={{ fontWeight: 600, fontSize: 15 }}>Soldier Profile</span>
              <button onClick={() => setEditingProfile(false)} style={{ background: "none", border: "none", cursor: "pointer", color: "#94a3b8", fontSize: 18, lineHeight: 1 }}>✕</button>
            </div>
            {([
              ["name_last_first",  "Name (Last, First, MI)"],
              ["rank",             "Rank (e.g. SPC, SSG)"],
              ["grade",            "Grade (e.g. E-4)"],
              ["ssn_last4",        "SSN Last 4 digits"],
              ["dod_id",           "DoD ID (10 digits)"],
              ["unit",             "Unit (e.g. 1-503 IN, 82nd ABN)"],
              ["installation",     "Installation (e.g. Fort Liberty)"],
              ["uic",              "UIC (e.g. W4XXXX)"],
              ["supervisor_name",  "Supervisor Name"],
              ["supervisor_title", "Supervisor Title (e.g. Squad Leader)"],
            ] as [keyof SoldierProfile, string][]).map(([f, label]) => (
              <input key={f} placeholder={label}
                style={{ display: "block", width: "100%", marginBottom: 8, border: "1px solid #e2e8f0", borderRadius: 6, padding: "8px 12px", fontSize: 13, color: "#1e293b", outline: "none" }}
                value={profileDraft[f]}
                onChange={e => setProfileDraft({ ...profileDraft, [f]: e.target.value })}
              />
            ))}
            <button onClick={saveProfile} style={{ width: "100%", padding: "10px 0", background: OLIVE, color: "#fff", border: "none", borderRadius: 6, fontSize: 13, fontWeight: 600, cursor: "pointer", marginTop: 4 }}>
              Save Profile
            </button>
          </div>
        </div>
      )}

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>

        {/* ── Sidebar ── */}
        <aside style={{ width: 220, background: "#fff", borderRight: "1px solid #e2e8f0", display: "flex", flexDirection: "column", flexShrink: 0 }}>

          <div style={{ flex: 1, overflowY: "auto", padding: 16 }}>

            {/* Modules */}
            <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 10 }}>Modules</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {MODULES.map(m => {
                const active = tab === m.key;
                return (
                  <button key={m.key} onClick={() => setTab(m.key)} style={{ display: "flex", alignItems: "flex-start", gap: 10, padding: "10px 12px", borderRadius: 8, border: "none", borderLeft: active ? `3px solid ${CRIMSON}` : "3px solid transparent", cursor: "pointer", background: active ? "#f0f4ff" : "transparent", color: active ? NAVY : "#475569", textAlign: "left", transition: "background 0.15s" }}
                    onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "#f8fafc"; }}
                    onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                  >
                    <span style={{ fontSize: 16, lineHeight: 1, marginTop: 1, flexShrink: 0 }}>{m.icon}</span>
                    <div>
                      <div style={{ fontSize: 13, fontWeight: 600, color: active ? NAVY : "#1e293b", lineHeight: 1.2 }}>{m.label}</div>
                      <div style={{ fontSize: 11, color: active ? "#64748b" : "#94a3b8", marginTop: 2 }}>{m.subtitle}</div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Session history */}
            {userMessages.length > 0 && (
              <div style={{ marginTop: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                  <span style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", letterSpacing: "0.1em", textTransform: "uppercase" }}>This Session</span>
                  <button onClick={clearHistory} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 10, color: "#94a3b8" }}
                    onMouseEnter={e => (e.currentTarget as HTMLElement).style.color = CRIMSON}
                    onMouseLeave={e => (e.currentTarget as HTMLElement).style.color = "#94a3b8"}
                  >Clear</button>
                </div>
                {userMessages.slice(-8).map(m => (
                  <div key={m.id} title={m.content} style={{ fontSize: 11, color: "#64748b", padding: "4px 6px", borderRadius: 4, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", marginBottom: 2 }}>
                    {m.content.length > 50 ? m.content.slice(0, 50) + "…" : m.content}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Health dots */}
          <div style={{ padding: "12px 16px", borderTop: "1px solid #f1f5f9" }}>
            {[
              { label: health ? (health.llm_provider === "ollama" ? "Ollama (local)" : "AI Model") : "AI Model", ok: llmReady },
              { label: `Vector Store (${health?.vector_store_chunks ?? "…"})`, ok: health ? (health.vector_store_chunks > 0) : null },
              { label: "GSA Rates Cached", ok: health ? health.gsa_cache_loaded : null },
              { label: health?.offline_ready ? "Offline Ready" : "Cloud Mode", ok: health ? true : null, color: health?.offline_ready ? "#22c55e" : "#f59e0b" },
            ].map(({ label, ok, color }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 7, marginBottom: 6 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", flexShrink: 0, background: ok === null ? "#cbd5e1" : ok ? (color || "#22c55e") : "#ef4444", display: "inline-block" }} />
                <span style={{ fontSize: 11, color: "#94a3b8", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{label}</span>
              </div>
            ))}
          </div>
        </aside>

        {/* ── Main ── */}
        <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, background: "#f8fafc" }}>

          {/* Tab bar */}
          <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "10px 20px", background: "#fff", borderBottom: "1px solid #e2e8f0", flexShrink: 0 }}>
            {MODULES.map(m => {
              const active = tab === m.key;
              return (
                <button key={m.key} onClick={() => setTab(m.key)} style={{ display: "flex", alignItems: "center", gap: 7, padding: "7px 16px", fontSize: 13, fontWeight: active ? 600 : 500, border: "none", borderRadius: 6, background: active ? NAVY : "transparent", color: active ? "#fff" : "#64748b", cursor: "pointer", transition: "background 0.15s, color 0.15s" }}
                  onMouseEnter={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "#f1f5f9"; }}
                  onMouseLeave={e => { if (!active) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
                >
                  <span style={{ fontSize: 14 }}>{m.icon}</span>
                  {m.label}
                </button>
              );
            })}
            {tabMessages.length > 0 && (
              <button onClick={clearHistory} style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", fontSize: 12, color: "#94a3b8", display: "flex", alignItems: "center", gap: 4 }}>
                ✕ Clear
              </button>
            )}
          </div>

          {/* Messages */}
          <div style={{ flex: 1, overflowY: "auto", padding: "24px", display: "flex", flexDirection: "column" }}>
            {tabMessages.length === 0 ? (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", maxWidth: 560, margin: "0 auto", width: "100%" }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: "#1e293b", marginBottom: 4 }}>{currentMod.icon} {currentMod.label}</div>
                <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 24 }}>{currentMod.subtitle} — describe your task below</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {STARTERS[tab].map(s => (
                    <button key={s} onClick={() => { setInput(s); inputRef.current?.focus(); }}
                      style={{ textAlign: "left", padding: "12px 16px", fontSize: 13, color: "#475569", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 10, cursor: "pointer", boxShadow: "0 1px 3px rgba(0,0,0,0.04)", transition: "border-color 0.15s" }}
                      onMouseEnter={e => (e.currentTarget as HTMLElement).style.borderColor = NAVY}
                      onMouseLeave={e => (e.currentTarget as HTMLElement).style.borderColor = "#e2e8f0"}
                    >{s}</button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 20, maxWidth: 680, margin: "0 auto", width: "100%" }}>
                {tabMessages.map(msg => (
                  <div key={msg.id}>
                    {msg.role === "user" ? (
                      <div style={{ display: "flex", justifyContent: "flex-end" }}>
                        <div style={{ maxWidth: "76%", background: NAVY, color: "#fff", padding: "12px 16px", borderRadius: "18px 18px 4px 18px", fontSize: 14, lineHeight: 1.6 }}>
                          {msg.content}
                        </div>
                      </div>
                    ) : (
                      <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "4px 18px 18px 18px", boxShadow: "0 1px 4px rgba(0,0,0,0.05)", overflow: "hidden" }}>

                        {/* Response — rendered markdown */}
                        <div style={{ padding: "12px 16px" }}>
                          {renderMarkdown(msg.content)}
                        </div>

                        {/* Tool trace / reasoning */}
                        {msg.tool_calls && msg.tool_calls.length > 0 && (
                          <div style={{ margin: "0 16px 12px", borderTop: "1px solid #f1f5f9", paddingTop: 10 }}>
                            <button onClick={() => toggleReasoning(msg.id)}
                              style={{ background: "none", border: "none", cursor: "pointer", padding: 0, fontSize: 12, color: "#64748b", fontWeight: 600, display: "flex", alignItems: "center", gap: 6 }}
                            >
                              <span>{showReasoning[msg.id] ? "▾" : "▸"}</span>
                              {msg.tool_calls.length} tool call{msg.tool_calls.length > 1 ? "s" : ""}
                            </button>
                            {showReasoning[msg.id] && (
                              <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                                {msg.tool_calls.map((tc, idx) => (
                                  <div key={idx} style={{ display: "flex", flexDirection: "column", gap: 2, background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 8, padding: "8px 12px" }}>
                                    <span style={{ fontSize: 12, fontWeight: 600, color: NAVY }}>{tc.label}</span>
                                    <span style={{ fontSize: 12, color: "#64748b" }}>{tc.result_summary}</span>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Form download */}
                        {msg.form_output && (
                          <div style={{ margin: "12px 16px", border: `2px solid ${CRIMSON}33`, borderRadius: 8, overflow: "hidden", background: "#fef2f2" }}>
                            <div style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                              <span style={{ fontSize: 13, fontWeight: 600, color: "#1e293b" }}>
                                {msg.form_output.form_name.replace(/_\d+.*/, "").replace(/_/g, " ")}
                              </span>
                              {msg.form_output.pdf_available && msg.form_output.pdf_url ? (
                                <a href={msg.form_output.pdf_url} download style={{ padding: "6px 16px", background: CRIMSON, color: "#fff", borderRadius: 6, fontSize: 13, fontWeight: 600, textDecoration: "none" }}>
                                  Download PDF
                                </a>
                              ) : (
                                <span style={{ fontSize: 12, color: "#94a3b8" }}>PDF not available</span>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Timestamp */}
                        <div style={{ padding: "0 16px 10px", fontSize: 10, color: "#cbd5e1" }}>
                          {msg.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {/* Typing indicator */}
                {busy && (
                  <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: "4px 18px 18px 18px", padding: "14px 18px", boxShadow: "0 1px 4px rgba(0,0,0,0.05)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, color: "#94a3b8", fontSize: 13 }}>
                      <span style={{ display: "inline-flex", gap: 4 }}>
                        {[0, 150, 300].map(d => (
                          <span key={d} style={{ width: 6, height: 6, borderRadius: "50%", background: "#94a3b8", display: "inline-block", animation: `bounce 1.2s ${d}ms infinite` }} />
                        ))}
                      </span>
                      {status === "thinking" ? "Reasoning through your request…" : "Streaming response…"}
                    </div>
                  </div>
                )}
                <div ref={endRef} />
              </div>
            )}
          </div>

          {/* Input */}
          <div style={{ padding: "16px 24px", borderTop: "1px solid #e2e8f0", background: "#fff", flexShrink: 0 }}>
            <form onSubmit={send} style={{ maxWidth: 680, margin: "0 auto", display: "flex", gap: 10 }}>
              <textarea ref={inputRef} value={input}
                onChange={e => {
                  setInput(e.target.value);
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
                }}
                onKeyDown={onKey}
                placeholder={`Ask about ${currentMod.label.toLowerCase()}…`}
                rows={1} disabled={busy}
                style={{ flex: 1, resize: "none", border: "1px solid #e2e8f0", borderRadius: 12, padding: "10px 14px", fontSize: 14, color: "#1e293b", background: "#f8fafc", outline: "none", fontFamily: "inherit", opacity: busy ? 0.5 : 1, overflowY: "hidden", lineHeight: 1.5 }}
                onFocus={e => (e.currentTarget as HTMLElement).style.borderColor = NAVY}
                onBlur={e => (e.currentTarget as HTMLElement).style.borderColor = "#e2e8f0"}
              />
              <button type="submit" disabled={!input.trim() || busy}
                style={{ padding: "10px 20px", background: (!input.trim() || busy) ? "#94a3b8" : NAVY, color: "#fff", border: "none", borderRadius: 12, fontSize: 14, fontWeight: 600, cursor: (!input.trim() || busy) ? "not-allowed" : "pointer", flexShrink: 0, transition: "background 0.15s" }}
              >Send</button>
            </form>
            <p style={{ textAlign: "center", fontSize: 11, color: "#94a3b8", marginTop: 8, marginBottom: 0 }}>
              AI may make mistakes — verify critical information before action
            </p>
          </div>
        </main>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-4px); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
}
