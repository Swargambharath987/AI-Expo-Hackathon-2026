import { CircleAlertIcon } from "lucide-react";
import { FormCard } from "@/components/chat/FormCard";
import { Markdown } from "@/components/chat/Markdown";
import { ToolSteps } from "@/components/chat/ToolSteps";
import { VerificationBadge } from "@/components/chat/VerificationBadge";
import { cn } from "@/lib/utils";
import type { Message } from "@/types";

const time = (d: Date) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export function MessageItem({ message }: { message: Message }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-navy px-4 py-2.5 text-sm leading-relaxed break-words whitespace-pre-wrap text-white sm:max-w-[76%]">
          {message.content}
        </div>
      </div>
    );
  }

  const tools = message.tool_calls ?? [];

  return (
    <article className={cn(
      "flex flex-col gap-3 rounded-2xl rounded-tl-sm border bg-card px-4 py-3 shadow-xs",
      message.isError && "border-crimson/25 bg-crimson/[0.03]",
    )}>
      {tools.length > 0 && <ToolSteps calls={tools} />}

      {message.isError ? (
        <p className="flex gap-2 text-sm text-crimson">
          <CircleAlertIcon className="mt-0.5 size-4 shrink-0" />
          <span>{message.content}</span>
        </p>
      ) : (
        <Markdown>{message.content}</Markdown>
      )}

      <VerificationBadge verification={message.verification} />
      {message.form_output && <FormCard form={message.form_output} />}

      <time className="text-[10px] text-slate-400">{time(message.timestamp)}</time>
    </article>
  );
}

export function TypingIndicator() {
  return (
    <div className="flex w-fit items-center gap-2 rounded-2xl rounded-tl-sm border bg-card px-4 py-3 text-sm text-muted-foreground shadow-xs" role="status">
      <span className="inline-flex gap-1" aria-hidden>
        {[0, 150, 300].map(d => (
          <span key={d} className="size-1.5 animate-typing rounded-full bg-slate-400" style={{ animationDelay: `${d}ms` }} />
        ))}
      </span>
      Reasoning through your request…
    </div>
  );
}
