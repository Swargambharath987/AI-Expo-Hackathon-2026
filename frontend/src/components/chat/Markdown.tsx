import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

// Assistant answers are rendered as markdown → React elements. Raw HTML is
// never interpreted (no rehype-raw): visitors' API keys live in localStorage,
// so model output must not be able to inject markup or scripts.
const components: Components = {
  p:          ({ node, ...p }) => <p className="my-2 leading-7 first:mt-0 last:mb-0" {...p} />,
  h1:         ({ node, ...p }) => <h3 className="mt-4 mb-1.5 text-base font-bold text-foreground first:mt-0" {...p} />,
  h2:         ({ node, ...p }) => <h3 className="mt-4 mb-1.5 text-[15px] font-bold text-foreground first:mt-0" {...p} />,
  h3:         ({ node, ...p }) => <h4 className="mt-3 mb-1 text-sm font-bold text-foreground first:mt-0" {...p} />,
  h4:         ({ node, ...p }) => <h4 className="mt-3 mb-1 text-sm font-semibold text-foreground first:mt-0" {...p} />,
  ul:         ({ node, ...p }) => <ul className="my-2 list-disc space-y-1 pl-5 marker:text-muted-foreground" {...p} />,
  ol:         ({ node, ...p }) => <ol className="my-2 list-decimal space-y-1 pl-5 marker:text-muted-foreground" {...p} />,
  li:         ({ node, ...p }) => <li className="pl-0.5 leading-6" {...p} />,
  strong:     ({ node, ...p }) => <strong className="font-semibold text-foreground" {...p} />,
  a:          ({ node, ...p }) => <a className="font-medium text-navy underline underline-offset-2 hover:text-crimson" target="_blank" rel="noopener noreferrer" {...p} />,
  blockquote: ({ node, ...p }) => <blockquote className="my-2 border-l-2 border-border pl-3 text-muted-foreground italic" {...p} />,
  hr:         ({ node, ...p }) => <hr className="my-3 border-border" {...p} />,
  code:       ({ node, className, ...p }) => <code className="rounded bg-muted px-1 py-0.5 font-mono text-[12px] text-foreground" {...p} />,
  pre:        ({ node, ...p }) => <pre className="my-2 overflow-x-auto rounded-lg bg-muted p-3 text-[12px] [&_code]:bg-transparent [&_code]:p-0" {...p} />,
  table:      ({ node, ...p }) => (
    <div className="my-3 overflow-x-auto rounded-lg border">
      <table className="w-full border-collapse text-[13px]" {...p} />
    </div>
  ),
  thead:      ({ node, ...p }) => <thead className="bg-muted/60" {...p} />,
  th:         ({ node, ...p }) => <th className="border-b px-3 py-2 text-left font-semibold whitespace-nowrap text-foreground" {...p} />,
  td:         ({ node, ...p }) => <td className="border-b px-3 py-2 align-top text-slate-600 [tr:last-child_&]:border-b-0" {...p} />,
};

export function Markdown({ children }: { children: string }) {
  return (
    <div className="text-sm text-slate-700 break-words">
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components} skipHtml>
        {children}
      </ReactMarkdown>
    </div>
  );
}
