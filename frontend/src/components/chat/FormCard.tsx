import { DownloadIcon, FileTextIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { formSubtitle, formTitle } from "@/lib/constants";
import type { FormOutput } from "@/types";

/** The filled form attached to an assistant answer. */
export function FormCard({ form }: { form: FormOutput }) {
  const title = formTitle(form.form_name);
  const missing = form.missing_fields ?? [];
  const filled = Object.keys(form.filled_fields ?? {}).length;

  return (
    <div className="rounded-xl border border-crimson/20 bg-crimson/[0.03] p-3 sm:p-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-crimson/10 text-crimson">
          <FileTextIcon className="size-4.5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-foreground">{title}</div>
          <div className="text-xs text-muted-foreground">
            {[formSubtitle(title), filled > 0 && `${filled} fields filled`].filter(Boolean).join(" · ")}
          </div>
        </div>
        {form.pdf_available && form.pdf_url ? (
          <Button asChild size="lg" className="bg-crimson px-3 text-white hover:bg-crimson-700">
            <a href={form.pdf_url} download>
              <DownloadIcon /> Download PDF
            </a>
          </Button>
        ) : (
          <span className="text-xs text-muted-foreground">PDF not available</span>
        )}
      </div>
      {missing.length > 0 && (
        <p className="mt-3 border-t border-crimson/10 pt-2.5 text-xs text-slate-600">
          <span className="font-semibold text-crimson">{missing.length} field{missing.length === 1 ? "" : "s"} to complete by hand:</span>{" "}
          {missing.join(", ")}
        </p>
      )}
    </div>
  );
}
