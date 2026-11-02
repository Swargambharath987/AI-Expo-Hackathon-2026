import { zodResolver } from "@hookform/resolvers/zod";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Field, FieldDescription, FieldError, FieldGroup, FieldLabel } from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import type { SoldierProfile } from "@/types";

const optional = z.string().trim();

export const profileSchema = z.object({
  name_last_first:  z.string().trim().min(1, "Enter your name as Last, First MI."),
  rank:             z.string().trim().min(1, "Enter your rank, e.g. SPC or SSG."),
  grade:            optional,
  ssn_last4:        optional.regex(/^(\d{4})?$/, "Must be exactly 4 digits, or leave it blank."),
  dod_id:           optional.regex(/^(\d{10})?$/, "Must be exactly 10 digits, or leave it blank."),
  unit:             optional,
  installation:     optional,
  uic:              optional,
  supervisor_name:  optional,
  supervisor_title: optional,
}) satisfies z.ZodType<SoldierProfile>;

type FieldSpec = {
  name: keyof SoldierProfile;
  label: string;
  placeholder?: string;
  hint?: string;
  required?: boolean;
  inputMode?: "numeric";
  maxLength?: number;
  wide?: boolean;
};

const FIELDS: FieldSpec[] = [
  { name: "name_last_first",  label: "Name (Last, First MI)", placeholder: "Rivera, Maria J.", required: true, wide: true },
  { name: "rank",             label: "Rank",                  placeholder: "SPC", required: true },
  { name: "grade",            label: "Grade",                 placeholder: "E-4" },
  { name: "unit",             label: "Unit",                  placeholder: "1-503 IN, 82nd ABN", wide: true },
  { name: "installation",     label: "Installation",          placeholder: "Fort Liberty" },
  { name: "uic",              label: "UIC",                   placeholder: "W4XXXX" },
  { name: "supervisor_name",  label: "Supervisor name",       placeholder: "SSG Doe, John" },
  { name: "supervisor_title", label: "Supervisor title",      placeholder: "Squad Leader" },
  { name: "ssn_last4",        label: "SSN — last 4",          placeholder: "1234", hint: "Optional. Only used on forms that ask for it.", inputMode: "numeric", maxLength: 4 },
  { name: "dod_id",           label: "DoD ID",                placeholder: "10 digits", hint: "Optional.", inputMode: "numeric", maxLength: 10 },
];

export function ProfileForm({
  defaultValues,
  onSubmit,
  onCancel,
  submitLabel = "Save profile",
  cancelLabel = "Cancel",
}: {
  defaultValues: SoldierProfile;
  onSubmit: (profile: SoldierProfile) => void;
  onCancel?: () => void;
  submitLabel?: string;
  cancelLabel?: string;
}) {
  const form = useForm<SoldierProfile>({
    resolver: zodResolver(profileSchema),
    defaultValues,
    mode: "onTouched",
  });

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} noValidate className="flex flex-col gap-6">
      <FieldGroup className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {FIELDS.map(spec => (
          <Controller
            key={spec.name}
            name={spec.name}
            control={form.control}
            render={({ field, fieldState }) => (
              <Field data-invalid={fieldState.invalid} className={spec.wide ? "sm:col-span-2" : undefined}>
                <FieldLabel htmlFor={`profile-${spec.name}`}>
                  {spec.label}
                  {spec.required && <span className="text-crimson" aria-hidden>*</span>}
                </FieldLabel>
                <Input
                  {...field}
                  id={`profile-${spec.name}`}
                  placeholder={spec.placeholder}
                  inputMode={spec.inputMode}
                  maxLength={spec.maxLength}
                  autoComplete="off"
                  aria-invalid={fieldState.invalid}
                  aria-required={spec.required}
                />
                {spec.hint && !fieldState.invalid && <FieldDescription>{spec.hint}</FieldDescription>}
                {fieldState.invalid && <FieldError errors={[fieldState.error]} />}
              </Field>
            )}
          />
        ))}
      </FieldGroup>

      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        {onCancel && (
          <Button type="button" variant="outline" size="lg" onClick={onCancel}>
            {cancelLabel}
          </Button>
        )}
        <Button type="submit" size="lg" className="bg-crimson px-5 text-white hover:bg-crimson-700">
          {submitLabel}
        </Button>
      </div>
    </form>
  );
}
