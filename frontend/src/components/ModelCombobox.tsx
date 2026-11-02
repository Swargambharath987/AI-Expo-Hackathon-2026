import { useState } from "react";
import { ChevronsUpDownIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

/** Searchable model picker — OpenRouter alone lists ~400 models. */
export function ModelCombobox({
  id,
  models,
  value,
  onChange,
}: {
  id?: string;
  models: string[];
  value: string;
  onChange: (model: string) => void;
}) {
  const [open, setOpen] = useState(false);

  return (
    <Popover open={open} onOpenChange={setOpen} modal>
      <PopoverTrigger asChild>
        <Button
          id={id}
          type="button"
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="h-9 w-full justify-between font-normal"
        >
          <span className="truncate">{value || "Choose a model…"}</span>
          <ChevronsUpDownIcon className="opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-(--radix-popover-trigger-width) p-0" align="start">
        <Command>
          <CommandInput placeholder={`Search ${models.length} models…`} />
          <CommandList className="max-h-64">
            <CommandEmpty>No model matches.</CommandEmpty>
            <CommandGroup>
              {models.map(m => (
                <CommandItem
                  key={m}
                  value={m}
                  data-checked={m === value}
                  onSelect={() => { onChange(m); setOpen(false); }}
                >
                  <span className="truncate">{m}</span>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
