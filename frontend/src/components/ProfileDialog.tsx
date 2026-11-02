import { ProfileForm } from "@/components/ProfileForm";
import {
  Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";
import type { SoldierProfile } from "@/types";

export function ProfileDialog({
  open,
  onOpenChange,
  profile,
  onSave,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  profile: SoldierProfile;
  onSave: (profile: SoldierProfile) => void;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>Soldier profile</DialogTitle>
          <DialogDescription>
            Auto-fills your forms. Stored in this browser only and sent with each request — never saved on the server.
          </DialogDescription>
        </DialogHeader>
        {/* DialogContent unmounts when closed, so every open starts from the saved profile (Cancel discards edits). */}
        <ProfileForm
          defaultValues={profile}
          onSubmit={p => { onSave(p); onOpenChange(false); }}
          onCancel={() => onOpenChange(false)}
        />
      </DialogContent>
    </Dialog>
  );
}
