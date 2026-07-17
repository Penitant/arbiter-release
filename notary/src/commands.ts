// /train-mode toggle: manual (proposals shown) vs auto (logged silently). In-memory, per-process.

export const TRAIN_MODE_COMMAND = "train-mode";

function defaultManual(): boolean {
  // Default manual ON; set ARBITER_TRAIN_MODE=auto (or off/0) to start in auto mode.
  const v = (process.env.ARBITER_TRAIN_MODE ?? "manual").toLowerCase();
  return !["auto", "off", "0", "false", "no"].includes(v);
}

let manual = defaultManual();

/** True when proposals should be shown for accept/override (manual approval). */
export function isManualMode(): boolean {
  return manual;
}

/** Flip the mode and return the new state. */
export function toggleManualMode(): boolean {
  manual = !manual;
  return manual;
}

/** Human-readable one-liner for a toast after toggling. */
export function modeMessage(on: boolean): string {
  return on
    ? "Arbiter train-mode ON — proposals shown for accept/override."
    : "Arbiter train-mode OFF — routing automatic, decisions logged silently.";
}
