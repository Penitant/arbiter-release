// Plan-mode menu rendering + override parsing. Pure string logic, no SDK/network deps.

import type { Decision } from "./client.js";

export const TIERS = ["local-simple", "cloud-mid", "cloud-heavy"] as const;
export type Tier = (typeof TIERS)[number];

/** One proposed subtask: the task text and the Server's resolved Decision. */
export interface Proposal {
  taskText: string;
  decision: Decision;
}

export type Override =
  | { kind: "accept" }
  | { kind: "override"; index: number; target: string; targetIsTier: boolean };

/** Display name for a "provider/id" model — the id half. */
export function shortModel(model: string): string {
  const slash = model.lastIndexOf("/");
  return slash >= 0 ? model.slice(slash + 1) : model;
}

export function isTier(value: string): value is Tier {
  return (TIERS as readonly string[]).includes(value);
}

/** Collapse a task to a single trimmed line for the menu. */
function oneLine(text: string, max = 60): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1)}…` : flat;
}

/** Render the numbered plan-mode menu; the `[tier -> model]` column is width-padded. */
export function renderProposal(proposals: Proposal[]): string {
  if (proposals.length === 0) return "";

  const tags = proposals.map(
    (p) => `[${p.decision.tier} -> ${shortModel(p.decision.model)}]`,
  );
  const width = Math.max(...tags.map((t) => t.length));

  const lines = proposals.map((p, i) => {
    const tag = tags[i].padEnd(width);
    return `${i + 1}. ${tag}   ${oneLine(p.taskText)}`;
  });

  const footer =
    proposals.length === 1
      ? [
          "",
          "Accept: enter",
          'Override: type a tier or a model (e.g. "gemini-3.5-flash" or "cloud-heavy")',
        ]
      : [
          "",
          "Accept all: enter",
          "Override: type subtask number, then a tier or a specific model",
          '          (e.g. "2 gemini-3.5-flash")',
        ];

  return ["Arbiter proposes:", "", ...lines, ...footer].join("\n");
}

const ACCEPT_TOKENS = new Set(["", "accept", "accept all", "y", "yes", "ok"]);

/** Parse a reply: accept token, "<n> <target>", bare "<target>" (single subtask), else null. */
export function parseOverride(input: string, count: number): Override | null {
  const trimmed = input.trim();
  if (ACCEPT_TOKENS.has(trimmed.toLowerCase())) return { kind: "accept" };

  // "<n> <target...>" — leading 1-based subtask index, then a tier or model.
  const numbered = trimmed.match(/^(\d+)\s+(.+)$/);
  if (numbered) {
    const index = Number(numbered[1]);
    if (index < 1 || index > count) return null; // out of range -> not a control input
    const target = numbered[2].trim();
    return { kind: "override", index, target, targetIsTier: isTier(target) };
  }

  // Bare "<target>" with a single subtask: unambiguous override of subtask 1.
  if (count === 1 && /^[\w./:-]+$/.test(trimmed)) {
    return { kind: "override", index: 1, target: trimmed, targetIsTier: isTier(trimmed) };
  }

  return null;
}
