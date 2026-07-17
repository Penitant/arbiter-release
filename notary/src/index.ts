// Notary plugin: plan-mode menu, override handling, correction logging. UI only — no routing logic.

import type { Plugin } from "@opencode-ai/plugin";

import { ServerClient, type Correction, type Decision } from "./client.js";
import { type Proposal, parseOverride, renderProposal } from "./menu.js";
import {
  TRAIN_MODE_COMMAND,
  isManualMode,
  modeMessage,
  toggleManualMode,
} from "./commands.js";

// Injected-menu prefix; skipped in chat.message so our own output isn't read as an override.
const MENU_PREFIX = "Arbiter proposes:";

// Only pending/in-progress subtasks are still routable.
const ACTIONABLE = new Set(["pending", "in_progress"]);

/** ~4 chars/token is a coarse but adequate estimate for the context-fit field. */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/** Pull the concatenated text out of a message's parts (text parts only). */
function extractText(parts: Array<{ type?: string; text?: string }>): string {
  return parts
    .filter((p) => p?.type === "text" && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("\n");
}

export const NotaryPlugin: Plugin = async ({ client }) => {
  const server = new ServerClient();

  // Proposals shown but not yet accepted/overridden, keyed by session.
  const pending = new Map<string, Proposal[]>();
  // Last todo-set signature per session, so status-only updates don't re-show the menu.
  const lastSignature = new Map<string, string>();

  const toast = (message: string, variant: "info" | "success" = "info") =>
    client.tui
      .showToast({ body: { message, variant } })
      .catch(() => {}); // a failed toast must never break the flow

  const logAccepted = (p: Proposal, sessionID: string) =>
    server.logCorrection({
      task_text: p.taskText,
      predicted_tier: p.decision.tier,
      predicted_model: p.decision.model,
      confidence: p.decision.confidence,
      session_id: sessionID,
    });

  // Classify subtask texts in parallel, dropping any the Server can't resolve (null).
  const classifyAll = async (
    texts: string[],
    sessionID: string,
  ): Promise<Proposal[]> => {
    const decisions = await Promise.all(
      texts.map((t) =>
        server.classify({
          task_text: t,
          session_id: sessionID,
          context_tokens_estimate: estimateTokens(t),
        }),
      ),
    );
    const out: Proposal[] = [];
    decisions.forEach((d: Decision | null, i) => {
      if (d) out.push({ taskText: texts[i], decision: d });
    });
    return out;
  };

  console.log("[notary] Arbiter plugin loaded — Server contract at", "127.0.0.1:4317");

  return {
    /** Register the /train-mode command so OpenCode routes it to us. */
    config: async (cfg) => {
      cfg.command ??= {};
      cfg.command[TRAIN_MODE_COMMAND] ??= {
        // No-op template: the toggle happens in command.execute.before.
        template: "",
        description: "Toggle Arbiter train-mode (verify/override per-subtask routing)",
      };
    },

    /** Intercept /train-mode: flip the mode, suppress the prompt, toast the result. */
    "command.execute.before": async (input, output) => {
      if (input.command !== TRAIN_MODE_COMMAND) return;
      const on = toggleManualMode();
      output.parts = []; // suppress — this is a UI toggle, not a model prompt
      await toast(modeMessage(on));
    },

    /** Plan interception: classify each actionable subtask; in train-mode show the menu. */
    event: async ({ event }) => {
      if (event.type !== "todo.updated") return;
      const { sessionID, todos } = event.properties;

      const actionable = todos.filter((t) => ACTIONABLE.has(t.status));
      const signature = JSON.stringify(actionable.map((t) => [t.id, t.content]));
      if (signature === lastSignature.get(sessionID)) return; // status-only churn
      lastSignature.set(sessionID, signature);

      if (actionable.length === 0) {
        pending.delete(sessionID);
        return;
      }

      const proposals = await classifyAll(
        actionable.map((t) => t.content),
        sessionID,
      );
      if (proposals.length === 0) return; // Server couldn't resolve anything -> stay silent

      if (isManualMode()) {
        await client.session
          .prompt({
            path: { id: sessionID },
            body: { noReply: true, parts: [{ type: "text", text: renderProposal(proposals) }] },
          })
          .catch(() => {});
        pending.set(sessionID, proposals);
      } else {
        // Auto mode: pass-through, every subtask's decision logged silently.
        for (const p of proposals) await logAccepted(p, sessionID);
      }
    },

    /** Override channel: interpret the reply to a pending menu and log the correction(s). */
    "chat.message": async (input, output) => {
      const sessionID = input.sessionID;
      const awaiting = pending.get(sessionID);
      if (!awaiting) return; // no proposal in flight -> not our concern

      const text = extractText(output.parts);
      if (!text.trim() || text.startsWith(MENU_PREFIX)) return; // empty / our own menu

      const ov = parseOverride(text, awaiting.length);
      if (!ov) {
        // Not a control reply: the user moved on; log the proposals as accepted-as-is.
        pending.delete(sessionID);
        for (const p of awaiting) await logAccepted(p, sessionID);
        return;
      }

      pending.delete(sessionID);
      if (ov.kind === "accept") {
        for (const p of awaiting) await logAccepted(p, sessionID);
        await toast("Arbiter: accepted.", "success");
        return;
      }

      // Override of one subtask; the rest stand as proposed (accepted).
      const target = awaiting[ov.index - 1];
      const correction: Correction = {
        task_text: target.taskText,
        predicted_tier: target.decision.tier,
        predicted_model: target.decision.model,
        confidence: target.decision.confidence,
        session_id: sessionID,
        corrected_tier: ov.targetIsTier ? ov.target : null,
        corrected_model: ov.targetIsTier ? null : ov.target,
      };
      await server.logCorrection(correction);
      for (let i = 0; i < awaiting.length; i++) {
        if (i !== ov.index - 1) await logAccepted(awaiting[i], sessionID);
      }
      await toast(`Arbiter: overrode #${ov.index} -> ${ov.target}.`, "success");
    },
  };
};
