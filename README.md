<div align="center">

# Arbiter

**A local-first model router for [OpenCode](https://opencode.ai).**

Automatically decides — per task, no manual `/models` switching — whether local (Ollama) or
cloud handles it, and manages the context handoff when that decision changes models mid-session.

[![npm](https://img.shields.io/npm/v/opencode-arbiter?color=0b0b0b&labelColor=0b0b0b)](https://www.npmjs.com/package/opencode-arbiter)
[![license](https://img.shields.io/badge/license-MIT-0b0b0b?labelColor=0b0b0b)](./LICENSE)

</div>

## Install

```bash
npm install -g opencode-arbiter
arbiter setup     # download + install the server, create a venv, patch OpenCode config
arbiter start     # run the router on 127.0.0.1:4317
```

`setup` needs **Python 3.11+** on your PATH (used for a CPU-only PyTorch venv) and, optionally,
**Ollama** for local models. OpenCode's provider `baseUrl` is pointed at the router, which
decides local vs cloud per task and forwards to the real provider using its own credentials.

## Commands

| Command | Does |
|---|---|
| `arbiter setup`  | Installs the server bundle + classifier checkpoint into `~/.arbiter`, verifies checksums, builds a venv, patches OpenCode config. |
| `arbiter start`  | Starts the server detached and records its PID. |
| `arbiter stop`   | Stops the running server. |
| `arbiter status` | Reports running/stopped and the server's `/health`. |

## How it routes

```
        OpenCode  ──baseUrl──►  Arbiter Server ──►  Ollama (local)
        + Notary plugin              │         └──►  Anthropic / OpenAI / … (cloud)
                                     │
                        Triage (classify → resolve)
                        Ferry  (context payload on switch)
                        Registry (models.dev + OpenRouter + Ollama)
```

- **Triage** classifies each task's difficulty with a trained classifier, then deterministically
  resolves difficulty → tier → a concrete model against a live model catalog.
- **Ferry** reacts to a model switch (or a new subagent session) and decides how much context
  travels with the task — scaled to the tier, not a separate judgment.
- **Notary** is the OpenCode plugin: it shows what Triage decided, lets you override per subtask,
  and logs corrections that fine-tune the classifier locally over time.

Everything routing-related lives in the server; the plugin is UI and logging only.

---

# Design

<!-- The high-level design, embedded. -->

## 1. Purpose

Arbiter is a local-first model router for OpenCode. It sits between OpenCode and the LLM
providers it talks to, automatically deciding — on every task, with no manual `/models`
switching — whether local (Ollama) or cloud handles it, based on task complexity and type. It
also solves a problem general coding agents don't: when an automatic decision causes a
*different* model to take over a task, Arbiter decides what context that model actually needs,
instead of either always dumping the full raw history or always compressing it.

## 2. Non-goals

- Centralized or cross-user shared learning. All classifier personalization stays local to one
  user's machine.
- Support for IDEs/agents other than OpenCode.
- A hosted backend, account system, or telemetry service.
- A general-purpose provider gateway (load balancing/failover across redundant deployments of
  the *same* model). Arbiter routes between *different* tiers of capability.
- Task decomposition. OpenCode's own Plan mode and subagent system split work into subtasks;
  Arbiter classifies and routes what's handed to it.

## 3. Two components

| Component | Runtime | Why |
|---|---|---|
| **Server** | Python (FastAPI) | Owns model inference (the PyTorch classifier), the actual provider HTTP calls, and request rewriting. Decoupled from OpenCode's plugin runtime — reached over localhost HTTP. |
| **Notary** | TypeScript (`@opencode-ai/plugin`) | OpenCode's plugin loader only accepts JS/TS. Owns everything user-facing: the plan-mode menu, overrides, correction logging. |

The split is deliberate: it isolates routing-critical logic (Triage, Ferry) from the less
stable surface (OpenCode's plugin API has a documented history of breaking across patch
versions). If the plugin API changes, only Notary needs patching; the Server keeps working.

## 4. Server modules

### 4.1 Triage — two independent halves

**Half 1 — difficulty classification (the ML part).** A small PyTorch MLP on top of frozen
pretrained sentence embeddings. Input: task text. Output: a difficulty label
(`easy` / `medium` / `hard`) plus a confidence score. It is pretrained on SWE-smith (52K real
bug-fix instances across 128 production repos, MIT licensed) and ships as the default weights —
nobody trains from a cold model. Per-user corrections fine-tune incrementally on top.

**Half 2 — difficulty → tier → model (deterministic, not ML).** Map the label to a tier
(`easy → local-simple`, `medium → cloud-mid`, `hard → cloud-heavy`), then filter candidate
models by context-window fit and a capability floor and rank the survivors. Ranking priority:
real OpenRouter benchmark score where it exists → a `reasoning`-flag-plus-cost fallback →
Ollama parameter count as a last resort for local-only resolutions. A per-request
`reasoning_effort` is applied on top as a separate lever, defaulting higher for higher tiers.

### 4.2 Ferry — reactive context payload

Ferry never picks a destination; it reacts to what Triage decided and answers one question:
what context accompanies this task to its new model. It fires on two triggers — a mid-session
model switch, or a new child session spawned by OpenCode's subagent system — and scales the
payload to the tier:

| Tier | Payload |
|---|---|
| `local-simple` | Minimal — the task text plus directly relevant files only. |
| `cloud-mid` | A structured handoff summary (goal, decisions, work done, remaining tasks, hard constraints). |
| `cloud-heavy` | Full raw history, uncompressed. |

Every compaction system in existing agents triggers on context-window overflow within one
model's session — none have a code path for the model *changing*. That gap is what Ferry fills.
Who *writes* the mid-tier summary is a separate decision from who receives it: the cheapest
available model, local-first.

### 4.3 Registry

A shared utility both Triage and Ferry call. It aggregates three sources: **models.dev**
(pricing, context window, the `reasoning` flag — broad provider coverage), **OpenRouter**
(populated benchmark scores for a real subset of models, treated as a property of the weights
and so transferable to a local copy), and **Ollama's `/api/tags`** (what's actually installed
locally right now). Matching model identity across the three naming schemes is real work, not a
lookup.

## 5. Notary (plugin)

The human-in-the-loop checkpoint. It intercepts OpenCode's own Plan-mode output and, per
subtask, shows Triage's resolved decision as a numbered accept/override menu:

```
1. [local-simple -> qwen3-coder]   rename variable in auth.ts
2. [cloud-mid -> sonnet]           implement JWT refresh logic
3. [cloud-heavy -> opus]           redesign session architecture

Accept all: enter
Override: type subtask number, then a tier or a specific model
```

`/train-mode` toggles the manual menu on/off. Every resolution (accepted or overridden) is
logged locally to SQLite and triggers a small, async, single-step fine-tune on top of Triage's
existing weights — never a full retrain, never synced anywhere.

## 6. Request lifecycle

1. A task is produced (by the user or OpenCode's Plan mode, per subtask).
2. Notary intercepts and calls the Server's classify operation.
3. Triage returns the resolved decision: tier, confidence, concrete model.
4. In manual mode, Notary shows the menu and waits for accept/override.
5. The decision is logged; an override also feeds the incremental fine-tune.
6. OpenCode's normal request flow hits the Server (its `baseUrl`), not the real provider.
7. The Server rewrites the request's `model` field and authenticates with its own credentials.
8. Ferry checks whether this request's model differs from the session's last turn; if so, it
   reshapes the context to the tier's payload, then forwards.
9. The response streams back through the Server unmodified.

## License

MIT — see [LICENSE](./LICENSE).
