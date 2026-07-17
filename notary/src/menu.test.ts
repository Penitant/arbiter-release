// Tests for menu rendering + override parsing; runs under Node's built-in test runner.

import assert from "node:assert/strict";
import { test } from "node:test";

import type { Decision } from "./client.ts";
import {
  isTier,
  parseOverride,
  renderProposal,
  shortModel,
} from "./menu.ts";

function decision(over: Partial<Decision> = {}): Decision {
  return {
    difficulty: "medium",
    confidence: 0.82,
    tier: "cloud-mid",
    model: "google/gemini-2.5-flash",
    reasoning_effort: "medium",
    reason: "cheapest reasoning-capable cloud-mid model",
    ...over,
  };
}

test("shortModel strips the provider prefix", () => {
  assert.equal(shortModel("google/gemini-3.5-flash"), "gemini-3.5-flash");
  assert.equal(shortModel("qwen3:8b"), "qwen3:8b"); // no slash -> unchanged
});

test("isTier recognizes only the three real tiers", () => {
  assert.ok(isTier("cloud-heavy"));
  assert.ok(!isTier("ultra"));
});

test("renderProposal shows the tier->model tag, task, and footer", () => {
  const out = renderProposal([
    { taskText: "implement JWT refresh logic", decision: decision() },
  ]);
  assert.match(out, /^Arbiter proposes:/);
  assert.match(out, /\[cloud-mid -> gemini-2\.5-flash\]/);
  assert.match(out, /implement JWT refresh logic/);
  assert.match(out, /Accept: enter/);
});

test("renderProposal aligns multiple subtasks and shows multi-footer", () => {
  const out = renderProposal([
    { taskText: "rename a var", decision: decision({ tier: "local-simple", model: "ollama/qwen3:8b" }) },
    { taskText: "redesign session arch", decision: decision({ tier: "cloud-heavy", model: "google/gemini-3.5-flash" }) },
  ]);
  const lines = out.split("\n").filter((l) => /^\d+\. /.test(l));
  assert.equal(lines.length, 2);
  // Tags padded to equal width -> the task text starts at the same column.
  assert.equal(lines[0].indexOf("rename"), lines[1].indexOf("redesign"));
  assert.match(out, /Accept all: enter/);
});

test("parseOverride treats empty/accept tokens as accept", () => {
  for (const t of ["", "  ", "accept", "ACCEPT ALL", "y", "ok"]) {
    assert.deepEqual(parseOverride(t, 1), { kind: "accept" });
  }
});

test("parseOverride parses '<n> <model>' as a model override", () => {
  assert.deepEqual(parseOverride("2 gemini-3.5-flash", 3), {
    kind: "override",
    index: 2,
    target: "gemini-3.5-flash",
    targetIsTier: false,
  });
});

test("parseOverride recognizes a tier target", () => {
  const ov = parseOverride("2 cloud-heavy", 3);
  assert.deepEqual(ov, { kind: "override", index: 2, target: "cloud-heavy", targetIsTier: true });
});

test("parseOverride allows a bare target only for a single subtask", () => {
  assert.deepEqual(parseOverride("gemini-3.5-flash", 1), {
    kind: "override",
    index: 1,
    target: "gemini-3.5-flash",
    targetIsTier: false,
  });
  // With multiple subtasks a bare target is ambiguous -> not a control input.
  assert.equal(parseOverride("gemini-3.5-flash", 3), null);
});

test("parseOverride rejects an out-of-range index", () => {
  assert.equal(parseOverride("9 gemini-3.5-flash", 3), null);
});

test("parseOverride returns null for an ordinary new task message", () => {
  assert.equal(parseOverride("now add tests for the auth module please", 1), null);
});
