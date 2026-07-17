"""Deterministic 50-prompt classifier evaluation (run: `uv run python -m triage.evaluate`)."""

# Checkpoint fa51096 results: overall 0.780 — easy 0.941 (16/17), medium 0.471 (8/17),
# hard 0.938 (15/16). Dominant error: medium predicted as easy (8 of 11 misses).

from __future__ import annotations

from triage.model import LABELS

# Fixed hand-written prompts (17 easy / 17 medium / 16 hard), disjoint from training templates.
EVAL_SET: tuple[tuple[str, str], ...] = (
    # easy: mechanical edits and lookups
    ("correct the misspelled word in CONTRIBUTING.md", "easy"),
    ("rename the function loadSettings to readSettings in settings.ts", "easy"),
    ("add a one-line comment above the retry loop in fetch.js", "easy"),
    ("remove the leftover console.log in checkout.tsx", "easy"),
    ("fix the broken import path in tests/helpers.py", "easy"),
    ("reindent the yaml block in deploy.yml", "easy"),
    ("add a docstring to the sum_totals helper", "easy"),
    ("delete the dead code at the bottom of legacy.py", "easy"),
    ("what does the --verbose flag do?", "easy"),
    ("show me the signature of create_order", "easy"),
    ("list the exported symbols in index.ts", "easy"),
    ("change the default port constant to 8080 in constants.go", "easy"),
    ("fix the typo in the error message in login.py", "easy"),
    ("run the formatter on src/models.rs", "easy"),
    ("what is a generator in python?", "easy"),
    ("capitalize the heading in docs/setup.md", "easy"),
    ("swap the order of these two imports in app.py", "easy"),
    # medium: bounded single-feature work
    ("add a DELETE endpoint for removing a saved search", "medium"),
    ("write tests for the slugify helper including unicode input", "medium"),
    ("add a retry with backoff to the webhook sender", "medium"),
    ("support a --json flag on the stats command", "medium"),
    ("validate the email field before saving the subscriber", "medium"),
    ("add a created_at timestamp to the invoices table and return it from the API", "medium"),
    ("read the API base url from an env var instead of the constant", "medium"),
    ("add structured logging to the export job", "medium"),
    ("implement the stubbed-out parse_headers function", "medium"),
    ("cache the results of get_exchange_rates for five minutes", "medium"),
    ("add a confirmation step before the purge command deletes anything", "medium"),
    ("return a 404 instead of a 500 when the record is missing", "medium"),
    ("add a healthz route that checks the database connection", "medium"),
    ("make the batch size a config option with a sane default", "medium"),
    ("add cursor-based paging to the activity feed", "medium"),
    ("wire the new template into the notification sender", "medium"),
    ("handle the empty-file case in the CSV importer", "medium"),
    # hard: cross-cutting debugging, redesign, perf
    ("the worker pool hangs when a job panics — track down why and fix it", "hard"),
    ("requests intermittently time out after the connection pool change, find the root cause", "hard"),
    ("restructure the billing module so plans and invoicing no longer depend on each other", "hard"),
    ("the cache and the database disagree after concurrent updates — fix the consistency bug", "hard"),
    ("memory usage grows unbounded in the long-running sync daemon, find the leak", "hard"),
    ("make the ingest pipeline safe to run from multiple processes at once", "hard"),
    ("the p99 latency doubled last release — profile and fix the regression", "hard"),
    ("convert the callback-based uploader to async without changing its public interface", "hard"),
    ("why do two schedulers sometimes pick up the same job?", "hard"),
    ("the integration tests deadlock on CI but pass locally — figure out why", "hard"),
    ("redesign the retry queue so it survives a process crash", "hard"),
    ("eliminate the N+1 queries that make the dashboard slow", "hard"),
    ("split the monolithic request handler so middleware can be tested in isolation", "hard"),
    ("Traceback (most recent call last):\n  File \"pipeline.py\", line 118, in flush\n"
     "RuntimeError: dictionary changed size during iteration\n"
     "this started after the concurrency change — fix it", "hard"),
    ("panic: send on closed channel\n\ngoroutine 42 [running]:\nmain.(*Broker).publish(...)\n"
     "this crashes the broker under load, debug it", "hard"),
    ("the migration corrupts rows with null timestamps — write a safe migration and a backfill", "hard"),
)


def evaluate() -> dict:
    from triage.classifier import classify

    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    misses: list[tuple[str, str, str, float]] = []
    for text, expected in EVAL_SET:
        result = classify(text)
        confusion[expected][result.difficulty] += 1
        if result.difficulty != expected:
            misses.append((text, expected, result.difficulty, result.confidence))

    per_class = {
        label: confusion[label][label] / max(1, sum(confusion[label].values()))
        for label in LABELS
    }
    correct = sum(confusion[label][label] for label in LABELS)
    return {
        "n": len(EVAL_SET),
        "accuracy": correct / len(EVAL_SET),
        "per_class": per_class,
        "confusion": confusion,
        "misses": misses,
    }


def main() -> None:
    r = evaluate()
    print(f"[eval] {r['n']} prompts, accuracy {r['accuracy']:.3f}")
    for label in LABELS:
        print(f"[eval]   {label:<6} acc {r['per_class'][label]:.3f}")
    header = " ".join(f"{p:>7}" for p in LABELS)
    print(f"[eval] confusion (rows=true, cols=predicted):\n[eval]          {header}")
    for t in LABELS:
        row = " ".join(f"{r['confusion'][t][p]:>7}" for p in LABELS)
        print(f"[eval]   {t:<6} {row}")
    if r["misses"]:
        print(f"[eval] {len(r['misses'])} misclassified:")
        for text, expected, got, conf in r["misses"]:
            one_line = text.splitlines()[0]
            print(f"[eval]   {expected}->{got} ({conf:.2f}): {one_line}")


if __name__ == "__main__":
    main()
