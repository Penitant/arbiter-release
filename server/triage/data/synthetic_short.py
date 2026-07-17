"""Generate the seeded synthetic short-prompt slice (run: `uv run python triage/data/synthetic_short.py`)."""

from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).parent / "synthetic_short.jsonl"
SEED = 42
PER_CLASS = 160

FILES = [
    "auth.ts", "main.py", "utils.js", "server.go", "config.yaml", "README.md",
    "api/routes.py", "src/parser.rs", "app/models.py", "handlers/user.ts",
    "lib/cache.js", "core/scheduler.py", "db/queries.sql", "cli.py",
]
IDENTS = [
    "getUser", "parse_config", "handleRequest", "MAX_RETRIES", "user_id",
    "fetchData", "validate_token", "conn_pool", "retry_count", "sessionStore",
]
MODULES = [
    "the auth module", "the request pipeline", "the job scheduler", "the cache layer",
    "the payment service", "the sync engine", "the import pipeline", "the websocket handler",
]
CONCEPTS = [
    "a mutex", "dependency injection", "the event loop", "a foreign key",
    "tail recursion", "the borrow checker", "a closure", "an index scan",
]
FEATURES = [
    "pagination for the list endpoint", "an endpoint that returns the user's profile",
    "input validation for the signup form", "a --dry-run flag for the CLI",
    "retry logic for the HTTP client", "a health check endpoint",
    "rate limiting on the login route", "CSV export for the report page",
    "sorting by date to the results table", "email notifications on task completion",
]

EASY_TEMPLATES = [
    "fix a typo in {file}",
    "fix the spelling mistake in {file}",
    "rename {ident} to {ident2} in {file}",
    "rename this variable in {file}",
    "add a docstring to {ident}",
    "add a comment explaining {ident}",
    "reformat {file}",
    "fix the indentation in {file}",
    "fix the import in {file}",
    "remove the unused import from {file}",
    "delete the commented-out code in {file}",
    "remove the print statement from {file}",
    "fix the lint warnings in {file}",
    "what is {concept}?",
    "what does {ident} do?",
    "list the functions in {file}",
    "show me the contents of {file}",
    "bump the version number in {file}",
]

MEDIUM_TEMPLATES = [
    "add {feature}",
    "implement {feature}",
    "write a unit test for {ident} covering the empty-input case",
    "add a test for {feature}",
    "update {file} to read the port from an environment variable",
    "add error handling to {ident} so it returns a 400 on bad input",
    "extend {file} to support JSON output as well",
    "wire up {feature} using the existing helpers in {file}",
    "add logging to {module} so failures are visible",
    "implement the TODO in {file}",
    "make {ident} configurable instead of hardcoded",
    "add a database column for last_login and expose it in the API",
]

HARD_TEMPLATES = [
    "refactor {module} to remove the circular dependency",
    "redesign {module} so it can run concurrently",
    "debug the race condition in {module}",
    "there's a deadlock somewhere in {module} when two workers start at once — find and fix it",
    "why does {module} deadlock under load?",
    "find the root cause of the memory leak in {module}",
    "optimize the performance of {module}, it's the main bottleneck",
    "profile {module} and fix whatever is making it slow",
    "why is {ident} returning stale data after a concurrent write?",
    "migrate {module} from callbacks to async/await without breaking the public API",
    "rewrite {module} to scale past a single process",
    "trace through {module} and explain why the regression appeared",
    "Traceback (most recent call last):\n  File \"{file}\", line 42, in {ident}\n"
    "ValueError: unexpected state\nwhy does this keep happening when I run the tests?",
    "TypeError: cannot read properties of undefined (reading '{ident}')\n"
    "    at {ident2} ({file}:87:13)\nthis only fails in production — debug it",
]


def _fill(template: str, rng: random.Random) -> str:
    ident, ident2 = rng.sample(IDENTS, 2)
    return template.format(
        file=rng.choice(FILES), ident=ident, ident2=ident2,
        module=rng.choice(MODULES), concept=rng.choice(CONCEPTS),
        feature=rng.choice(FEATURES),
    )


def generate() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []
    for label, templates in (
        ("easy", EASY_TEMPLATES), ("medium", MEDIUM_TEMPLATES), ("hard", HARD_TEMPLATES)
    ):
        seen: set[str] = set()
        i = 0
        while len(seen) < PER_CLASS:
            text = _fill(templates[i % len(templates)], rng)
            i += 1
            if text in seen:
                continue
            seen.add(text)
            rows.append({
                "instance_id": f"synthetic__{label}_{len(seen):03d}",
                "text": text,
                "label": label,
            })
    return rows


if __name__ == "__main__":
    rows = generate()
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[synthetic] wrote {len(rows)} rows -> {OUT.name}")
