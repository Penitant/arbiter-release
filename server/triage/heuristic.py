"""Keyword + structural difficulty heuristic; fallback when no trained checkpoint exists."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Substrings signaling work likely to need a capable cloud model.
_HARD_SIGNALS = (
    "refactor", "architect", "design ", "concurren", "race condition", "deadlock",
    "optimi", "performance", "profil", "migrat", "security", "vulnerab",
    "distributed", "algorithm", "debug", "root cause", "regression", "threading",
    "async", "memory leak", "scal", "why is", "why does", "trace through",
    "rewrite", "redesign", "bottleneck",
)

# Substrings signaling a trivial, mechanical edit better served locally.
_EASY_SIGNALS = (
    "typo", "rename", "reformat", "format ", "indent", "add a comment",
    "docstring", "spelling", "lint", "fix the import", "print statement",
    "what is", "list the", "show me the", "rename the", "add a test for",
)

_CODE_FENCE = re.compile(r"```")
# A traceback or error dump is a strong "real debugging" tell.
_TRACEBACK = re.compile(
    r"traceback \(most recent call last\)|^\s*at .+\(.+:\d+\)|Exception:|Error:",
    re.IGNORECASE | re.MULTILINE,
)

# Score bucket thresholds; medium is the default for ambiguous input.
_HARD_AT = 3
_EASY_AT = -2


@dataclass(frozen=True)
class Classification:
    difficulty: str          # "easy" | "medium" | "hard"
    confidence: float        # 0.0-1.0
    score: int               # raw heuristic score, for logging
    signals: tuple[str, ...]  # which signals fired, for observability


def _extract_text(body: dict[str, Any] | None) -> tuple[str, int]:
    """Return (concatenated message text, message count); unparseable bodies yield ("", 0)."""
    if not isinstance(body, dict):
        return "", 0
    messages = body.get("messages")
    if not isinstance(messages, list):
        return "", 0

    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # Content blocks: pull text out of each.
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
    return "\n".join(parts), len(messages)


def classify(body: dict[str, Any] | None) -> Classification:
    """Classify a parsed request body into easy / medium / hard."""
    text, n_messages = _extract_text(body)
    lowered = text.lower()
    fired: list[str] = []

    score = 0
    for kw in _HARD_SIGNALS:
        if kw in lowered:
            score += 2
            fired.append(f"hard:{kw.strip()}")
    for kw in _EASY_SIGNALS:
        if kw in lowered:
            score -= 2
            fired.append(f"easy:{kw.strip()}")

    # Structural signals: supporting, not dominant.
    length = len(text)
    if length > 2000:
        score += 2
        fired.append("struct:long")
    elif 0 < length < 120:
        score -= 2
        fired.append("struct:short")
    if n_messages > 12:
        score += 1
        fired.append("struct:long-convo")
    if _CODE_FENCE.search(text):
        score += 1
        fired.append("struct:code-fence")
    if _TRACEBACK.search(text):
        score += 2
        fired.append("struct:traceback")

    if score >= _HARD_AT:
        difficulty = "hard"
        margin = score - _HARD_AT
    elif score <= _EASY_AT:
        difficulty = "easy"
        margin = _EASY_AT - score
    else:
        difficulty = "medium"
        margin = 0

    # Confidence floor at 0.5, rising with distance past the bucket edge.
    confidence = round(min(1.0, 0.5 + margin * 0.1), 2)
    return Classification(difficulty, confidence, score, tuple(fired))
