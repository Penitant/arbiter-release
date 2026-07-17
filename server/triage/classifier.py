"""Trained difficulty classifier: task text -> easy/medium/hard + confidence."""

from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn.functional as F

from triage import heuristic
from triage.heuristic import Classification, _extract_text
from triage.model import CHECKPOINT_PATH, LABELS, TriageMLP, get_embedder, load_checkpoint

__all__ = ["Classification", "classify", "reload_model"]

# One lock guards both the model reference and the forward pass.
_lock = threading.Lock()
_model: TriageMLP | None = None

# Load once at import; a missing checkpoint falls back to the keyword heuristic.
if CHECKPOINT_PATH.exists():
    _model = load_checkpoint(CHECKPOINT_PATH)
else:
    print(f"[triage] no checkpoint at {CHECKPOINT_PATH} — falling back to keyword heuristic")


def reload_model() -> None:
    """Re-read the checkpoint from disk (called after each fine-tune update)."""
    global _model
    model = load_checkpoint(CHECKPOINT_PATH)
    with _lock:
        _model = model


def classify(body: dict[str, Any] | str | None) -> Classification:
    """Classify a parsed request body (or bare task string) into easy/medium/hard."""
    if isinstance(body, str):
        text = body
    else:
        text, _ = _extract_text(body)
    if _model is None:
        return heuristic.classify(
            {"messages": [{"role": "user", "content": text}]} if isinstance(body, str) else body
        )
    if not text.strip():
        # Empty/unparseable body: medium default without running the model.
        return Classification("medium", 0.5, LABELS.index("medium"), ("mlp:empty-body",))

    x = get_embedder().embed([text])
    with _lock, torch.no_grad():
        probs = F.softmax(_model(x), dim=-1)[0]
    idx = int(probs.argmax())
    signals = tuple(f"mlp:p_{name}={p:.3f}" for name, p in zip(LABELS, probs.tolist()))
    return Classification(LABELS[idx], round(float(probs[idx]), 4), idx, signals)
