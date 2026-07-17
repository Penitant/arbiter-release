"""Incremental per-correction fine-tune of the Triage MLP, on top of the shipped checkpoint."""

from __future__ import annotations

import sqlite3
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from config import DIFFICULTY_TO_TIER, config
from triage.model import CHECKPOINT_PATH, LABELS, get_embedder, load_checkpoint, save_checkpoint

# Corrections arrive in tier vocabulary; the classifier learns in difficulty vocabulary.
_TIER_TO_DIFFICULTY = {tier: diff for diff, tier in DIFFICULTY_TO_TIER.items()}

STEPS = 3      # gradient steps per correction — a nudge, not a retrain
LR = 1e-4      # same LR as initial training

# Single worker: queued corrections run strictly one at a time, never racing on the checkpoint.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="finetune")


def enqueue(db_path: str | Path | None = None) -> Future:
    """Submit a fine-tune pass to the single-worker queue. Never blocks."""
    return _executor.submit(_run_safe, db_path)


def _run_safe(db_path: str | Path | None) -> int:
    # Surface failures: nobody reads the Future, so exceptions would die silently.
    try:
        return run_finetune(db_path)
    except Exception:
        print("[finetune] pass failed:")
        traceback.print_exc()
        return 0


def run_finetune(db_path: str | Path | None = None) -> int:
    """Process all unprocessed override rows. Returns how many were applied."""
    path = Path(db_path) if db_path is not None else Path(config.corrections_db_path)
    if not path.exists() or not CHECKPOINT_PATH.exists():
        return 0  # nothing logged yet, or no pretrained weights to tune on top of

    conn = sqlite3.connect(str(path))
    try:
        last = conn.execute(
            "SELECT last_processed_id FROM finetune_state WHERE id = 1"
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT id, task_text, corrected_tier FROM corrections
            WHERE id > ? AND corrected_tier IS NOT NULL
            ORDER BY id
            """,
            (last,),
        ).fetchall()
        if not rows:
            return 0

        applied = 0
        for row_id, task_text, corrected_tier in rows:
            difficulty = _TIER_TO_DIFFICULTY.get(corrected_tier)
            if difficulty is None:
                # Skipping (and advancing past) a bad row beats wedging the queue.
                _advance(conn, row_id)
                continue
            _apply_one(task_text, difficulty)
            # Cursor advances only after _apply_one's atomic checkpoint replace.
            _advance(conn, row_id)
            applied += 1
        return applied
    finally:
        conn.close()


def _apply_one(task_text: str, difficulty: str) -> None:
    """STEPS gradient steps on one example, then atomic checkpoint swap."""
    x = get_embedder().embed([task_text])  # frozen stack: no_grad inside
    y = torch.tensor([LABELS.index(difficulty)])

    model = load_checkpoint(CHECKPOINT_PATH)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    model.eval()

    save_checkpoint(model, CHECKPOINT_PATH, finetuned_at=_now())

    # Hot-swap the in-memory model; late import keeps this module lightweight to import.
    try:
        from triage import classifier

        classifier.reload_model()
    except Exception:
        pass  # inference keeps the old weights until next restart — never fatal


def _advance(conn: sqlite3.Connection, row_id: int) -> None:
    conn.execute(
        "UPDATE finetune_state SET last_processed_id = ?, last_run_at = ? WHERE id = 1",
        (row_id, _now()),
    )
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
