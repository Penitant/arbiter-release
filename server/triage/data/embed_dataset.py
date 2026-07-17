"""Embed the labeled sets into embeddings.pt {"X", "y"} (run: `uv run python triage/data/embed_dataset.py`)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from triage.model import LABELS, Embedder  # noqa: E402

DATA_DIR = Path(__file__).parent
SOURCES = (DATA_DIR / "train_labeled.jsonl", DATA_DIR / "synthetic_short.jsonl")
OUT = DATA_DIR / "embeddings.pt"


def main() -> None:
    rows = [json.loads(line) for src in SOURCES for line in src.open()]
    texts = [r["text"] for r in rows]
    y = torch.tensor([LABELS.index(r["label"]) for r in rows], dtype=torch.int64)
    print(f"[embed] {len(texts)} texts")

    embedder = Embedder()
    start = time.time()
    chunks = []
    step = 64
    for i in range(0, len(texts), step):
        chunks.append(embedder.embed(texts[i : i + step]))
        done = min(i + step, len(texts))
        rate = done / (time.time() - start)
        print(f"[embed] {done}/{len(texts)} ({rate:.1f} texts/s)", flush=True)
    X = torch.cat(chunks, dim=0)

    torch.save({"X": X, "y": y}, OUT)
    print(f"[embed] saved X={tuple(X.shape)} y={tuple(y.shape)} -> {OUT.name}")


if __name__ == "__main__":
    main()
