"""Train the Triage difficulty MLP on precomputed embeddings (run: `uv run python triage/train.py`)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triage.model import CHECKPOINT_PATH, LABELS, TriageMLP, save_checkpoint  # noqa: E402

EMBEDDINGS = Path(__file__).parent / "data" / "embeddings.pt"

SEED = 42
LR = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 3


def stratified_split(y: torch.Tensor, val_frac: float = 0.2):
    """Per-class 80/20 index split so val keeps the true class mix."""
    g = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = [], []
    for cls in range(len(LABELS)):
        idx = (y == cls).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_val = max(1, int(len(idx) * val_frac))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return torch.cat(train_idx), torch.cat(val_idx)


def per_class_accuracy(logits: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    out = {}
    for cls, name in enumerate(LABELS):
        mask = y == cls
        out[name] = (pred[mask] == cls).float().mean().item() if mask.any() else float("nan")
    return out


def main() -> None:
    torch.manual_seed(SEED)
    data = torch.load(EMBEDDINGS, weights_only=True)
    X, y = data["X"], data["y"]
    print(f"[train] X={tuple(X.shape)} labels={dict(zip(LABELS, torch.bincount(y).tolist()))}")

    train_idx, val_idx = stratified_split(y)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f"[train] split: {len(train_idx)} train / {len(val_idx)} val")

    # Inverse-frequency class weights (mean 1) so the head can't collapse to always-medium.
    counts = torch.bincount(y_train, minlength=len(LABELS)).float()
    weights = counts.sum() / (len(LABELS) * counts)
    print(f"[train] class weights: {dict(zip(LABELS, [round(w, 3) for w in weights.tolist()]))}")

    model = TriageMLP()
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    stale = 0
    g = torch.Generator().manual_seed(SEED)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_train), generator=g)
        total = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            batch = perm[i : i + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch]), y_train[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = criterion(val_logits, y_val).item()
        acc = per_class_accuracy(val_logits, y_val)
        overall = (val_logits.argmax(-1) == y_val).float().mean().item()
        print(
            f"[train] epoch {epoch:3d}  train_loss={total / len(perm):.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={overall:.3f}  "
            f"per-class={{{', '.join(f'{k}: {v:.3f}' for k, v in acc.items())}}}",
            flush=True,
        )

        if val_loss < best_val:
            best_val = val_loss
            stale = 0
            save_checkpoint(model, CHECKPOINT_PATH, val_loss=val_loss, epoch=epoch,
                            val_per_class_acc=acc, val_acc=overall)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"[train] early stop at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"[train] best val_loss={best_val:.4f} -> {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
