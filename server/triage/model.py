"""Shared Triage model pieces: frozen embedding stack, MLP head, checkpoint format."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Restore the Conv1D re-export (dropped in transformers 4.5x) that codesage's remote code imports.
import transformers.modeling_utils as _mu
from transformers.pytorch_utils import Conv1D as _Conv1D

if not hasattr(_mu, "Conv1D"):
    _mu.Conv1D = _Conv1D

from transformers import AutoModel, AutoTokenizer  # noqa: E402

LABELS: tuple[str, ...] = ("easy", "medium", "hard")

CODESAGE = "codesage/codesage-small-v2"
E5 = "intfloat/multilingual-e5-small"
CODESAGE_MAX_TOKENS = 1024
E5_MAX_TOKENS = 512  # e5's hard cap

EMBED_DIM = 1024 + 384  # codesage pooler + e5 mean-pooled

CHECKPOINT_PATH = Path(__file__).parent / "weights" / "checkpoint.pt"


class Embedder:
    """Frozen two-model embedding stack; CPU, inference-only, lazily loaded."""

    def __init__(self) -> None:
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._cs_tok = AutoTokenizer.from_pretrained(CODESAGE, trust_remote_code=True)
        self._cs = AutoModel.from_pretrained(CODESAGE, trust_remote_code=True)
        self._e5_tok = AutoTokenizer.from_pretrained(E5)
        self._e5 = AutoModel.from_pretrained(E5)
        for m in (self._cs, self._e5):
            m.eval()
            for p in m.parameters():  # frozen: no gradient ever flows here
                p.requires_grad_(False)
        self._loaded = True

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 16) -> torch.Tensor:
        """Embed texts -> float32 tensor [len(texts), EMBED_DIM]."""
        self._load()
        chunks: list[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            enc = self._cs_tok(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=CODESAGE_MAX_TOKENS,
            )
            cs_vec = F.normalize(self._cs(**enc).pooler_output, dim=-1)

            # e5 requires its "query: " instruction prefix.
            enc = self._e5_tok(
                [f"query: {t}" for t in batch], return_tensors="pt", padding=True,
                truncation=True, max_length=E5_MAX_TOKENS,
            )
            out = self._e5(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            e5_vec = F.normalize(pooled, dim=-1)

            # Each half L2-normalized so neither dominates on raw magnitude.
            chunks.append(torch.cat([cs_vec, e5_vec], dim=-1).float())
        return torch.cat(chunks, dim=0)


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Process-wide embedder singleton."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER


class TriageMLP(nn.Module):
    """Trained head: concatenated embedding -> 3 raw difficulty logits (no softmax)."""

    def __init__(self, input_dim: int = EMBED_DIM,
                 hidden1: int = 256, hidden2: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, len(LABELS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_checkpoint(model: TriageMLP, path: Path, **meta) -> None:
    """Atomic checkpoint write: temp file in the same directory, then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), "labels": LABELS,
               "input_dim": EMBED_DIM, **meta}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> TriageMLP:
    """Load the MLP head from a checkpoint written by save_checkpoint()."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(payload.get("labels", ())) != LABELS:
        raise ValueError(f"checkpoint label order {payload.get('labels')} != {LABELS}")
    model = TriageMLP(input_dim=payload.get("input_dim", EMBED_DIM))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
