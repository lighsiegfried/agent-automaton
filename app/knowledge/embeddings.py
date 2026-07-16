"""Local text embeddings (Phase 6A, item 4).

Documents are NEVER sent to an external API. The default ``HashEmbedder`` is a
deterministic, dependency-free local embedder (hashed word + character n-gram features
projected to a fixed dimension and L2-normalized) — a genuine local text embedding
that is CPU-only, reproducible, and always available (so tests are deterministic and a
GPU-less host still works). A neural ``SentenceTransformerEmbedder`` is used instead
when it is installed and configured; its model cache lives in the configured host/Docker
cache. The embedder is separate from the generation model — the two never share
responsibilities.
"""

from __future__ import annotations

import hashlib
import math
import re

from app.core.logger import get_logger

log = get_logger(__name__)

_TOKEN = re.compile(r"[0-9a-záéíóúüñ]+", re.IGNORECASE)


class HashEmbedder:
    """Deterministic local embedder — the default and CPU fallback."""

    name = "hash-local-v1"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        words = _TOKEN.findall((text or "").lower())
        feats = list(words)
        # Character 3-grams add sub-word signal without any model.
        for w in words:
            padded = f"#{w}#"
            feats.extend(padded[i:i + 3] for i in range(len(padded) - 2))
        return feats

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in self._tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:  # pragma: no cover - requires optional heavy deps
    """Optional neural embedder (lazy import; never external)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


def resolve_device(configured: str) -> str:
    if configured == "cpu":
        return "cpu"
    try:  # pragma: no cover - depends on torch availability
        import torch
        if configured in ("auto", "cuda") and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_embedder(settings):
    """Resolve the local embedder. Falls back to the deterministic HashEmbedder when
    sentence-transformers is unavailable — always local, always available."""
    try:  # pragma: no cover - optional
        import sentence_transformers  # noqa: F401
        device = resolve_device(settings.knowledge_embedding_device)
        return SentenceTransformerEmbedder(device=device)
    except Exception:
        return HashEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
