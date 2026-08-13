"""Embedding.

Runs `BAAI/bge-base-en-v1.5` through ONNX Runtime via `fastembed`, not PyTorch. The
model is ~90 MB more either way; the difference is the runtime — a torch install is
roughly 2 GB and this service never trains anything, so it is 2 GB of cold-start and
image size bought for nothing. ONNX also gives deterministic CPU inference, which
makes the conformance-style tests below meaningful.

**The asymmetry that matters.** bge-*-en-v1.5 is trained so a query and a passage are
encoded differently: the query gets a short instruction prefix, the passage gets
nothing. Embedding both the same way costs a few points of recall and reports no
error at all — retrieval simply gets quietly worse. `embed_query` and `embed_passages`
are separate methods here specifically so a caller cannot use the wrong one by
accident, and `Embedder` is a Protocol so a different backend has to honour the same
split.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from .config import DEFAULT_DIM, DEFAULT_MODEL, DEFAULT_QUERY_PREFIX


@runtime_checkable
class Embedder(Protocol):
    """The contract retrieval depends on. Keeps the store backend-agnostic."""

    dim: int

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    """Scale to unit length so cosine distance and inner product agree.

    pgvector's HNSW index is built for one operator class. Normalising here means
    `vector_cosine_ops` and a dot product rank identically, so switching operator
    later cannot silently reorder results.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]


class FastEmbedEmbedder:
    """`fastembed` backend. Downloads the model on first use, then caches it."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        dim: int = DEFAULT_DIM,
        query_prefix: str = DEFAULT_QUERY_PREFIX,
        cache_dir: str | None = None,
        batch_size: int = 64,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "fastembed is not installed. Run: pip install -e '.[embed]'"
            ) from exc

        self.model_name = model_name
        self.dim = dim
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def _encode(self, texts: Iterable[str]) -> list[list[float]]:
        # fastembed exposes `query_embed`/`passage_embed` helpers that apply their own
        # prefixes. We deliberately call plain `embed` and own the prefix ourselves:
        # the transformation applied to the text is then visible in this file and
        # testable, rather than depending on a helper whose behaviour varies by model
        # and library version. Double-prefixing is the failure that would otherwise
        # be invisible.
        vectors = [list(map(float, v)) for v in self._model.embed(list(texts))]
        out = [l2_normalize(v) for v in vectors]
        for v in out:
            if len(v) != self.dim:
                raise ValueError(
                    f"{self.model_name} produced {len(v)}-dim vectors, config says {self.dim}. "
                    "The schema's vector(N) must match; change both together."
                )
        return out

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]


class HashingEmbedder:
    """Deterministic stand-in for tests and CI.

    Not a real embedding model and makes no semantic claim — it exists so the store,
    schema, and retrieval plumbing can be exercised without a 90 MB download in CI.
    Same vector for the same text, different vectors for different text, correct
    dimensionality, unit length. That is enough to test everything except relevance,
    which is what the eval harness is for.
    """

    def __init__(self, dim: int = DEFAULT_DIM, query_prefix: str = DEFAULT_QUERY_PREFIX) -> None:
        self.dim = dim
        self.query_prefix = query_prefix

    def _vec(self, text: str) -> list[float]:
        import hashlib

        # Expand a digest to `dim` floats by re-hashing with a counter.
        raw = bytearray()
        counter = 0
        while len(raw) < self.dim * 2:
            raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
            counter += 1
        values = [
            int.from_bytes(raw[i : i + 2], "big") / 65535.0 - 0.5 for i in range(0, self.dim * 2, 2)
        ]
        return l2_normalize(values)

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(self.query_prefix + text)


def build_embedder(
    model_name: str = DEFAULT_MODEL,
    *,
    dim: int = DEFAULT_DIM,
    query_prefix: str = DEFAULT_QUERY_PREFIX,
    cache_dir: str | None = None,
    batch_size: int = 64,
) -> Embedder:
    """Return the real embedder, or the test stand-in when `model_name` is 'hashing'."""
    if model_name == "hashing":
        return HashingEmbedder(dim=dim, query_prefix=query_prefix)
    return FastEmbedEmbedder(
        model_name,
        dim=dim,
        query_prefix=query_prefix,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
