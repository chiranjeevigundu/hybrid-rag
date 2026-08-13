"""Cross-encoder reranking.

The retrieval arms are bi-encoders and lexical matching: both score a query against a
passage **without the two ever meeting**. A passage's vector is computed once, at
index time, with no knowledge of what will be asked. That is what makes retrieval fast
over a large corpus, and it is also its ceiling — the model cannot attend to the query
while reading the passage, so it cannot tell that "the window begins on the delivery
date" answers "when does the clock start" while "the window is 30 days" does not.

A cross-encoder concatenates query and passage and runs them through the model
together. Attention crosses between them, which is strictly more informative and
strictly more expensive: cost is O(candidates) forward passes per query rather than
one. That economics dictates the shape used here — retrieve broadly and cheaply, then
rerank a shortlist. Reranking the whole corpus would be correct and unusable.

Default is `Xenova/ms-marco-MiniLM-L-6-v2` (~80 MB) rather than
`BAAI/bge-reranker-base` (~1 GB). MiniLM is the long-standing MS MARCO baseline and
keeps a clone-and-run repository honest. Whether the 13x larger model earns its size
on a given corpus is measurable, not obvious, and the eval harness is where that
question gets settled rather than guessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .config import RERANK_MODEL as DEFAULT_RERANK_MODEL


@runtime_checkable
class Reranker(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


class CrossEncoderReranker:
    """`fastembed` cross-encoder. Downloads the model on first use, then caches it."""

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, *, cache_dir: str | None = None):
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover - import-time guard
            raise RuntimeError(
                "fastembed is not installed. Run: pip install -e '.[embed]'"
            ) from exc
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance logits, one per passage, aligned with the input order.

        These are raw logits, not probabilities — unbounded and typically negative.
        They are meaningful only *relative to each other for one query*, so they are
        used to sort a shortlist and must never be compared across queries or
        thresholded against a fixed constant.
        """
        if not passages:
            return []
        return [float(s) for s in self._model.rerank(query, list(passages))]


class IdentityReranker:
    """No-op reranker for tests: preserves the order it was given.

    Returns descending scores so that sorting by score is a stable identity. Lets the
    rerank code path be exercised in CI without a model download.
    """

    model_name = "identity"

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        return [float(len(passages) - i) for i in range(len(passages))]


def build_reranker(
    model_name: str | None = DEFAULT_RERANK_MODEL, *, cache_dir: str | None = None
) -> Reranker | None:
    """Return a reranker, or None when reranking is disabled.

    `None` and the string "none" both disable it; "identity" selects the test stand-in.
    """
    if model_name in (None, "", "none"):
        return None
    if model_name == "identity":
        return IdentityReranker()
    return CrossEncoderReranker(model_name, cache_dir=cache_dir)
