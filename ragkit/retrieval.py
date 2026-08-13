"""Retrieval: dense, lexical, and the hybrid pipeline over both.

The pipeline is retrieve-broadly-then-narrow:

    query ──┬─► dense  (bi-encoder, pgvector HNSW)   ─┐
            └─► lexical (Postgres FTS, cover density) ─┴─► RRF ─► rerank ─► top-k

Each stage exists because the one before it cannot do its job. The two arms have
opposite blind spots — embeddings miss rare identifiers, lexical matching misses
paraphrase — so running both is what makes the union reliable. RRF combines them
without needing their scores to be comparable. The reranker then puts calibrated,
query-aware scores back on the shortlist, which fusion deliberately threw away.

`candidates` is the width of the funnel: each arm returns that many, fusion sees up to
twice that, and the reranker scores the top `rerank_candidates` of it. Wider costs
reranker forward passes and buys recall. 50 is a reasonable default for a corpus of
this size; the eval harness is how you find the right number for yours rather than
inheriting mine.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum

from .embedding import Embedder
from .fusion import DEFAULT_K, reciprocal_rank_fusion
from .rerank import Reranker
from .store import Hit, Store


class SearchMode(str, Enum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


@dataclass
class Retriever:
    store: Store
    embedder: Embedder
    reranker: Reranker | None = None
    rrf_k: int = DEFAULT_K
    weights: dict[str, float] = field(default_factory=dict)
    """Per-arm RRF weights, e.g. {"dense": 1.0, "lexical": 1.5}. Defaults to equal.

    Left equal by default on purpose. Tilting the weights is the most tempting knob
    here and the easiest one to turn on a hunch — it should be moved in response to
    eval numbers, not intuition about which arm "feels" better.
    """

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        mode: SearchMode | str = SearchMode.HYBRID,
        candidates: int = 50,
        rerank_candidates: int = 25,
        rerank: bool | None = None,
    ) -> list[Hit]:
        """Retrieve the top-k passages.

        `rerank` defaults to True when a reranker is configured. Pass False to
        measure the pipeline without it — which is exactly what the eval harness does
        to establish whether reranking is earning its latency.
        """
        if not query.strip():
            return []
        mode = SearchMode(mode)

        if mode is SearchMode.DENSE:
            hits = self._dense(query, candidates)
        elif mode is SearchMode.LEXICAL:
            hits = self._lexical(query, candidates)
        else:
            hits = self._hybrid(query, candidates)

        should_rerank = (self.reranker is not None) if rerank is None else rerank
        if should_rerank and self.reranker is not None:
            hits = self._rerank(query, hits, rerank_candidates)

        return hits[:k]

    # ------------------------------------------------------------------ arms

    def _dense(self, query: str, k: int) -> list[Hit]:
        vector = self.embedder.embed_query(query)
        return [
            dataclasses.replace(hit, dense_rank=i + 1)
            for i, hit in enumerate(self.store.search_dense(vector, k=k))
        ]

    def _lexical(self, query: str, k: int) -> list[Hit]:
        return [
            dataclasses.replace(hit, lexical_rank=i + 1)
            for i, hit in enumerate(self.store.search_lexical(query, k=k))
        ]

    # ---------------------------------------------------------------- hybrid

    def _hybrid(self, query: str, candidates: int) -> list[Hit]:
        dense = self._dense(query, candidates)
        lexical = self._lexical(query, candidates)

        # One arm returning nothing is normal, not an error: a pure-paraphrase query
        # can miss lexically, and a query of rare tokens can miss densely. Fusion over
        # a single list degrades to that list's own order, which is the right
        # behaviour — but the surviving arm's ranks still travel on the Hit, so the
        # asymmetry stays visible downstream instead of being silently smoothed over.
        by_id: dict[int, Hit] = {}
        for hit in lexical + dense:  # dense last so its rank field wins on merge
            existing = by_id.get(hit.chunk_id)
            if existing is None:
                by_id[hit.chunk_id] = hit
            else:
                by_id[hit.chunk_id] = dataclasses.replace(
                    existing,
                    dense_rank=hit.dense_rank
                    if hit.dense_rank is not None
                    else existing.dense_rank,
                    lexical_rank=(
                        hit.lexical_rank if hit.lexical_rank is not None else existing.lexical_rank
                    ),
                )

        fused = reciprocal_rank_fusion(
            {
                "dense": [h.chunk_id for h in dense],
                "lexical": [h.chunk_id for h in lexical],
            },
            k=self.rrf_k,
            weights=self.weights,
        )
        # The RRF score replaces the arm-specific one. They are not comparable and
        # keeping a cosine value on a fused result would invite exactly that mistake.
        return [dataclasses.replace(by_id[f.key], score=f.score) for f in fused]

    # --------------------------------------------------------------- rerank

    def _rerank(self, query: str, hits: list[Hit], limit: int) -> list[Hit]:
        """Rescore the head of the list with a cross-encoder; leave the tail in place.

        Only the top `limit` are rescored, since each is a forward pass. The remainder
        keeps its fused order and is appended below — a candidate the reranker never
        saw must not be able to outrank one it scored, whatever the two numbers look
        like, because they come from different scales entirely.
        """
        if self.reranker is None or not hits:
            return hits
        head, tail = hits[:limit], hits[limit:]
        # Rerank on the same text the user will read, not embed_text: the heading
        # prefix is an indexing aid, and feeding it to a cross-encoder trained on
        # natural query/passage pairs is off-distribution.
        scores = self.reranker.score(query, [h.text for h in head])
        rescored = [
            dataclasses.replace(hit, rerank_score=score)
            for hit, score in zip(head, scores, strict=True)
        ]
        rescored.sort(key=lambda h: (-(h.rerank_score or 0.0), h.chunk_id))
        return rescored + tail
