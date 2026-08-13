"""Retrieval.

Dense-only at this stage. The lexical arm (BM25 over Postgres `tsvector`), Reciprocal
Rank Fusion, and cross-encoder reranking land in the next step; `Retriever.search`
is the seam they slot into, so callers do not change when they arrive.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import Embedder
from .store import Hit, Store


@dataclass
class Retriever:
    store: Store
    embedder: Embedder

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """Top-k passages for a natural-language query.

        `embed_query` rather than `embed_passages` is load-bearing — see embedding.py
        for why using the wrong one costs recall without raising anything.
        """
        if not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        return self.store.search_dense(vector, k=k)
