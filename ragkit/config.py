"""Configuration.

A frozen dataclass with an explicit `from_env()`, rather than reading `os.environ`
at each use site. Two reasons: the settings a caller must supply are visible in one
place, and tests construct a config object instead of mutating process state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_DIM = 768

#: bge-*-en-v1.5 is trained asymmetrically: queries carry a short instruction, passages
#: do not. Applying the same treatment to both is a quiet accuracy loss — everything
#: still runs, recall is just worse than it should be. See embedding.py.
DEFAULT_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

#: ~80 MB, against ~1 GB for BAAI/bge-reranker-base. Chosen to keep clone-and-run
#: honest; whether the larger model earns its size is an eval question, not a guess.
DEFAULT_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class Config:
    database_url: str = "postgresql://rag:rag@localhost:5433/rag"
    model_name: str = DEFAULT_MODEL
    embedding_dim: int = DEFAULT_DIM
    query_prefix: str = DEFAULT_QUERY_PREFIX
    target_chars: int = 1200
    overlap_chars: int = 150
    max_chars: int = 2000
    batch_size: int = 64
    cache_dir: str | None = None

    # Retrieval. "none" disables reranking; "identity" selects the test stand-in.
    rerank_model: str = DEFAULT_RERANK_MODEL
    rrf_k: int = 60
    candidates: int = 50
    rerank_candidates: int = 25

    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            model_name=os.getenv("RAG_MODEL", cls.model_name),
            embedding_dim=int(os.getenv("RAG_EMBEDDING_DIM", str(cls.embedding_dim))),
            query_prefix=os.getenv("RAG_QUERY_PREFIX", cls.query_prefix),
            target_chars=int(os.getenv("RAG_TARGET_CHARS", str(cls.target_chars))),
            overlap_chars=int(os.getenv("RAG_OVERLAP_CHARS", str(cls.overlap_chars))),
            max_chars=int(os.getenv("RAG_MAX_CHARS", str(cls.max_chars))),
            batch_size=int(os.getenv("RAG_BATCH_SIZE", str(cls.batch_size))),
            cache_dir=os.getenv("RAG_CACHE_DIR") or None,
            rerank_model=os.getenv("RAG_RERANK_MODEL", cls.rerank_model),
            rrf_k=int(os.getenv("RAG_RRF_K", str(cls.rrf_k))),
            candidates=int(os.getenv("RAG_CANDIDATES", str(cls.candidates))),
            rerank_candidates=int(os.getenv("RAG_RERANK_CANDIDATES", str(cls.rerank_candidates))),
        )
