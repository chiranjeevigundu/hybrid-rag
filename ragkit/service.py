"""HTTP API — the service-to-service face.

    uvicorn ragkit.service:app --port 8080

Separate from the MCP server on purpose. MCP carries tool schemas and descriptions
written for a model to read; a backend calling this from another container has no use
for any of that and wants a plain JSON contract it can type against. Same core, two
faces, because the consumers genuinely differ.

**Read-only, deliberately.** There is no POST /index. Indexing needs filesystem
access, runs for minutes on a real corpus, and would put a caller-supplied path in
front of the disk — a long-running request wrapped around a path-traversal surface.
It stays a batch job invoked from the CLI, and this service only reads what that job
produced. The narrower surface is the point.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .config import Config
from .embedding import build_embedder
from .rerank import build_reranker
from .retrieval import Retriever, SearchMode
from .store import Store

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the model and connection pool once, at startup.

    Doing it lazily on first request means the first caller pays a multi-second ONNX
    load and probably times out — and under any real concurrency, several callers pay
    it simultaneously.
    """
    config = Config.from_env()
    store = Store(config.database_url)
    _state["config"] = config
    _state["store"] = store
    _state["retriever"] = Retriever(
        store,
        build_embedder(
            config.model_name,
            dim=config.embedding_dim,
            query_prefix=config.query_prefix,
            cache_dir=config.cache_dir,
        ),
        reranker=build_reranker(config.rerank_model, cache_dir=config.cache_dir),
        rrf_k=config.rrf_k,
    )
    yield
    _state.clear()


app = FastAPI(
    title="ragkit",
    version="0.3.0",
    description="Hybrid retrieval over a local corpus. Read-only; indexing is a batch job.",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    mode: Literal["hybrid", "dense", "lexical"] = "hybrid"
    rerank: bool | None = Field(
        None, description="Override the configured default. Null means use the default."
    )


class SearchHit(BaseModel):
    rank: int
    chunk_id: int
    citation: str
    source: str
    section: str
    text: str
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    took_ms: int
    results: list[SearchHit]


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus a real database round trip.

    A health check that only confirms the process is running will report healthy while
    every query fails, which is worse than no health check — it actively suppresses
    the alert.
    """
    try:
        stats = _state["store"].stats()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"database unreachable: {exc}") from exc
    return {"status": "ok", "documents": stats["documents"], "chunks": stats["chunks"]}


@app.get("/stats")
def stats() -> dict[str, int]:
    return _state["store"].stats()


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    config: Config = _state["config"]
    hits = _state["retriever"].search(
        request.query,
        k=request.k,
        mode=SearchMode(request.mode),
        rerank=request.rerank,
        candidates=config.candidates,
        rerank_candidates=config.rerank_candidates,
    )
    return SearchResponse(
        query=request.query,
        mode=request.mode,
        count=len(hits),
        took_ms=int((time.perf_counter() - started) * 1000),
        results=[
            SearchHit(
                rank=i,
                chunk_id=h.chunk_id,
                citation=h.citation,
                source=h.source,
                section=" > ".join(h.heading_path),
                text=h.text,
                score=round(h.score, 6),
                dense_rank=h.dense_rank,
                lexical_rank=h.lexical_rank,
                rerank_score=h.rerank_score,
            )
            for i, h in enumerate(hits, 1)
        ],
    )


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: int) -> dict[str, Any]:
    hit = _state["store"].get_chunk(chunk_id)
    if hit is None:
        raise HTTPException(status_code=404, detail=f"no chunk {chunk_id}")
    return {
        "chunk_id": hit.chunk_id,
        "citation": hit.citation,
        "source": hit.source,
        "section": " > ".join(hit.heading_path),
        "kind": hit.kind,
        "text": hit.text,
    }


@app.get("/sources")
def sources(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    with _state["store"].connect() as conn:
        rows = conn.execute(
            "SELECT source, title, chunk_count, indexed_at FROM documents ORDER BY source LIMIT %s",
            (limit,),
        ).fetchall()
    return {
        "count": len(rows),
        "sources": [
            {"source": r[0], "title": r[1], "chunks": r[2], "indexed_at": r[3].isoformat()}
            for r in rows
        ],
    }
