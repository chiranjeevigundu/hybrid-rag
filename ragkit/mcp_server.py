"""MCP server — exposes retrieval as tools an LLM client can call.

    claude mcp add hybrid-rag -- python -m ragkit.mcp_server

Three tools, not thirty. A large tool surface costs context on every single request
and gives the model more ways to pick the wrong one; the useful shape here is one good
search tool plus the two things a model actually needs around it — a way to see more of
a truncated passage, and a way to find out what is in the corpus before deciding the
corpus is relevant at all.

All three are annotated `read_only_hint=True`, which is accurate: nothing here writes.
(The MCP 2.x field names are snake_case. Passing the camelCase spelling is silently
ignored rather than rejected, so the annotation vanishes and clients never learn the
tools are safe — worth asserting in a test, which `tests/test_faces.py` does.)
Indexing is deliberately not exposed. It needs filesystem access, takes minutes on a
real corpus, and would hand a model a path argument that reaches the disk — none of
which belongs behind a tool call. Indexing is a batch job run from the CLI.

Results are trimmed before they are returned. A retrieval tool that hands back ten
full passages can spend several thousand tokens answering a question the model then
has to re-read, so each result carries a snippet and a `chunk_id`, and `get_chunk`
expands the one that matters.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from .config import Config
from .embedding import build_embedder
from .rerank import build_reranker
from .retrieval import Retriever, SearchMode
from .store import Store

SNIPPET_CHARS = 400

server = MCPServer(
    name="hybrid-rag",
    version="0.3.0",
    instructions=(
        "Hybrid retrieval (dense + lexical) over a local document corpus. "
        "Use search_corpus to find passages that answer a question, then get_chunk to "
        "read one in full if the snippet is truncated. Every result carries a citation "
        "identifying the source file and section, which should be preserved when "
        "quoting the passage. Call corpus_stats first if you are unsure whether this "
        "corpus covers the topic at all."
    ),
)

_state: dict[str, Any] = {}


def _retriever() -> Retriever:
    """Build once and reuse. Loading the ONNX model on every call would dominate latency."""
    if "retriever" not in _state:
        config = Config.from_env()
        _state["config"] = config
        _state["retriever"] = Retriever(
            Store(config.database_url),
            build_embedder(
                config.model_name,
                dim=config.embedding_dim,
                query_prefix=config.query_prefix,
                cache_dir=config.cache_dir,
            ),
            reranker=build_reranker(config.rerank_model, cache_dir=config.cache_dir),
            rrf_k=config.rrf_k,
        )
    return _state["retriever"]


def _snippet(text: str) -> tuple[str, bool]:
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_CHARS:
        return flat, False
    return flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " …", True


@server.tool(
    description=(
        "Search the indexed corpus for passages relevant to a question. Combines "
        "semantic (embedding) and keyword search, so it handles both paraphrased "
        "questions and exact identifiers such as error codes or document numbers. "
        "Returns ranked passages with citations. Snippets are truncated; use "
        "get_chunk with the returned chunk_id to read one in full."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def search_corpus(query: str, k: int = 5, mode: str = "hybrid") -> dict[str, Any]:
    """Args:
    query: A natural-language question, or an exact identifier to look up.
    k: How many passages to return. 1-20; defaults to 5.
    mode: "hybrid" (default, recommended), "dense" (semantic only), or
        "lexical" (keyword only). Prefer hybrid unless isolating one arm.
    """
    try:
        search_mode = SearchMode(mode)
    except ValueError:
        return {
            "error": f"unknown mode {mode!r}; expected one of "
            + ", ".join(m.value for m in SearchMode)
        }
    # Clamp rather than reject: a model asking for 100 results has made a judgement
    # call about breadth, not an error, and failing the call wastes a turn.
    k = max(1, min(int(k), 20))

    try:
        config = Config.from_env()
        hits = _retriever().search(query, k=k, mode=search_mode, candidates=config.candidates)
    except Exception as exc:  # noqa: BLE001 - surface as data, not a protocol error
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Is the database running and the corpus indexed? `make up && make index`.",
        }

    results = []
    for rank, hit in enumerate(hits, 1):
        snippet, truncated = _snippet(hit.text)
        results.append(
            {
                "rank": rank,
                "chunk_id": hit.chunk_id,
                "citation": hit.citation,
                "source": hit.source,
                "section": " > ".join(hit.heading_path),
                "snippet": snippet,
                "truncated": truncated,
                "score": round(hit.score, 4),
                "found_by": hit.provenance,
            }
        )
    return {"query": query, "mode": search_mode.value, "count": len(results), "results": results}


@server.tool(
    description=(
        "Read one passage in full, by the chunk_id returned from search_corpus. Use "
        "this when a search snippet was truncated and the detail you need is cut off."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def get_chunk(chunk_id: int) -> dict[str, Any]:
    """Args:
    chunk_id: The chunk_id field from a search_corpus result.
    """
    try:
        config = Config.from_env()
        hit = Store(config.database_url).get_chunk(int(chunk_id))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if hit is None:
        return {"error": f"no chunk with id {chunk_id}. Run search_corpus for valid ids."}
    return {
        "chunk_id": hit.chunk_id,
        "citation": hit.citation,
        "source": hit.source,
        "section": " > ".join(hit.heading_path),
        "kind": hit.kind,
        "text": hit.text,
    }


@server.tool(
    description=(
        "Describe what is in the corpus: how many documents and passages are indexed, "
        "and the source filenames. Worth calling before concluding the corpus cannot "
        "answer a question, and before telling a user something is not covered."
    ),
    annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False),
)
def corpus_stats() -> dict[str, Any]:
    try:
        config = Config.from_env()
        store = Store(config.database_url)
        stats = store.stats()
        with store.connect() as conn:
            sources = [
                r[0]
                for r in conn.execute("SELECT source FROM documents ORDER BY source").fetchall()
            ]
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "hint": "Is the database running? `make up`.",
        }
    if not sources:
        return {
            "documents": 0,
            "chunks": 0,
            "sources": [],
            "note": "The corpus is empty. Nothing has been indexed yet.",
        }
    return {"documents": stats["documents"], "chunks": stats["chunks"], "sources": sources}


def main() -> None:
    # stdio is the transport Claude Code and Claude Desktop launch servers over. HTTP
    # deployments should use ragkit.service instead, which is a plain REST API rather
    # than MCP-over-HTTP — a service-to-service caller has no use for tool schemas.
    if os.getenv("RAGKIT_MCP_TRANSPORT", "stdio") != "stdio":  # pragma: no cover
        server.run(transport=os.environ["RAGKIT_MCP_TRANSPORT"])
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
