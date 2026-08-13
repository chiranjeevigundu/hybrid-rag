"""The MCP and HTTP faces over the same core.

Both are thin, so these tests concentrate on the parts that are easy to get subtly
wrong and impossible to notice: whether the read-only annotations actually reach a
client, whether a failure arrives as data the model can act on or as a protocol error
that ends the turn, and whether the write surface that was deliberately left out is
genuinely absent.
"""

from __future__ import annotations

import asyncio

import pytest

from ragkit.chunking import BlockKind, Chunk

pytestmark = pytest.mark.postgres


@pytest.fixture
def indexed(store, embedder, monkeypatch, database_url):
    """Point both faces at the isolated test database, with a small corpus loaded.

    Without the monkeypatch these read DATABASE_URL and would run against the working
    index instead of the test one.
    """
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RAG_MODEL", "hashing")
    monkeypatch.setenv("RAG_RERANK_MODEL", "none")

    chunks = [
        Chunk(
            text="Refund policy POL-4471 supersedes POL-3980 for every region we ship to.",
            ordinal=0,
            heading_path=("Refunds", "Identifiers"),
            kind=BlockKind.PROSE,
        ),
        Chunk(
            text="x" * 900,  # long enough to force snippet truncation
            ordinal=1,
            heading_path=("Refunds", "Long"),
            kind=BlockKind.PROSE,
        ),
    ]
    store.upsert_document(
        source="policy.md",
        title="Policy",
        sha="s",
        chunks=chunks,
        embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
    )
    return store


# ------------------------------------------------------------------- MCP


def call(tool: str, args: dict):
    from ragkit.mcp_server import server

    return asyncio.run(server.call_tool(tool, args)).structured_content


def test_every_tool_is_annotated_read_only():
    # MCP 2.x field names are snake_case, and the camelCase spelling is silently
    # ignored rather than rejected — the annotation just vanishes and a client never
    # learns these tools are safe to call. Nothing else catches that.
    from ragkit.mcp_server import server

    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == {"search_corpus", "get_chunk", "corpus_stats"}
    for t in tools:
        assert t.annotations is not None, f"{t.name} has no annotations"
        assert t.annotations.read_only_hint is True, f"{t.name} is not marked read-only"
        assert t.description and len(t.description) > 60, f"{t.name} needs a usable description"


def test_no_tool_can_write():
    # The write surface is deliberately absent: indexing needs filesystem access and
    # would put a caller-supplied path in front of the disk.
    from ragkit.mcp_server import server

    names = {t.name for t in asyncio.run(server.list_tools())}
    assert not {n for n in names if any(w in n for w in ("index", "delete", "write", "reset"))}


def test_search_returns_ranked_results_with_provenance(indexed):
    out = call("search_corpus", {"query": "POL-4471", "k": 3})
    assert out["count"] >= 1
    top = out["results"][0]
    assert "POL-4471" in top["snippet"]
    assert top["rank"] == 1 and isinstance(top["chunk_id"], int)
    assert "lexical#" in top["found_by"], "provenance lost on the way through MCP"


def test_long_passages_are_truncated_and_flagged(indexed):
    out = call("search_corpus", {"query": "xxxxx", "k": 5})
    long_hit = next((h for h in out["results"] if h["truncated"]), None)
    assert long_hit is not None, "a 900-char passage should have been truncated"
    assert len(long_hit["snippet"]) < 500
    # Truncation is only acceptable because the full text stays reachable.
    full = call("get_chunk", {"chunk_id": long_hit["chunk_id"]})
    assert len(full["text"]) > len(long_hit["snippet"])


def test_an_unknown_mode_is_returned_as_data_not_raised(indexed):
    # A raised exception ends the model's turn; an error field lets it retry with a
    # valid argument. The message names the valid options for that reason.
    out = call("search_corpus", {"query": "anything", "mode": "bogus"})
    assert "error" in out and "hybrid" in out["error"]


def test_an_unknown_chunk_id_is_returned_as_data(indexed):
    out = call("get_chunk", {"chunk_id": -1})
    assert "error" in out and "search_corpus" in out["error"]


def test_k_is_clamped_rather_than_rejected(indexed):
    # A model asking for 100 results made a breadth judgement, not an error. Failing
    # the call would waste a turn to teach it a bound it cannot see.
    assert call("search_corpus", {"query": "policy", "k": 999})["count"] <= 20
    assert call("search_corpus", {"query": "policy", "k": 0})["count"] <= 1


def test_corpus_stats_lists_what_is_indexed(indexed):
    out = call("corpus_stats", {})
    assert out["documents"] == 1 and out["chunks"] == 2
    assert out["sources"] == ["policy.md"]


def test_corpus_stats_says_so_when_empty(store, monkeypatch, database_url):
    # "0 documents" alone reads like a failure. The note tells the model the corpus is
    # empty rather than broken, which is a different thing to tell a user.
    monkeypatch.setenv("DATABASE_URL", database_url)
    out = call("corpus_stats", {})
    assert out["documents"] == 0 and "empty" in out["note"].lower()


# ------------------------------------------------------------------ HTTP


@pytest.fixture
def client(indexed):
    from fastapi.testclient import TestClient

    from ragkit.service import app

    with TestClient(app) as c:
        yield c


def test_health_does_a_real_database_round_trip(client):
    # A health check that only proves the process is alive reports healthy while every
    # query fails, which suppresses the alert rather than raising it.
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["documents"] == 1


def test_search_returns_full_text_and_timing(client):
    body = client.post("/search", json={"query": "POL-4471", "k": 2}).json()
    assert body["count"] >= 1
    assert "POL-4471" in body["results"][0]["text"], "HTTP returns full text, not a snippet"
    assert body["took_ms"] >= 0


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "x", "k": 0},
        {"query": "x", "k": 99},
        {"query": "x", "mode": "nonsense"},
        {},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_missing_chunk_is_404_not_500(client):
    assert client.get("/chunks/999999").status_code == 404


def test_there_is_no_write_endpoint(client):
    # Indexing is a batch job. Exposing it here would mean a long-running request
    # wrapped around a caller-supplied filesystem path.
    for path in ("/index", "/documents", "/chunks"):
        assert client.post(path, json={}).status_code in (404, 405)


def test_sources_lists_indexed_documents(client):
    body = client.get("/sources").json()
    assert body["count"] == 1 and body["sources"][0]["source"] == "policy.md"
