"""Store tests. Require a live Postgres with pgvector."""

from __future__ import annotations

import pytest

from ragkit.chunking import BlockKind, Chunk
from ragkit.store import Store, content_sha, to_pgvector

pytestmark = pytest.mark.postgres


def _chunks(n: int = 3) -> list[Chunk]:
    return [
        Chunk(text=f"passage {i}", ordinal=i, heading_path=("Doc", f"S{i}"), kind=BlockKind.PROSE)
        for i in range(n)
    ]


def test_to_pgvector_formats_for_postgres():
    assert to_pgvector([1.0, -0.5]) == "[1,-0.5]"


def test_schema_is_idempotent(store: Store):
    # ensure_schema runs on every CLI invocation, so a second run must be a no-op
    # rather than an error. Derived from the directory rather than hardcoded, so
    # adding a migration does not break this test for the wrong reason.
    from ragkit.store import MIGRATIONS_DIR

    expected = [p.name for p in sorted(MIGRATIONS_DIR.glob("*.sql"))]
    assert expected, "no migrations found — the glob or the path is wrong"
    assert store.ensure_schema() == expected
    assert store.ensure_schema() == expected, "re-applying migrations was not a no-op"


def test_upsert_then_search_round_trip(store: Store, embedder):
    chunks = _chunks()
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    store.upsert_document(source="a.md", title="A", sha="sha1", chunks=chunks, embeddings=vectors)

    # Query with the target's own `embed_text`, not a paraphrase of it. HashingEmbedder
    # has no notion of similarity — only identity — so this asserts what it can
    # actually prove: vectors survive the round trip and ORDER BY ranks by distance.
    # Whether a *paraphrase* retrieves the right passage is a relevance question, and
    # relevance belongs to the eval harness with the real model behind it.
    hits = store.search_dense(embedder.embed_passages([chunks[1].embed_text])[0], k=3)
    assert len(hits) == 3
    assert hits[0].text == "passage 1", "identical vector was not ranked first"
    assert hits[0].source == "a.md"
    assert hits[0].heading_path == ("Doc", "S1")
    assert hits[0].score == pytest.approx(1.0, abs=1e-6), "self-similarity should be ~1.0"
    # Ordered scores; unordered means the ORDER BY is not driving the result.
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_reindexing_replaces_chunks_rather_than_accumulating(store: Store, embedder):
    first = _chunks(5)
    store.upsert_document(
        source="a.md",
        title="A",
        sha="s1",
        chunks=first,
        embeddings=embedder.embed_passages([c.embed_text for c in first]),
    )
    second = _chunks(2)
    store.upsert_document(
        source="a.md",
        title="A",
        sha="s2",
        chunks=second,
        embeddings=embedder.embed_passages([c.embed_text for c in second]),
    )
    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == 2, "old chunks survived a re-index and are now orphaned"


def test_stored_sha_drives_the_incremental_skip(store: Store, embedder):
    chunks = _chunks(1)
    store.upsert_document(
        source="a.md",
        title="A",
        sha=content_sha("body"),
        chunks=chunks,
        embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
    )
    assert store.stored_sha("a.md") == content_sha("body")
    assert store.stored_sha("missing.md") is None


def test_mismatched_chunk_and_embedding_counts_are_rejected(store: Store):
    # Would otherwise zip() short and silently drop chunks.
    with pytest.raises(ValueError, match="3 chunks but 1 embeddings"):
        store.upsert_document(
            source="a.md", title=None, sha="s", chunks=_chunks(3), embeddings=[[0.0]]
        )


def test_deleting_a_document_cascades_to_its_chunks(store: Store, embedder):
    chunks = _chunks(4)
    store.upsert_document(
        source="a.md",
        title="A",
        sha="s",
        chunks=chunks,
        embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
    )
    assert store.delete_document("a.md") is True
    assert store.stats() == {
        "documents": 0,
        "chunks": 0,
        "oversized_chunks": 0,
        "unembedded_chunks": 0,
    }
    assert store.delete_document("a.md") is False


def test_prune_removes_only_documents_whose_source_is_gone(store: Store, embedder):
    for name in ("a.md", "b.md", "c.md"):
        chunks = _chunks(1)
        store.upsert_document(
            source=name,
            title=name,
            sha="s",
            chunks=chunks,
            embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
        )
    removed = store.prune_missing(["a.md", "c.md"])
    assert removed == ["b.md"]
    assert store.stats()["documents"] == 2


def test_get_chunk_returns_full_context(store: Store, embedder):
    chunks = _chunks(2)
    store.upsert_document(
        source="a.md",
        title="A",
        sha="s",
        chunks=chunks,
        embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
    )
    hit = store.search_dense(embedder.embed_passages(["passage 0"])[0], k=1)[0]
    fetched = store.get_chunk(hit.chunk_id)
    assert fetched is not None and fetched.text == hit.text
    assert store.get_chunk(-1) is None


def test_citation_renders_source_and_heading_path(store: Store, embedder):
    chunks = _chunks(1)
    store.upsert_document(
        source="policies/refunds.md",
        title="R",
        sha="s",
        chunks=chunks,
        embeddings=embedder.embed_passages([c.embed_text for c in chunks]),
    )
    hit = store.search_dense(embedder.embed_passages(["passage 0"])[0], k=1)[0]
    assert hit.citation == "policies/refunds.md § Doc > S0"


def test_oversized_flag_survives_the_round_trip(store: Store, embedder):
    chunk = Chunk(
        text="| a | b |", ordinal=0, heading_path=(), kind=BlockKind.TABLE, oversized=True
    )
    store.upsert_document(
        source="t.md",
        title="T",
        sha="s",
        chunks=[chunk],
        embeddings=embedder.embed_passages([chunk.embed_text]),
    )
    assert store.stats()["oversized_chunks"] == 1
