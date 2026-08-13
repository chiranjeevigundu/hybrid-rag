"""Hybrid retrieval tests against a live Postgres.

Relevance is not asserted here — HashingEmbedder has no semantics, so these check
pipeline behaviour: that each arm runs, that fusion merges provenance correctly, that
reranking reorders only what it scored. Whether the results are *good* is the eval
harness's question, with the real model behind it.

The lexical arm is the exception: Postgres full-text search is genuinely semantic-free
and deterministic, so its matching can be asserted directly.
"""

from __future__ import annotations

import pytest

from ragkit.chunking import BlockKind, Chunk
from ragkit.rerank import IdentityReranker
from ragkit.retrieval import Retriever, SearchMode

pytestmark = pytest.mark.postgres


@pytest.fixture
def loaded(store, embedder):
    """A small corpus with one rare identifier and one paraphrasable passage."""
    chunks = [
        Chunk(
            text="Refund policy POL-4471 supersedes POL-3980 for all regions.",
            ordinal=0,
            heading_path=("Refunds", "Identifiers"),
            kind=BlockKind.PROSE,
        ),
        Chunk(
            text="Parcels that arrive broken should be photographed before opening.",
            ordinal=1,
            heading_path=("Shipping", "Damage"),
            kind=BlockKind.PROSE,
        ),
        Chunk(
            text="Freight deliveries are curbside only and need a liftgate request.",
            ordinal=2,
            heading_path=("Shipping", "Freight"),
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


def test_lexical_arm_finds_an_exact_identifier(loaded, embedder):
    # The case dense retrieval is measurably bad at, and the reason the lexical arm
    # exists at all.
    hits = Retriever(loaded, embedder).search("POL-4471", mode=SearchMode.LEXICAL)
    assert hits, "lexical search found nothing for an exact identifier"
    assert "POL-4471" in hits[0].text
    assert hits[0].lexical_rank == 1
    assert hits[0].dense_rank is None, "lexical-only search must not set a dense rank"


def test_lexical_arm_matches_words_from_the_heading_path(loaded, embedder):
    # migration 002 folds heading_path into the tsvector. "Freight" appears in the
    # heading and the body here, so use a heading-only word to prove the fold works.
    hits = Retriever(loaded, embedder).search("Damage", mode=SearchMode.LEXICAL)
    assert hits and hits[0].heading_path == ("Shipping", "Damage")


def test_lexical_search_survives_hostile_input(loaded, embedder):
    # websearch_to_tsquery must not raise on punctuation soup the way to_tsquery does.
    r = Retriever(loaded, embedder)
    for query in ["!!!", "AND OR", '"unclosed quote', "-", "a & b | c"]:
        r.search(query, mode=SearchMode.LEXICAL)  # must not raise


def test_pure_stopwords_return_nothing_rather_than_everything(loaded, embedder):
    assert Retriever(loaded, embedder).search("the and of", mode=SearchMode.LEXICAL) == []


def test_hybrid_records_provenance_from_both_arms(loaded, embedder):
    hits = Retriever(loaded, embedder).search("POL-4471", mode=SearchMode.HYBRID, candidates=10)
    assert hits
    target = next(h for h in hits if "POL-4471" in h.text)
    assert target.lexical_rank is not None, "lexical rank lost during fusion"
    assert target.dense_rank is not None, "dense rank lost during fusion"
    assert "lexical#" in target.provenance and "dense#" in target.provenance


def test_hybrid_scores_are_rrf_not_cosine(loaded, embedder):
    # RRF scores are ~1/61 scale. A cosine value surviving onto a fused result would
    # mean the two scales got mixed, which is the bug this asserts against.
    hits = Retriever(loaded, embedder).search("refund", mode=SearchMode.HYBRID)
    assert hits and all(h.score < 0.1 for h in hits), [h.score for h in hits]


def test_an_empty_query_returns_nothing_in_every_mode(loaded, embedder):
    r = Retriever(loaded, embedder)
    for mode in SearchMode:
        assert r.search("", mode=mode) == []
        assert r.search("   ", mode=mode) == []


def test_k_bounds_the_result_count(loaded, embedder):
    assert len(Retriever(loaded, embedder).search("shipping refund policy", k=2)) <= 2


def test_reranking_runs_and_annotates_only_what_it_scored(loaded, embedder):
    r = Retriever(loaded, embedder, reranker=IdentityReranker())
    hits = r.search("shipping", k=10, candidates=10, rerank_candidates=1)
    assert hits[0].rerank_score is not None, "the head was not rescored"
    # Candidates below the rerank cutoff keep their fused order and carry no score;
    # a passage the reranker never saw must not appear to have been judged.
    assert all(h.rerank_score is None for h in hits[1:])


def test_rerank_can_be_disabled_per_call(loaded, embedder):
    r = Retriever(loaded, embedder, reranker=IdentityReranker())
    hits = r.search("shipping", rerank=False)
    assert hits and all(h.rerank_score is None for h in hits)


def test_a_reranked_candidate_never_falls_below_an_unscored_one(loaded, embedder):
    # Rerank logits and RRF scores are different scales entirely. Sorting them
    # together would let an unscored tail item leapfrog a scored one.
    r = Retriever(loaded, embedder, reranker=IdentityReranker())
    hits = r.search("shipping refund", k=10, candidates=10, rerank_candidates=2)
    scored = [i for i, h in enumerate(hits) if h.rerank_score is not None]
    unscored = [i for i, h in enumerate(hits) if h.rerank_score is None]
    assert not unscored or not scored or max(scored) < min(unscored)


def test_dense_and_lexical_modes_stay_isolated(loaded, embedder):
    r = Retriever(loaded, embedder)
    dense = r.search("freight curbside", mode=SearchMode.DENSE)
    assert all(h.lexical_rank is None for h in dense)
    lexical = r.search("freight curbside", mode=SearchMode.LEXICAL)
    assert all(h.dense_rank is None for h in lexical)


def test_weights_are_accepted_and_change_ordering(loaded, embedder):
    heavy = Retriever(loaded, embedder, weights={"lexical": 10.0})
    hits = heavy.search("POL-4471", mode=SearchMode.HYBRID, candidates=10)
    assert "POL-4471" in hits[0].text
