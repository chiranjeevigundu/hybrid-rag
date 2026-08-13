"""Embedding tests.

The one that matters is `test_query_and_passage_encodings_differ`. Encoding both
sides identically is the single easiest way to lose retrieval accuracy with a bge
model, and it raises nothing — the only way it surfaces is as worse results.
"""

from __future__ import annotations

import math

import pytest

from ragkit.embedding import DEFAULT_DIM, Embedder, HashingEmbedder, l2_normalize


def test_l2_normalize_produces_unit_vectors():
    v = l2_normalize([3.0, 4.0])
    assert math.isclose(math.sqrt(sum(x * x for x in v)), 1.0, rel_tol=1e-9)
    assert math.isclose(v[0], 0.6) and math.isclose(v[1], 0.8)


def test_l2_normalize_survives_the_zero_vector():
    # Guards a division by zero that would otherwise only appear on a pathological
    # input, at write time, inside a batch.
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_query_and_passage_encodings_differ_for_identical_text():
    e = HashingEmbedder()
    text = "how long is the refund window"
    assert e.embed_query(text) != e.embed_passages([text])[0], (
        "query and passage encodings are identical — the bge instruction prefix is "
        "not being applied, which silently costs recall"
    )


def test_embeddings_are_deterministic():
    a, b = HashingEmbedder(), HashingEmbedder()
    assert a.embed_passages(["stable"]) == b.embed_passages(["stable"])


def test_different_text_yields_different_vectors():
    e = HashingEmbedder()
    [x, y] = e.embed_passages(["alpha", "beta"])
    assert x != y


def test_dimensionality_is_honoured():
    e = HashingEmbedder(dim=384)
    assert len(e.embed_passages(["x"])[0]) == 384
    assert len(e.embed_query("x")) == 384


def test_default_dim_matches_the_schema():
    # migrations/001_schema.sql declares vector(768). If these drift, every insert
    # fails at runtime with a dimension mismatch.
    assert DEFAULT_DIM == 768


def test_empty_batch_is_allowed():
    assert HashingEmbedder().embed_passages([]) == []


def test_hashing_embedder_satisfies_the_protocol():
    assert isinstance(HashingEmbedder(), Embedder)


@pytest.mark.model
def test_real_model_round_trip():
    """Only runs with `-m model`. Downloads ~90 MB."""
    from ragkit.embedding import FastEmbedEmbedder

    e = FastEmbedEmbedder()
    vectors = e.embed_passages(["refunds are processed within thirty days"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 768
    assert math.isclose(math.sqrt(sum(x * x for x in vectors[0])), 1.0, rel_tol=1e-5)
