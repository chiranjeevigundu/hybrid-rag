"""Reciprocal Rank Fusion tests. Pure arithmetic — no database, no model."""

from __future__ import annotations

import pytest

from ragkit.fusion import reciprocal_rank_fusion


def test_a_document_found_by_both_arms_outranks_one_found_by_either():
    # The core claim of hybrid retrieval. Doc 2 is second in both lists and must beat
    # doc 1 and doc 3, each of which is first in one list but absent from the other.
    fused = reciprocal_rank_fusion({"dense": [1, 2], "lexical": [3, 2]})
    assert fused[0].key == 2
    assert fused[0].ranks == {"dense": 2, "lexical": 2}


def test_scores_match_the_published_formula():
    # rank 1 in one list, k=60 → 1/61.
    fused = reciprocal_rank_fusion({"dense": [7]}, k=60)
    assert fused[0].score == pytest.approx(1 / 61)


def test_a_single_arm_degrades_to_that_arms_order():
    # A paraphrase query can miss lexically and a rare-token query can miss densely.
    # Neither is an error, and fusion must not reorder the surviving list.
    fused = reciprocal_rank_fusion({"dense": [5, 3, 9], "lexical": []})
    assert [f.key for f in fused] == [5, 3, 9]


def test_empty_input_yields_empty_output():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "lexical": []}) == []


def test_weights_bias_fusion_without_changing_the_mechanism():
    # Equal weights: doc 1 (dense#1) and doc 2 (lexical#1) tie, broken by key.
    even = reciprocal_rank_fusion({"dense": [1], "lexical": [2]})
    assert [f.key for f in even] == [1, 2]
    # Favour lexical and doc 2 wins on score, not on tie-breaking.
    tilted = reciprocal_rank_fusion({"dense": [1], "lexical": [2]}, weights={"lexical": 3.0})
    assert tilted[0].key == 2
    assert tilted[0].score > tilted[1].score


def test_ranks_record_which_arm_contributed():
    fused = reciprocal_rank_fusion({"dense": [1, 2], "lexical": [2]})
    by_key = {f.key: f.ranks for f in fused}
    assert by_key[1] == {"dense": 1}, "doc 1 was dense-only"
    assert by_key[2] == {"dense": 2, "lexical": 1}


def test_output_is_deterministic_under_ties():
    # An eval harness comparing runs cannot tell real drift from arbitrary
    # tie-breaking, so identical input must produce byte-identical ordering.
    lists = {"dense": [4, 1, 3], "lexical": [3, 1, 4]}
    first = [f.key for f in reciprocal_rank_fusion(lists)]
    for _ in range(5):
        assert [f.key for f in reciprocal_rank_fusion(lists)] == first


def test_smaller_k_sharpens_the_top_of_each_list():
    # With k=1, rank 1 (1/2) hugely outweighs rank 2 (1/3). With k=1000 they are
    # nearly equal. This is the knob's whole purpose, so it should be observable.
    sharp = reciprocal_rank_fusion({"a": [1, 2]}, k=1)
    flat = reciprocal_rank_fusion({"a": [1, 2]}, k=1000)
    sharp_gap = sharp[0].score - sharp[1].score
    flat_gap = flat[0].score - flat[1].score
    assert sharp_gap > flat_gap * 100


def test_non_positive_k_is_rejected():
    # k=0 would divide by zero at rank 0; a negative k inverts the ranking silently.
    with pytest.raises(ValueError, match="k must be positive"):
        reciprocal_rank_fusion({"a": [1]}, k=0)
