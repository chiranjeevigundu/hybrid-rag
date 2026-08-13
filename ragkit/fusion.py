"""Reciprocal Rank Fusion.

Combining a dense result list with a lexical one is harder than it looks, because
the two produce numbers that mean different things. Cosine similarity lives in
[-1, 1] and clusters tightly around 0.5-0.8 for a decent corpus; `ts_rank_cd` is
unbounded, corpus-dependent, and routinely differs by an order of magnitude between
queries. Adding or averaging them requires normalising two distributions that have no
stable relationship, and every normalisation choice silently privileges one arm.

RRF sidesteps it by discarding the scores and fusing **ranks**:

    score(d) = Σ  weight_r / (K + rank_r(d))

A document's contribution depends only on where it placed in each list, so the two
arms become comparable by construction. The cost is real and worth stating: rank
fusion throws away confidence. A document that dense retrieval ranked first with 0.95
and one it ranked first with 0.31 contribute identically. That is a fair trade here
precisely *because* the score distributions are untrustworthy — and it is why a
reranker runs afterwards, to put calibrated scores back on the shortlist.

K defaults to 60, from Cormack et al. (2009), which introduced the method. Its role
is to damp the top of each list: with K=60 the difference between rank 1 and rank 2
is small, so a single arm cannot dominate the fusion on the strength of one confident
guess. Lowering K sharpens the top; raising it flattens toward a democratic vote.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_K = 60


@dataclass(frozen=True)
class Fused:
    """One fused result: the key, its combined score, and where it came from."""

    key: int
    score: float
    ranks: dict[str, int]
    """Per-arm 1-based rank, keyed by arm name. Absent arms did not return it.

    Kept rather than discarded because it is the debugging surface: "found by lexical
    at rank 2, dense missed it entirely" is the observation that tells you which arm
    is carrying a query, and the eval harness reports exactly this.
    """


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Sequence[int]],
    *,
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
) -> list[Fused]:
    """Fuse named ranked lists of keys into one ordered list.

    `ranked_lists` maps an arm name to its result keys in rank order. Weights default
    to 1.0 per arm; raising one biases fusion toward that arm without changing the
    rank-based mechanism.

    Ties are broken by best single rank, then by key, so the output is deterministic.
    Two runs over the same inputs must not reorder — an eval harness comparing runs
    cannot distinguish real drift from arbitrary tie-breaking otherwise.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    weights = weights or {}
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for arm, keys in ranked_lists.items():
        weight = weights.get(arm, 1.0)
        for zero_based, key in enumerate(keys):
            rank = zero_based + 1
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
            ranks.setdefault(key, {})[arm] = rank

    return sorted(
        (Fused(key=key, score=score, ranks=ranks[key]) for key, score in scores.items()),
        key=lambda f: (-f.score, min(f.ranks.values()), f.key),
    )
