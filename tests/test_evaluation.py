"""Eval harness tests.

The harness is the thing that decides whether a retrieval change shipped or got
reverted, so its arithmetic needs to be right for reasons that have nothing to do with
retrieval. A metric that quietly reports 1.0 when nothing was found does not fail
loudly — it just stops catching regressions, and nobody notices for a month.

All pure logic: no database, no model.
"""

from __future__ import annotations

import json
import math

import pytest

from ragkit.chunking import BlockKind
from ragkit.evaluation import (
    CaseResult,
    EvalResult,
    GoldenCase,
    Relevant,
    check_floor,
    load_golden_set,
    save_baseline,
    verify_golden_set,
)
from ragkit.store import Hit


def hit(text: str, source: str = "a.md", chunk_id: int = 1) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        text=text,
        heading_path=(),
        kind=BlockKind.PROSE.value,
        source=source,
        title=None,
        score=1.0,
    )


def case(*relevant: Relevant, kind: str = "test") -> GoldenCase:
    return GoldenCase(id="c", query="q", kind=kind, relevant=tuple(relevant))


# ------------------------------------------------------------------ matching


def test_relevance_matching_is_whitespace_insensitive():
    # Ground truth must survive the chunker changing where a line wraps, otherwise
    # tuning the chunker breaks the very set that measures whether tuning helped.
    rel = Relevant("a.md", "three to five business days")
    assert rel.matches(hit("refunds appear in three\nto  five business days"))


def test_relevance_requires_the_right_source_file():
    rel = Relevant("refunds.md", "30 days")
    assert not rel.matches(hit("30 days", source="shipping.md"))


def test_ranks_are_reported_per_relevant_passage():
    c = case(Relevant("a.md", "alpha"), Relevant("a.md", "beta"))
    hits = [hit("gamma", chunk_id=1), hit("beta", chunk_id=2), hit("alpha", chunk_id=3)]
    assert c.ranks_in(hits) == [2, 3]


def test_a_missing_passage_is_omitted_not_zero_filled():
    c = case(Relevant("a.md", "alpha"), Relevant("a.md", "absent"))
    assert c.ranks_in([hit("alpha")]) == [1]


# ------------------------------------------------------------------- metrics


def test_reciprocal_rank_matches_the_definition():
    assert CaseResult(case(), [1], 1).reciprocal_rank() == 1.0
    assert CaseResult(case(), [4], 1).reciprocal_rank() == pytest.approx(0.25)


def test_a_miss_scores_zero_rather_than_being_skipped():
    # Averaging only over the queries that succeeded is how a retriever appears
    # healthy while failing a third of its traffic.
    r = CaseResult(case(), [], 1)
    assert r.missed and r.reciprocal_rank() == 0.0
    assert r.recall_at(10) == 0.0 and r.ndcg_at(10) == 0.0


def test_recall_counts_only_hits_inside_k():
    r = CaseResult(case(), [1, 8], 2)
    assert r.recall_at(1) == pytest.approx(0.5)
    assert r.recall_at(10) == pytest.approx(1.0)


def test_ndcg_applies_position_decay():
    # One relevant passage: nDCG reduces to 1/log2(rank+1).
    assert CaseResult(case(), [1], 1).ndcg_at(10) == pytest.approx(1.0)
    assert CaseResult(case(), [3], 1).ndcg_at(10) == pytest.approx(1 / math.log2(4))
    assert CaseResult(case(), [2], 1).ndcg_at(10) > CaseResult(case(), [5], 1).ndcg_at(10)


def test_ndcg_is_normalised_so_a_perfect_run_scores_one():
    # Two relevant passages at ranks 1 and 2 is the ideal ordering, so exactly 1.0 —
    # not some smaller number that a missing normalisation would produce.
    assert CaseResult(case(), [1, 2], 2).ndcg_at(10) == pytest.approx(1.0)


def test_aggregates_average_over_every_case_including_misses():
    result = EvalResult(label="t", k=10)
    result.results = [CaseResult(case(), [1], 1), CaseResult(case(), [], 1)]
    assert result.mrr() == pytest.approx(0.5)
    assert len(result.misses()) == 1


def test_subset_filters_by_kind():
    result = EvalResult(label="t", k=10)
    result.results = [
        CaseResult(case(kind="exact-id"), [1], 1),
        CaseResult(case(kind="paraphrase"), [4], 1),
    ]
    assert result.subset("exact-id").mrr() == pytest.approx(1.0)
    assert result.subset("paraphrase").mrr() == pytest.approx(0.25)
    assert result.kinds == ["exact-id", "paraphrase"]


def test_empty_result_does_not_divide_by_zero():
    empty = EvalResult(label="t", k=10)
    assert empty.mrr() == 0.0 and empty.recall_at(3) == 0.0 and empty.ndcg_at(10) == 0.0


# ------------------------------------------------------------- golden set io


def test_the_committed_golden_set_loads_and_matches_the_corpus():
    from pathlib import Path

    cases = load_golden_set()
    assert len(cases) >= 20, "the golden set shrank unexpectedly"
    corpus = Path(__file__).resolve().parent.parent / "corpus"
    assert verify_golden_set(cases, corpus) == []


def test_every_committed_case_explains_why_it_exists():
    # A case with no rationale becomes impossible to judge later: when it starts
    # failing, nobody can tell whether the retriever regressed or the case was wrong.
    for c in load_golden_set():
        assert c.why.strip(), f"case {c.id} has no 'why'"


def test_duplicate_case_ids_are_rejected(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(
        json.dumps(
            {
                "cases": [
                    {"id": "dup", "query": "a", "relevant": [{"source": "s", "contains": "x"}]},
                    {"id": "dup", "query": "b", "relevant": [{"source": "s", "contains": "y"}]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate case id"):
        load_golden_set(p)


def test_a_case_with_no_relevant_passages_is_rejected(tmp_path):
    # It would silently score 0 forever and drag the aggregate down for no reason.
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"cases": [{"id": "x", "query": "q", "relevant": []}]}))
    with pytest.raises(ValueError, match="no relevant passages"):
        load_golden_set(p)


def test_verification_catches_ground_truth_that_does_not_exist(tmp_path):
    (tmp_path / "real.md").write_text("the actual text")
    cases = [
        GoldenCase("bad-file", "q", "k", (Relevant("missing.md", "x"),)),
        GoldenCase("bad-text", "q", "k", (Relevant("real.md", "not present"),)),
        GoldenCase("good", "q", "k", (Relevant("real.md", "actual text"),)),
    ]
    problems = verify_golden_set(cases, tmp_path)
    assert len(problems) == 2
    assert any("no such file" in p for p in problems)
    assert any("not found in" in p for p in problems)


# ------------------------------------------------------------ regression floor


def _result(mrr_rank: int) -> EvalResult:
    r = EvalResult(label="t", k=10)
    r.results = [CaseResult(case(), [mrr_rank], 1)]
    return r


def test_floor_passes_when_metrics_hold(tmp_path):
    path = tmp_path / "b.json"
    save_baseline(_result(1), path)
    ok, _ = check_floor(_result(1), path)
    assert ok


def test_floor_fails_on_a_real_regression(tmp_path):
    path = tmp_path / "b.json"
    save_baseline(_result(1), path)
    ok, messages = check_floor(_result(5), path)
    assert not ok
    assert any("FAIL" in m for m in messages)


def test_floor_tolerates_noise_below_the_threshold(tmp_path):
    # A floor that trips on a 0.001 model-version drift gets disabled within a month,
    # and a disabled check protects nothing.
    path = tmp_path / "b.json"
    base = EvalResult(label="t", k=10)
    base.results = [CaseResult(case(), [1], 1)] * 100
    save_baseline(base, path)
    slightly_worse = EvalResult(label="t", k=10)
    slightly_worse.results = [CaseResult(case(), [1], 1)] * 99 + [CaseResult(case(), [2], 1)]
    ok, _ = check_floor(slightly_worse, path, tolerance=0.02)
    assert ok, "a 0.005 drop should not fail the build"


def test_floor_refuses_to_compare_across_different_golden_set_sizes(tmp_path):
    # Metrics from 22 cases and 40 cases are not comparable, and quietly comparing
    # them turns "we added hard cases" into a phantom regression.
    path = tmp_path / "b.json"
    save_baseline(_result(1), path)
    bigger = EvalResult(label="t", k=10)
    bigger.results = [CaseResult(case(), [1], 1), CaseResult(case(), [1], 1)]
    ok, messages = check_floor(bigger, path)
    assert not ok
    assert any("golden set changed size" in m for m in messages)


def test_a_missing_baseline_is_not_a_failure(tmp_path):
    ok, messages = check_floor(_result(1), tmp_path / "absent.json")
    assert ok and any("no baseline" in m for m in messages)


def test_the_committed_baseline_is_consistent_with_the_committed_golden_set():
    from pathlib import Path

    from ragkit.evaluation import DEFAULT_BASELINE

    baseline = json.loads(Path(DEFAULT_BASELINE).read_text())
    assert baseline["n"] == len(load_golden_set()), (
        "baseline n does not match the golden set — re-run `ragkit eval --save-baseline`"
    )
