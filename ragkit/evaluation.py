"""Retrieval evaluation.

Named `evaluation` rather than `eval` so it does not shadow the builtin.

Four principles, carried over from an agent-eval design that was written for an
architecture that no longer exists. The substrate changed; these did not:

**Deterministic assertions before model judging.** Everything measured here is
positional — did the passage containing the answer appear, and where. No LLM judges
anything. Judge-based scoring is expensive, noisy, and unfalsifiable without a written
rubric, so it is the last resort rather than the first tool. Retrieval is fortunate:
almost everything that matters about it is checkable without a model.

**Report rates with n attached, never a bare pass/fail.** A single query moving one
position is noise. A recall@3 that drops from 0.93 to 0.71 across twenty queries is
signal. Every number below carries its sample size, because a metric without one
invites conclusions the sample cannot support — this corpus has 29 chunks, and no
amount of decimal places changes that.

**Ground truth survives re-chunking.** Cases identify a passage by source file plus a
substring it must contain, never by chunk id or ordinal. Chunk ids are assigned by the
database and ordinals move the moment chunking parameters change — which is precisely
the change you most want to measure. A golden set that breaks when you tune the
chunker cannot tell you whether the tuning helped.

**Misses are reported, not averaged away.** A query where nothing relevant was
retrieved contributes 0 to MRR, and the harness also prints which queries those were.
An aggregate that quietly absorbs failures is how a retriever regresses for a month
without anyone noticing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import Retriever, SearchMode
from .store import Hit

DEFAULT_GOLDEN_SET = Path(__file__).resolve().parent.parent / "evals" / "golden.json"
DEFAULT_BASELINE = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"


@dataclass(frozen=True)
class Relevant:
    """One passage that answers a query, identified durably rather than by id."""

    source: str
    contains: str

    def matches(self, hit: Hit) -> bool:
        # Whitespace-normalised so a case does not break when the chunker changes
        # where a line wraps.
        if hit.source != self.source:
            return False
        return " ".join(self.contains.split()) in " ".join(hit.text.split())


@dataclass(frozen=True)
class GoldenCase:
    id: str
    query: str
    kind: str
    relevant: tuple[Relevant, ...]
    why: str = ""

    def ranks_in(self, hits: Sequence[Hit]) -> list[int]:
        """1-based ranks at which each relevant passage appears; absent ones omitted."""
        found: list[int] = []
        for rel in self.relevant:
            for i, hit in enumerate(hits, 1):
                if rel.matches(hit):
                    found.append(i)
                    break
        return sorted(found)


@dataclass
class CaseResult:
    case: GoldenCase
    ranks: list[int]
    n_relevant: int

    @property
    def first_rank(self) -> int | None:
        return self.ranks[0] if self.ranks else None

    @property
    def missed(self) -> bool:
        return not self.ranks

    def reciprocal_rank(self) -> float:
        return 1.0 / self.first_rank if self.first_rank else 0.0

    def recall_at(self, k: int) -> float:
        if not self.n_relevant:
            return 0.0
        return sum(1 for r in self.ranks if r <= k) / self.n_relevant

    def ndcg_at(self, k: int) -> float:
        """Binary-gain nDCG.

        Binary rather than graded because the golden set records whether a passage
        answers the query, not how well — inventing a 0-3 scale would be a judgement
        call dressed as data. With one relevant passage this reduces to
        1/log2(rank+1), which is a reasonable position-decay curve.
        """
        if not self.n_relevant:
            return 0.0
        gains = [0.0] * k
        for r in self.ranks:
            if r <= k:
                gains[r - 1] = 1.0
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(self.n_relevant, k)))
        return dcg / ideal if ideal else 0.0


@dataclass
class EvalResult:
    label: str
    k: int
    results: list[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    def subset(self, kind: str) -> EvalResult:
        return EvalResult(
            label=self.label, k=self.k, results=[r for r in self.results if r.case.kind == kind]
        )

    @property
    def kinds(self) -> list[str]:
        return sorted({r.case.kind for r in self.results})

    def _mean(self, fn) -> float:
        return sum(fn(r) for r in self.results) / self.n if self.n else 0.0

    def mrr(self) -> float:
        return self._mean(lambda r: r.reciprocal_rank())

    def recall_at(self, k: int) -> float:
        return self._mean(lambda r: r.recall_at(k))

    def ndcg_at(self, k: int) -> float:
        return self._mean(lambda r: r.ndcg_at(k))

    def misses(self) -> list[CaseResult]:
        return [r for r in self.results if r.missed]

    def metrics(self) -> dict[str, float]:
        return {
            "mrr": round(self.mrr(), 4),
            "recall@1": round(self.recall_at(1), 4),
            "recall@3": round(self.recall_at(3), 4),
            f"recall@{self.k}": round(self.recall_at(self.k), 4),
            "ndcg@10": round(self.ndcg_at(10), 4),
        }


# ------------------------------------------------------------------ loading


def load_golden_set(path: Path | str = DEFAULT_GOLDEN_SET) -> list[GoldenCase]:
    """Load and validate the committed golden set.

    Validation is strict and fails loudly. A golden set with a typo'd source filename
    silently reports a miss, which reads as a retrieval regression — the most
    misleading possible failure for a tool whose entire purpose is telling you whether
    retrieval regressed.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    seen: set[str] = set()
    for raw in data["cases"]:
        case_id = raw["id"]
        if case_id in seen:
            raise ValueError(f"duplicate case id {case_id!r} — ids must be unique")
        seen.add(case_id)
        relevant = tuple(Relevant(r["source"], r["contains"]) for r in raw["relevant"])
        if not relevant:
            raise ValueError(f"case {case_id!r} lists no relevant passages")
        cases.append(
            GoldenCase(
                id=case_id,
                query=raw["query"],
                kind=raw.get("kind", "general"),
                relevant=relevant,
                why=raw.get("why", ""),
            )
        )
    if not cases:
        raise ValueError(f"{path} contains no cases")
    return cases


def verify_golden_set(cases: Sequence[GoldenCase], corpus_dir: Path | str) -> list[str]:
    """Check every case points at text that actually exists in the corpus.

    Run before trusting any number the harness produces. Without it a stale case is
    indistinguishable from a genuine retrieval failure.
    """
    corpus = Path(corpus_dir)
    problems: list[str] = []
    cache: dict[str, str] = {}
    for case in cases:
        for rel in case.relevant:
            path = corpus / rel.source
            if rel.source not in cache:
                if not path.exists():
                    problems.append(f"{case.id}: no such file {rel.source}")
                    cache[rel.source] = ""
                    continue
                cache[rel.source] = " ".join(path.read_text(encoding="utf-8").split())
            body = cache[rel.source]
            if body and " ".join(rel.contains.split()) not in body:
                problems.append(f"{case.id}: {rel.contains!r} not found in {rel.source}")
    return problems


# --------------------------------------------------------------- evaluating


def evaluate(
    retriever: Retriever,
    cases: Sequence[GoldenCase],
    *,
    label: str = "default",
    k: int = 10,
    mode: SearchMode | str = SearchMode.HYBRID,
    rerank: bool | None = None,
    candidates: int = 50,
    rerank_candidates: int = 25,
) -> EvalResult:
    """Run every case through one retrieval configuration."""
    result = EvalResult(label=label, k=k)
    for case in cases:
        hits = retriever.search(
            case.query,
            k=k,
            mode=mode,
            rerank=rerank,
            candidates=candidates,
            rerank_candidates=rerank_candidates,
        )
        result.results.append(
            CaseResult(case=case, ranks=case.ranks_in(hits), n_relevant=len(case.relevant))
        )
    return result


# ------------------------------------------------------------------ display


def format_comparison(results: Sequence[EvalResult]) -> str:
    """A metric-by-configuration table, broken out by case kind.

    The per-kind breakdown is the point. An overall average hides the finding that
    matters — that the two arms fail on opposite query types — behind a single number
    that looks merely mediocre.
    """
    if not results:
        return "(no results)"
    lines: list[str] = []
    width = 22
    labels = [r.label for r in results]
    header = f"{'metric':<{width}}" + "".join(f"{lab:>12}" for lab in labels)
    lines.append(header)
    lines.append("-" * len(header))

    kinds = sorted({k for r in results for k in r.kinds})
    for scope, subsets in [("ALL", results)] + [
        (kind, [r.subset(kind) for r in results]) for kind in kinds
    ]:
        n = subsets[0].n
        lines.append(f"{scope} (n={n})")
        for metric in ("mrr", "recall@1", "recall@3", "ndcg@10"):
            row = f"  {metric:<{width - 2}}"
            for sub in subsets:
                row += f"{sub.metrics()[metric]:>12.3f}"
            lines.append(row)
        lines.append("")

    for r in results:
        missed = r.misses()
        if missed:
            lines.append(f"misses [{r.label}]: " + ", ".join(m.case.id for m in missed))
    return "\n".join(lines)


def format_detail(result: EvalResult) -> str:
    """Per-case ranks for one configuration.

    Column width is derived from the data rather than fixed, so a case id exactly as
    long as the column does not run into the next field.
    """
    if not result.results:
        return "(no cases)"
    id_w = max(len(r.case.id) for r in result.results) + 2
    kind_w = max(len(r.case.kind) for r in result.results) + 2
    header = f"{'case':<{id_w}}{'kind':<{kind_w}}{'rank':>6}  query"
    lines = [header, "-" * (len(header) + 34)]
    for r in result.results:
        rank = str(r.first_rank) if r.first_rank else "MISS"
        lines.append(f"{r.case.id:<{id_w}}{r.case.kind:<{kind_w}}{rank:>6}  {r.case.query[:44]}")
    return "\n".join(lines)


# ----------------------------------------------------------- regression floor


def check_floor(
    result: EvalResult, baseline_path: Path | str = DEFAULT_BASELINE, *, tolerance: float = 0.02
) -> tuple[bool, list[str]]:
    """Compare against the committed baseline. Returns (ok, messages).

    `tolerance` absorbs the small movement that comes from a model or library upgrade
    without absorbing a real regression. It is deliberately not zero: a floor that
    fails on noise gets disabled within a month, and a disabled check protects nothing.
    """
    path = Path(baseline_path)
    if not path.exists():
        return True, [f"no baseline at {path} — run `ragkit eval --save-baseline` to create one"]

    baseline = json.loads(path.read_text(encoding="utf-8"))
    expected = baseline.get("metrics", {})
    actual = result.metrics()
    messages: list[str] = []
    ok = True

    if baseline.get("n") != result.n:
        messages.append(
            f"golden set changed size: baseline n={baseline.get('n')}, now n={result.n}. "
            "Metrics are not comparable across different sets — re-save the baseline."
        )
        ok = False

    for metric, want in expected.items():
        got = actual.get(metric)
        if got is None:
            continue
        delta = got - want
        flag = "FAIL" if delta < -tolerance else "ok"
        if flag == "FAIL":
            ok = False
        messages.append(
            f"  {flag:<4} {metric:<12} baseline {want:.3f}  now {got:.3f}  ({delta:+.3f})"
        )
    return ok, messages


def save_baseline(
    result: EvalResult, baseline_path: Path | str = DEFAULT_BASELINE, *, note: str = ""
) -> None:
    path = Path(baseline_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": result.label,
                "n": result.n,
                "k": result.k,
                "note": note,
                "metrics": result.metrics(),
                "per_kind": {kind: result.subset(kind).metrics() for kind in result.kinds},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
