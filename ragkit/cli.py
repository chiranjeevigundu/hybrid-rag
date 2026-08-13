"""Command line interface.

argparse rather than a CLI framework: two commands do not justify a dependency in a
tool whose selling point is that `pip install` and `docker compose up` are the whole
setup.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

from .config import RERANK_MODEL, Config
from .embedding import build_embedder
from .evaluation import (
    DEFAULT_BASELINE,
    DEFAULT_GOLDEN_SET,
    check_floor,
    evaluate,
    format_comparison,
    format_detail,
    load_golden_set,
    save_baseline,
    verify_golden_set,
)
from .indexer import index_directory
from .rerank import build_reranker
from .retrieval import Retriever, SearchMode
from .store import Store


def _build(config: Config):
    return Store(config.database_url), build_embedder(
        config.model_name,
        dim=config.embedding_dim,
        query_prefix=config.query_prefix,
        cache_dir=config.cache_dir,
        batch_size=config.batch_size,
    )


def cmd_init(args: argparse.Namespace, config: Config) -> int:
    store = Store(config.database_url)
    applied = store.ensure_schema()
    print("applied: " + ", ".join(applied))
    return 0


def cmd_index(args: argparse.Namespace, config: Config) -> int:
    root = Path(args.path).resolve()
    if not root.exists():
        print(f"no such path: {root}", file=sys.stderr)
        return 2
    store, embedder = _build(config)
    store.ensure_schema()
    report = index_directory(
        root,
        store=store,
        embedder=embedder,
        config=config,
        force=args.force,
        prune=not args.no_prune,
    )
    print(report.summary())
    return 1 if report.failed else 0


def cmd_search(args: argparse.Namespace, config: Config) -> int:
    store, embedder = _build(config)
    reranker = (
        None if args.no_rerank else build_reranker(config.rerank_model, cache_dir=config.cache_dir)
    )
    retriever = Retriever(store, embedder, reranker=reranker, rrf_k=config.rrf_k)
    hits = retriever.search(
        args.query,
        k=args.k,
        mode=args.mode,
        candidates=config.candidates,
        rerank_candidates=config.rerank_candidates,
    )
    if not hits:
        print("no results")
        return 0
    for rank, hit in enumerate(hits, 1):
        # The leading number must always be the one that produced this ordering.
        # When a reranker runs it overrides the retrieval score, so printing the
        # cosine or RRF value here shows a column that is visibly not sorted — which
        # reads as a bug in the ranking rather than a bug in the display.
        if hit.rerank_score is not None:
            lead = f"rerank {hit.rerank_score:+.2f}"
            trailing = f"{hit.provenance}, retrieval {hit.score:.4f}"
        else:
            lead = f"{hit.score:.4f}"
            trailing = hit.provenance
        # Provenance on every line: which arm found a passage is what you actually
        # need when a result looks wrong, and the score alone cannot tell you.
        print(f"\n{rank}. [{lead}] {hit.citation}   ({trailing})")
        body = " ".join(hit.text.split())
        print(textwrap.indent(textwrap.fill(body[:400], width=88), "   "))
    print()
    return 0


def cmd_eval(args: argparse.Namespace, config: Config) -> int:
    cases = load_golden_set(args.golden)

    # Ground truth is checked before anything is measured. A stale case reports a
    # miss, which is indistinguishable from a retrieval regression — the single most
    # misleading failure mode a tool like this can have.
    problems = verify_golden_set(cases, args.corpus)
    if problems:
        print("golden set does not match the corpus:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 2
    if args.verify:
        print(f"ok: {len(cases)} cases, all ground-truth strings present in {args.corpus}")
        return 0

    store, embedder = _build(config)
    # --compare must always produce a real rerank column, even though reranking is
    # disabled by default. Building it from config.rerank_model would silently make
    # that column a duplicate of `hybrid` and hide the very comparison being asked for.
    rerank_model = RERANK_MODEL if args.compare else config.rerank_model
    reranker = build_reranker(rerank_model, cache_dir=config.cache_dir)
    if args.compare and reranker is None:  # pragma: no cover - defensive
        print("could not build a reranker for comparison", file=sys.stderr)
        return 2

    def run(label: str, mode: SearchMode, rerank: bool):
        return evaluate(
            Retriever(store, embedder, reranker=reranker if rerank else None, rrf_k=config.rrf_k),
            cases,
            label=label,
            k=args.k,
            mode=mode,
            rerank=rerank,
            candidates=config.candidates,
            rerank_candidates=config.rerank_candidates,
        )

    if args.compare:
        results = [
            run("dense", SearchMode.DENSE, False),
            run("lexical", SearchMode.LEXICAL, False),
            run("hybrid", SearchMode.HYBRID, False),
            run("hybrid+rr", SearchMode.HYBRID, True),
        ]
        print(format_comparison(results))
        primary = results[-1] if not args.no_rerank else results[-2]
    else:
        # The label must describe what actually ran, not what was asked for. With
        # reranking disabled by default, `--no-rerank` being absent does not mean a
        # reranker exists — labelling on the flag alone writes "hybrid+rr" into a
        # committed baseline that contains no reranked numbers.
        reranking = reranker is not None and not args.no_rerank
        label = f"{args.mode}{'+rr' if reranking else ''}"
        primary = run(label, SearchMode(args.mode), reranking)
        print(format_comparison([primary]))

    if args.detail:
        print()
        print(format_detail(primary))

    if args.save_baseline:
        save_baseline(primary, args.baseline, note=args.note)
        print(f"\nbaseline saved to {args.baseline}")
        return 0

    if args.check_floor:
        ok, messages = check_floor(primary, args.baseline, tolerance=args.tolerance)
        print("\nregression floor:")
        for m in messages:
            print(m)
        if not ok:
            print(
                "\nFAILED: retrieval quality dropped below the committed baseline.", file=sys.stderr
            )
            return 1
        print("ok: no regression")
    return 0


def cmd_stats(args: argparse.Namespace, config: Config) -> int:
    for key, value in Store(config.database_url).stats().items():
        print(f"{key:20} {value}")
    return 0


def cmd_reset(args: argparse.Namespace, config: Config) -> int:
    if not args.yes:
        print("refusing to drop tables without --yes", file=sys.stderr)
        return 2
    store = Store(config.database_url)
    store.reset()
    store.ensure_schema()
    print("schema dropped and recreated")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragkit",
        description="Hybrid retrieval over a local corpus. No API keys required.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create the schema").set_defaults(func=cmd_init)

    p_index = sub.add_parser("index", help="index a file or directory")
    p_index.add_argument("path", help="file or directory to index")
    p_index.add_argument(
        "--force",
        action="store_true",
        help="re-embed even when the content hash is unchanged "
        "(use after a model or chunking change)",
    )
    p_index.add_argument(
        "--no-prune",
        action="store_true",
        help="keep documents whose source file has disappeared",
    )
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="search the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5, help="number of results (default 5)")
    p_search.add_argument(
        "--mode",
        choices=[m.value for m in SearchMode],
        default=SearchMode.HYBRID.value,
        help="retrieval mode (default hybrid). dense/lexical isolate a single arm, "
        "which is how you see what each contributes",
    )
    p_search.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip cross-encoder reranking (faster, and the honest A/B against it)",
    )
    p_search.set_defaults(func=cmd_search)

    p_eval = sub.add_parser("eval", help="measure retrieval against the golden set")
    p_eval.add_argument("--golden", default=str(DEFAULT_GOLDEN_SET), help="golden set JSON")
    p_eval.add_argument("--corpus", default="corpus", help="corpus directory, for verification")
    p_eval.add_argument("-k", type=int, default=10, help="retrieval depth (default 10)")
    p_eval.add_argument(
        "--mode", choices=[m.value for m in SearchMode], default=SearchMode.HYBRID.value
    )
    p_eval.add_argument("--no-rerank", action="store_true", help="measure without reranking")
    p_eval.add_argument(
        "--compare",
        action="store_true",
        help="run dense, lexical, hybrid and hybrid+rerank side by side",
    )
    p_eval.add_argument("--detail", action="store_true", help="per-case ranks")
    p_eval.add_argument(
        "--verify", action="store_true", help="only check the golden set against the corpus"
    )
    p_eval.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    p_eval.add_argument("--save-baseline", action="store_true", help="write the current metrics")
    p_eval.add_argument("--note", default="", help="note recorded with a saved baseline")
    p_eval.add_argument(
        "--check-floor", action="store_true", help="fail if metrics dropped below the baseline"
    )
    p_eval.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="how far a metric may drop before failing (default 0.02)",
    )
    p_eval.set_defaults(func=cmd_eval)

    sub.add_parser("stats", help="index statistics").set_defaults(func=cmd_stats)

    p_reset = sub.add_parser("reset", help="drop and recreate the schema")
    p_reset.add_argument("--yes", action="store_true", help="confirm the drop")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config.from_env()
    try:
        return args.func(args, config)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
