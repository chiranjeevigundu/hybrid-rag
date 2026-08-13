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

from .config import Config
from .embedding import build_embedder
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
