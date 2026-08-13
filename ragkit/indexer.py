"""The indexing pipeline: files → blocks → chunks → vectors → Postgres."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_document
from .config import Config
from .embedding import Embedder
from .store import Store, content_sha

#: Extensions treated as text. Everything else is skipped rather than guessed at —
#: a PDF read as UTF-8 produces plausible-looking garbage that embeds without error.
TEXT_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".rst"})


@dataclass
class IndexReport:
    indexed: list[str] = field(default_factory=list)
    skipped_unchanged: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    chunks_written: int = 0
    oversized: int = 0
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"indexed          {len(self.indexed)}",
            f"skipped (same)   {len(self.skipped_unchanged)}",
            f"pruned (gone)    {len(self.pruned)}",
            f"chunks written   {self.chunks_written}",
            f"elapsed          {self.elapsed_s:.1f}s",
        ]
        if self.oversized:
            lines.append(
                f"oversized        {self.oversized}  "
                "(atomic blocks kept whole rather than split — see chunking.py)"
            )
        if self.failed:
            lines.append(f"FAILED           {len(self.failed)}")
            lines += [f"  {src}: {err}" for src, err in self.failed]
        return "\n".join(lines)


def discover(root: Path) -> list[Path]:
    """Find indexable files under `root`, or return `root` itself if it is a file."""
    if root.is_file():
        return [root] if root.suffix.lower() in TEXT_SUFFIXES else []
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in TEXT_SUFFIXES
        and not any(part.startswith(".") for part in p.parts)
    )


def _title_of(text: str, path: Path) -> str:
    """First markdown H1 if there is one, else the filename."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").replace("_", " ")


def index_paths(
    paths: Iterable[Path],
    *,
    store: Store,
    embedder: Embedder,
    config: Config,
    root: Path | None = None,
    force: bool = False,
    prune: bool = True,
) -> IndexReport:
    """Index files, skipping any whose content hash is unchanged.

    The hash check is what makes re-indexing cheap: embedding dominates the cost, so
    a scheduled re-run over an unchanged corpus does almost no work. `force=True`
    bypasses it — needed after a chunking or model change, where the file is the same
    but the vectors it should produce are not.
    """
    started = time.monotonic()
    report = IndexReport()
    paths = list(paths)

    for path in paths:
        source = str(path.relative_to(root)) if root else str(path)
        try:
            raw = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            report.failed.append((source, f"unreadable: {exc}"))
            continue

        sha = content_sha(raw)
        if not force and store.stored_sha(source) == sha:
            report.skipped_unchanged.append(source)
            continue

        try:
            chunks = chunk_document(
                raw,
                target_chars=config.target_chars,
                overlap_chars=config.overlap_chars,
                max_chars=config.max_chars,
            )
            # Embed in batches. The list is materialised per document, which is fine
            # for documents but would need streaming for very large single files.
            vectors: list[list[float]] = []
            batch = config.batch_size
            texts = [c.embed_text for c in chunks]
            for i in range(0, len(texts), batch):
                vectors.extend(embedder.embed_passages(texts[i : i + batch]))

            store.upsert_document(
                source=source,
                title=_title_of(raw, path),
                sha=sha,
                chunks=chunks,
                embeddings=vectors,
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            report.failed.append((source, f"{type(exc).__name__}: {exc}"))
            continue

        report.indexed.append(source)
        report.chunks_written += len(chunks)
        report.oversized += sum(1 for c in chunks if c.oversized)

    if prune and root is not None:
        live = [str(p.relative_to(root)) for p in paths]
        report.pruned = store.prune_missing(live)

    report.elapsed_s = time.monotonic() - started
    return report


def index_directory(
    root: Path,
    *,
    store: Store,
    embedder: Embedder,
    config: Config,
    force: bool = False,
    prune: bool = True,
) -> IndexReport:
    return index_paths(
        discover(root),
        store=store,
        embedder=embedder,
        config=config,
        root=root if root.is_dir() else root.parent,
        force=force,
        prune=prune,
    )
