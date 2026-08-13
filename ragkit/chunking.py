"""Structure-aware chunking.

The failure this module exists to prevent: a fixed-size character window cuts a
markdown table in half. The second half reaches the embedder as a run of pipes and
numbers with no header row attached, so it embeds as noise, never retrieves, and the
fact it held is silently unreachable. Nothing errors. You find out when a user asks a
question whose answer is in row 14.

So blocks are parsed first and packed second. A table, a fenced code block, and a
heading are **atomic** — never split, even when that means emitting a chunk over the
target size. Prose is the only thing divided, and only at sentence boundaries.

Each chunk also carries the heading path it lives under. That is not decoration: a
chunk reading "must be filed within 30 days" is useless without knowing it sits under
"Refunds > International Orders", and the retriever has no way to recover that from
the chunk body alone. `Chunk.text` is what you show a user; `Chunk.embed_text`
prepends the heading path and is what actually gets embedded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

# Budgets are in characters, not tokens, to avoid dragging a tokenizer into the
# import graph. For English, ~4 chars/token is the usual approximation, so the 1200
# default lands near 300 tokens and MAX_CHARS near 500 — comfortably inside
# bge-base's 512-token window. Exceeding that window is a silent truncation, not an
# error, which is why the ceiling is deliberately conservative.
TARGET_CHARS = 1200
OVERLAP_CHARS = 150
MAX_CHARS = 2000

_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+")

# Split after ., !, or ? followed by whitespace — but not when the preceding token
# looks like an abbreviation or initial. Deliberately not a full sentence tokenizer;
# it only has to find safe cut points inside prose, and a missed split just yields a
# slightly larger chunk rather than a wrong one.
_ABBREV = "".join(
    (
        r"(?<!\b[A-Z])",  # a single capital is an initial, not a sentence end
        r"(?<!\bNo)(?<!\bMr)(?<!\bMs)(?<!\bDr)(?<!\bSt)",
        r"(?<!\bvs)(?<!\betc)(?<!\bi\.e)(?<!\be\.g)",
    )
)
_SENTENCE_END = re.compile(_ABBREV + r"(?<=[.!?])\s+")


class BlockKind(str, Enum):
    PROSE = "prose"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    HEADING = "heading"


#: Blocks that must never be divided, whatever the size budget says.
ATOMIC = frozenset({BlockKind.TABLE, BlockKind.CODE})


@dataclass(frozen=True)
class Block:
    """One structural unit of a document, before size-based packing."""

    kind: BlockKind
    text: str
    heading_path: tuple[str, ...]
    level: int = 0  # heading depth; 0 for everything else

    @property
    def atomic(self) -> bool:
        return self.kind in ATOMIC


@dataclass(frozen=True)
class Chunk:
    """A packed, embeddable unit."""

    text: str
    ordinal: int
    heading_path: tuple[str, ...]
    kind: BlockKind
    oversized: bool = False
    """True when an atomic block exceeded MAX_CHARS and was emitted whole anyway.

    Kept as a flag rather than silently split, because a truncated table is worse
    than a large one. Surfaced in the index summary so it is visible, not buried.
    """

    @property
    def embed_text(self) -> str:
        """What goes to the embedder — heading context prepended.

        A chunk retrieved for "how long do I have to file" needs to know it lives
        under "Refunds > International Orders". That context is in the document
        structure, not in the chunk body, so it has to be carried in explicitly.
        """
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


def parse_blocks(markdown: str) -> list[Block]:
    """Split a document into structural blocks, tracking the heading stack.

    Line-oriented on purpose: fenced code can contain anything at all, including
    text that looks like headings or table rows, so the fence state has to win over
    every other pattern.
    """
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_kind = BlockKind.PROSE
    fence: str | None = None

    def current_path() -> tuple[str, ...]:
        return tuple(title for _, title in heading_stack)

    def flush() -> None:
        nonlocal buf, buf_kind
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(Block(buf_kind, text, current_path()))
        buf = []
        buf_kind = BlockKind.PROSE

    for line in markdown.splitlines():
        fence_match = _FENCE.match(line)

        # Inside a fence, nothing else is interpreted. This is the rule that keeps a
        # ``` block containing '# not a heading' from corrupting the heading stack.
        if fence is not None:
            buf.append(line)
            if fence_match and fence_match.group(1)[0] == fence[0]:
                flush()
                fence = None
            continue

        if fence_match:
            flush()
            fence = fence_match.group(1)
            buf_kind = BlockKind.CODE
            buf.append(line)
            continue

        heading_match = _HEADING.match(line)
        if heading_match:
            flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            # Pop siblings and deeper levels so the path reflects real nesting.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            blocks.append(Block(BlockKind.HEADING, title, current_path() + (title,), level))
            heading_stack.append((level, title))
            continue

        if _TABLE_ROW.match(line):
            if buf_kind is not BlockKind.TABLE:
                flush()
                buf_kind = BlockKind.TABLE
            buf.append(line)
            continue

        if _LIST_ITEM.match(line):
            if buf_kind is not BlockKind.LIST:
                flush()
                buf_kind = BlockKind.LIST
            buf.append(line)
            continue

        if not line.strip():
            flush()
            continue

        # A non-matching line ends a table or list and starts prose.
        if buf_kind in (BlockKind.TABLE, BlockKind.LIST):
            flush()
        buf.append(line)

    flush()
    return blocks


def _split_prose(text: str, target: int, overlap: int) -> list[str]:
    """Divide prose at sentence boundaries, with overlap for context continuity."""
    if len(text) <= target:
        return [text]

    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    parts: list[str] = []
    current: list[str] = []
    size = 0

    for sentence in sentences:
        # A single sentence longer than the target is split on whitespace as a last
        # resort. Rare, but a 4000-character run-on would otherwise blow the window.
        if len(sentence) > target:
            if current:
                parts.append(" ".join(current))
                current, size = [], 0
            words = sentence.split()
            piece: list[str] = []
            for word in words:
                if sum(len(w) + 1 for w in piece) + len(word) > target and piece:
                    parts.append(" ".join(piece))
                    piece = []
                piece.append(word)
            if piece:
                parts.append(" ".join(piece))
            continue

        if size + len(sentence) > target and current:
            parts.append(" ".join(current))
            # Carry the tail of the previous chunk forward so a fact spanning a
            # boundary is retrievable from either side.
            tail: list[str] = []
            carried = 0
            for s in reversed(current):
                if carried + len(s) > overlap:
                    break
                tail.insert(0, s)
                carried += len(s)
            current, size = tail, carried

        current.append(sentence)
        size += len(sentence)

    if current:
        parts.append(" ".join(current))
    return parts


def chunk_document(
    markdown: str,
    *,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[Chunk]:
    """Parse a document into blocks, then pack blocks into embeddable chunks.

    Packing rules, in precedence order:
      1. An atomic block (table, code) is never split. If it alone exceeds
         `max_chars` it is emitted whole and flagged `oversized`.
      2. A heading never trails at the end of a chunk — it attaches to the content
         that follows it, since a heading alone retrieves nothing useful.
      3. Everything else fills greedily to `target_chars`.
    """
    blocks = parse_blocks(markdown)
    chunks: list[Chunk] = []
    pending: list[Block] = []
    ordinal = 0

    def emit(group: list[Block]) -> None:
        nonlocal ordinal
        if not group:
            return
        body = "\n\n".join(b.text for b in group).strip()
        if not body:
            return
        # A group's kind is the most specific non-prose kind it contains, so a chunk
        # holding a table reports itself as a table for downstream filtering.
        kind = next((b.kind for b in group if b.kind in ATOMIC), group[0].kind)
        chunks.append(
            Chunk(
                text=body,
                ordinal=ordinal,
                heading_path=group[0].heading_path,
                kind=kind,
                oversized=len(body) > max_chars,
            )
        )
        ordinal += 1

    def pending_size() -> int:
        return sum(len(b.text) + 2 for b in pending)

    for block in blocks:
        if block.atomic:
            # Flush whatever is buffered, then let the atomic block stand alone
            # rather than risk it being divided by a later packing decision.
            if pending_size() + len(block.text) > target_chars and pending:
                emit(pending)
                pending = []
            pending.append(block)
            emit(pending)
            pending = []
            continue

        if block.kind is BlockKind.HEADING:
            # Rule 2: start a fresh chunk at a heading, so the heading leads its
            # section instead of dangling off the end of the previous one.
            if pending:
                emit(pending)
                pending = []
            pending.append(block)
            continue

        if len(block.text) > target_chars:
            if pending:
                emit(pending)
                pending = []
            for part in _split_prose(block.text, target_chars, overlap_chars):
                emit([Block(block.kind, part, block.heading_path)])
            continue

        if pending_size() + len(block.text) > target_chars and pending:
            emit(pending)
            pending = []
        pending.append(block)

    emit(pending)

    # A heading with no body following it produces a chunk that is just a title.
    # It can only add noise to the index, so drop it.
    return [
        c
        for c in chunks
        if not (c.kind is BlockKind.HEADING and c.text.count("\n") == 0 and len(c.text) < 80)
    ]
