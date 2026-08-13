"""Chunking tests.

These are the tests that would have caught the bug the module exists to prevent, so
they assert on structure rather than on exact output strings — an implementation that
merely produces *some* chunks of *roughly* the right size still fails here.
"""

from __future__ import annotations

from ragkit.chunking import BlockKind, chunk_document, parse_blocks

TABLE_DOC = """# Refund Policy

Refunds are processed within the windows below.

| Region        | Window   | Fee    |
|---------------|----------|--------|
| Domestic      | 30 days  | $0     |
| International | 60 days  | $12.50 |
| Wholesale     | 90 days  | $0     |

Contact support to begin.
"""


def test_a_table_is_never_split_across_chunks():
    # The whole point of the module. A tiny budget would slice a naive chunker
    # straight through the middle of the table.
    chunks = chunk_document(TABLE_DOC, target_chars=80, overlap_chars=10, max_chars=100)
    tables = [c for c in chunks if c.kind is BlockKind.TABLE]
    assert len(tables) == 1, "the table was split or lost"
    body = tables[0].text
    for row in ("Domestic", "International", "Wholesale", "Region"):
        assert row in body, f"{row!r} was separated from its table"


def test_an_oversized_table_is_flagged_rather_than_truncated():
    rows = "\n".join(f"| item-{i} | value-{i} | {i * 7} |" for i in range(200))
    doc = f"# Inventory\n\n| A | B | C |\n|---|---|---|\n{rows}\n"
    chunks = chunk_document(doc, target_chars=500, max_chars=800)
    tables = [c for c in chunks if c.kind is BlockKind.TABLE]
    assert len(tables) == 1
    assert tables[0].oversized is True
    assert "item-199" in tables[0].text, "rows were dropped off the end"


def test_a_fenced_code_block_cannot_corrupt_the_heading_stack():
    # '# not a heading' inside a fence must stay inert. If the fence state does not
    # win, every chunk after this point carries a wrong heading path.
    doc = """# Real Heading

```python
# not a heading
| not | a | table |
```

Body text after the fence.
"""
    blocks = parse_blocks(doc)
    headings = [b.text for b in blocks if b.kind is BlockKind.HEADING]
    assert headings == ["Real Heading"]
    code = [b for b in blocks if b.kind is BlockKind.CODE]
    assert len(code) == 1
    assert "# not a heading" in code[0].text


def test_heading_path_tracks_nesting_and_pops_siblings():
    doc = """# Policies

## Refunds

Refund body.

### International

International body.

## Shipping

Shipping body.
"""
    blocks = parse_blocks(doc)
    by_text = {b.text: b.heading_path for b in blocks if b.kind is not BlockKind.HEADING}
    assert by_text["Refund body."] == ("Policies", "Refunds")
    assert by_text["International body."] == ("Policies", "Refunds", "International")
    # "Shipping" is a sibling of "Refunds", so "Refunds" and "International" must pop.
    assert by_text["Shipping body."] == ("Policies", "Shipping")


def test_embed_text_carries_heading_context_but_display_text_does_not():
    doc = "# Refunds\n\n## Timeframes\n\nMust be filed within 30 days.\n"
    chunk = next(c for c in chunk_document(doc) if "30 days" in c.text)
    assert "Refunds > Timeframes" in chunk.embed_text
    assert "Refunds > Timeframes" not in chunk.text, "display text should stay clean"
    assert chunk.text.endswith("Must be filed within 30 days.")


def test_prose_splits_at_sentence_boundaries_with_overlap():
    sentences = [f"Sentence number {i} carries a distinct fact." for i in range(40)]
    doc = "# Doc\n\n" + " ".join(sentences) + "\n"
    chunks = chunk_document(doc, target_chars=300, overlap_chars=100)
    assert len(chunks) > 1, "long prose should have been divided"
    # No chunk should end mid-sentence.
    for c in chunks:
        assert c.text.rstrip().endswith("."), f"chunk cut mid-sentence: {c.text[-60:]!r}"
    # Overlap means consecutive chunks share text.
    # strict=False is correct here: chunks[1:] is intentionally one shorter, since we
    # are comparing each chunk against its successor.
    assert any(
        set(a.text.split(".")) & set(b.text.split("."))
        for a, b in zip(chunks, chunks[1:], strict=False)
    ), "no overlap between adjacent chunks"


def test_a_heading_leads_its_section_rather_than_trailing_the_previous_one():
    doc = "# One\n\nBody one.\n\n# Two\n\nBody two.\n"
    chunks = chunk_document(doc, target_chars=4000)
    # With a huge budget everything could pack into one chunk; the heading rule must
    # still force a split so each section is independently retrievable.
    assert len(chunks) >= 2
    assert not any(c.text.rstrip().endswith("Two") for c in chunks), "heading dangled"


def test_bare_headings_produce_no_chunk():
    chunks = chunk_document("# Title\n\n## Subtitle\n")
    assert chunks == [], "a heading with no body is noise in the index"


def test_ordinals_are_contiguous_and_ordered():
    doc = TABLE_DOC * 3
    chunks = chunk_document(doc, target_chars=200)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_empty_and_whitespace_documents_are_safe():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  \t\n") == []


def test_a_list_is_kept_together_as_one_block():
    doc = "# Steps\n\n- first\n- second\n- third\n\nAfter.\n"
    blocks = parse_blocks(doc)
    lists = [b for b in blocks if b.kind is BlockKind.LIST]
    assert len(lists) == 1
    assert lists[0].text.count("\n") == 2
