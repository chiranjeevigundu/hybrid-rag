"""Indexer tests — file discovery and title extraction run without a database."""

from __future__ import annotations

import pytest

from ragkit.indexer import _title_of, discover, index_directory
from ragkit.store import content_sha


def test_discover_finds_text_files_recursively(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("b")
    found = {p.name for p in discover(tmp_path)}
    assert found == {"a.md", "b.txt"}


def test_discover_skips_binary_and_unknown_extensions(tmp_path):
    # A PDF decoded as UTF-8 produces plausible garbage that embeds without error, so
    # unknown types are skipped rather than guessed at.
    (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 binary")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "real.md").write_text("# Real")
    assert [p.name for p in discover(tmp_path)] == ["real.md"]


def test_discover_skips_dotfiles_and_dot_directories(tmp_path):
    (tmp_path / ".hidden.md").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "COMMIT_EDITMSG.txt").write_text("x")
    (tmp_path / "visible.md").write_text("# V")
    assert [p.name for p in discover(tmp_path)] == ["visible.md"]


def test_discover_accepts_a_single_file(tmp_path):
    f = tmp_path / "one.md"
    f.write_text("# One")
    assert discover(f) == [f]
    other = tmp_path / "one.bin"
    other.write_bytes(b"\x00")
    assert discover(other) == []


def test_title_prefers_the_first_h1(tmp_path):
    assert _title_of("# Refund Policy\n\nbody", tmp_path / "x.md") == "Refund Policy"


def test_title_falls_back_to_a_readable_filename(tmp_path):
    assert _title_of("no heading here", tmp_path / "refund_policy-v2.md") == "refund policy v2"


def test_content_sha_is_stable_and_content_sensitive():
    assert content_sha("abc") == content_sha("abc")
    assert content_sha("abc") != content_sha("abd")


@pytest.mark.postgres
def test_unreadable_files_are_reported_without_aborting_the_run(tmp_path, store, embedder, config):
    (tmp_path / "good.md").write_text("# Good\n\nSome body text.")
    (tmp_path / "bad.md").write_bytes(b"\xff\xfe invalid utf-8 \xff")

    report = index_directory(tmp_path, store=store, embedder=embedder, config=config)

    assert report.indexed == ["good.md"], "a bad file stopped the run"
    assert len(report.failed) == 1 and report.failed[0][0] == "bad.md"


@pytest.mark.postgres
def test_second_run_skips_unchanged_and_force_overrides(tmp_path, store, embedder, config):
    (tmp_path / "a.md").write_text("# A\n\nBody.")

    first = index_directory(tmp_path, store=store, embedder=embedder, config=config)
    assert first.indexed == ["a.md"]

    second = index_directory(tmp_path, store=store, embedder=embedder, config=config)
    assert second.indexed == [] and second.skipped_unchanged == ["a.md"]

    # --force is what you reach for after a chunking or model change: same bytes on
    # disk, different vectors required.
    forced = index_directory(tmp_path, store=store, embedder=embedder, config=config, force=True)
    assert forced.indexed == ["a.md"]


@pytest.mark.postgres
def test_deleting_a_file_prunes_it_from_the_index(tmp_path, store, embedder, config):
    (tmp_path / "a.md").write_text("# A\n\nBody.")
    (tmp_path / "b.md").write_text("# B\n\nBody.")
    index_directory(tmp_path, store=store, embedder=embedder, config=config)
    assert store.stats()["documents"] == 2

    (tmp_path / "b.md").unlink()
    report = index_directory(tmp_path, store=store, embedder=embedder, config=config)

    assert report.pruned == ["b.md"], "a deleted file stayed retrievable"
    assert store.stats()["documents"] == 1
