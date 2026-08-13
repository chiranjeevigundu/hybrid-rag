"""Shared fixtures.

Database-backed tests are marked `postgres` and skip themselves when no server is
reachable, so `pytest` on a fresh clone runs the pure-logic suite and passes. CI
starts a real pgvector service and runs everything.
"""

from __future__ import annotations

import os

import pytest

from ragkit.config import Config
from ragkit.embedding import HashingEmbedder
from ragkit.store import Store

DEFAULT_URL = "postgresql://rag:rag@localhost:5433/rag"


def _reachable(url: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=2) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("DATABASE_URL", DEFAULT_URL)
    if not _reachable(url):
        pytest.skip(f"no Postgres at {url} — run `make up`")
    return url


@pytest.fixture
def store(database_url: str) -> Store:
    """A store with a clean schema. Dropped and rebuilt per test for isolation."""
    s = Store(database_url)
    s.reset()
    s.ensure_schema()
    return s


@pytest.fixture
def embedder() -> HashingEmbedder:
    """Deterministic stand-in — no download, no network, no semantic claim."""
    return HashingEmbedder()


@pytest.fixture
def config() -> Config:
    return Config(model_name="hashing")
