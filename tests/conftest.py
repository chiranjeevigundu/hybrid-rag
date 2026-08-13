"""Shared fixtures.

Database-backed tests are marked `postgres` and skip themselves when no server is
reachable, so `pytest` on a fresh clone runs the pure-logic suite and passes. CI starts
a real pgvector service and runs everything.

**Tests never touch the database you index into.** The `store` fixture drops and
recreates the schema between tests, so pointing it at the working database would
silently destroy your index every time you ran the suite — the kind of thing you
discover by searching for a document that was there five minutes ago. The test
database name is therefore derived from `DATABASE_URL` with a `_test` suffix and
created on demand. Override it with `TEST_DATABASE_URL` if you want it elsewhere.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse, urlunparse

import pytest

from ragkit.config import Config
from ragkit.embedding import HashingEmbedder
from ragkit.store import Store

DEFAULT_URL = "postgresql://rag:rag@localhost:5433/rag"


def _derive_test_url(url: str) -> str:
    """Return `url` with `_test` appended to the database name."""
    parts = urlparse(url)
    name = (parts.path or "/").lstrip("/") or "postgres"
    if name.endswith("_test"):
        return url
    return urlunparse(parts._replace(path=f"/{name}_test"))


def _ensure_database(url: str) -> bool:
    """Create the target database if it does not exist. True when it is usable."""
    import psycopg

    parts = urlparse(url)
    target = parts.path.lstrip("/")
    admin = urlunparse(parts._replace(path="/postgres"))
    try:
        # autocommit: CREATE DATABASE cannot run inside a transaction block.
        with psycopg.connect(admin, connect_timeout=3, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (target,)
            ).fetchone()
            if not exists:
                # Identifier cannot be parameterised; the name is derived from our own
                # config rather than user input, and psycopg's Identifier quotes it.
                from psycopg import sql

                conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target)))
        with psycopg.connect(url, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.getenv("TEST_DATABASE_URL") or _derive_test_url(os.getenv("DATABASE_URL", DEFAULT_URL))
    if not _ensure_database(url):
        pytest.skip(f"no Postgres reachable for {url} — run `make up`")
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
