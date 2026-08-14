"""Assembling the DSN from parts.

`DATABASE_URL` stays the primary knob. These cover the fallback, which exists because
no orchestrator hands you a DSN: ECS, Kubernetes and Compose all inject a secret as its
own variable, and managed-database services generate the password themselves.
"""

import pytest

from ragkit.config import Config

PARTS = {
    "RAG_DB_HOST": "db.example.internal",
    "RAG_DB_PORT": "5432",
    "RAG_DB_NAME": "rag",
    "RAG_DB_USER": "raguser",
    "RAG_DB_PASSWORD": "plainpassword",
}


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    for key in ("DATABASE_URL", *PARTS):
        monkeypatch.delenv(key, raising=False)


def test_the_default_is_untouched_when_nothing_is_set():
    assert Config.from_env().database_url == Config.database_url


def test_an_explicit_url_wins(monkeypatch):
    for k, v in PARTS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DATABASE_URL", "postgresql://someone@elsewhere/db")
    assert Config.from_env().database_url == "postgresql://someone@elsewhere/db"


def test_it_assembles_from_parts(monkeypatch):
    for k, v in PARTS.items():
        monkeypatch.setenv(k, v)
    assert (
        Config.from_env().database_url
        == "postgresql://raguser:plainpassword@db.example.internal:5432/rag"
    )


def test_the_host_alone_is_what_triggers_assembly(monkeypatch):
    # Without a host there is nothing to connect to, so partial configuration must not
    # produce a half-built URL pointing at localhost with production credentials.
    monkeypatch.setenv("RAG_DB_PASSWORD", "secret")
    monkeypatch.setenv("RAG_DB_NAME", "rag")
    assert Config.from_env().database_url == Config.database_url


def test_a_password_with_url_syntax_does_not_move_the_host(monkeypatch):
    """The failure this function exists to prevent.

    RDS and friends generate passwords containing `@`, `/` and `:`. Concatenated, a
    password like `pa@ss/word` yields a URL that parses *successfully* into a different
    host, and the connection error then names a server nobody configured.
    """
    for k, v in PARTS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RAG_DB_PASSWORD", "pa@ss/wo:rd")

    url = Config.from_env().database_url
    assert "pa@ss" not in url, "an unquoted @ splits the authority in the wrong place"
    assert url == "postgresql://raguser:pa%40ss%2Fwo%3Ard@db.example.internal:5432/rag"

    # And it must still parse back to the intended host and password.
    from urllib.parse import unquote, urlsplit

    parts = urlsplit(url)
    assert parts.hostname == "db.example.internal"
    assert unquote(parts.password or "") == "pa@ss/wo:rd"


def test_a_username_with_url_syntax_is_quoted_too(monkeypatch):
    for k, v in PARTS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RAG_DB_USER", "iam:user@corp")

    from urllib.parse import unquote, urlsplit

    parts = urlsplit(Config.from_env().database_url)
    assert parts.hostname == "db.example.internal"
    assert unquote(parts.username or "") == "iam:user@corp"


def test_no_password_omits_the_colon(monkeypatch):
    # IAM database authentication and local trust setups have no password. A trailing
    # colon before the @ is accepted by some drivers and rejected by others.
    monkeypatch.setenv("RAG_DB_HOST", "db.example.internal")
    monkeypatch.setenv("RAG_DB_USER", "raguser")
    assert Config.from_env().database_url == "postgresql://raguser@db.example.internal:5432/rag"
