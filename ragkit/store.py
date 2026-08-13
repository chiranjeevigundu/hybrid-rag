"""Postgres + pgvector storage.

Vectors are sent as their text form (`'[0.1,0.2,...]'::vector`) rather than through
the `pgvector` adapter. It costs one small helper and removes a dependency whose
`register_vector` has to be re-applied per connection — an easy thing to forget in a
pooled setup, where the failure is a confusing type error at query time rather than
at startup.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import psycopg

from .chunking import Chunk

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def to_pgvector(vector: Sequence[float]) -> str:
    """Render a float sequence in pgvector's text input format."""
    return "[" + ",".join(f"{v:.7g}" for v in vector) + "]"


def content_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    text: str
    heading_path: tuple[str, ...]
    kind: str
    source: str
    title: str | None
    score: float

    @property
    def citation(self) -> str:
        """Where this came from, in a form worth showing a user."""
        where = " > ".join(self.heading_path)
        return f"{self.source}" + (f" § {where}" if where else "")


class Store:
    def __init__(self, database_url: str) -> None:
        self.database_url = database_url

    def connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url)

    # ---------------------------------------------------------------- schema

    def ensure_schema(self) -> list[str]:
        """Apply every migration in order. Each file is idempotent."""
        applied: list[str] = []
        with self.connect() as conn:
            for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
                conn.execute(path.read_text(encoding="utf-8"))
                applied.append(path.name)
            conn.commit()
        return applied

    def reset(self) -> None:
        with self.connect() as conn:
            conn.execute("DROP TABLE IF EXISTS chunks, documents CASCADE")
            conn.commit()

    # ------------------------------------------------------------- indexing

    def stored_sha(self, source: str) -> str | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT content_sha FROM documents WHERE source = %s", (source,)
            ).fetchone()
        return row[0] if row else None

    def upsert_document(
        self,
        *,
        source: str,
        title: str | None,
        sha: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Replace a document and all its chunks in one transaction.

        Replace rather than diff: chunk boundaries move when a document is edited, so
        ordinal N before and after an edit are not the same passage. Deleting and
        re-inserting keeps the index consistent with the file; a partial update would
        leave chunks that no longer correspond to any text in the source.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

        with self.connect() as conn:
            with conn.transaction():
                doc_id = conn.execute(
                    """
                    INSERT INTO documents (source, title, content_sha, chunk_count, indexed_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (source) DO UPDATE
                        SET title = EXCLUDED.title,
                            content_sha = EXCLUDED.content_sha,
                            chunk_count = EXCLUDED.chunk_count,
                            indexed_at = now()
                    RETURNING id
                    """,
                    (source, title, sha, len(chunks)),
                ).fetchone()[0]

                conn.execute("DELETE FROM chunks WHERE document_id = %s", (doc_id,))

                if chunks:
                    with conn.cursor() as cur:
                        cur.executemany(
                            """
                            INSERT INTO chunks
                                (document_id, ordinal, text, heading_path,
                                 kind, oversized, embedding)
                            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            """,
                            [
                                (
                                    doc_id,
                                    c.ordinal,
                                    c.text,
                                    list(c.heading_path),
                                    c.kind.value,
                                    c.oversized,
                                    to_pgvector(e),
                                )
                                # strict=True: the length check above already
                                # guarantees this, and a silent zip() truncation here
                                # would drop chunks without writing anything to say so.
                                for c, e in zip(chunks, embeddings, strict=True)
                            ],
                        )
        return doc_id

    def delete_document(self, source: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE source = %s", (source,))
            conn.commit()
            return cur.rowcount > 0

    def prune_missing(self, live_sources: Sequence[str]) -> list[str]:
        """Drop documents whose source file no longer exists.

        Without this, deleting a file leaves its chunks retrievable forever — the
        index keeps answering from a document that is gone, which is worse than
        having no answer.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT source FROM documents WHERE NOT (source = ANY(%s))",
                (list(live_sources),),
            ).fetchall()
            removed = [r[0] for r in rows]
            if removed:
                conn.execute(
                    "DELETE FROM documents WHERE NOT (source = ANY(%s))",
                    (list(live_sources),),
                )
            conn.commit()
        return removed

    # ------------------------------------------------------------ retrieval

    def search_dense(self, query_vector: Sequence[float], k: int = 10) -> list[Hit]:
        """Nearest neighbours by cosine distance.

        `ORDER BY embedding <=> query` is what lets the HNSW index serve this; any
        other expression there silently falls back to a sequential scan.
        """
        vec = to_pgvector(query_vector)
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.text, c.heading_path, c.kind,
                       d.source, d.title,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, k),
            ).fetchall()
        return [
            Hit(
                chunk_id=r[0],
                text=r[1],
                heading_path=tuple(r[2] or ()),
                kind=r[3],
                source=r[4],
                title=r[5],
                score=float(r[6]),
            )
            for r in rows
        ]

    def get_chunk(self, chunk_id: int) -> Hit | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT c.id, c.text, c.heading_path, c.kind, d.source, d.title
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.id = %s
                """,
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        return Hit(
            chunk_id=row[0],
            text=row[1],
            heading_path=tuple(row[2] or ()),
            kind=row[3],
            source=row[4],
            title=row[5],
            score=1.0,
        )

    # ---------------------------------------------------------------- stats

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT (SELECT count(*) FROM documents),
                       (SELECT count(*) FROM chunks),
                       (SELECT count(*) FROM chunks WHERE oversized),
                       (SELECT count(*) FROM chunks WHERE embedding IS NULL)
                """
            ).fetchone()
        return {
            "documents": row[0],
            "chunks": row[1],
            "oversized_chunks": row[2],
            "unembedded_chunks": row[3],
        }
