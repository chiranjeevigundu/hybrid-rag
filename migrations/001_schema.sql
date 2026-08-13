-- Schema for dense retrieval.
--
-- Step 2 adds the lexical half (a tsvector column and a GIN index) in its own
-- migration rather than here, so each migration reflects what the code at that
-- commit actually uses.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           bigserial PRIMARY KEY,
    source       text        NOT NULL UNIQUE,
    title        text,
    -- SHA-256 of the raw file. Re-indexing compares against this and skips the
    -- document when it matches, so a re-run over 10k files re-embeds only what
    -- actually changed. Embedding is the expensive step; this is what makes
    -- re-indexing cheap enough to run on a schedule.
    content_sha  text        NOT NULL,
    chunk_count  integer     NOT NULL DEFAULT 0,
    indexed_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    document_id  bigint      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      integer     NOT NULL,
    text         text        NOT NULL,
    heading_path text[]      NOT NULL DEFAULT '{}',
    kind         text        NOT NULL,
    -- An atomic block (table, code fence) that exceeded the size ceiling and was
    -- stored whole rather than split. Recorded so it can be reported, not hidden:
    -- an oversized table is a known trade, a truncated one is a silent data loss.
    oversized    boolean     NOT NULL DEFAULT false,
    -- The dimension is fixed by the model. BAAI/bge-base-en-v1.5 emits 768.
    -- Changing the model means changing this number in a new migration and
    -- re-indexing everything; there is no way to reinterpret existing vectors.
    embedding    vector(768),
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- HNSW, not IVFFlat: IVFFlat needs a populated table to build meaningful lists and
-- degrades when the corpus grows past what it was trained on. HNSW can be created on
-- an empty table and stays correct as rows arrive, which matters for a tool people
-- will point at an empty database and then fill.
--
-- vector_cosine_ops pairs with the `<=>` operator. Embeddings are L2-normalised on
-- write (see embedding.l2_normalize), so cosine and inner product rank identically.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
