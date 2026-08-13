-- The lexical half of hybrid retrieval.
--
-- Dense retrieval is measurably weak on rare tokens: on this repo's corpus the
-- identifier POL-4471 retrieves its correct chunk at 0.54 with only a 0.15 margin
-- over an unrelated one, while a paraphrase query manages 0.74 with a 0.33 margin.
-- Embeddings encode meaning, and an order number has no meaning to encode. Lexical
-- matching has the opposite bias, which is the entire reason to run both.

-- Why this wrapper exists, since asserting IMMUTABLE is not something to do casually.
--
-- A STORED generated column may only call immutable functions, and
-- `array_to_string(anyarray, text)` is marked **stable**. The reason is polymorphism:
-- for an arbitrary element type the output function may consult session settings —
-- timestamptz renders against TimeZone, for instance — so the planner cannot assume a
-- fixed result across sessions.
--
-- That reasoning does not apply once the signature is pinned to `text[]`. The element
-- output function for text is the identity, and the text search configuration is
-- named explicitly rather than read from default_text_search_config, so both sources
-- of session-dependence are gone and the expression is genuinely immutable.
--
-- The narrow `text[]` signature is therefore load-bearing, not incidental. Widening
-- it to anyarray would make the IMMUTABLE assertion false, and the failure mode is
-- nasty: the index silently disagrees with the data rather than raising anything.
CREATE OR REPLACE FUNCTION chunk_tsv(heading_path text[], body text)
RETURNS tsvector
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
AS $$
    SELECT to_tsvector(
        'english',
        coalesce(array_to_string(heading_path, ' '), '') || ' ' || coalesce(body, '')
    )
$$;

-- A STORED generated column rather than a trigger: Postgres recomputes it on write
-- automatically, so the tsvector cannot drift out of sync with `text` the way a
-- trigger-maintained column does the first time someone writes an UPDATE that forgets
-- to fire it.
--
-- heading_path is folded in so a query naming a section ("damaged parcels") matches
-- chunks under that heading even when the body never repeats the words. This mirrors
-- what Chunk.embed_text does for the dense side, keeping both arms over the same text.
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (chunk_tsv(heading_path, text)) STORED;

-- GIN, not GiST: GIN is slower to build and larger on disk, but substantially faster
-- to search, and this index is written once per re-index and read on every query.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
