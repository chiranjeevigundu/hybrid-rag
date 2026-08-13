# hybrid-rag

Retrieval over a local corpus, built to be **measured rather than asserted**. Runs
entirely on your machine — no API keys, no hosted vector database, no model provider.

```bash
git clone https://github.com/chiranjeevigundu/hybrid-rag && cd hybrid-rag
make demo
```

That starts Postgres, installs the package, indexes `./corpus`, and runs a query.

---

## Why this exists

Most RAG examples stop at "embed the chunks, take the top 5." Two things go wrong in
practice, and neither raises an error:

**Chunking destroys structure.** A fixed-size window cuts a markdown table in half.
The second half reaches the embedder as a run of pipes and numbers with no header row,
so it embeds as noise and never retrieves. You find out when someone asks a question
whose answer was in row 14.

**Dense retrieval is weak on exact identifiers.** Embeddings capture meaning, and an
order number has no meaning to capture. Searching `POL-4471`, the correct chunk scores
**0.54** against **0.39** for an unrelated one — a 0.15 margin, where the paraphrase
query "do you train models on my support conversations" manages 0.74 against 0.41, a
margin more than twice as wide. Same index, same model.

A narrow margin is a fragility signal, not a failure, and the difference is worth
stating precisely: dense retrieval still ranked `POL-4471` first. Across the 22-case
golden set, dense loses the top spot on exactly one identifier query (`EXP-2DA cutoff
time`, rank 2) — which the lexical arm fixes, taking that category from 0.917 to a
clean 1.000. The full numbers are below, including the one that went against the
design.

## Measured results

22 queries against the bundled corpus (4 documents, 29 chunks), scored at **chunk**
level: ground truth is the specific passage containing the answer, not merely the right
document. That distinction matters — with four documents, document-level scoring
saturates at 1.000 for every configuration and measures nothing at all.

```bash
python -m ragkit eval --compare
```

| MRR@10 | dense | lexical | hybrid | hybrid + rerank |
|---|---|---|---|---|
| exact-id (n=6) | 0.917 | 0.833 | **1.000** | 1.000 |
| paraphrase (n=9) | 0.917 | 0.000 | **0.917** | 0.856 |
| table (n=4) | 0.750 | 0.250 | **0.750** | 0.708 |
| discriminate (n=3) | 1.000 | 0.000 | **1.000** | 1.000 |
| **Overall (n=22)** | 0.898 | 0.273 | **0.920** | 0.888 |

**The arms have genuinely opposite blind spots.** Lexical misses **every paraphrase
query** — 0.000, sixteen misses out of twenty-two overall — while beating nothing else
on identifiers. Dense is the mirror image. That complementarity is the entire premise
of hybrid retrieval, and here it is visible rather than asserted.

**Fusion takes the better arm without paying for the worse one.** Hybrid beats dense on
identifiers (0.917 → 1.000) while matching it exactly on paraphrase, tables, and
discrimination. An arm returning nothing degrades to the other arm's ordering rather
than dragging the result down — the RRF property the whole design leans on.

**Reranking was measured, and it is off by default because it lost.** A cross-encoder
costs 0.032 MRR here: −0.061 on paraphrase, −0.042 on tables, no gain anywhere. The
code path stays and one environment variable turns it on, but the default follows the
measurement rather than the convention. The honest scope of that claim is narrow — 22
queries over 29 chunks, and `ms-marco-MiniLM` is trained on web search passages rather
than policy documentation, so this says something about *this corpus*, not about
cross-encoders. Which is exactly why the harness exists: run it on yours before
trusting either default.

**Tables are the weakest area** at 0.750 MRR and 0.500 recall@1 — the answer is found,
but often at rank 2 rather than rank 1. Recorded as a known gap rather than smoothed
into the average.

### The regression floor

`evals/baseline.json` holds the committed metrics; CI runs `ragkit eval --check-floor`
and fails the build when any metric drops more than 0.02 below it. The tolerance is
deliberately not zero — a gate that trips on model-version noise gets disabled within a
month, and a disabled gate protects nothing.

Ground truth is checked against the corpus before anything is measured
(`ragkit eval --verify`, also in CI). A case pointing at text somebody edited away
reports a miss, which is indistinguishable from a genuine regression — the single most
misleading failure a tool like this can have.

Cases identify passages by source file plus a substring, never by chunk id or ordinal,
so the set survives re-chunking. A golden set that breaks when you tune the chunker
cannot tell you whether the tuning helped.

Reproduce with `make demo`, or compare arms directly:

```bash
python -m ragkit search "EXP-2DA cutoff time" --mode dense
python -m ragkit search "EXP-2DA cutoff time" --mode lexical
python -m ragkit search "EXP-2DA cutoff time"                    # hybrid (the default)
RAG_RERANK_MODEL=Xenova/ms-marco-MiniLM-L-6-v2 \
  python -m ragkit search "EXP-2DA cutoff time"                  # ...with reranking
```

Every result line carries its provenance, so which arm found a passage — and which one
missed it — is visible without instrumenting anything:

```
1. [rerank +5.43] shipping.md § Shipping and Delivery > Service levels
                  (dense#2 lexical#1, retrieval 0.0325)
2. [rerank -2.85] api-reference.md § Orders API > Create an order
                  (dense#20, retrieval 0.0125)
```

The leading number is always the one that produced the ordering: the rerank logit when
reranking is on, the retrieval score when it is off. Showing a cosine value beside a
rerank-sorted list prints a column that visibly is not sorted, which reads as a broken
ranker rather than a mislabelled column.

## Three faces, one core

The same retrieval engine is reachable three ways, because the consumers genuinely
differ — not because three interfaces sounded thorough.

**Library** — `from ragkit import Retriever`. The eval harness and indexing jobs import
it directly; putting HTTP in front of a batch job would be pure overhead.

**HTTP** — `uvicorn ragkit.service:app`. For service-to-service calls. A backend in
another container wants a typed JSON contract, not tool schemas written for a model to
read.

```bash
curl -s localhost:8080/search \
  -H 'content-type: application/json' \
  -d '{"query": "how long to return an international order?", "k": 3}'
```

**MCP** — `python -m ragkit.mcp_server`. For LLM clients. Three tools: `search_corpus`,
`get_chunk`, `corpus_stats`, all annotated read-only.

```bash
claude mcp add hybrid-rag -- python -m ragkit.mcp_server
```

A `.mcp.json` is committed, so cloning the repo and opening it in Claude Code is enough
— no global config to edit.

Two things are deliberately *not* exposed on either remote face:

**No indexing endpoint.** It needs filesystem access, runs for minutes on a real
corpus, and would put a caller-supplied path in front of the disk — a long-running
request wrapped around a path-traversal surface. Indexing stays a batch job.

**MCP results are truncated; HTTP results are not.** A model paying per token wants a
400-character snippet and a `chunk_id` it can expand on demand; a backend rendering
citations wants the passage. Same core, different obligations.

## Design decisions worth the words

**Structure-aware chunking.** Documents are parsed into blocks *before* being packed
into chunks. Tables and fenced code blocks are atomic — never split, even when that
means emitting an oversized chunk, which is recorded and reported rather than hidden.
A truncated table is worse than a large one. See [`ragkit/chunking.py`](ragkit/chunking.py).

**Heading paths travel with the chunk.** A passage reading "must be filed within 30
days" is useless without knowing it sits under *Refunds > International Orders*, and
the retriever cannot recover that from the chunk body. `Chunk.text` is what you show a
user; `Chunk.embed_text` prepends the heading path and is what gets embedded.

**Queries and passages are encoded differently.** `bge-*-en-v1.5` is trained
asymmetrically: the query takes a short instruction prefix, the passage does not.
Encoding both the same way costs recall and reports nothing at all. `embed_query` and
`embed_passages` are separate methods so the wrong one cannot be reached by accident.

**ONNX Runtime, not PyTorch.** The model is ~90 MB either way; a torch install is
about 2 GB. This service never trains anything, so that is 2 GB of cold-start and
image size bought for nothing.

**HNSW over IVFFlat.** IVFFlat needs a populated table to build meaningful lists and
degrades as the corpus outgrows what it was built against. HNSW can be created on an
empty table and stays correct as rows arrive — which matters for a tool people point
at an empty database and then fill.

**Content-hash incremental indexing.** Each document stores a SHA-256 of its source.
Re-indexing skips anything unchanged, so a scheduled re-run over a large corpus
re-embeds only what actually moved. Deleted files are pruned, because an index that
keeps answering from a document that no longer exists is worse than one that returns
nothing.

## Usage

```bash
make up                              # Postgres + pgvector on :5433
make install                         # pip install -e ".[embed,dev]"
python -m ragkit init                # create the schema
python -m ragkit index corpus        # chunk, embed, store
python -m ragkit search "how long do I have to return an international order?"
python -m ragkit stats
```

As a library:

```python
from ragkit import Config, Retriever, Store, build_embedder

config = Config.from_env()
retriever = Retriever(Store(config.database_url), build_embedder(config.model_name))

for hit in retriever.search("what happens if my package shows up broken?", k=5):
    print(f"{hit.score:.3f}  {hit.citation}")
    print(hit.text[:200])
```

Configuration is environment-driven with working defaults for everything — see
[`.env.example`](.env.example).

## Tests

```bash
make test        # pure logic; no database, no download
make test-all    # adds the Postgres-backed suite
make eval        # retrieval quality against the golden set
make eval-compare  # dense vs lexical vs hybrid vs +rerank
```

Database tests skip themselves when no server is reachable, so `pytest` on a fresh
clone passes. CI runs a real `pgvector/pgvector:pg16` service and executes everything,
because the store tests assert on pgvector operator behaviour and HNSW ordering, and a
fake would reproduce neither.

**Tests never touch the database you index into.** The `store` fixture drops and
recreates the schema between tests, so running the suite against your working database
would destroy your index — silently, and you would only notice later when a search
came back empty. The test database name is derived from `DATABASE_URL` with a `_test`
suffix and created on demand, so `rag` and `rag_test` sit side by side. Set
`TEST_DATABASE_URL` to put it somewhere else.

`HashingEmbedder` is a deterministic stand-in used throughout the suite. It makes no
semantic claim — it exists so the schema, storage, and ranking plumbing can be tested
without a 90 MB download. **Relevance is deliberately not tested there**; that is what
the eval harness is for.

## Status

| | |
|---|---|
| ✅ | Structure-aware chunking, heading paths, atomic tables and code blocks |
| ✅ | Dense retrieval — bge-base-en-v1.5 via ONNX, pgvector, HNSW |
| ✅ | Incremental re-indexing by content hash, pruning of deleted sources |
| ✅ | Lexical retrieval — Postgres full-text search, generated `tsvector`, GIN |
| ✅ | Reciprocal Rank Fusion with per-arm provenance on every result |
| ✅ | Cross-encoder reranking (measured; off by default because it lost) |
| ✅ | Eval harness — 22-case golden set, MRR / recall@k / nDCG@10, CI regression floor |
| ✅ | Three faces over one core — Python library, HTTP API, MCP server |

Known limits, stated rather than discovered later:

- **The corpus is small enough to flatter the results.** 29 chunks over 4 documents.
  Document-level retrieval saturates completely, and even chunk-level leaves dense
  alone at 0.875 — there is not much room for fusion to demonstrate itself. The
  measurements are honest but they are not evidence about corpora three orders of
  magnitude larger.
- **`ts_rank_cd` is cover density, not BM25.** No IDF term. See `store.search_lexical`
  for why that is acceptable under rank fusion and what it would take to change.
- **Reranking is off by default** because it measured negative here; see above.
- **Tables retrieve at 0.750 MRR / 0.500 recall@1** — the weakest category, and the
  most obvious place to spend the next round of effort.
- **English only.** The text search configuration is pinned to `'english'` in the
  generated column, so stemming and stopwords are wrong for anything else.

## History

This repository began in August 2025 as `mini-rag-tfidf`, a TF-IDF retriever with a
Streamlit UI. TF-IDF is the direct ancestor of BM25, so the lexical arm arriving in the
next step is a return to where this started, with the dense half now beside it. The
early commits are still in the log.

## License

MIT.
