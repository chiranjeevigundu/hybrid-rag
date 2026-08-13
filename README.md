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
order number has no meaning to capture. Searching `POL-4471` against this corpus, the
correct chunk scores **0.54** while an unrelated API chunk scores **0.39** — a 0.15
margin. The paraphrase query "do you train models on my support conversations" scores
**0.74** against **0.41**, a 0.33 margin. Same index, same model; the identifier query
is far closer to a coin flip. That gap is the argument for a lexical arm, and it is
measured here rather than assumed.

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
```

Database tests skip themselves when no server is reachable, so `pytest` on a fresh
clone passes. CI runs a real `pgvector/pgvector:pg16` service and executes everything,
because the store tests assert on pgvector operator behaviour and HNSW ordering, and a
fake would reproduce neither.

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
| 🔜 | BM25 over Postgres `tsvector`, Reciprocal Rank Fusion |
| 🔜 | Cross-encoder reranking |
| 🔜 | Eval harness — recall@k, MRR, nDCG@10 against a committed golden set, with a CI regression floor |
| 🔜 | HTTP and MCP interfaces over the same core |

The numbers quoted above come from the corpus in this repository and are reproducible
with `make demo`. When the lexical arm lands, the same queries become the before/after
measurement — which is the point of writing the baseline down now.

## History

This repository began in August 2025 as `mini-rag-tfidf`, a TF-IDF retriever with a
Streamlit UI. TF-IDF is the direct ancestor of BM25, so the lexical arm arriving in the
next step is a return to where this started, with the dense half now beside it. The
early commits are still in the log.

## License

MIT.
