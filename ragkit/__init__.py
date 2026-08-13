"""ragkit — hybrid retrieval over a local corpus.

from ragkit import Config, Retriever, Store, build_embedder, build_reranker

config = Config.from_env()
retriever = Retriever(
    Store(config.database_url),
    build_embedder(config.model_name),
    reranker=build_reranker(config.rerank_model),
)
for hit in retriever.search("how long do I have to file a refund?"):
    print(hit.score, hit.citation, hit.provenance)
"""

from .chunking import Block, BlockKind, Chunk, chunk_document, parse_blocks
from .config import Config
from .embedding import Embedder, HashingEmbedder, build_embedder, l2_normalize
from .fusion import Fused, reciprocal_rank_fusion
from .indexer import IndexReport, discover, index_directory, index_paths
from .rerank import CrossEncoderReranker, IdentityReranker, Reranker, build_reranker
from .retrieval import Retriever, SearchMode
from .store import Hit, Store, content_sha, to_pgvector

__all__ = [
    "Block",
    "BlockKind",
    "Chunk",
    "Config",
    "CrossEncoderReranker",
    "Embedder",
    "Fused",
    "HashingEmbedder",
    "Hit",
    "IdentityReranker",
    "IndexReport",
    "Reranker",
    "Retriever",
    "SearchMode",
    "Store",
    "build_embedder",
    "build_reranker",
    "chunk_document",
    "content_sha",
    "discover",
    "index_directory",
    "index_paths",
    "l2_normalize",
    "parse_blocks",
    "reciprocal_rank_fusion",
    "to_pgvector",
]

__version__ = "0.2.0"
