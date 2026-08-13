"""ragkit — hybrid retrieval over a local corpus.

from ragkit import Config, Retriever, Store, build_embedder

config = Config.from_env()
store = Store(config.database_url)
retriever = Retriever(store, build_embedder(config.model_name))
for hit in retriever.search("how long do I have to file a refund?"):
    print(hit.score, hit.citation)
"""

from .chunking import Block, BlockKind, Chunk, chunk_document, parse_blocks
from .config import Config
from .embedding import Embedder, HashingEmbedder, build_embedder, l2_normalize
from .indexer import IndexReport, discover, index_directory, index_paths
from .retrieval import Retriever
from .store import Hit, Store, content_sha, to_pgvector

__all__ = [
    "Block",
    "BlockKind",
    "Chunk",
    "Config",
    "Embedder",
    "Hit",
    "HashingEmbedder",
    "IndexReport",
    "Retriever",
    "Store",
    "build_embedder",
    "chunk_document",
    "content_sha",
    "discover",
    "index_directory",
    "index_paths",
    "l2_normalize",
    "parse_blocks",
    "to_pgvector",
]

__version__ = "0.1.0"
