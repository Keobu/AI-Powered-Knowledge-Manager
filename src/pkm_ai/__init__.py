"""PKM AI core package."""

from .embeddings import (
    BaseVectorStore,
    ChromaVectorStore,
    EmbeddingError,
    FaissVectorStore,
    VectorResult,
    VectorStoreError,
    build_vector_store,
    create_embeddings,
)
from .ingestion import Document, DocumentLoaderError, load_document, split_text
from .pipeline import DocumentIngestionPipeline, IngestionResult
from .storage import (
    BaseMetadataStore,
    ChunkInput,
    ChunkRecord,
    DocumentRecord,
    JSONMetadataStore,
    MetadataStoreError,
    SQLiteMetadataStore,
)

__all__ = [
    "Document",
    "DocumentLoaderError",
    "load_document",
    "split_text",
    "create_embeddings",
    "EmbeddingError",
    "VectorResult",
    "VectorStoreError",
    "BaseVectorStore",
    "FaissVectorStore",
    "ChromaVectorStore",
    "build_vector_store",
    "BaseMetadataStore",
    "MetadataStoreError",
    "SQLiteMetadataStore",
    "JSONMetadataStore",
    "DocumentRecord",
    "ChunkRecord",
    "ChunkInput",
    "DocumentIngestionPipeline",
    "IngestionResult",
]
