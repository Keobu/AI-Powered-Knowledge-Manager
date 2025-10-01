"""PKM AI core package."""

from .app_state import AppState, UploadedDocument
from .chat import ChatEngine, ChatError, ChatResponse, RetrievedChunk
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
from .export import (
    ExportError,
    SummarySection,
    export_context_to_json,
    export_summary_to_markdown,
    export_summary_to_pdf,
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
    "ChatEngine",
    "ChatError",
    "ChatResponse",
    "RetrievedChunk",
    "AppState",
    "UploadedDocument",
    "export_summary_to_markdown",
    "export_summary_to_pdf",
    "export_context_to_json",
    "SummarySection",
    "ExportError",
]
