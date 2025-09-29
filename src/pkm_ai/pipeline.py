"""Document ingestion pipeline wiring metadata store and vector store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .embeddings import BaseVectorStore, create_embeddings
from .ingestion import load_document, split_text
from .storage import BaseMetadataStore, ChunkInput, ChunkRecord, DocumentRecord


EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass
class IngestionResult:
    document: DocumentRecord
    chunks: list[ChunkRecord]


class DocumentIngestionPipeline:
    """High-level pipeline that loads, splits, embeds, and stores documents."""

    def __init__(
        self,
        metadata_store: BaseMetadataStore,
        vector_store: BaseVectorStore,
        *,
        embedding_fn: Optional[EmbedFn] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._metadata_store = metadata_store
        self._vector_store = vector_store
        self._embedding_model_name = embedding_model_name
        if embedding_fn is None:
            self._embedding_fn = lambda texts: create_embeddings(texts, model_name=self._embedding_model_name)
        else:
            self._embedding_fn = embedding_fn

    def ingest_file(
        self,
        path: str,
        *,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> IngestionResult:
        document = load_document(path)
        chunk_texts = split_text(document.content, chunk_size=chunk_size, overlap=overlap)

        document_record = self._metadata_store.upsert_document(
            path=document.metadata["path"],
            extension=document.metadata["extension"],
            num_chars=document.metadata["num_chars"],
        )

        chunk_inputs = [ChunkInput(text=text, position=index) for index, text in enumerate(chunk_texts)]
        chunk_records = self._metadata_store.replace_document_chunks(document_record.id, chunk_inputs)

        if chunk_records:
            embeddings = self._embedding_fn([chunk.text for chunk in chunk_records])
            metadatas = [
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "position": chunk.position,
                    "path": document_record.path,
                }
                for chunk in chunk_records
            ]
            self._vector_store.add(embeddings, metadatas)

        return IngestionResult(document=document_record, chunks=list(chunk_records))


__all__ = ["IngestionResult", "DocumentIngestionPipeline"]
