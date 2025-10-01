"""Simple in-memory application state for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .chat import ChatEngine, ChatResponse
from .pipeline import DocumentIngestionPipeline
from .storage import BaseMetadataStore, ChunkRecord, DocumentRecord


@dataclass
class UploadedDocument:
    record: DocumentRecord
    chunks: List[ChunkRecord]


@dataclass
class AppState:
    metadata_store: BaseMetadataStore
    ingestion_pipeline: DocumentIngestionPipeline
    chat_engine: ChatEngine
    documents: Dict[str, UploadedDocument] = field(default_factory=dict)

    def ingest_file(self, file_path: str, *, chunk_size: int = 1000, overlap: int = 200) -> UploadedDocument:
        result = self.ingestion_pipeline.ingest_file(file_path, chunk_size=chunk_size, overlap=overlap)
        document = UploadedDocument(record=result.document, chunks=result.chunks)
        self.documents[result.document.id] = document
        return document

    def refresh_documents(self) -> List[UploadedDocument]:
        refreshed: Dict[str, UploadedDocument] = {}
        for document in self.documents.values():
            chunks = self.metadata_store.list_document_chunks(document.record.id)
            refreshed[document.record.id] = UploadedDocument(record=document.record, chunks=chunks)
        self.documents = refreshed
        return list(self.documents.values())

    def search(self, query: str, *, top_k: int = 5) -> Sequence[ChunkRecord]:
        response = self.chat_engine.ask(query, top_k=top_k)
        ordered_chunks = []
        for chunk in response.chunks:
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id:
                chunk_record = self.metadata_store.get_chunk(chunk_id)
                if chunk_record:
                    ordered_chunks.append(chunk_record)
        return ordered_chunks

    def chat(self, question: str) -> ChatResponse:
        return self.chat_engine.ask(question)
