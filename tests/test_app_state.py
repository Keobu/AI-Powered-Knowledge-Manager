from pkm_ai.app_state import AppState, UploadedDocument
from pkm_ai.chat import ChatResponse, RetrievedChunk
from pkm_ai.embeddings import BaseVectorStore, VectorResult
from pkm_ai.pipeline import DocumentIngestionPipeline, IngestionResult
from pkm_ai.storage import BaseMetadataStore, ChunkInput, ChunkRecord, DocumentRecord


class DummyMetadataStore(BaseMetadataStore):
    def __init__(self) -> None:
        self.documents = {}
        self.chunks = {}

    def upsert_document(self, *, path, extension, num_chars):
        doc_id = path
        record = DocumentRecord(id=doc_id, path=path, extension=extension, num_chars=num_chars)
        self.documents[doc_id] = record
        return record

    def replace_document_chunks(self, document_id, chunks):
        chunk_records = [
            ChunkRecord(id=f"chunk-{i}", document_id=document_id, position=c.position, text=c.text)
            for i, c in enumerate(chunks)
        ]
        self.chunks[document_id] = chunk_records
        return chunk_records

    def list_document_chunks(self, document_id):
        return self.chunks.get(document_id, [])

    def get_chunk(self, chunk_id):
        for chunk_list in self.chunks.values():
            for chunk in chunk_list:
                if chunk.id == chunk_id:
                    return chunk
        return None


class DummyPipeline(DocumentIngestionPipeline):
    def __init__(self, metadata_store):
        self._metadata_store = metadata_store

    def ingest_file(self, path, chunk_size=1000, overlap=200):
        document = self._metadata_store.upsert_document(path=path, extension=".txt", num_chars=10)
        chunks = self._metadata_store.replace_document_chunks(
            document.id,
            [ChunkInput(text="chunk text", position=0)],
        )
        return IngestionResult(document=document, chunks=chunks)


class DummyChatEngine:
    def __init__(self):
        self.last_question = None

    def ask(self, question, top_k=None):
        self.last_question = question
        return ChatResponse(
            answer="dummy answer",
            prompt="prompt",
            chunks=[RetrievedChunk(text="chunk text", metadata={"chunk_id": "chunk-0"}, score=0.9)],
        )


def test_app_state_ingest_and_refresh(tmp_path):
    metadata_store = DummyMetadataStore()
    pipeline = DummyPipeline(metadata_store)
    chat_engine = DummyChatEngine()

    state = AppState(metadata_store=metadata_store, ingestion_pipeline=pipeline, chat_engine=chat_engine)

    doc = state.ingest_file("/tmp/doc.txt")
    assert isinstance(doc, UploadedDocument)
    assert doc.record.id == "/tmp/doc.txt"

    refreshed = state.refresh_documents()
    assert len(refreshed) == 1
    assert refreshed[0].chunks[0].text == "chunk text"

    search_results = state.search("question")
    assert len(search_results) == 1
    assert search_results[0].text == "chunk text"

    chat_response = state.chat("question")
    assert chat_response.answer == "dummy answer"
    assert chat_engine.last_question == "question"
