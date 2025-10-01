from typing import Sequence

from pkm_ai.embeddings import BaseVectorStore, VectorResult
from pkm_ai.pipeline import DocumentIngestionPipeline
from pkm_ai.storage import SQLiteMetadataStore


class DummyVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self.calls: list[tuple[Sequence[Sequence[float]], Sequence[dict]]] = []

    def add(self, embeddings, metadatas):
        self.calls.append((list(embeddings), list(metadatas)))

    def similarity_search(self, query_embedding, *, k: int = 5):  # noqa: D401 - test stub
        return [VectorResult(metadata={}, score=0.0)] * 0


def simple_embedder(texts: Sequence[str]) -> Sequence[Sequence[float]]:
    return [[float(len(text))] for text in texts]


def test_pipeline_ingests_document(tmp_path):
    store_path = tmp_path / "metadata.db"
    metadata_store = SQLiteMetadataStore(store_path)
    vector_store = DummyVectorStore()

    document_path = tmp_path / "doc.txt"
    document_path.write_text("alpha beta gamma delta", encoding="utf-8")

    pipeline = DocumentIngestionPipeline(
        metadata_store,
        vector_store,
        embedding_fn=simple_embedder,
    )

    result = pipeline.ingest_file(str(document_path), chunk_size=6, overlap=1)

    assert result.document.path == str(document_path.resolve())
    assert len(result.chunks) >= 1
    assert len(vector_store.calls) == 1
    embeddings, metadatas = vector_store.calls[0]
    assert len(embeddings) == len(result.chunks)
    assert metadatas[0]["document_id"] == result.document.id
    assert metadatas[0]["text"] == result.chunks[0].text

    initial_chunk_texts = [chunk.text for chunk in result.chunks]

    document_path.write_text("updated content for document", encoding="utf-8")
    second_result = pipeline.ingest_file(str(document_path), chunk_size=7, overlap=0)

    assert len(vector_store.calls) == 2
    assert second_result.document.id == result.document.id

    stored_chunk_texts = [chunk.text for chunk in metadata_store.list_document_chunks(result.document.id)]
    assert stored_chunk_texts != initial_chunk_texts

    metadata_store.close()


def test_pipeline_handles_empty_document(tmp_path):
    store_path = tmp_path / "metadata.db"
    metadata_store = SQLiteMetadataStore(store_path)
    vector_store = DummyVectorStore()

    document_path = tmp_path / "empty.txt"
    document_path.write_text("   ", encoding="utf-8")

    pipeline = DocumentIngestionPipeline(
        metadata_store,
        vector_store,
        embedding_fn=simple_embedder,
    )

    result = pipeline.ingest_file(str(document_path))

    assert result.chunks == []
    assert vector_store.calls == []

    metadata_store.close()
