from typing import Any, Sequence

import pytest

from pkm_ai.chat import ChatEngine, ChatError
from pkm_ai.embeddings import BaseVectorStore, VectorResult
from pkm_ai.storage import ChunkInput, SQLiteMetadataStore


class StaticVectorStore(BaseVectorStore):
    def __init__(self, results: Sequence[VectorResult]) -> None:
        self._results = list(results)

    def add(self, embeddings, metadatas):  # pragma: no cover - not used in tests
        raise NotImplementedError

    def similarity_search(self, query_embedding, *, k: int = 5):
        return list(self._results)[:k]


def test_chat_engine_builds_prompt_and_invokes_llm(tmp_path):
    vector_store = StaticVectorStore(
        [
            VectorResult(
                metadata={
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "position": 0,
                    "path": "doc.txt",
                    "text": "Alpha beta gamma",
                },
                score=0.9,
            ),
            VectorResult(
                metadata={
                    "chunk_id": "chunk-2",
                    "document_id": "doc-2",
                    "position": 1,
                    "path": "doc.txt",
                    "text": "Delta epsilon zeta",
                },
                score=0.5,
            ),
        ]
    )

    captured_prompt = {}

    def fake_llm(prompt: str) -> str:
        captured_prompt["value"] = prompt
        assert "Context:" in prompt
        assert "Alpha beta gamma" in prompt
        assert "User question" in prompt
        return "This is a context-aware answer."

    engine = ChatEngine(
        vector_store,
        llm=fake_llm,
        embed_query_fn=lambda question: [float(len(question))],
        max_context_chunks=2,
    )

    response = engine.ask("What is covered?")

    assert "context-aware" in response.answer
    assert len(response.chunks) == 2
    assert "Alpha beta gamma" in captured_prompt["value"]


def test_chat_engine_fetches_chunk_from_store(tmp_path):
    metadata_store = SQLiteMetadataStore(tmp_path / "metadata.db")
    document = metadata_store.upsert_document(path="/tmp/doc.txt", extension=".txt", num_chars=10)
    chunks = metadata_store.replace_document_chunks(document.id, [ChunkInput(text="Stored chunk", position=0)])

    vector_store = StaticVectorStore(
        [
            VectorResult(
                metadata={
                    "chunk_id": chunks[0].id,
                    "document_id": document.id,
                    "path": document.path,
                },
                score=0.8,
            )
        ]
    )

    engine = ChatEngine(
        vector_store,
        metadata_store=metadata_store,
        llm=lambda prompt: "Answer",
        embed_query_fn=lambda question: [1.0],
    )

    response = engine.ask("Explain?", top_k=1)

    assert response.chunks[0].text == "Stored chunk"
    metadata_store.close()


def test_chat_engine_requires_llm_or_model(vector_store=None):
    vector_store = vector_store or StaticVectorStore([])
    with pytest.raises(ChatError):
        ChatEngine(vector_store)
