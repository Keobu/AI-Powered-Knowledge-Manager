import numpy as np
import pytest

import pkm_ai.embeddings as embeddings_module
from pkm_ai.embeddings import (
    BaseVectorStore,
    ChromaVectorStore,
    EmbeddingError,
    FaissVectorStore,
    VectorStoreError,
    build_vector_store,
    create_embeddings,
)


def test_create_embeddings_with_custom_encoder():
    texts = ["alpha", "beta"]
    encoder = lambda items: [[float(len(item))] for item in items]
    vectors = create_embeddings(texts, encoder=encoder)

    assert vectors == [[5.0], [4.0]]


def test_create_embeddings_empty_list():
    with pytest.raises(ValueError):
        create_embeddings([])


def test_create_embeddings_without_sentence_transformers(monkeypatch):
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", None)

    with pytest.raises(EmbeddingError):
        create_embeddings(["text"])


class DummyFaissIndex:
    def __init__(self, dim):
        self.vectors = np.zeros((0, dim), dtype="float32")

    def add(self, vectors):
        self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query, k):
        query_vec = query[0]
        scores = self.vectors @ query_vec
        top_indices = np.argsort(scores)[::-1][:k]
        return (
            np.array([scores[top_indices]], dtype="float32"),
            np.array([top_indices], dtype="int64"),
        )


class DummyFaissModule:
    def __init__(self, index_cls):
        self._index_cls = index_cls

    def IndexFlatIP(self, dim):
        return self._index_cls(dim)

    def IndexFlatL2(self, dim):
        return self._index_cls(dim)


def test_faiss_vector_store_add_and_search(monkeypatch):
    dummy_faiss = DummyFaissModule(DummyFaissIndex)
    monkeypatch.setattr(embeddings_module, "faiss", dummy_faiss)

    store = FaissVectorStore(dim=3)
    store.add([[1, 0, 0], [0, 1, 0]], metadatas=[{"id": 1}, {"id": 2}])

    results = store.similarity_search([0.9, 0.1, 0], k=1)

    assert len(results) == 1
    assert results[0].metadata["id"] == 1
    assert results[0].score == pytest.approx(0.9, rel=1e-6)


def test_faiss_vector_store_requires_same_lengths():
    store = FaissVectorStore.__new__(FaissVectorStore)
    store._index = DummyFaissIndex(3)
    store._metadata = []

    with pytest.raises(ValueError):
        store.add([[1, 2, 3]], metadatas=[])


class FakeCollection:
    def __init__(self):
        self.records = []

    def add(self, ids, embeddings, metadatas):
        for identifier, embedding, metadata in zip(ids, embeddings, metadatas):
            self.records.append((identifier, np.asarray(embedding, dtype="float32"), metadata))

    def query(self, query_embeddings, n_results):
        query_vec = np.asarray(query_embeddings[0], dtype="float32")
        scores = []
        metas = []
        for _, emb, meta in self.records:
            score = float(np.dot(emb, query_vec))
            scores.append(score)
            metas.append(meta)
        order = np.argsort(scores)[::-1][:n_results]
        return {
            "metadatas": [[metas[i] for i in order]],
            "distances": [[scores[i] for i in order]],
        }


class FakeClient:
    def __init__(self, collection):
        self._collection = collection

    def get_or_create_collection(self, name, embedding_function):  # noqa: ARG002
        return self._collection


class DummyEmbeddingFunctions:
    class SentenceTransformerEmbeddingFunction:  # noqa: D401 - simple stub
        def __init__(self, model_name):  # noqa: D401 - simple stub
            self.model_name = model_name


def test_chroma_vector_store_with_fake_client(monkeypatch):
    monkeypatch.setattr(embeddings_module, "SentenceTransformer", object())
    monkeypatch.setattr(embeddings_module, "embedding_functions", DummyEmbeddingFunctions)

    collection = FakeCollection()
    client = FakeClient(collection)

    store = ChromaVectorStore("test", client=client, model_name="dummy-model")
    store.add([[1, 0]], metadatas=[{"id": "a"}])

    results = store.similarity_search([0.5, 0], k=1)

    assert len(results) == 1
    assert results[0].metadata["id"] == "a"
    assert isinstance(results[0].score, float)


def test_build_vector_store_factory(monkeypatch):
    dummy_faiss = DummyFaissModule(DummyFaissIndex)
    monkeypatch.setattr(embeddings_module, "faiss", dummy_faiss)

    faiss_store = build_vector_store("faiss", dim=2)
    assert isinstance(faiss_store, BaseVectorStore)

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", object())
    monkeypatch.setattr(embeddings_module, "embedding_functions", DummyEmbeddingFunctions)

    chroma_store = build_vector_store("chroma", collection_name="test", client=FakeClient(FakeCollection()))
    assert isinstance(chroma_store, BaseVectorStore)

    with pytest.raises(ValueError):
        build_vector_store("unknown")

    with pytest.raises(ValueError):
        build_vector_store("faiss")
