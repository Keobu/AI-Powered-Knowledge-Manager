"""Embedding utilities and vector store integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Literal, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - handled gracefully
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import faiss
except ImportError:  # pragma: no cover - handled gracefully
    faiss = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings as ChromaSettings
except ImportError:  # pragma: no cover - handled gracefully
    chromadb = None  # type: ignore[assignment]
    embedding_functions = None  # type: ignore[assignment]
    ChromaSettings = None  # type: ignore[assignment]

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when embeddings cannot be generated."""


def create_embeddings(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    encoder: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
) -> List[List[float]]:
    """Return embeddings for input texts using SentenceTransformers or a custom encoder."""

    if not texts:
        raise ValueError("The text list is empty")

    if encoder is None:
        if SentenceTransformer is None:
            raise EmbeddingError("SentenceTransformers is not available. Install 'sentence-transformers'.")
        encoder_model = SentenceTransformer(model_name)
        embeddings = encoder_model.encode(list(texts), convert_to_numpy=True).tolist()
    else:
        embeddings = [list(vector) for vector in encoder(texts)]

    if not embeddings:
        raise EmbeddingError("No embeddings were generated")

    return embeddings


@dataclass
class VectorResult:
    metadata: dict[str, Any]
    score: float


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class BaseVectorStore:
    """Common interface for vector stores."""

    def add(self, embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> List[VectorResult]:
        raise NotImplementedError


class FaissVectorStore(BaseVectorStore):
    """FAISS-backed vector store."""

    def __init__(self, dim: int, *, metric: Literal["l2", "ip"] = "ip") -> None:
        if faiss is None:
            raise VectorStoreError("FAISS is not available. Install 'faiss-cpu'.")

        if metric == "ip":
            self._index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            self._index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError("Metric must be 'ip' or 'l2'")

        self._metadata: List[dict[str, Any]] = []

    def add(self, embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict[str, Any]]) -> None:
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings and metadata entries must match")

        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("Embeddings must have shape (n, dim)")

        self._index.add(vectors)
        self._metadata.extend(dict(meta) for meta in metadatas)

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> List[VectorResult]:
        if not self._metadata:
            return []

        query = np.asarray([query_embedding], dtype="float32")
        scores, indices = self._index.search(query, min(k, len(self._metadata)))

        results: List[VectorResult] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append(VectorResult(metadata=self._metadata[idx], score=float(score)))
        return results


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        collection_name: str,
        *,
        client: Any | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        if client is None:
            if chromadb is None:
                raise VectorStoreError("ChromaDB is not available. Install 'chromadb'.")
            settings = ChromaSettings(anonymized_telemetry=False) if ChromaSettings else None
            client = chromadb.Client(settings)

        if embedding_functions is None:
            raise VectorStoreError("Chroma embedding functions are not available.")

        if SentenceTransformer is None:
            raise VectorStoreError("SentenceTransformers is required by ChromaVectorStore.")

        model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

        self._collection = client.get_or_create_collection(collection_name, embedding_function=model)
        self._id_counter = 0

    def add(self, embeddings: Sequence[Sequence[float]], metadatas: Sequence[dict[str, Any]]) -> None:
        if len(embeddings) != len(metadatas):
            raise ValueError("Number of embeddings and metadata entries must match")

        ids = [f"doc-{self._id_counter + i}" for i in range(len(embeddings))]
        self._id_counter += len(ids)
        self._collection.add(ids=ids, embeddings=list(embeddings), metadatas=list(metadatas))

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 5,
    ) -> List[VectorResult]:
        result = self._collection.query(query_embeddings=[list(query_embedding)], n_results=k)
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [VectorResult(metadata=meta, score=float(score)) for meta, score in zip(metadatas, distances)]


def build_vector_store(
    backend: Literal["faiss", "chroma"],
    *,
    dim: Optional[int] = None,
    collection_name: str = "pkm_ai",
    client: Any | None = None,
    metric: Literal["l2", "ip"] = "ip",
    model_name: str = DEFAULT_MODEL_NAME,
) -> BaseVectorStore:
    """Factory to instantiate a vector store backend."""

    if backend == "faiss":
        if dim is None:
            raise ValueError("FAISS backend requires the 'dim' parameter")
        return FaissVectorStore(dim=dim, metric=metric)
    if backend == "chroma":
        return ChromaVectorStore(collection_name=collection_name, client=client, model_name=model_name)

    raise ValueError("Unsupported backend")


__all__ = [
    "create_embeddings",
    "EmbeddingError",
    "VectorStoreError",
    "VectorResult",
    "BaseVectorStore",
    "FaissVectorStore",
    "ChromaVectorStore",
    "build_vector_store",
]
