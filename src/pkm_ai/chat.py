"""Retrieval-augmented chat helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from .embeddings import BaseVectorStore, VectorResult, create_embeddings
from .storage import BaseMetadataStore

try:  # pragma: no cover - optional dependency
    from transformers import pipeline as hf_pipeline
except ImportError:  # pragma: no cover - handled dynamically
    hf_pipeline = None

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user question using only the provided context. "
    "If the context does not contain the answer, reply with 'I do not know based on the provided documents.'"
)


class ChatError(Exception):
    """Raised when the chat engine cannot fulfill a request."""


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass
class ChatResponse:
    answer: str
    prompt: str
    chunks: List[RetrievedChunk]


LLMCallable = Callable[[str], Any]
EmbedQueryFn = Callable[[str], Sequence[float]]


class ChatEngine:
    """Coordinate retrieval-augmented prompts with a pluggable LLM backend."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        *,
        metadata_store: BaseMetadataStore | None = None,
        llm: LLMCallable | None = None,
        huggingface_model: str | None = None,
        huggingface_task: str = "text-generation",
        huggingface_kwargs: Optional[dict[str, Any]] = None,
        embed_query_fn: EmbedQueryFn | None = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_context_chunks: int = 4,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if llm is None and huggingface_model is None:
            raise ChatError("Provide either an 'llm' callable or a 'huggingface_model' name.")

        if llm is None:
            if hf_pipeline is None:
                raise ChatError("transformers is required to load a HuggingFace model.")
            kwargs = huggingface_kwargs or {}
            pipeline = hf_pipeline(huggingface_task, model=huggingface_model, **kwargs)

            def _hf_llm(prompt: str) -> Any:
                return pipeline(prompt)

            self._llm = _hf_llm
        else:
            self._llm = llm

        if embed_query_fn is None:
            self._embed_query_fn = lambda question: create_embeddings(
                [question], model_name=embedding_model_name
            )[0]
        else:
            self._embed_query_fn = embed_query_fn

        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._max_context_chunks = max_context_chunks
        self._system_prompt = system_prompt.strip()

    def ask(self, question: str, *, top_k: Optional[int] = None) -> ChatResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")

        query_embedding = self._embed_query_fn(question)
        results = self._vector_store.similarity_search(query_embedding, k=top_k or self._max_context_chunks)

        chunks = [self._convert_result(result) for result in results]
        prompt = self._build_prompt(question, chunks)
        answer_text = self._format_llm_output(self._llm(prompt))

        return ChatResponse(answer=answer_text.strip(), prompt=prompt, chunks=chunks)

    def _convert_result(self, result: VectorResult) -> RetrievedChunk:
        metadata = dict(result.metadata)
        text = metadata.get("text")
        if text is None and self._metadata_store is not None:
            chunk_id = metadata.get("chunk_id")
            if chunk_id is not None:
                chunk_record = self._metadata_store.get_chunk(chunk_id)
                if chunk_record is not None:
                    text = chunk_record.text
                    metadata.setdefault("document_id", chunk_record.document_id)
                    metadata.setdefault("position", chunk_record.position)
        if text is None:
            raise ChatError("Vector metadata must contain 'text' or metadata_store must provide it.")
        metadata["text"] = text
        return RetrievedChunk(text=text, metadata=metadata, score=result.score)

    def _build_prompt(self, question: str, chunks: Sequence[RetrievedChunk]) -> str:
        if chunks:
            context_blocks = []
            for idx, chunk in enumerate(chunks, start=1):
                doc_path = chunk.metadata.get("path", "unknown")
                position = chunk.metadata.get("position")
                context_blocks.append(
                    f"Chunk {idx} (document: {doc_path}, position: {position}, score: {chunk.score:.3f}):\n{chunk.text}"
                )
            context_section = "\n\n".join(context_blocks)
        else:
            context_section = "No matching context found."

        return (
            f"{self._system_prompt}\n\n"
            f"Context:\n{context_section}\n\n"
            f"User question:\n{question}\n\n"
            "Assistant response:"
        )

    @staticmethod
    def _format_llm_output(raw_output: Any) -> str:
        if isinstance(raw_output, str):
            return raw_output
        if isinstance(raw_output, list) and raw_output:
            first = raw_output[0]
            if isinstance(first, str):
                return first
            if isinstance(first, dict):
                for key in ("generated_text", "text"):  # common pipeline outputs
                    if key in first:
                        return str(first[key])
        if isinstance(raw_output, dict):
            for key in ("generated_text", "text"):
                if key in raw_output:
                    return str(raw_output[key])
        raise ChatError("Unable to parse LLM output.")


__all__ = ["ChatEngine", "ChatError", "ChatResponse", "RetrievedChunk"]
