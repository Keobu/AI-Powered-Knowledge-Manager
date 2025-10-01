"""Streamlit dashboard for PKM AI."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import streamlit as st

from .app_state import AppState
from .chat import ChatEngine
from .embeddings import build_vector_store
from .pipeline import DocumentIngestionPipeline
from .storage import SQLiteMetadataStore

st.set_page_config(page_title="PKM AI", layout="wide")


def initialize_state() -> AppState:
    if "app_state" in st.session_state:
        return st.session_state["app_state"]

    data_dir = Path(st.secrets.get("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    metadata_store = SQLiteMetadataStore(data_dir / "metadata.db")
    vector_store = build_vector_store("faiss", dim=384)

    pipeline = DocumentIngestionPipeline(metadata_store, vector_store)
    chat_engine = ChatEngine(vector_store, metadata_store=metadata_store, llm=lambda prompt: "LLM response placeholder")

    app_state = AppState(metadata_store=metadata_store, ingestion_pipeline=pipeline, chat_engine=chat_engine)
    st.session_state["app_state"] = app_state
    return app_state


def render_sidebar(state: AppState) -> None:
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Upload PDF/TXT/MD", type=["pdf", "txt", "md"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name
            document = state.ingest_file(tmp_path)
            st.sidebar.success(f"Ingested {uploaded_file.name} → {len(document.chunks)} chunks")

    st.sidebar.button("Refresh document list", on_click=state.refresh_documents)


def render_document_list(state: AppState) -> None:
    st.subheader("Document List")
    documents = state.refresh_documents()
    if not documents:
        st.info("No documents ingested yet.")
        return

    for doc in documents:
        with st.expander(Path(doc.record.path).name, expanded=False):
            st.write(f"Path: {doc.record.path}")
            st.write(f"Chunks: {len(doc.chunks)}")
            preview = "\n\n".join(chunk.text[:300] for chunk in doc.chunks[:3])
            st.text_area("Preview", preview or "<empty>", height=150)


def render_semantic_search(state: AppState) -> None:
    st.subheader("Semantic Search")
    query = st.text_input("Search your knowledge base")
    if query:
        results = state.chat_engine.ask(query, top_k=5)
        for idx, chunk in enumerate(results.chunks, start=1):
            st.markdown(f"**Result {idx}** — Score: {chunk.score:.3f}, Path: {chunk.metadata.get('path')}" )
            st.write(chunk.text)


def render_chat(state: AppState) -> None:
    st.subheader("Chat")
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    with st.form("chat_form"):
        question = st.text_input("Ask a question")
        submitted = st.form_submit_button("Send")

    if submitted and question.strip():
        response = state.chat(question)
        st.session_state["chat_history"].append((question, response.answer, response.chunks))

    for idx, (user_question, answer, chunks) in enumerate(reversed(st.session_state["chat_history"])):
        st.markdown(f"**You:** {user_question}")
        st.markdown(f"**PKM AI:** {answer}")
        with st.expander("Context", expanded=False):
            for chunk in chunks:
                st.write(f"Score: {chunk.score:.3f} | Document: {chunk.metadata.get('path')} | Position: {chunk.metadata.get('position')}")
                st.write(chunk.text)
        st.markdown("---")


def main() -> None:
    st.title("PKM AI Dashboard")

    state = initialize_state()

    render_sidebar(state)

    col1, col2 = st.columns(2)
    with col1:
        render_document_list(state)
        render_semantic_search(state)
    with col2:
        render_chat(state)


if __name__ == "__main__":
    main()
