# PKM AI

PKM AI is my personal knowledge hub powered by AI. Drop in PDFs, notes, or markdown files, and it indexes everything so future-you (or an LLM teammate) can instantly recall the good parts.

## Where Things Stand
- ✅ Phase 1 – Project scaffolding (src/tests/docs, dependencies, tooling)
- ✅ Phase 2 – Document ingestion (`load_document`, `split_text`, pytest coverage)
- ✅ Phase 3 – Embeddings, vector stores, and metadata pipeline (SentenceTransformers, FAISS/Chroma wrappers, persistence tests)
- ☐ Phase 4 – AI chat layer
- ☐ Phase 5 – Streamlit dashboard
- ☐ Phase 6 – Bonus polish (exports, highlights, auth)

## What PKM AI Tries To Do
- Collect PDFs, TXT, and Markdown into a searchable knowledge base.
- Extract key chunks with NLP-friendly splits.
- Encode everything into vector space for semantic lookup.
- Answer questions using only your own documents.
- Surface everything in an approachable Streamlit dashboard.
- Let you export concise summaries when you’re done.

## Getting Started
1. Spin up a Python ≥3.10 virtual environment and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. (Optional, recommended) install the package locally so imports just work: `pip install -e .`.

## Running Tests
- Full suite: `pytest`

## How the Pieces Fit
### Ingestion
- `pkm_ai.ingestion.load_document(path)` handles PDF/TXT/MD and returns raw text plus metadata (path, extension, char count).
- `pkm_ai.ingestion.split_text(text, chunk_size, overlap)` slices clean, overlapping context windows.
- Coverage lives in `tests/test_ingestion.py`.

### Embeddings & Vector Search
- `pkm_ai.create_embeddings(texts, encoder=None)` defaults to SentenceTransformers but accepts any encoder.
- `pkm_ai.build_vector_store("faiss", dim=...)` spins up a FAISS index; `build_vector_store("chroma", ...)` opens a Chroma collection.
- Custom wrappers (`FaissVectorStore`, `ChromaVectorStore`) keep metadata alongside similarity scores.
- Validated in `tests/test_embeddings.py`.

### Metadata Persistence
- `pkm_ai.storage.SQLiteMetadataStore` is the durable option with constraints, indexes, and upserts.
- `pkm_ai.storage.JSONMetadataStore` is a lightweight alternative for quick experiments.
- Each exposes `upsert_document`, `replace_document_chunks`, `list_document_chunks` so you can swap backends without rewiring.
- See `tests/test_storage.py` for scenarios.

### Ingestion Pipeline
- `pkm_ai.pipeline.DocumentIngestionPipeline` threads ingestion, metadata, embeddings, and vector indexing together.
- Returns `IngestionResult` with the stored document plus the chunk records that made it into the vector DB.
- Exercised end-to-end in `tests/test_pipeline.py` via a stub embedder/vector store.

### Tiny Quickstart
```python
from pkm_ai import DocumentIngestionPipeline, SQLiteMetadataStore, build_vector_store

metadata_store = SQLiteMetadataStore("./data/metadata.db")
vector_store = build_vector_store("faiss", dim=384)  # match the embedding model dimension

pipeline = DocumentIngestionPipeline(metadata_store, vector_store)
result = pipeline.ingest_file("./docs/meeting-notes.md")
print(result.document)
print(len(result.chunks), "chunks indexed")
```

## Roadmap
### Phase 4 – AI Chat (next up)
- Plug in an LLM (OpenAI API, Ollama, or HuggingFace).
- Shape retrieval-augmented prompts anchored to stored chunks.
- Add regression tests for answer grounding and hallucination checks.

### Phase 5 – Streamlit Dashboard
- Drag-and-drop uploads with automatic ingestion.
- Document browser with previews and semantic search.
- Conversational panel powered by the retrieval+LLM stack.

### Phase 6 – Nice-to-haves
- Export polished summaries (PDF/Markdown).
- Highlight the source snippets in each answer.
- Optional user accounts / auth if sharing the workspace.

## Repo Layout
```
PKM AI/
├── docs/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── pkm_ai/
│       ├── __init__.py
│       ├── embeddings.py
│       ├── ingestion.py
│       ├── pipeline.py
│       └── storage.py
├── tests/
│   ├── conftest.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_storage.py
├── pytest.ini
└── .vscode/
    └── settings.json
```

## What’s Next
1. Flesh out retrieval queries so the LLM receives curated context for every question.
2. Wire in the preferred LLM provider and expose a FastAPI endpoint for chat.
3. Layer on the Streamlit experience so the workflow feels as friendly as the README.
