# PKM AI

PKM AI is my personal knowledge hub powered by AI. Drop in PDFs, notes, or markdown files, and it indexes everything so future-you (or an LLM teammate) can instantly recall the good parts.

## Where Things Stand
- ✅ Phase 1 – Project scaffolding (src/tests/docs, dependencies, tooling)
- ✅ Phase 2 – Document ingestion (`load_document`, `split_text`, pytest coverage)
- ✅ Phase 3 – Embeddings, vector stores, and metadata pipeline (SentenceTransformers, FAISS/Chroma wrappers, persistence tests)
- ✅ Phase 4 – Retrieval-augmented chat (LLM hooks, prompts, coherence tests)
- ✅ Phase 5 – Streamlit dashboard (upload/ingest, document list, semantic search, chat pane)
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

### Launch the Dashboard
```bash
streamlit run src/pkm_ai/streamlit_app.py
```
By default the app stores SQLite metadata under `./data/metadata.db`. Override via `STREAMLIT_SECRETS` entry `DATA_DIR` if you want another path.

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
- Each exposes `upsert_document`, `replace_document_chunks`, `list_document_chunks`, `get_chunk` so you can swap backends without rewiring.
- See `tests/test_storage.py` for scenarios.

### Ingestion Pipeline
- `pkm_ai.pipeline.DocumentIngestionPipeline` threads ingestion, metadata, embeddings, and vector indexing together.
- Stores chunk text alongside metadata so downstream retrieval can respond without extra lookups.
- Returns `IngestionResult` with the stored document plus the chunk records that made it into the vector DB.
- Exercised end-to-end in `tests/test_pipeline.py` via a stub embedder/vector store.

### Retrieval-Augmented Chat
- `pkm_ai.chat.ChatEngine` wraps similarity search, prompt building, and LLM calls.
- Works with any callable LLM, or spin up a HuggingFace `transformers` pipeline by passing `huggingface_model="sentence-transformers/all-mpnet-base-v2"` (for example).
- Prompts instruct the assistant to stick to the retrieved context; missing answers trigger an "I do not know" response.
- Tests (`tests/test_chat.py`) verify prompt assembly, metadata fallbacks, and error handling.

### Streamlit Dashboard
- `src/pkm_ai/streamlit_app.py` bootstraps metadata, vector store, ingestion pipeline, and chat engine.
- Sidebar supports multi-file upload with automatic ingestion; feedback surfaces chunk counts per document.
- Main content splits into document list (preview first chunks), semantic search results, and a chat transcript with expandable context snippets.
- Under the hood the dashboard shares a session-level `AppState` (`src/pkm_ai/app_state.py`) so uploads, searches, and chats stay in sync. Tested in `tests/test_app_state.py`.

### Tiny Quickstart
```python
from pkm_ai import (
    AppState,
    ChatEngine,
    DocumentIngestionPipeline,
    SQLiteMetadataStore,
    build_vector_store,
)

metadata_store = SQLiteMetadataStore("./data/metadata.db")
vector_store = build_vector_store("faiss", dim=384)  # match the embedding model dimension

pipeline = DocumentIngestionPipeline(metadata_store, vector_store)
chat = ChatEngine(
    vector_store,
    metadata_store=metadata_store,
    llm=lambda prompt: "This is where the LLM answer would go.",
)

state = AppState(metadata_store=metadata_store, ingestion_pipeline=pipeline, chat_engine=chat)
state.ingest_file("./docs/meeting-notes.md")
print(state.chat("What were the action items?").answer)
```

## Roadmap
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
│       ├── app_state.py
│       ├── chat.py
│       ├── embeddings.py
│       ├── ingestion.py
│       ├── pipeline.py
│       ├── storage.py
│       └── streamlit_app.py
├── tests/
│   ├── conftest.py
│   ├── test_app_state.py
│   ├── test_chat.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_storage.py
├── pytest.ini
└── .vscode/
    └── settings.json
```

## What’s Next
1. Hook an actual LLM backend into the Streamlit chat (OpenAI, Ollama, HF) and expose FastAPI endpoints.
2. Build export/highlighting features so answers cite the exact snippets.
3. Add authentication if the PKM workspace needs to be shared.
