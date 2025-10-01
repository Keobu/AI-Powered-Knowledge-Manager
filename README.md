# PKM AI

Welcome to **PKM AI**, the personal knowledge sidekick that helps you tame PDFs, notes, and Markdown files. Drop in documents, let the app chew through them with embeddings, and chat with an LLM that only answers with information pulled from your own library.

## Sneak Peek
Here’s a glimpse of the desktop app powered by CustomTkinter:

![Dashboard](docs/assets/dashboard.png)
![Search & Chat](docs/assets/search_chat.png)
![Settings](docs/assets/settings.png)

## What You Get
- **Document ingestion** for PDF/TXT/MD with automatic chunking and metadata tracking.
- **Semantic search + RAG chat** backed by FAISS/Chroma and SentenceTransformers.
- **Desktop GUI** (dark mode!) with tabs for uploads, chat, configuration, and exports.
- **Export helpers** to turn AI answers into Markdown, PDF, or JSON digests.
- **Streamlit dashboard** and pytest suite for the web & testing crowd.

## Getting Started Quickly
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install customtkinter python-dotenv reportlab
```

### Launch the Desktop App
```bash
python3 -m pkm_ai.gui
```
The GUI lives in `src/pkm_ai/gui.py`. It ships with a placeholder LLM response—hook it up to a real provider (see below) for the full experience.

### Launch the Streamlit Dashboard
```bash
streamlit run src/pkm_ai/streamlit_app.py
```

## Hooking Up the LLM (Ollama Recommended)
For local, privacy-friendly answers, install [Ollama](https://ollama.ai/download) and pull a model such as `llama3`:
```bash
ollama run llama3
```
Then adjust the chat engine in `gui.py` (and/or `chat.py`) to call Ollama’s HTTP endpoint. The GUI is already wired with placeholders—swap the lambda for a small client that posts to `http://localhost:11434/api/generate` and you’re set. Feel free to point to OpenAI or HuggingFace instead if you prefer the cloud.

Store API keys in `.env` (use the GUI Settings tab or edit the file manually). The app reads `OPENAI_API_KEY` and `HUGGINGFACEHUB_API_TOKEN` out of the box.

## Behind the Scenes
- **Embeddings & Vector DB**: SentenceTransformers models with FAISS or Chroma, configurable via the settings tab.
- **Metadata persistence**: SQLite (default) or JSON for quick experiments.
- **Pipeline**: `DocumentIngestionPipeline` wires ingestion, chunking, embeddings, and vector persistence.
- **Chat engine**: Retrieval-augmented prompts with pluggable LLM backends.
- **Exports**: Markdown/PDF/JSON via `pkm_ai.export`.
- **Tests**: `pytest` covers ingestion, embeddings, chat, storage, export, and GUI state helpers.

## Project Layout
```
PKM AI/
├── docs/
│   └── assets/            # screenshots for the README
├── src/
│   └── pkm_ai/
│       ├── app_state.py
│       ├── chat.py
│       ├── embeddings.py
│       ├── export.py
│       ├── gui.py         # CustomTkinter desktop app
│       ├── ingestion.py
│       ├── pipeline.py
│       ├── storage.py
│       └── streamlit_app.py
├── tests/
│   └── …                  # pytest coverage for the core modules
├── requirements.txt
└── README.md
```

## Roadmap & Nice-to-haves
- Swap the LLM placeholder in the GUI with an Ollama client by default.
- Surface export buttons directly inside the Streamlit dashboard.
- Highlight answer sources inline (both GUI and Streamlit).
- Add optional workspace auth/multi-user support before sharing with teammates.

Enjoy building out your personal knowledge garden! If you run into issues, start by checking the `.env` credentials, the Ollama service status, and the vector store dimension (`384`) to make sure everything lines up.
