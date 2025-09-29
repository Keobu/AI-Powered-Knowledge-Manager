# PKM AI

Personal Knowledge Management con AI per raccogliere, indicizzare e interrogare documenti personali con intelligenza artificiale.

## Stato Avanzamento
- [x] Fase 1 – Setup iniziale (struttura repo, dipendenze, README)
- [x] Fase 2 – Document ingestion (`load_document`, `split_text`, test pytest)
- [ ] Fase 3 – Embedding & Vector DB
- [ ] Fase 4 – AI Chat
- [ ] Fase 5 – Dashboard Streamlit
- [ ] Fase 6 – Extra features

## Obiettivi
- Caricare documenti (PDF, TXT, MD) e organizzarli in un archivio consultabile.
- Estrarre informazioni chiave con modelli NLP.
- Creare embedding vettoriali per la ricerca semantica.
- Fornire una chat AI che risponde basandosi sui documenti caricati.
- Visualizzare il tutto tramite una dashboard Streamlit.
- Esportare note sintetiche in PDF/Markdown.

## Setup Ambiente
1. Creare un virtual environment Python (>=3.10) e attivarlo.
2. Installare le dipendenze core: `pip install -r requirements.txt`.
3. (Opzionale) Installare il pacchetto in editable mode per avere `pkm_ai` disponibile ovunque: `pip install -e .`.

## Test
- Eseguire l'intera suite: `pytest`

## Document Ingestion
- `pkm_ai.ingestion.load_document(path)` gestisce PDF, TXT e MD, restituendo contenuto e metadati (percorso, estensione, numero caratteri).
- `pkm_ai.ingestion.split_text(text, chunk_size, overlap)` suddivide il testo in chunk sovrapposti per alimentare pipeline NLP successive.
- La copertura è validata con `tests/test_ingestion.py`, incluse le condizioni di errore e i parametri non validi.

## Roadmap Dettagliata
### Fase 1 – Setup ✅
- Struttura base del progetto (`src/`, `tests/`, `docs/`).
- `requirements.txt` con dipendenze principali.
- `pyproject.toml` per configurare il package e integrare strumenti (pytest, IDE).

### Fase 2 – Document Ingestion ✅
- Funzioni `load_document()` e `split_text()` pubblicate nel package.
- Gestione robusta degli errori (file mancanti, formati non supportati, PDF senza dipendenze).
- Test unitari per tutti i percorsi principali.

### Fase 3 – Embedding & Vector DB
- Funzione `create_embeddings()` con SentenceTransformers.
- Integrazione con FAISS o ChromaDB per la persistenza.
- Ricerca semantica su input utente.

### Fase 4 – AI Chat
- Integrazione con un modello LLM (OpenAI API, Ollama locale, HuggingFace).
- Prompt engineering per garantire risposte basate sui documenti.
- Test di coerenza delle risposte.

### Fase 5 – Dashboard Streamlit
- Upload file con ingestione automatica.
- Lista documenti + anteprima contenuti.
- Barra di ricerca semantica.
- Chat AI interattiva.

### Fase 6 – Extra Features
- Esportazione di note sintetiche.
- Evidenziazione dei passaggi di origine nelle risposte.
- Supporto multi-utente/autenticazione (opzionale).

## Struttura Cartelle
```
PKM AI/
├── docs/
├── pyproject.toml
├── requirements.txt
├── src/
│   └── pkm_ai/
│       ├── __init__.py
│       └── ingestion.py
├── tests/
│   ├── conftest.py
│   └── test_ingestion.py
└── pytest.ini
```

## Prossimi passi
1. Definire lo schema per i metadati (SQLite/JSON) e collegarlo al loader.
2. Implementare la pipeline di embedding con FAISS/ChromaDB (Fase 3).
3. Progettare test per la persistenza e la ricerca semantica.
