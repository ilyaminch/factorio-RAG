# Streamlit chat

A Factorio Wiki chat with cited sources and session memory. Successful question/answer pairs are stored per session; only the latest `HISTORY_TURNS` pairs are passed to RAG. Failed turns are displayed but excluded from model history. Conversations are not saved after the session ends.

## Run

Build the Qdrant database first, then run from the repository root:

```powershell
python -m pip install -r deploy/requirements.txt
$env:MISTRAL_API_KEY = '<private API key>'
$env:MISTRAL_MODEL = '<model ID>'
$env:MISTRAL_BASE_URL = '<compatible API base URL>'
python -m streamlit run streamlit-app/app.py
```

Omit `MISTRAL_BASE_URL` when using the provider's default endpoint. Local `.env` files are not loaded automatically. Open the local URL printed by Streamlit; stop with Ctrl+C. Never put real credentials into tracked files.

## Integration

The interface loads `src.config.load_settings`, `src.database.retriever.create_retriever`, `src.rag.generation.create_llm` and `src.rag.service.answer`. Follow-ups are rewritten into standalone search queries using history; retrieved chunks supply the answer context and numbered sources.

Defaults: local Qdrant, collection `factorio_wiki`, multilingual E5 large, 1024 dimensions, `TOP_K=4`, `HISTORY_TURNS=5`. Override `QDRANT_PATH` for another storage location. The local database can be opened by one process at a time; stop indexing/CLI search before starting the app.

[Project setup](../README.md) · [Database settings and retrieval results](../src/database/README.md)
