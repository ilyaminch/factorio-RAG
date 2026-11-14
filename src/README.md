# Source modules

| Module | Scripts | Purpose |
| --- | --- | --- |
| [data_pipeline](data_pipeline/README.md) | `parser.py`, `preprocess.py` | Collect Wiki pages, clean content and create chunks |
| [database](database/README.md) | `index.py`, `query.py`, `retriever.py` | Generate embeddings, store them in Qdrant and search |
| rag | `service.py`, `generation.py` | Rewrite follow-ups using history and generate cited answers |

Shared settings live in `config.py`; payload and history contracts live in `contracts.py`.

The pipeline is `Wiki → data/raw/ → data/processed/chunks.jsonl → qdrant_db/ → retrieval → LLM + history → Streamlit`. Collection and preprocessing share one folder; cleanup scripts live at the repository root.

Run Python scripts from the repository root using `python -m src.<module>.<script>` in the configured environment. Default paths resolve relative to the project root; database scripts use shared environment settings. `__init__.py` files mark Python packages; the former `qdrant.py` is now `database/index.py`.

[Setup and complete workflow](../README.md)
