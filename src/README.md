# Source modules

| Module | Scripts | Purpose |
| --- | --- | --- |
| [data_pipeline](data_pipeline/README.md) | `parser.py`, `preprocess.py` | Collect Wiki pages, clean content and create chunks |
| [database](database/README.md) | `index.py`, `query.py` | Generate embeddings, store them in Qdrant and search |

The pipeline is `Wiki → data/raw/ → data/processed/chunks.jsonl → qdrant_db/ → search`. Collection and preprocessing share one folder; cleanup scripts live at the repository root.

Run Python scripts from the repository root using `python -m src.<module>.<script>` in the configured environment. Data pipeline paths and the indexer's input resolve relative to the project root; database storage uses `./qdrant_db` relative to the working directory. `__init__.py` files mark Python packages; the former `qdrant.py` is now `database/index.py`.

[Setup and complete workflow](../README.md)
