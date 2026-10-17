# Data pipeline

Collect English Factorio Wiki articles and prepare source-linked chunks for retrieval. Both scripts use functions; chunking is a separate function in `preprocess.py`.

| Script | Input | Output |
| --- | --- | --- |
| `parser.py` | MediaWiki API | `data/raw/<page_id>.json` |
| `preprocess.py` | Raw JSON files | `data/processed/pages/*.json` and `chunks.jsonl` |

Run from the repository root in the configured Python environment:

```powershell
python -m src.data_pipeline.parser
python -m src.data_pipeline.preprocess
```

## Collection

Requests and BeautifulSoup extract descriptions, infoboxes, recipes and article HTML from main and Tutorial pages, including Space Age. Translations, redirects and selected technical/archive subpages are excluded. Requests have a 0.75-second delay; existing JSON files are skipped on subsequent runs. Errors are printed and produce a nonzero exit code.

## Preprocessing

The script removes navigation, history and service content, labels table columns and separates list items. Base game and Space Age infobox tabs stay distinct. Recipe amounts and probabilities remain as displayed on the Wiki.

Chunks contain `id`, `text` and `metadata`: page ID, title, URL, revision, language, source file, section, chunk indices, section URL and game variant when explicit. Each text includes the article and section titles. UUIDs are deterministic for unchanged text and position.

`CHUNK_SIZE = 1600` limits **characters**, including headings; `OVERLAP = 200` targets overlap within a section. Splitting prefers paragraphs, then sentences and words. Both constants are defined in `preprocess.py`.

Before refreshing raw or changing preprocessing rules, use the root [cleanup scripts](../../README.md#cleanup). Processed JSONL is overwritten, but obsolete per-page files are not automatically removed. Pass the resulting chunks to [database](../database/README.md).

[Project overview](../../README.md)
