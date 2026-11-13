"""Load shared Factorio RAG settings from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from src.contracts import ProjectError


class Settings(TypedDict):
    CHUNKS_PATH: str
    QDRANT_PATH: str
    COLLECTION_NAME: str
    EMBEDDING_MODEL: str
    VECTOR_SIZE: int
    TOP_K: int
    MISTRAL_API_KEY: str | None
    MISTRAL_MODEL: str | None
    HISTORY_TURNS: int
    MAX_QUESTION_CHARS: int
    ANSWER_MAX_TOKENS: int
    REWRITE_MAX_TOKENS: int
    LLM_TIMEOUT_SECONDS: int


_INTEGER_DEFAULTS = {
    "VECTOR_SIZE": 384,
    "TOP_K": 4,
    "HISTORY_TURNS": 5,
    "MAX_QUESTION_CHARS": 1000,
    "ANSWER_MAX_TOKENS": 512,
    "REWRITE_MAX_TOKENS": 128,
    "LLM_TIMEOUT_SECONDS": 60,
}


def _optional_env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _positive_integer(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ProjectError(
            "CONFIGURATION", f"Настройка {name} должна быть целым числом."
        ) from exc
    if value <= 0:
        raise ProjectError(
            "CONFIGURATION", f"Настройка {name} должна быть больше нуля."
        )
    return value


def _absolute_path(name: str, default: str, project_root: Path) -> str:
    raw_value = os.getenv(name, default).strip()
    if not raw_value:
        raise ProjectError("CONFIGURATION", f"Настройка {name} не может быть пустой.")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_settings() -> Settings:
    """Return validated settings without creating database or LLM resources."""

    project_root = Path(__file__).resolve().parent.parent
    integer_values = {
        name: _positive_integer(name, default)
        for name, default in _INTEGER_DEFAULTS.items()
    }

    return {
        "CHUNKS_PATH": _absolute_path(
            "CHUNKS_PATH", "data/processed/chunks.jsonl", project_root
        ),
        "QDRANT_PATH": _absolute_path("QDRANT_PATH", "qdrant_db", project_root),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "factorio_wiki").strip()
        or "factorio_wiki",
        "EMBEDDING_MODEL": os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        ).strip()
        or "BAAI/bge-small-en-v1.5",
        "MISTRAL_API_KEY": _optional_env("MISTRAL_API_KEY"),
        "MISTRAL_MODEL": _optional_env("MISTRAL_MODEL"),
        **integer_values,
    }
