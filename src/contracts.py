"""Shared data contracts and errors for the Factorio RAG pipeline."""

from __future__ import annotations

from typing import Literal, TypedDict


class SearchHit(TypedDict):
    chunk_id: str
    text: str
    title: str
    section: str
    url: str
    game_variant: str | None
    score: float


class ContextChunk(SearchHit):
    source_id: int


class Source(TypedDict):
    source_id: int
    chunk_id: str
    title: str
    section: str
    url: str
    game_variant: str | None


class HistoryTurn(TypedDict):
    question: str
    answer: str
    sources: list[Source]


class RagError(TypedDict):
    code: str
    message: str


class RagResult(TypedDict):
    status: Literal["ok", "error"]
    answer: str | None
    sources: list[Source]
    search_query: str | None
    retrieved_chunks: list[ContextChunk]
    error: RagError | None


class ProjectError(Exception):
    """Expected project failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)
