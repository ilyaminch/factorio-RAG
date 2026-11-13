"""Orchestrate query rewriting, retrieval, and answer generation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from src.config import Settings
from src.contracts import (
    ContextChunk,
    HistoryTurn,
    ProjectError,
    RagResult,
    SearchHit,
    Source,
)

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I could not find enough information in the available sources."


class Retriever(Protocol):
    def search(self, query: str, limit: int = 4) -> list[SearchHit]: ...


class LlmGateway(Protocol):
    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str: ...


GenerateAnswer = Callable[
    [str, list[HistoryTurn], list[ContextChunk], LlmGateway, Settings], str
]


def _error_result(error: ProjectError) -> RagResult:
    return {
        "status": "error",
        "answer": None,
        "sources": [],
        "search_query": None,
        "retrieved_chunks": [],
        "error": {"code": error.code, "message": error.message},
    }


def _validate_question(question: object, max_chars: int) -> str:
    if not isinstance(question, str) or not question.strip():
        raise ProjectError("INVALID_INPUT", "Вопрос не может быть пустым.")
    normalized = question.strip()
    if len(normalized) > max_chars:
        raise ProjectError(
            "INVALID_INPUT",
            f"Вопрос не должен превышать {max_chars} символов.",
        )
    return normalized


def _limited_history(history: object, limit: int) -> list[HistoryTurn]:
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes)):
        raise ProjectError("INVALID_INPUT", "История должна быть списком завершённых пар.")

    result: list[HistoryTurn] = []
    for turn in history[-limit:]:
        if not isinstance(turn, Mapping):
            raise ProjectError("INVALID_INPUT", "История содержит неверную запись.")
        if not isinstance(turn.get("question"), str) or not isinstance(
            turn.get("answer"), str
        ):
            raise ProjectError("INVALID_INPUT", "История содержит неверную запись.")
        sources = turn.get("sources")
        if not isinstance(sources, list):
            raise ProjectError("INVALID_INPUT", "История содержит неверную запись.")
        result.append(cast(HistoryTurn, dict(turn)))
    return result


def _rewrite_query(
    question: str,
    history: list[HistoryTurn],
    llm: LlmGateway,
    max_tokens: int,
) -> str:
    transcript = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
    )
    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user question as one standalone English search query "
                "for the Factorio Wiki. Resolve references using the dialogue history, but "
                "preserve an explicit topic change. Return only the query and do not answer it."
            ),
        },
        {
            "role": "user",
            "content": f"Dialogue history:\n{transcript}\n\nLatest question:\n{question}",
        },
    ]
    rewritten = llm.complete(messages, max_tokens=max_tokens)
    if not isinstance(rewritten, str) or not rewritten.strip():
        raise ProjectError(
            "LLM_RESPONSE", "Модель вернула пустой поисковый запрос."
        )
    return rewritten.strip()


def _number_hits(hits: object) -> tuple[list[ContextChunk], list[Source]]:
    if not isinstance(hits, list):
        raise ProjectError("INTERNAL_ERROR", "Поиск вернул данные неверного формата.")

    required = (
        "chunk_id",
        "text",
        "title",
        "section",
        "url",
        "game_variant",
        "score",
    )
    chunks: list[ContextChunk] = []
    sources: list[Source] = []
    for source_id, hit in enumerate(hits, start=1):
        if not isinstance(hit, Mapping) or any(key not in hit for key in required):
            raise ProjectError("INTERNAL_ERROR", "Поиск вернул данные неверного формата.")
        chunk = cast(ContextChunk, {**hit, "source_id": source_id})
        chunks.append(chunk)
        sources.append(
            {
                "source_id": source_id,
                "chunk_id": cast(str, hit["chunk_id"]),
                "title": cast(str, hit["title"]),
                "section": cast(str, hit["section"]),
                "url": cast(str, hit["url"]),
                "game_variant": cast(str | None, hit["game_variant"]),
            }
        )
    return chunks, sources


def answer(
    question: str,
    history: list[HistoryTurn],
    retriever: Retriever,
    llm: LlmGateway,
    settings: Settings,
    generate_fn: GenerateAnswer | None = None,
) -> RagResult:
    """Run the synchronous RAG chain and return a stable result dictionary."""

    try:
        clean_question = _validate_question(question, settings["MAX_QUESTION_CHARS"])
        recent_history = _limited_history(history, settings["HISTORY_TURNS"])
        search_query = (
            clean_question
            if not recent_history
            else _rewrite_query(
                clean_question,
                recent_history,
                llm,
                settings["REWRITE_MAX_TOKENS"],
            )
        )
        hits = retriever.search(search_query, limit=settings["TOP_K"])
        chunks, sources = _number_hits(hits)

        if not chunks:
            return {
                "status": "ok",
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "search_query": search_query,
                "retrieved_chunks": [],
                "error": None,
            }

        if generate_fn is None:
            from src.rag.generation import generate_answer

            generate_fn = cast(GenerateAnswer, generate_answer)

        generated_answer = generate_fn(
            clean_question, recent_history, chunks, llm, settings
        )
        if not isinstance(generated_answer, str) or not generated_answer.strip():
            raise ProjectError("LLM_RESPONSE", "Модель вернула пустой ответ.")

        return {
            "status": "ok",
            "answer": generated_answer.strip(),
            "sources": sources,
            "search_query": search_query,
            "retrieved_chunks": chunks,
            "error": None,
        }
    except ProjectError as exc:
        return _error_result(exc)
    except Exception:
        logger.exception("Unexpected RAG service failure")
        return _error_result(
            ProjectError("INTERNAL_ERROR", "Не удалось обработать вопрос.")
        )
