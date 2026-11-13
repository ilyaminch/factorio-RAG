"""Минимальный интерфейс Factorio Wiki RAG."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from urllib.parse import urlparse

import streamlit as st


HISTORY_LIMIT_DEFAULT = 5
QUESTION_LIMIT_DEFAULT = 1000
_RESOURCE_LOCK = Lock()


class IntegrationUnavailable(RuntimeError):
    """Общие модули RAG ещё не подключены."""


@st.cache_resource(show_spinner=False)
def _load_resources() -> dict:
    try:
        from src.config import load_settings
        from src.database.retriever import create_retriever
        from src.rag.generation import create_llm
        from src.rag.service import answer
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("src."):
            raise IntegrationUnavailable(
                "Модули RAG ещё не подключены."
            ) from exc
        raise IntegrationUnavailable(
            f"Не установлена зависимость: {exc.name or 'неизвестный модуль'}."
        ) from exc

    settings = load_settings()
    return {
        "answer": answer,
        "history_limit": max(
            1, int(settings.get("HISTORY_TURNS", HISTORY_LIMIT_DEFAULT))
        ),
        "llm": create_llm(settings),
        "question_limit": int(
            settings.get("MAX_QUESTION_CHARS", QUESTION_LIMIT_DEFAULT)
        ),
        "retriever": create_retriever(settings),
        "settings": settings,
    }


def _init_state() -> None:
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "last_attempt" not in st.session_state:
        st.session_state.last_attempt = None


def _show_sources(sources: object) -> None:
    if not isinstance(sources, list) or not sources:
        st.write("Источники не найдены.")
        return

    for index, source in enumerate(sources):
        with st.container(border=True):
            if not isinstance(source, dict):
                st.write("Некорректный источник.")
                continue

            source_id = source.get("source_id", index + 1)
            title = str(source.get("title") or "Источник без названия")
            section = source.get("section")
            st.write(f"[{source_id}] {title}")
            if section:
                st.write(section)

            url = source.get("url")
            if (
                isinstance(url, str)
                and urlparse(url).scheme in {"http", "https"}
                and urlparse(url).netloc
            ):
                st.markdown(f"[Открыть источник]({url})")


def _show_turn(turn: dict, index: int) -> None:
    with st.container(border=True):
        st.write("Вопрос")
        st.write(turn["question"])
    with st.container(border=True):
        st.write("Ответ")
        st.write(turn["answer"])
    with st.container(border=True):
        st.write("Источники")
        _show_sources(turn.get("sources"))


def _error_message(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    message = getattr(exc, "message", None)
    if isinstance(code, str) and isinstance(message, str) and message.strip():
        return f"{code}: {message.strip()}"
    if isinstance(exc, IntegrationUnavailable):
        return str(exc)
    return "Не удалось получить ответ. Проверьте модули RAG и настройки."


def _valid_result(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("status") == "ok":
        return (
            isinstance(result.get("answer"), str)
            and bool(result["answer"].strip())
            and isinstance(result.get("sources"), list)
        )
    if result.get("status") == "error":
        error = result.get("error")
        return (
            isinstance(error, dict)
            and isinstance(error.get("code"), str)
            and isinstance(error.get("message"), str)
        )
    return False


def _submit_question(question: str) -> None:
    try:
        resources = _load_resources()
        history_limit = resources["history_limit"]
        history = deepcopy(st.session_state.conversation[-history_limit:])

        with _RESOURCE_LOCK:
            result = resources["answer"](
                question=question,
                history=history,
                retriever=resources["retriever"],
                llm=resources["llm"],
                settings=resources["settings"],
            )

        if not _valid_result(result):
            raise ValueError("Ответ RAG не соответствует общему формату.")
        if result["status"] == "error":
            error = result["error"]
            st.session_state.last_attempt = {
                "question": question,
                "message": f'{error["code"]}: {error["message"]}',
            }
            return

        st.session_state.conversation = (
            st.session_state.conversation
            + [
                {
                    "question": question,
                    "answer": result["answer"],
                    "sources": deepcopy(result["sources"]),
                }
            ]
        )[-history_limit:]
        st.session_state.last_attempt = None
    except Exception as exc:
        st.session_state.last_attempt = {
            "question": question,
            "message": _error_message(exc),
        }


def main() -> None:
    st.set_page_config(page_title="Factorio Wiki RAG")
    _init_state()

    st.title("Factorio Wiki RAG")

    if st.button("Очистить разговор"):
        st.session_state.conversation = []
        st.session_state.last_attempt = None
        st.rerun()

    if not st.session_state.conversation and not st.session_state.last_attempt:
        with st.container(border=True):
            st.write("Вопрос")
        with st.container(border=True):
            st.write("Ответ")
        with st.container(border=True):
            st.write("Источники")

    for index, turn in enumerate(st.session_state.conversation):
        _show_turn(turn, index)

    attempt = st.session_state.last_attempt
    if attempt:
        with st.container(border=True):
            st.write("Вопрос")
            st.write(attempt["question"])
        with st.container(border=True):
            st.write("Ответ")
            st.write(attempt["message"])

    question = st.chat_input("Введите вопрос")
    if question is not None:
        question = question.strip()
        if question:
            _submit_question(question)
        else:
            st.session_state.last_attempt = {
                "question": question,
                "message": "Введите вопрос.",
            }
        st.rerun()


if __name__ == "__main__":
    main()
