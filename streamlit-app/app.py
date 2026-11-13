"""Минимальный интерфейс Factorio Wiki RAG."""

from __future__ import annotations

from base64 import b64encode
from copy import deepcopy
import logging
import re
import traceback
from pathlib import Path
from threading import Lock
from urllib.parse import quote, urlparse

import streamlit as st


HISTORY_LIMIT_DEFAULT = 5
QUESTION_LIMIT_DEFAULT = 1000
_RESOURCE_LOCK = Lock()
_LOGGER = logging.getLogger(__name__)


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


def _show_sources(sources: object) -> None:
    if not isinstance(sources, list):
        return

    for index, source in enumerate(sources):
        with st.container(border=True, width="content"):
            if not isinstance(source, dict):
                st.write("Некорректный источник.")
                continue

            source_id = source.get("source_id", index + 1)
            title = str(source.get("title") or "Источник без названия")
            section = source.get("section")
            label = f"[{source_id}] {title}"
            label = re.sub(r"([\\`*_{}\[\]()<>#!|])", r"\\\1", label)
            url = source.get("url")
            if (
                isinstance(url, str)
                and urlparse(url).scheme in {"http", "https"}
                and urlparse(url).netloc
            ):
                label = f"[{label}]({quote(url, safe=':/?#[]@!$&%*+,;=~')})"
            if section:
                escaped_section = re.sub(
                    r"([\\`*_{}\[\]()<>#!|])", r"\\\1", str(section)
                )
                label += f" — {escaped_section}"
            st.markdown(label)


def _show_question(question: str) -> None:
    with st.container(horizontal=True, horizontal_alignment="right"):
        with st.container(border=True, width="content"):
            st.write(question)


def _show_turn(turn: dict) -> None:
    _show_question(turn["question"])
    if "error" in turn:
        st.write(turn["error"])
    else:
        st.write(turn["answer"])
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
    turn = {"question": question}
    try:
        if not question:
            raise IntegrationUnavailable("Введите вопрос.")
        resources = _load_resources()
        if len(question) > resources["question_limit"]:
            turn["error"] = (
                f'Вопрос слишком длинный. Максимум: {resources["question_limit"]} символов.'
            )
            st.session_state.conversation.append(turn)
            return
        history_limit = resources["history_limit"]
        successful_turns = [
            item for item in st.session_state.conversation if "answer" in item
        ]
        history = deepcopy(successful_turns[-history_limit:])

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
            turn["error"] = f'{error["code"]}: {error["message"]}'
        else:
            turn["answer"] = result["answer"]
            turn["sources"] = deepcopy(result["sources"])
    except Exception as exc:
        if not isinstance(exc, IntegrationUnavailable):
            location = traceback.extract_tb(exc.__traceback__)[-1]
            _LOGGER.error(
                "Ошибка обработки вопроса: %s (%s:%d, %s)",
                type(exc).__name__,
                Path(location.filename).name,
                location.lineno,
                location.name,
            )
        turn["error"] = _error_message(exc)
    st.session_state.conversation.append(turn)


@st.cache_data(show_spinner=False)
def _load_styles(file_versions: tuple[int, ...]) -> str:
    """Подготавливает локальные стили и шрифты один раз до изменения файлов."""
    app_dir = Path(__file__).parent
    styles = (app_dir / "style.css").read_text(encoding="utf-8")
    for name, filename in (
        ("FULL_AUTOMATION", "FullAutomation.otf"),
        ("EXO_2", "Exo2.ttf"),
    ):
        font = b64encode((app_dir / "fonts" / filename).read_bytes()).decode("ascii")
        styles = styles.replace(f"{{{{{name}}}}}", font)
    return styles


def main() -> None:
    st.set_page_config(page_title="Factorio Wiki")
    _init_state()

    app_dir = Path(__file__).parent
    style_files = (
        app_dir / "style.css",
        app_dir / "fonts" / "FullAutomation.otf",
        app_dir / "fonts" / "Exo2.ttf",
    )
    styles = _load_styles(tuple(path.stat().st_mtime_ns for path in style_files))
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    st.markdown(
        """
        <header class="factory-header">
          <div class="factory-heading">
            <h1>Factorio Wiki</h1>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )

    for turn in st.session_state.conversation:
        _show_turn(turn)

    question = st.chat_input("Введите вопрос")
    if question is not None:
        _show_question(question.strip())
        with st.spinner("Готовлю ответ…"):
            _submit_question(question.strip())
        st.rerun()


if __name__ == "__main__":
    main()
