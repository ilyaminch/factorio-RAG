from __future__ import annotations

import copy
import unittest
from typing import Any

from src.config import Settings
from src.contracts import HistoryTurn, ProjectError, SearchHit
from src.rag.service import NO_CONTEXT_ANSWER, answer


SETTINGS: Settings = {
    "CHUNKS_PATH": "unused",
    "QDRANT_PATH": "unused",
    "COLLECTION_NAME": "factorio_wiki",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "VECTOR_SIZE": 384,
    "TOP_K": 4,
    "MISTRAL_API_KEY": None,
    "MISTRAL_MODEL": None,
    "HISTORY_TURNS": 5,
    "MAX_QUESTION_CHARS": 1000,
    "ANSWER_MAX_TOKENS": 512,
    "REWRITE_MAX_TOKENS": 128,
    "LLM_TIMEOUT_SECONDS": 60,
}

HIT: SearchHit = {
    "chunk_id": "steel-1",
    "text": "Steel plate is smelted from iron plates.",
    "title": "Steel plate",
    "section": "Recipe",
    "url": "https://wiki.factorio.com/Steel_plate",
    "game_variant": None,
    "score": 0.9,
}


class FakeRetriever:
    def __init__(self, hits: list[SearchHit] | None = None) -> None:
        self.hits = [] if hits is None else hits
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int = 4) -> list[SearchHit]:
        self.calls.append((query, limit))
        return self.hits


class FakeLlm:
    def __init__(self, response: str = "How is steel plate made in Factorio?") -> None:
        self.response = response
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        self.calls.append((messages, max_tokens))
        return self.response


def history_turn(number: int) -> HistoryTurn:
    return {"question": f"Question {number}", "answer": f"Answer {number}", "sources": []}


class AnswerTests(unittest.TestCase):
    def test_without_history_searches_original_question_and_numbers_sources(self) -> None:
        retriever = FakeRetriever([HIT])
        llm = FakeLlm()
        generator_calls: list[tuple[Any, ...]] = []

        def generator(*args: Any) -> str:
            generator_calls.append(args)
            return "Steel plate is made in a furnace. [1]"

        result = answer(" What is steel plate? ", [], retriever, llm, SETTINGS, generator)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["search_query"], "What is steel plate?")
        self.assertEqual(retriever.calls, [("What is steel plate?", 4)])
        self.assertEqual(llm.calls, [])
        self.assertEqual(result["sources"][0]["source_id"], 1)
        self.assertEqual(result["retrieved_chunks"][0]["source_id"], 1)
        self.assertNotIn("text", result["sources"][0])
        self.assertEqual(generator_calls[0][0], "What is steel plate?")

    def test_history_is_rewritten_limited_and_not_mutated(self) -> None:
        history = [history_turn(number) for number in range(1, 7)]
        original = copy.deepcopy(history)
        retriever = FakeRetriever([HIT])
        llm = FakeLlm()
        received_history: list[HistoryTurn] = []

        def generator(
            question: str,
            recent_history: list[HistoryTurn],
            chunks: Any,
            gateway: Any,
            settings: Any,
        ) -> str:
            received_history.extend(recent_history)
            return "It is smelted from iron plates. [1]"

        result = answer("How is it made?", history, retriever, llm, SETTINGS, generator)

        self.assertEqual(result["search_query"], "How is steel plate made in Factorio?")
        self.assertEqual(retriever.calls[0], ("How is steel plate made in Factorio?", 4))
        self.assertEqual(llm.calls[0][1], 128)
        self.assertEqual([turn["question"] for turn in received_history], [f"Question {n}" for n in range(2, 7)])
        self.assertEqual(history, original)

    def test_empty_search_skips_generator(self) -> None:
        called = False

        def generator(*args: Any) -> str:
            nonlocal called
            called = True
            return "unexpected"

        result = answer("Unknown topic", [], FakeRetriever(), FakeLlm(), SETTINGS, generator)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["answer"], NO_CONTEXT_ANSWER)
        self.assertEqual(result["search_query"], "Unknown topic")
        self.assertEqual(result["sources"], [])
        self.assertFalse(called)

    def test_invalid_question_returns_invalid_input(self) -> None:
        result = answer("   ", [], FakeRetriever(), FakeLlm(), SETTINGS, lambda *args: "x")
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "INVALID_INPUT")  # type: ignore[index]

    def test_empty_rewrite_returns_llm_response_error(self) -> None:
        result = answer(
            "How?",
            [history_turn(1)],
            FakeRetriever([HIT]),
            FakeLlm("   "),
            SETTINGS,
            lambda *args: "x",
        )
        self.assertEqual(result["error"]["code"], "LLM_RESPONSE")  # type: ignore[index]
        self.assertIsNone(result["search_query"])

    def test_project_error_from_retriever_is_preserved(self) -> None:
        class BrokenRetriever:
            def search(self, query: str, limit: int = 4) -> list[SearchHit]:
                raise ProjectError("DATABASE_NOT_READY", "Database is not ready.")

        result = answer("Question", [], BrokenRetriever(), FakeLlm(), SETTINGS, lambda *args: "x")
        self.assertEqual(result["error"]["code"], "DATABASE_NOT_READY")  # type: ignore[index]

    def test_unexpected_error_is_hidden_as_internal_error(self) -> None:
        class BrokenRetriever:
            def search(self, query: str, limit: int = 4) -> list[SearchHit]:
                raise RuntimeError("sensitive details")

        result = answer("Question", [], BrokenRetriever(), FakeLlm(), SETTINGS, lambda *args: "x")
        self.assertEqual(result["error"]["code"], "INTERNAL_ERROR")  # type: ignore[index]
        self.assertNotIn("sensitive", result["error"]["message"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
