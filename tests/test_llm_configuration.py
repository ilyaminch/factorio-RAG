from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from src.config import load_settings
from src.rag.generation import create_llm, generate_answer


class LlmConfigurationTests(unittest.TestCase):
    def test_generation_uses_configured_history_window(self) -> None:
        llm = Mock()
        llm.complete.return_value = 'Answer [1]'
        history = [{'question': f'Q{i}', 'answer': f'A{i}', 'sources': []} for i in range(4)]
        generate_answer(
            'Follow-up', history,
            [{'source_id': 1, 'title': 'Steel plate', 'text': 'Recipe'}],
            llm, {'HISTORY_TURNS': 2, 'ANSWER_MAX_TOKENS': 100},
        )
        messages = llm.complete.call_args.kwargs['messages']
        self.assertEqual([m['content'] for m in messages[1:-1]], ['Q2', 'A2', 'Q3', 'A3'])
        self.assertEqual(llm.complete.call_args.kwargs['max_tokens'], 100)

    def test_environment_settings_reach_openai_client(self) -> None:
        environment = {
            "MISTRAL_API_KEY": "test-key",
            "MISTRAL_MODEL": "gpt-6-luna",
            "MISTRAL_BASE_URL": "https://panel.safronspace.ru/v1",
        }
        with patch.dict(os.environ, environment, clear=True):
            with patch("src.rag.generation.OpenAI") as client:
                gateway = create_llm(load_settings())

        self.assertEqual(gateway.model, "gpt-6-luna")
        client.assert_called_once_with(
            api_key="test-key",
            base_url="https://panel.safronspace.ru/v1",
            timeout=60.0,
        )


if __name__ == "__main__":
    unittest.main()
