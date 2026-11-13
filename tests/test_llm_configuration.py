from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config import load_settings
from src.rag.generation import create_llm


class LlmConfigurationTests(unittest.TestCase):
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
