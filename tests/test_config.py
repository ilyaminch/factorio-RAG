from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import load_settings
from src.contracts import ProjectError


class LoadSettingsTests(unittest.TestCase):
    def test_defaults_have_expected_types_and_absolute_paths(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = load_settings()

        self.assertTrue(Path(settings["CHUNKS_PATH"]).is_absolute())
        self.assertTrue(Path(settings["QDRANT_PATH"]).is_absolute())
        self.assertEqual(settings["COLLECTION_NAME"], "factorio_wiki")
        self.assertEqual(settings["VECTOR_SIZE"], 384)
        self.assertEqual(settings["TOP_K"], 4)
        self.assertIsNone(settings["MISTRAL_API_KEY"])
        self.assertIsNone(settings["MISTRAL_MODEL"])
        self.assertIsNone(settings["MISTRAL_BASE_URL"])

    def test_environment_overrides_are_normalized(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CHUNKS_PATH": "custom/chunks.jsonl",
                "QDRANT_PATH": "custom/db",
                "TOP_K": "7",
                "MISTRAL_API_KEY": "  secret  ",
                "MISTRAL_MODEL": "  mistral-small  ",
                "MISTRAL_BASE_URL": "  https://example.com/v1  ",
            },
            clear=True,
        ):
            settings = load_settings()

        self.assertTrue(settings["CHUNKS_PATH"].endswith("custom\\chunks.jsonl"))
        self.assertTrue(settings["QDRANT_PATH"].endswith("custom\\db"))
        self.assertEqual(settings["TOP_K"], 7)
        self.assertEqual(settings["MISTRAL_API_KEY"], "secret")
        self.assertEqual(settings["MISTRAL_MODEL"], "mistral-small")
        self.assertEqual(settings["MISTRAL_BASE_URL"], "https://example.com/v1")

    def test_invalid_integer_raises_configuration_error(self) -> None:
        with patch.dict(os.environ, {"TOP_K": "many"}, clear=True):
            with self.assertRaises(ProjectError) as raised:
                load_settings()
        self.assertEqual(raised.exception.code, "CONFIGURATION")

    def test_non_positive_integer_raises_configuration_error(self) -> None:
        with patch.dict(os.environ, {"HISTORY_TURNS": "0"}, clear=True):
            with self.assertRaises(ProjectError) as raised:
                load_settings()
        self.assertEqual(raised.exception.code, "CONFIGURATION")


if __name__ == "__main__":
    unittest.main()
