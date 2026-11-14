"""Check the retriever contract against temporary real Qdrant storage."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qdrant_client import QdrantClient, models

from src.config import load_settings
from src.contracts import ProjectError
from src.database.retriever import create_retriever


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.settings = load_settings()
        self.settings['QDRANT_PATH'] = str(Path(self.directory.name) / 'db')
        self.settings['EMBEDDING_MODEL'] = 'intfloat/multilingual-e5-large'
        self.settings['VECTOR_SIZE'] = 1024
        self.settings['COLLECTION_NAME'] = 'test_wiki'

    def seed(self, dimension=1024, with_point=True):
        client = QdrantClient(path=self.settings['QDRANT_PATH'])
        try:
            client.create_collection('test_wiki', vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE,
            ))
            if with_point:
                client.upsert('test_wiki', points=[models.PointStruct(
                    id=1, vector=[1.0] + [0.0] * (dimension - 1),
                    payload={'text': 'Iron makes steel.', 'title': 'Steel plate',
                             'section': 'Introduction', 'url': 'https://wiki.factorio.com/Steel_plate',
                             'section_url': 'https://wiki.factorio.com/Steel_plate#Introduction',
                             'game_variant': None},
                )])
        finally:
            client.close()

    def test_search_uses_e5_prefix_and_returns_contract_without_section_filter(self):
        self.seed()
        with patch('src.database.retriever.TextEmbedding') as model:
            model.return_value.embed.return_value = iter([np.array([1.0] + [0.0] * 1023)])
            retriever = create_retriever(self.settings)
            try:
                hits = retriever.search(' How is steel made? ', limit=1)
                model.return_value.embed.assert_called_once_with(['query: How is steel made?'])
                self.assertEqual(len(hits), 1)
                self.assertEqual(set(hits[0]), {'chunk_id', 'text', 'title', 'section', 'url', 'game_variant', 'score'})
                self.assertEqual(hits[0]['chunk_id'], '1')
                self.assertEqual(hits[0]['section'], 'Introduction')
                self.assertTrue(hits[0]['url'].endswith('#Introduction'))
            finally:
                retriever.close()
        # Closing must release the file lock for the next process/resource.
        client = QdrantClient(path=self.settings['QDRANT_PATH'])
        client.close()

    def test_missing_database_does_not_create_empty_storage(self):
        with self.assertRaises(ProjectError) as raised:
            create_retriever(self.settings)
        self.assertEqual(raised.exception.code, 'DATABASE_NOT_READY')
        self.assertFalse(Path(self.settings['QDRANT_PATH']).exists())

    def test_old_dimensions_rejected_before_model_load_and_lock_released(self):
        self.seed(dimension=384)
        with patch('src.database.retriever.TextEmbedding') as model:
            with self.assertRaises(ProjectError) as raised:
                create_retriever(self.settings)
            self.assertEqual(raised.exception.code, 'DATABASE_NOT_READY')
            model.assert_not_called()
        client = QdrantClient(path=self.settings['QDRANT_PATH'])
        client.close()

    def test_empty_collection_rejected_before_model_load(self):
        self.seed(with_point=False)
        with patch('src.database.retriever.TextEmbedding') as model:
            with self.assertRaises(ProjectError) as raised:
                create_retriever(self.settings)
            self.assertEqual(raised.exception.code, 'DATABASE_NOT_READY')
            model.assert_not_called()
