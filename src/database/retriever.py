"""Поиск контекста в Qdrant для RAG-сервиса и Streamlit."""

import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from src.config import Settings
from src.contracts import ProjectError, SearchHit


ROOT = Path(__file__).resolve().parents[2]


class WikiRetriever:
    """Хранить одно подключение к базе и одну загруженную модель."""

    def __init__(self, settings: Settings):
        if settings['EMBEDDING_MODEL'] != 'intfloat/multilingual-e5-large' or settings['VECTOR_SIZE'] != 1024:
            raise ProjectError('CONFIGURATION', 'Для этой базы нужны multilingual E5 large и VECTOR_SIZE=1024.')
        if not Path(settings['QDRANT_PATH']).is_dir():
            raise ProjectError('DATABASE_NOT_READY', 'База Qdrant не найдена. Сначала выполните индексацию.')

        self.collection = settings['COLLECTION_NAME']
        self.client = None
        try:
            self.client = QdrantClient(path=settings['QDRANT_PATH'])
            if not self.client.collection_exists(self.collection):
                raise ProjectError('DATABASE_NOT_READY', 'Коллекция Qdrant не найдена.')
            params = self.client.get_collection(self.collection).config.params.vectors
            if not isinstance(params, models.VectorParams) or params.size != 1024 or params.distance != models.Distance.COSINE:
                raise ProjectError('DATABASE_NOT_READY', 'Нужна переиндексация: ожидаются векторы 1024 и метрика Cosine.')
            if self.client.count(self.collection, exact=True).count == 0:
                raise ProjectError('DATABASE_NOT_READY', 'Коллекция Qdrant пуста. Выполните индексацию.')
            # В контейнере используем настроенный кэш, локально — уже скачанные веса.
            self.model = TextEmbedding(
                model_name=settings['EMBEDDING_MODEL'],
                cache_dir=os.getenv('FASTEMBED_CACHE_PATH') or str(ROOT / 'data' / 'models'),
                threads=4,
            )
        except Exception as error:
            self.close()
            if isinstance(error, ProjectError):
                raise
            raise ProjectError('DATABASE_NOT_READY', 'Не удалось открыть базу или загрузить модель поиска.') from error

    def search(self, query: str, limit: int = 4) -> list[SearchHit]:
        """Найти чанки во всех разделах и привести их к контракту SearchHit.

        Args:
            query: Самостоятельный поисковый вопрос без служебного префикса.
            limit: Максимальное количество результатов.
        """
        if not isinstance(query, str) or not query.strip():
            raise ProjectError('INVALID_INPUT', 'Поисковый запрос не может быть пустым.')
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ProjectError('INVALID_INPUT', 'Лимит поиска должен быть положительным целым числом.')
        try:
            # Документы индексируются с passage:, вопросы — с query:.
            vector = next(iter(self.model.embed(['query: ' + query.strip()]))).tolist()
            hits = self.client.query_points(
                collection_name=self.collection, query=vector,
                limit=limit, with_payload=True,
            ).points
            return [{
                'chunk_id': str(hit.id),
                'text': hit.payload['text'],
                'title': hit.payload['title'],
                'section': hit.payload['section'],
                'url': hit.payload.get('section_url') or hit.payload['url'],
                'game_variant': hit.payload.get('game_variant'),
                'score': float(hit.score),
            } for hit in hits]
        except Exception as error:
            raise ProjectError('RETRIEVAL_ERROR', 'Не удалось выполнить поиск в Qdrant.') from error

    def close(self):
        """Освободить блокировку локальной базы после завершения работы."""
        if self.client is not None:
            self.client.close()
            self.client = None


def create_retriever(settings: Settings) -> WikiRetriever:
    """Создать retriever, совместимый с src.rag.service.answer."""
    return WikiRetriever(settings)
