"""Индексация чанков Factorio Wiki в локальный Qdrant с multilingual E5 large.

Запуск из корня: python -m src.database.index.
Модель использует векторы размерности 1024 и префикс passage: для документов.
"""

import json
import os
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import load_settings


ROOT = Path(__file__).resolve().parents[2]
settings = load_settings()
jsonl_file_path = Path(settings['CHUNKS_PATH'])
model_name = settings['EMBEDDING_MODEL']
vector_size = settings['VECTOR_SIZE']
collection_name = settings['COLLECTION_NAME']
if model_name != 'intfloat/multilingual-e5-large' or vector_size != 1024:
    raise SystemExit('Для индексации нужны multilingual E5 large и VECTOR_SIZE=1024.')
batch_size = 100  # Количество точек в одном запросе к Qdrant.
embedding_batch_size = 4  # Небольшой пакет ограничивает память large-модели.

if not jsonl_file_path.is_file():
    raise SystemExit('Сначала запустите python -m src.data_pipeline.preprocess')

client = QdrantClient(path=settings['QDRANT_PATH'])
if client.collection_exists(collection_name):
    # Векторы старой small-модели (384) нельзя смешивать с E5 (1024).
    params = client.get_collection(collection_name).config.params.vectors
    if not isinstance(params, VectorParams) or params.size != vector_size or params.distance != Distance.COSINE:
        client.close()
        raise SystemExit('Коллекция несовместима с E5 large. Сохраните старую qdrant_db в резервную папку и выполните полную индексацию.')
else:
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Коллекция '{collection_name}' успешно создана.", flush=True)

# Кэш сохраняется внутри data/, поэтому веса модели не попадают в Git.
embedding_model = TextEmbedding(
    model_name=model_name,
    cache_dir=os.getenv('FASTEMBED_CACHE_PATH') or str(ROOT / 'data' / 'models'),
    threads=4,
)


def process_batch(records, texts):
    """Вычислить эмбеддинги и загрузить один пакет чанков.

    Args:
        records: Записи id, text и metadata из chunks.jsonl.
        texts: Тексты соответствующих записей без служебных префиксов.
    """
    # E5 требует passage: и для английских, и для русских документов.
    embeddings = list(embedding_model.embed(
        ['passage: ' + text for text in texts], batch_size=embedding_batch_size,
    ))
    points = []
    for record, embedding in zip(records, embeddings):
        # В payload хранится исходный текст; префикс нужен только модели.
        payload = {'text': record['text'], **record['metadata']}
        points.append(PointStruct(
            id=record['id'], vector=embedding.tolist(), payload=payload,
        ))
    client.upsert(collection_name=collection_name, points=points)
    print(f'Загружено {len(points)} записей.', flush=True)


records_batch = []
texts_batch = []
total = 0
try:
    with jsonl_file_path.open(encoding='utf-8') as file:
        for line in file:
            record = json.loads(line)
            records_batch.append(record)
            texts_batch.append(record['text'])
            if len(records_batch) >= batch_size:
                process_batch(records_batch, texts_batch)
                total += len(records_batch)
                records_batch = []
                texts_batch = []
    # Последний пакет может содержать меньше batch_size записей.
    if records_batch:
        process_batch(records_batch, texts_batch)
        total += len(records_batch)
    print(f'Готово: загружено {total} чанков.', flush=True)
finally:
    client.close()
