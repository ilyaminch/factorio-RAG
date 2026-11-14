"""Векторный поиск по Factorio Wiki с multilingual E5 large.

Запуск из корня: python -m src.database.query "How to make steel plates?"
Можно передать несколько запросов в кавычках: модель загрузится один раз.
"""

import sys

from src.config import load_settings
from src.database.retriever import create_retriever


retriever = create_retriever(load_settings())


queries = sys.argv[1:] or ['Recipe: steel plate']
try:
    for query in queries:
        print(f'\nЗапрос: {query}', flush=True)
        print('--- Результаты поиска ---')
        for rank, hit in enumerate(retriever.search(query, limit=5), start=1):
            print(f"\nМесто: {rank} | Сходство: {hit['score']:.4f}")
            print(f"Заголовок: {hit.get('title')}")
            print(f"Раздел: {hit.get('section')}")
            print(f"Текст: {hit.get('text')}")
            print(f"Источник: {hit.get('url')}")
finally:
    retriever.close()
