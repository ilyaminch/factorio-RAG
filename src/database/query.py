from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding

client = QdrantClient(path="./qdrant_db")
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
collection_name = "factorio_wiki"

query = "Recipe: steel plate"

query_embedding = list(embedding_model.embed([query]))[0].tolist()

query_filter = models.Filter(
    must=[
        models.FieldCondition(key="section", match=models.MatchValue(value="Characteristics"))
    ]
)

response = client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    query_filter=query_filter,
    limit=10,
    with_payload=True
)

print("\n--- Результаты поиска ---")
for hit in response.points:
    print(f"Сходство (Score): {hit.score:.4f}")
    print(f"Заголовок: {hit.payload.get('title')}")
    print(f"Раздел: {hit.payload.get('section')}")
    print(f"Текст: {hit.payload.get('text').replace(chr(10), ' ')}")
    print("-" * 50)

client.close()