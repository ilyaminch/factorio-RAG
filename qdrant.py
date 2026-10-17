import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from fastembed import TextEmbedding

client = QdrantClient(path="./qdrant_db")

model_name = "BAAI/bge-small-en-v1.5"
embedding_model = TextEmbedding(model_name=model_name)
vector_size = 384 

collection_name = "factorio_wiki"

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print(f"Коллекция '{collection_name}' успешно создана.")

ROOT = Path(__file__).parent
jsonl_file_path = ROOT / 'data' / 'processed' / 'chunks.jsonl'
batch_size = 100

records_batch = []
texts_batch = []

def process_batch(records, texts):
    embeddings = list(embedding_model.embed(texts))
    
    points = []
    for record, embedding in zip(records, embeddings):
        payload = {
            "text": record["text"],
            **record["metadata"]
        }
        
        points.append(PointStruct(
            id=record["id"],
            vector=embedding.tolist(),
            payload=payload
        ))
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"Загружено {len(points)} записей.")

with open(jsonl_file_path, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        records_batch.append(record)
        texts_batch.append(record["text"])
        
        if len(records_batch) >= batch_size:
            process_batch(records_batch, texts_batch)
            records_batch = []
            texts_batch = []

if records_batch:
    process_batch(records_batch, texts_batch)

print('все!')