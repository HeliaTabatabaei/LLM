# ingestion.py
import json
import hashlib
from tqdm import tqdm
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from vectorstore import get_client, ensure_collection
from config import JSON_PATH, COLLECTION_NAME, BATCH_SIZE, OPENAI_API_KEY, EMBED_MODEL
from bm25 import PersianBM25Encoder
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.gapgpt.app/v1"
)


def embed_batch(texts):
    res = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in res.data]


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def already_indexed(client, hash_value):
    flt = Filter(must=[FieldCondition(key="text_hash", match=MatchValue(value=hash_value))])
    res = client.scroll(collection_name=COLLECTION_NAME, scroll_filter=flt, limit=1)
    return len(res[0]) > 0


def ingest():
    qdrant = get_client()
    ensure_collection(qdrant)

    # Initialize sparse encoder
    sparse_encoder = PersianBM25Encoder()

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # ساخت vocabulary از تمام chunks (یک بار)
    print("Building BM25 vocabulary...")
    all_texts = [c["embedding_text"].strip() for c in chunks if c["embedding_text"].strip()]
    sparse_encoder.build_vocab_from_texts(all_texts)
    print(f"Vocabulary size: {len(sparse_encoder.vocab)}")

    batch_texts, batch_points = [], []

    for chunk in tqdm(chunks, desc="Ingesting chunks"):
        text = chunk["embedding_text"].strip()
        if not text:
            continue

        h = text_hash(text)
        if already_indexed(qdrant, h):
            continue

        payload = {"text": text, "text_hash": h, **chunk.get("metadata", {})}
        batch_texts.append(text)
        batch_points.append((chunk["id"], payload))

        if len(batch_texts) >= BATCH_SIZE:
            flush_batch(qdrant, batch_texts, batch_points, sparse_encoder)
            batch_texts.clear()
            batch_points.clear()

    if batch_texts:
        flush_batch(qdrant, batch_texts, batch_points, sparse_encoder)

    # ✅ ذخیره مدل
    sparse_encoder.save("bm25_model.pkl")
    print("BM25 model saved")

    print("✅ Ingestion completed successfully")


def flush_batch(qdrant, batch_texts, batch_points, sparse_encoder):
    # Dense embeddings (OpenAI)
    dense_embeddings = embed_batch(batch_texts)

    # Sparse embeddings (BM25)
    sparse_embeddings = [sparse_encoder.encode_document(text) for text in batch_texts]

    # ساخت points با هر دو vector
    points = [
        PointStruct(
            id=i,  # chunk ID
            vector={
                "dense": dense_embeddings[i],
                "sparse": {
                    "indices": list(sparse_embeddings[i].keys()),
                    "values": list(sparse_embeddings[i].values())
                }
            },
            payload=batch_points[i][1]
        )
        for i in range(len(batch_texts))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


if __name__ == "__main__":
    ingest()
