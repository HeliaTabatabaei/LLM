from datetime import datetime
import hashlib
import os
from typing import List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    Distance
)
from Models.mainModels import SearchFilters
from bm25 import PersianBM25Encoder
from config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT
from OpenAIManagment import embed_query
COLLECTION_HISTORY = "chat_memory"
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

sparse_encoder = PersianBM25Encoder()
if os.path.exists("bm25_model.pkl"):
    sparse_encoder.load("bm25_model.pkl")
else:
    print("⚠️ bm25_model.pkl not found. Sparse search will be ineffective until ingestion is run.")
def init_history_collection():
    collections = qdrant.get_collections().collections
    exists = any(c.name == COLLECTION_HISTORY for c in collections)
    if not exists:
        qdrant.recreate_collection(
            collection_name=COLLECTION_HISTORY,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

# فراخوانی در ابتدای برنامه
init_history_collection()

def already_indexed(client, hash_value):
    flt = Filter(must=[FieldCondition(key="text_hash", match=MatchValue(value=hash_value))])
    res = client.scroll(collection_name=COLLECTION_HISTORY, scroll_filter=flt, limit=1)
    return len(res[0]) > 0

def save_message_to_qdrant(conversation_id, role, content, user_key=None):
    """ذخیره هر پیام در کیودرانت برای جستجوی معنایی در آینده"""
    try:
        hash=str(conversation_id)+":"+ content
        vector = embed_query(content) # از تابع موجود در کد خودت استفاده میکنیم
        h = text_hash(hash)
        if already_indexed(qdrant, h):
            raise ValueError("داده در وکتور استور وجود دارد ")
        qdrant.upsert(
            collection_name=COLLECTION_HISTORY,
            points=[{
                "id": str(uuid.uuid4()),
                "vector": vector,
                "payload": {
                    "conversation_id": str(conversation_id),
                    "user_key": user_key,
                    "role": role,
                    "content": content,
                    "text_hash":h,
                    "timestamp": datetime.now().isoformat()
                }
            }]
        )
    except Exception as e:
        print(f"Error saving to Qdrant history: {e}")

def search_chat_history(query_vector, conversation_id, limit=3):
    """جستجوی پیام‌های مرتبط قبلی در همین کانورسیشن"""
    try:
        hits = qdrant.query_points(
            collection_name=COLLECTION_HISTORY,
            query=query_vector,
            query_filter={
                "must": [
                    {"key": "conversation_id", "match": {"value": str(conversation_id)}}
                ]
            },
            limit=limit
        )
       
        # تبدیل نتایج به متن برای تزریق به پرامپت
        past_memories = "\n".join([f"{res.payload['role']}: {res.payload['content']}" for res in hits.points])
        print(past_memories)
        return past_memories
    except Exception as e:
        print(f"Error searching Qdrant history: {e}")
        return ""


def build_filter(filters: Optional[SearchFilters]) -> Optional[models.Filter]:
    """ساخت Qdrant filter از SearchFilters"""
    if not filters:
        return None

    conditions = []

    if filters.doc_ids:
        conditions.append(
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=filters.doc_ids)
            )
        )

    if filters.tags:
        conditions.append(
            models.FieldCondition(
                key="tags",
                match=models.MatchAny(any=filters.tags)
            )
        )

    if filters.keywords:
        conditions.append(
            models.FieldCondition(
                key="keywords",
                match=models.MatchAny(any=filters.keywords)
            )
        )

    if filters.date_from:
        conditions.append(
            models.FieldCondition(
                key="date",
                range=models.Range(gte=filters.date_from)
            )
        )

    if filters.date_to:
        conditions.append(
            models.FieldCondition(
                key="date",
                range=models.Range(lte=filters.date_to)
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)
def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def search(query_vector: List[float], limit: int = 5, filters: Optional[SearchFilters] = None):
    """جستجوی dense معمولی"""
    query_filter = build_filter(filters)

    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        using="dense",  # using dense vector
        limit=limit,
        query_filter=query_filter
    )
    return hits.points


def hybrid_search(
        query: str,
        limit: int = 5,
        filters: Optional[SearchFilters] = None
):
    """جستجوی hybrid (dense + sparse با RRF fusion)"""

    # 1. Dense embedding
    dense_vector = embed_query(query)

    # 2. Sparse vector (BM25)
    sparse_vector = sparse_encoder.encode_query(query)
    sparse_indices = list(sparse_vector.keys())
    sparse_values = list(sparse_vector.values())

    # 3. Build filter
    query_filter = build_filter(filters)

    prefetch = [
        models.Prefetch(
            query=dense_vector,
            using="dense",
            limit=limit * 2,
        )
    ]

    if sparse_indices:
        prefetch.append(
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                ),
                using="sparse",
                limit=limit * 2,
            )
        )

    # 4. Hybrid query با RRF fusion
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF
        ),
        limit=limit,
        query_filter=query_filter
    )

    return hits.points


def build_context(results) -> str:

    chunks = []

    for i, r in enumerate(results, start=1):

        if isinstance(r, dict):
            payload = r["payload"]
        else:
            payload = r.payload

        text = payload.get("text", "")
        doc_id = payload.get("doc_id", "نامشخص")
        title = payload.get("title", "")
        heading = payload.get("heading", "")

        header = f"[سند {i}"

        if doc_id:
            header += f" - {doc_id}"

        if title:
            header += f" - {title}"

        if heading:
            header += f" > {heading}"

        header += "]"

        chunks.append(f"{header}\n{text}")

    return "\n\n".join(chunks)
def checkQudrant():    
    
        qdrant.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collection": COLLECTION_NAME
        }
    