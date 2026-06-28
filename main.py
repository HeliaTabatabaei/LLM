from fastapi import FastAPI, HTTPException
from fastapi_swagger import patch_fastapi
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT
from prompts_config import SYSTEM_PROMPT, USER_PROMPT
from qdrant_client import QdrantClient
from qdrant_client import models
from bm25 import PersianBM25Encoder
import uvicorn
import os
from ingestion import ingest  

app = FastAPI(
    docs_url=None,
    swagger_ui_oauth2_redirect_url=None,
    title="Adonis Docs Assistant API",
    description="RAG-based technical support API for Adonis technicians",
    version="1.0.0"
)
patch_fastapi(app, docs_url="/docs")

# Initialize clients
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://api.gapgpt.app/v1"
)

#qdrant = QdrantClient("http://localhost:6333")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

sparse_encoder = PersianBM25Encoder()
if os.path.exists("bm25_model.pkl"):
    sparse_encoder.load("bm25_model.pkl")
else:
    print("⚠️ bm25_model.pkl not found. Sparse search will be ineffective until ingestion is run.")



# Request/Response models
class SearchFilters(BaseModel):
    """فیلترهای metadata برای جستجو"""
    doc_ids: Optional[List[str]] = Field(None, description="فیلتر بر اساس doc_id")
    tags: Optional[List[str]] = Field(None, description="فیلتر بر اساس tags")
    keywords: Optional[List[str]] = Field(None, description="فیلتر بر اساس keywords")
    date_from: Optional[str] = Field(None, description="تاریخ شروع (ISO format)")
    date_to: Optional[str] = Field(None, description="تاریخ پایان (ISO format)")


class QueryRequest(BaseModel):
    query: str = Field(..., description="سوال فنی تکنسین", min_length=1)
    limit: Optional[int] = Field(5, description="تعداد نتایج جستجو", ge=1, le=20)
    temperature: Optional[float] = Field(0.1, description="دمای مدل", ge=0.0, le=1.0)
    use_hybrid: Optional[bool] = Field(True, description="استفاده از hybrid search (dense + sparse)")
    filters: Optional[SearchFilters] = Field(None, description="فیلترهای metadata")


class SearchResult(BaseModel):
    id: str
    score: float
    text: str
    doc_id: Optional[str] = None
    title: Optional[str] = None
    heading: Optional[str] = None
    date: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    source_file: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    search_mode: str  # "dense" یا "hybrid"


# Core functions
def embed_query(text: str) -> List[float]:
    """تبدیل متن به embedding vector"""
    res = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return res.data[0].embedding


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


def answer_with_rag(query: str, results, temperature: float = 0.1) -> str:
    """تولید پاسخ با RAG"""
    context = build_context(results)

    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(
        query=query,
        context=context
    )

    response = client.responses.create(
        model=LLM_MODEL,
        temperature=temperature,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    print(response.output_text)

# Get token usage
    print(response.usage)
    print("Input tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
    print("id:", response.id)
    return response.output_text

sessions = {}

def answer_with_rag(user_id: str, query: str, results, temperature: float = 0.1) -> str:
    context = build_context(results)

    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT.format(
        query=query,
        context=context
    )

    if user_id not in sessions:
        sessions[user_id] = []

    messages = [{"role": "system", "content": system_prompt}]
    messages += sessions[user_id]
    messages.append({"role": "user", "content": user_prompt})

    response = client.responses.create(
        model=LLM_MODEL,
        temperature=temperature,
        input=messages,
    )

    answer = response.output_text

    sessions[user_id].append({"role": "user", "content": query})
    sessions[user_id].append({"role": "assistant", "content": answer})

    sessions[user_id] = sessions[user_id][-20:]

    return answer
@app.get("/test")
def test():
    from openai import OpenAI

    client = OpenAI(
    api_key="sk-bVyAADjnBq7laj5jYKkhxFCA2W6iGhBC7dNUfdka0b99wiRw",
    base_url="https://api.gapgpt.app/v1"
    )

    response = client.responses.create(
    model="gapgpt-qwen-3.5",
    input="salam"
    )

# Get the text output
    print(response.output_text)

# Get token usage
    print(response.usage)
    print("Input tokens:", response.usage.input_tokens)
    print("Output tokens:", response.usage.output_tokens)
    print("Total tokens:", response.usage.total_tokens)
    print("Total tokens:", response.id)
# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Adonis Tech Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """بررسی سلامت سرویس"""
    try:
        qdrant.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    پردازش سوال فنی و بازگشت پاسخ RAG-based

    - **query**: سوال فنی تکنسین
    - **limit**: تعداد اسناد مرتبط (پیش‌فرض: 5)
    - **temperature**: دمای مدل (پیش‌فرض: 0.1)
    - **use_hybrid**: استفاده از hybrid search (dense + sparse)
    - **filters**: فیلترهای metadata (doc_ids, tags, keywords, date)
    """
    try:
        # انتخاب نوع جستجو
        if request.use_hybrid:
            results = hybrid_search(
                query=request.query,
                limit=request.limit,
                filters=request.filters
            )
            search_mode = "hybrid"
        else:
            query_vector = embed_query(request.query)
            results = search(
                query_vector=query_vector,
                limit=request.limit,
                filters=request.filters
            )
            search_mode = "dense"
  
        if not results:
            raise HTTPException(
                status_code=404,
                detail="هیچ سند مرتبطی یافت نشد"
            )

        # تولید پاسخ
        answer = answer_with_rag(request.query, results, temperature=request.temperature)

        # فرمت sources با metadata
        sources = []
        for r in results:

            if isinstance(r, dict):
                payload = r["payload"]
                rid = r["id"]
                score = r["score"]
            else:
                payload = r.payload
                rid = r.id
                score = r.score

            sources.append(
                SearchResult(
                    id=str(rid),
                    score=score,
                    text=payload.get("text", "")[:200] + "...",
                    doc_id=payload.get("doc_id"),
                    title=payload.get("title"),
                    heading=payload.get("heading"),
                    date=payload.get("date"),
                    tags=payload.get("tags"),
                    keywords=payload.get("keywords"),
                    source_file=payload.get("source_file")
                )
            )

        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            search_mode=search_mode
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"خطا در پردازش درخواست: {str(e)}"
        )


@app.get('/api/ingestion')
async def get_ingestion():
    ingest()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
