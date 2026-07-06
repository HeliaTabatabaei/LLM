from fastapi import FastAPI, HTTPException
from fastapi_swagger import patch_fastapi
from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI
from config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT,connection_string
from db import DatabaseConnection
from log import log_message
from prompts_config import SYSTEM_PROMPT, USER_PROMPT
from qdrant_client import QdrantClient
from qdrant_client import models
from bm25 import PersianBM25Encoder
import uvicorn
import os
from ingestion import ingest  
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from typing import Tuple
SQL_SERVER_CONNECTION_STRING =connection_string
import uuid

def normalize_conversation_id(conversation_id: Optional[str]) -> Tuple[str, bool]:
    """
    اگر conversation_id معتبر باشد:
        (conversation_id, False)
    اگر خالی/نامعتبر باشد:
        (new_uuid, True)
    """
    try:
        if conversation_id is None:
            raise ValueError

        conversation_id = str(conversation_id).strip()

        if conversation_id in ("", "undefined", "null", "None"):
            raise ValueError

        normalized = str(uuid.UUID(conversation_id))
        return normalized, False

    except (ValueError, TypeError, AttributeError):
        return str(uuid.uuid4()), True



app = FastAPI(
    docs_url=None,
    swagger_ui_oauth2_redirect_url=None,
    title="Adonis Docs Assistant API",
    description="RAG-based technical support API for Adonis technicians",
    version="1.0.0"
)
patch_fastapi(app, docs_url="/docs")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # برای محیط توسعه؛ در محیط پروداکشن دامنه‌های خود را مشخص کنید
    allow_credentials=True,
    allow_methods=["*"],  # اجازه به تمام متدها (POST, GET, OPTIONS و...)
    allow_headers=["*"],  # اجازه به تمام هدرها
)
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
class QueryRequestWithHistory(BaseModel):
    query: str
    limit: int = 5
    temperature: float = 0.1
    use_hybrid: bool = True
    conversation_id: Optional[str] = None
    user_key: Optional[str] = None
    filters: Optional[SearchFilters] = None


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
class QueryResponseHistory(BaseModel):
    answer: str
    sources: List[SearchResult]
    query: str
    search_mode: str
    conversation_id: Optional[str] = None  # اضافه شد

def save_conversation(cursor, conversation_id: str, title: str, user_key: Optional[str] = None, model_id: Optional[str] = None) -> str:
    # تبدیل و نرمال‌سازی
    conversation_guid = str(uuid.UUID(conversation_id))
    
    cursor.execute(
        """
        INSERT INTO dbo.Conversations (chatId, Title, Userkey, modelId, createDate)
        VALUES (?, ?, ?, ?, ?)
        """,
        (conversation_guid, title[:255], user_key, model_id, datetime.now())
    )
    return conversation_guid 

def save_message(cursor, conversation_id: str, role: str, content: str, provider_response_id: Optional[int] = None):
    cursor.execute(
        """
        INSERT INTO dbo.Messages (ConversationId, role, content, providerResponseId, createDate)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(uuid.UUID(conversation_id)),
            role,
            content,
            provider_response_id,
            datetime.now()
        )
    )
def get_conversation_history(cursor, conversation_id: str, limit: int = 6):
    cursor.execute(
        """
        SELECT TOP (?) role, content
        FROM (
            SELECT TOP (?) role, content, createDate, Id
            FROM dbo.Messages
            WHERE ConversationId = ?
            ORDER BY createDate DESC, Id DESC
        ) AS recent
        ORDER BY createDate ASC, Id ASC
        """,
        (limit, limit, str(uuid.UUID(conversation_id)))
    )
    rows = cursor.fetchall()
    return [{"role": row.role, "content": row.content} for row in rows]


def update_conversation_summary_task(conversation_id: str, new_user_msg: str, new_assistant_msg: str):
    """
    این تابع در پس‌زمینه اجرا می‌شود و خلاصه مکالمه را به‌روز می‌کند.
    """
    try:
        with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
            # ۱. دریافت خلاصه قبلی از دیتابیس
            cursor.execute("SELECT Summary FROM dbo.Conversations WHERE chatId = ?", (conversation_id,))
            row = cursor.fetchone()
            old_summary = row[0] if row and row[0] else "مکالمه به تازگی شروع شده است."

            # ۲. ساخت پرامپت برای مدل خلاصه‌ساز
            summary_prompt = f"""
            با توجه به خلاصه قبلی مکالمه و پیام‌های جدید مبادله شده، یک خلاصه کوتاه، جامع و به زبان فارسی از کل مکالمه تا این لحظه بنویس. 
            جزئیات فنی مهم (مانند نام ابزارها، پورت‌ها، خطاها یا تصمیمات کلیدی) را حفظ کن اما خلاصه را تا حد امکان فشرده نگه‌دار.

            خلاصه قبلی:
            {old_summary}

            پیام‌های جدید:
            کاربر: {new_user_msg}
            دستیار: {new_assistant_msg}

            خلاصه جدید به‌روزشده:
            """

            # ۳. فراخوانی مدل برای خلاصه‌سازی (یک مدل سبک‌تر و سریع‌تر ترجیح داده می‌شود)
            response = client.responses.create(
                model=LLM_MODEL,  # یا یک مدل سریع‌تر/ارزان‌تر
                input=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
            new_summary = response.output_text.strip()

            # ۴. ذخیره خلاصه جدید در دیتابیس
            cursor.execute(
                "UPDATE dbo.Conversations SET Summary = ? WHERE chatId = ?",
                (new_summary, conversation_id)
            )
            
    except Exception as e:
        # اینجا لاگ خطا را ثبت کنید تا در صورت بروز مشکل، جریان اصلی چت متوقف نشود
        log_message(f"Error updating summary for {conversation_id}: {str(e)}")

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
from fastapi import BackgroundTasks

def answer_with_rag_with_summary(
    query: str,
    results: List[Any],
    background_tasks: BackgroundTasks,
    temperature: float = 0.1,
    conversation_id: Optional[str] = None,
    user_key: Optional[str] = None
) -> Dict[str, Any]:
    
    # ۱. نرمال‌سازی شناسه مکالمه
    conversation_id, is_new_chat = normalize_conversation_id(conversation_id)
    context = build_context(results)
    current_summary = ""
    history = []

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        if not is_new_chat:
            cursor.execute(
                "SELECT 1 FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            if not cursor.fetchone():
                is_new_chat = True

        if is_new_chat:
            save_conversation(
                cursor=cursor,
                conversation_id=conversation_id,
                title=query,
                user_key=user_key,
                model_id=1
            )
        else:
            # خواندن خلاصه فعلی مکالمه
            cursor.execute(
                "SELECT Summary FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                current_summary = row[0]

        # دریافت چند پیام آخر
        history = get_conversation_history(
            cursor=cursor,
            conversation_id=conversation_id,
            limit=4
        )

        # ذخیره پیام جدید کاربر
        save_message(cursor, conversation_id, "user", query)

    # ۲. ساخت پرامپت نهایی
    prompt_content = USER_PROMPT.format(
        context=context,
        history=current_summary if current_summary else "سابقه قبلی وجود ندارد.",
        query=query
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": prompt_content}
    ]

    # ۳. دریافت پاسخ از مدل
    response = client.responses.create(
        model=LLM_MODEL,
        input=messages,
        temperature=temperature
    )

    answer = response.output_text
    response_id = getattr(response, "id", None)

    # ۴. ذخیره پاسخ دستیار
    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            provider_response_id=response_id
        )

    # ۵. خلاصه‌سازی در پس‌زمینه
    background_tasks.add_task(
        update_conversation_summary_task,
        conversation_id=conversation_id,
        new_user_msg=query,
        new_assistant_msg=answer
    )

    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "provider_response_id": response_id
    }

def answer_with_rag_withHistory(
    query: str,
    results: List[Any],
    temperature: float = 0.1,
    conversation_id: Optional[str] = None,
    user_key: Optional[str] = None
) -> Dict[str, Any]:
    conversation_id, is_new_chat = normalize_conversation_id(conversation_id)
    log_message("444444444444444444444444444444")
    log_message(conversation_id)
    context = build_context(results)

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        if not is_new_chat:
            cursor.execute(
                "SELECT 1 FROM dbo.Conversations WHERE chatId = ?",
                (conversation_id,)
            )
            if not cursor.fetchone():
                is_new_chat = True


        if is_new_chat:
            conversation_id=save_conversation(
                cursor=cursor,
                conversation_id=conversation_id,
                title=query,
                user_key=user_key,
                model_id=1
            )

        history = get_conversation_history(
            cursor=cursor,
            conversation_id=conversation_id,
            limit=6
        )

        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="user",
            content=query
        )
    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {
            "role": "user",
            "content": USER_PROMPT.format(
                context=context,
                query=query,
                history=history_text
                
            )
        }
    ]
    #log_message(messages)
    response = client.responses.create(
        model=LLM_MODEL,
        input=messages,
        temperature=temperature
    )

    answer = response.output_text
    response_id = getattr(response, "id", None)

    with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
        save_message(
            cursor=cursor,
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
            provider_response_id=response_id
        )
    print("*********************")
    print(conversation_id)
    return {
        "conversation_id": conversation_id,
        "answer": answer,
        "provider_response_id": response_id
    }

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

@app.get("/test11")
def test1():
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
@app.post("/api/queryHistory", response_model=QueryResponseHistory)
async def api_queryHistory(request: QueryRequestWithHistory):
    try:
        query_vector = embed_query(request.query)

        if request.use_hybrid:
            results = hybrid_search(request.query, request.limit, request.filters)
            search_mode = "hybrid"
        else:
            results = search(query_vector, request.limit, request.filters)
            search_mode = "dense"

        rag_result = answer_with_rag_withHistory(
            query=request.query,
            results=results,
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            user_key=getattr(request, "user_key", None)
        )

        sources = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            sources.append({
                "id": str(getattr(r, "id", "")),
                "score": getattr(r, "score", None),
                "text": payload.get("text"),
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "heading": payload.get("heading"),
                "date": payload.get("date"),
                "tags": payload.get("tags"),
                "keywords": payload.get("keywords"),
                "source_file": payload.get("source_file"),
            })

        return {
            "answer": rag_result["answer"],
            "sources": sources,
            "query": request.query,
            "search_mode": search_mode,
            "conversation_id": rag_result["conversation_id"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/api/queryHistory failed: {str(e)}")


@app.post("/api/querySummeryHistory", response_model=QueryResponseHistory)
async def api_querySummeryHistory(
    request: QueryRequestWithHistory, 
    background_tasks: BackgroundTasks  # ۱. اضافه کردن BackgroundTasks به ورودی API
):
    try:
        query_vector = embed_query(request.query)

        if request.use_hybrid:
            results = hybrid_search(request.query, request.limit, request.filters)
            search_mode = "hybrid"
        else:
            results = search(query_vector, request.limit, request.filters)
            search_mode = "dense"

        # ۲. ارسال background_tasks به تابع RAG (مطمئن شوید تابع را طبق پیام قبلی اصلاح کرده‌اید)
        rag_result = answer_with_rag_with_summary(
            query=request.query,
            results=results,
            background_tasks=background_tasks, # ارسال تسک منیجر
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            user_key=getattr(request, "user_key", None)
        )

        sources = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            sources.append({
                "id": str(getattr(r, "id", "")),
                "score": getattr(r, "score", None),
                "text": payload.get("text"),
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "heading": payload.get("heading"),
                "date": payload.get("date"),
                "tags": payload.get("tags"),
                "keywords": payload.get("keywords"),
                "source_file": payload.get("source_file"),
            })

        return {
            "answer": rag_result["answer"],
            "sources": sources,
            "query": request.query,
            "search_mode": search_mode,
            "conversation_id": rag_result["conversation_id"]
        }

    except HTTPException:
        raise
    except Exception as e:
        # لاگ کردن خطا برای دیباگ راحت‌تر
        print(f"Error in queryHistory: {e}") 
        raise HTTPException(status_code=500, detail=f"/api/queryHistory failed: {str(e)}")
   
#https://gapgpt.app/api/v1/get_chat/token/0c48c8c8-f054-420b-ae8f-070479e92789
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
