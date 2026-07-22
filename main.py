import json
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi_swagger import patch_fastapi
from pydantic import BaseModel, Field
from typing import List, Optional
# from openai import OpenAI
# from QdrantManagment import init_history_collection
# from config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT,connection_string
# from db import DatabaseConnection
# from log import log_message
# from prompts_config import SYSTEM_PROMPT, USER_PROMPT
# from qdrant_client import QdrantClient
# from qdrant_client import models
# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     VectorParams,
#     Distance,
#     SparseVectorParams,
#     SparseIndexParams,
#     Modifier
# )
from Models.mainModels import QueryRequest, QueryRequestWithHistory, QueryResponse, QueryResponseHistory, SearchResult
from OpenAIManagment import embed_query
from QdrantManagment import checkQudrant, hybrid_search, init_history_collection, search
from answerWithRAG import answer_stream_only_llm, answer_with_rag, answer_with_rag_stream, answer_with_rag_with_summary, answer_with_rag_withHistory, answer_with_rag_withHistoryAndVectorDB
from bm25 import PersianBM25Encoder
import uvicorn
# import os
from config import LLM_MODEL, OPENAI_API_KEY
from ingestion import ingest  
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi import BackgroundTasks
from typing import Tuple
import uuid

from providers.factory import create_provider
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


#qdrant = QdrantClient("http://localhost:6333")


#==============================Qdrant History
# ایجاد کالکشن تاریخچه اگر وجود نداشته باشد
init_history_collection()
from fastapi import BackgroundTasks


def search_documents(request: QueryRequest):
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

    return results, search_mode
def build_sources(results):
    sources = []

    for result in results:
        if isinstance(result, dict):
            payload = result["payload"]
            result_id = result["id"]
            score = result["score"]
        else:
            payload = result.payload
            result_id = result.id
            score = result.score

        text = payload.get("text", "")

        sources.append(
            SearchResult(
                id=str(result_id),
                score=score,
                text=text[:200] + ("..." if len(text) > 200 else ""),
                doc_id=payload.get("doc_id"),
                title=payload.get("title"),
                heading=payload.get("heading"),
                date=payload.get("date"),
                tags=payload.get("tags"),
                keywords=payload.get("keywords"),
                source_file=payload.get("source_file")
            )
        )

    return sources
 
sessions = {}

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
        # qdrant.get_collections()
        # return {
        #     "status": "healthy",
        #     "qdrant": "connected",
        #     "collection": COLLECTION_NAME
        # }
        checkQudrant()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
@app.post("/api/queryHistoryWithVectorDB", response_model=QueryResponseHistory)
async def api_queryHistoryWithVectorDB(request: QueryRequestWithHistory,
                         background_tasks: BackgroundTasks):
    try:
        query_vector = embed_query(request.query)

        if request.use_hybrid:
            results = hybrid_search(request.query, request.limit, request.filters)
            search_mode = "hybrid"
        else:
            results = search(query_vector, request.limit, request.filters)
            search_mode = "dense"

        rag_result = answer_with_rag_withHistoryAndVectorDB(
            query=request.query,
            results=results,
            background_tasks=background_tasks,
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
def api_querySummeryHistory(
    request: QueryRequestWithHistory,
    background_tasks: BackgroundTasks  # 1. Add BackgroundTasks to API inputs
):
    try:
        print("\n⏱️ === Operation Timing Start ===")
        start_total = time.time()

        start = time.time()
        query_vector = embed_query(request.query)
        print(f"1️⃣ First embedding generation time: {time.time() - start:.2f} seconds")

        start = time.time()
        if request.use_hybrid:
            results = hybrid_search(request.query, request.limit, request.filters)
            search_mode = "hybrid"
        else:
            results = search(query_vector, request.limit, request.filters)
            search_mode = "dense"
        print(f"2️⃣ Qdrant search execution time ({search_mode}): {time.time() - start:.2f} seconds")

        start = time.time()
        rag_result = answer_with_rag_with_summary(
            query=request.query,
            results=results,
            background_tasks=background_tasks,  # Pass background tasks manager
            temperature=request.temperature,
            conversation_id=request.conversation_id,
            user_key=getattr(request, "user_key", None)
        )
        print(f"3️⃣ RAG execution, SQL DB load + LLM time: {time.time() - start:.2f} seconds")

        start = time.time()
        sources = []
        for r in results:
            payload = getattr(r, "payload", {}) or {}
            sources.append({
                "id": str(getattr(r, "id", "")),
                "score": getattr(r, "score", None),
                "text": payload.get("text", "")[:200] + "...",
                "doc_id": payload.get("doc_id"),
                "title": payload.get("title"),
                "heading": payload.get("heading"),
                "date": payload.get("date"),
                "tags": payload.get("tags"),
                "keywords": payload.get("keywords"),
                "source_file": payload.get("source_file"),
            })
        print(f"4️⃣ Output sources formatting time: {time.time() - start:.2f} seconds")

        print(f"Total Time: {time.time() - start_total:.2f} seconds")
        print("⏱️ === Operation Timing End ===\n")

        return {
            "answer": rag_result["answer"],
            "sources": sources,
            "query": request.query,
            "search_mode": search_mode,
            "conversation_id": rag_result["conversation_id"]
        }

    except HTTPException as e:
        print(f"HTTPException Get! {e} ")
        raise
    except Exception as e:
        print(f"Exception Get2! {e} ")
        raise HTTPException(status_code=500, detail=f"Request processing failed: {str(e)}")


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

@app.post("/api/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    try:
        results, search_mode = search_documents(request)
        sources = build_sources(results)
        
        def event_stream():
            try:
                source_data = {
                    "query": request.query,
                    "search_mode": search_mode,
                    "sources": [
                        source.model_dump()
                        for source in sources
                    ]
                }

                yield (
                    "event: sources\n"
                    f"data: {json.dumps(source_data, ensure_ascii=False)}\n\n"
                )

                for chunk in answer_with_rag_stream(
                    query=request.query,
                    results=results,
                    temperature=request.temperature
                ):
                    chunk_data = {
                        "text": chunk
                    }

                    yield (
                        "event: token\n"
                        f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"
                    )

                yield "event: done\ndata: [DONE]\n\n"

            except Exception as error:
                error_data = {
                    "error": str(error)
                }

                yield (
                    "event: error\n"
                    f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"خطا در جست‌وجو: {str(error)}"
        )

@app.get('/api/ingestion')
async def get_ingestion():
    ingest()

class ChatRequest(BaseModel):
    provider_name: str
    base_uri: str | None = None
    api_key: str
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):

    provider = create_provider(
        provider_name="openai",
        base_uri="https://api.gapgpt.app/v1",
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
        auth_header_name="Authorization",
        auth_token_prefix="Bearer",
        api_path=""
    )

    def generate():
        chunks = []

        def on_chunk(text):
            chunks.append(text)

        provider.chat_stream(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,

            on_chunk=on_chunk
        )

        for chunk in chunks:
            yield chunk


    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )
class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.1


@app.post("/api/query/stream_test")
async def query_stream_test_endpoint(request: QueryRequest):
    try:
        def event_stream():
            try:
                yield (
                    "event: start\n"
                    f"data: {json.dumps({'message': 'stream started'}, ensure_ascii=False)}\n\n"
                )

                for chunk in answer_stream_only_llm(
                    query=request.query,
                    temperature=request.temperature
                ):
                    yield (
                        "event: token\n"
                        f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
                    )

                yield "event: done\ndata: [DONE]\n\n"

            except Exception as error:
                yield (
                    "event: error\n"
                    f"data: {json.dumps({'error': str(error)}, ensure_ascii=False)}\n\n"
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"خطا: {str(error)}"
        )
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
