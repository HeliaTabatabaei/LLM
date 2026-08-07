from __future__ import annotations

import json
import os
from queue import Queue
from threading import Thread
from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from Models.mainModels import QueryRequestStream
from SQlDB.wallet import InsertIntoWallet
from config import QDRANT_HOST, QDRANT_PORT

from providers.factory import create_provider

from agent.chat_agent import ChatAgent
from agent.document_agent import DocumentAgent
from agent.router import RouterAgent

# اگر RAGService در پروژه‌ات در فایل دیگری است،
# فقط همین import را اصلاح کن.
from service.rag_service import RAGService


router = APIRouter(
    prefix="/api",
    tags=["query"],
)


STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def build_router_agent() -> RouterAgent:
    """
    ساخت کامل Provider، Qdrant، RAGService و Agentها.
    """

    provider = create_provider(
        provider_name="openai",
        base_uri="https://api.gapgpt.app/v1",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("LLM_MODEL", ""),
        embed_model=os.getenv("EMBED_MODEL", ""),
    )

    if not QDRANT_HOST:
        raise RuntimeError(
            "QDRANT_HOST is not configured"
        )

    if not QDRANT_PORT:
        raise RuntimeError(
            "QDRANT_PORT is not configured"
        )

    qdrant_client = QdrantClient(
        host=QDRANT_HOST,
        port=int(QDRANT_PORT),
    )

    rag_service = RAGService(
        llm=provider,
        qdrant_client=qdrant_client,
    )

    chat_agent = ChatAgent(
        llm=provider,
    )

    document_agent = DocumentAgent(
        llm_provider=provider,
        rag_service=rag_service,
    )

    return RouterAgent(
        llm=provider,
        chat_agent=chat_agent,
        document_agent=document_agent,
    )


# یک نمونه مشترک برای API
router_agent = build_router_agent()


@router.post("/StreamQuery")
async def stream_query_endpoint(
    request: QueryRequestStream,
    background_tasks: BackgroundTasks
):
    """
    Endpoint نهایی SSE.

    چون router دارای prefix="/api" است،
    مسیر نهایی این endpoint برابر است با:

    POST /api/StreamQuery
    """
    user_key='9a6b7ba9-abfe-4207-97fe-02a1da750cb7'
    chunks: Queue[Any] = Queue()
    
    def on_chunk(chunk: Any) -> None:
        """
        دریافت chunk از ChatAgent یا DocumentAgent
        و قرار دادن آن در صف SSE.
        """

        chunks.put(chunk)

    def produce() -> None:
        """
        اجرای RouterAgent در thread جداگانه.
        """

        try:
            router_agent.handle_stream(
                query=request.query,
                on_chunk=on_chunk,
                temperature=request.temperature,
            )

        except Exception as error:
            chunks.put(
                {
                    "type": "error",
                    "error": str(error),
                }
            )

        finally:
            # علامت پایان stream
            chunks.put(None)

    # شروع تولید پاسخ در پس‌زمینه
    Thread(
        target=produce,
        daemon=True,
    ).start()

    def event_stream():
        """
        تبدیل chunkهای صف به فرمت SSE با تفکیک نوع رویداد.
        """
        while True:
            chunk = chunks.get()

            # پایان stream
            if chunk is None:
                break

            # ۱. مدیریت خطاها
            if isinstance(chunk, dict) and chunk.get("type") == "error":
                yield (
                    "event: error\n"
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                )
                continue

            # ۲. تفکیک متادیتا و Usage (ارسالی از openai_provider)
            if isinstance(chunk, dict) and chunk.get("type") == "meta":
                final_response_id = chunk.get("response_id")
                final_usage = chunk.get("usage", {}) # دریافت دیکشنری usage
                yield (
                    "event: meta\n"
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                )
                continue

            # ۳. مدیریت توکن‌های متنی (Tokens)
            if isinstance(chunk, dict) and chunk.get("type") == "token":
                # تبدیل ساختار OpenAIProvider به ساختار مورد انتظار فرانت (text)
                payload = {"text": chunk.get("content", "")}
            elif isinstance(chunk, dict):
                payload = chunk
            else:
                payload = {"text": str(chunk)}

            yield (
                "event: token\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )
        if final_usage and final_response_id:
            background_tasks.add_task(
                InsertIntoWallet,
                final_usage.get("total_tokens", 0) * -1,
                final_usage.get("output_tokens", 0), # output_tokens
                final_usage.get("input_tokens", 0),     # input_tokens
                user_key,
                final_response_id
            )
                               
        # ارسال پایان قطعی استریم
        yield "event: done\ndata: [DONE]\n\n"
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
