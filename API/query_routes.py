from __future__ import annotations

import json
import os
from queue import Queue
from threading import Thread
from typing import Any, Optional, Tuple
import uuid


from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient

from Models.mainModels import QueryRequestStream, QueryRequestStreamٌwithConversionId
from SQlDB.db import DatabaseConnection
from SQlDB.wallet import InsertIntoWallet
from config import QDRANT_HOST, QDRANT_PORT

from dbManagement import SQL_SERVER_CONNECTION_STRING, get_conversation_history, save_conversation, save_message
from log import append_qa_to_file
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

     
def normalize_conversation_id(conversation_id: Optional[str]) -> Tuple[str, bool]:
        
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


def get_recent_history(
        
        conversation_id: str,
        query:str,
        user_key:str,
        limit: int = 3
    ):
        conversation_id, is_new_chat = normalize_conversation_id(conversation_id)

        print(conversation_id,flush=True)
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
            return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]),conversation_id

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


# @router.post("/StreamQuery")
# async def stream_query_endpoint(
#     request: QueryRequestStream,
#     background_tasks: BackgroundTasks
# ):
   
#     user_key='9a6b7ba9-abfe-4207-97fe-02a1da750cb7'
#     chunks: Queue[Any] = Queue()
    
#     def on_chunk(chunk: Any) -> None:    
#         chunks.put(chunk)

#     def produce() -> None:
#         try:
#             router_agent.handle_stream(
#                 query=request.query,
#                 on_chunk=on_chunk,
#                 temperature=request.temperature,
#             )

#         except Exception as error:
#             chunks.put(
#                 {
#                     "type": "error",
#                     "error": str(error),
#                 }
#             )

#         finally:
#             # علامت پایان stream
#             chunks.put(None)

#     # شروع تولید پاسخ در پس‌زمینه
#     Thread(
#         target=produce,
#         daemon=True,
#     ).start()

#     def event_stream():
        
#         """
#         تبدیل chunkهای صف به فرمت SSE با تفکیک نوع رویداد.
#         """
#         answer_parts = []
#         while True:
#             chunk = chunks.get()

#             # پایان stream
#             if chunk is None:
#                 break

#             # ۱. مدیریت خطاها
#             if isinstance(chunk, dict) and chunk.get("type") == "error":
#                 yield (
#                     "event: error\n"
#                     f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
#                 )
#                 continue

#             # ۲. تفکیک متادیتا و Usage (ارسالی از openai_provider)
#             if isinstance(chunk, dict) and chunk.get("type") == "meta":
#                 print("ssssssssssssssssss",flush=True)
#                 final_response_id = chunk.get("response_id")
#                 final_usage = chunk.get("usage", {}) # دریافت دیکشنری usage
#                 print(final_usage,flush=True)
#                 yield (
#                     "event: meta\n"
#                     f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
#                 )
#                 continue

#             # ۳. مدیریت توکن‌های متنی (Tokens)
#             if isinstance(chunk, dict) and chunk.get("type") == "token":
#                 print("ttttttttttttttttt",flush=True)
#                 text = chunk.get("content", "")
#                 answer_parts.append(text)
#                 payload = {"text": chunk.get("content", "")}
#                 # تبدیل ساختار OpenAIProvider به ساختار مورد انتظار فرانت (text)
             
#             elif isinstance(chunk, dict):
#                 print("hhhhhhhhhhh",flush=True)
#                 text = chunk.get("text")
#                 if text:
#                     answer_parts.append(str(text))
#                     payload = chunk
#             else:
#                 print("eeeeeeeeeeeeeeeeeee",flush=True)
#                 text = str(chunk)
#                 answer_parts.append(text)
#                 payload = {"text": text}

#             yield (
#                 "event: token\n"
#                 f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
#             )
#         final_answer = "".join(answer_parts).strip()
#         print(f"append_qa_to_file", flush=True)   
      
#     # ذخیره سؤال و جواب در فایل
#         try:
           
#             append_qa_to_file(
#             question=request.query,
#             answer=final_answer         
#         )
#         except Exception as e:
#             print(f"Failed to save QA log: {e}", flush=True)    
#         if final_usage and final_response_id:
#             print("ssssssssssssssssss2222",flush=True)          
#             background_tasks.add_task(
#                 InsertIntoWallet,
#                 final_usage.get("total_tokens", 0) * -1,
#                 final_usage.get("output_tokens", 0), # output_tokens
#                 final_usage.get("input_tokens", 0),     # input_tokens
#                 user_key,
#                 final_response_id
#             )                              
#         # ارسال پایان قطعی استریم
#         yield "event: done\ndata: [DONE]\n\n"
#     return StreamingResponse(
#         event_stream(),
#         media_type="text/event-stream",
#         headers=STREAM_HEADERS,
#     )

@router.post("/StreamQueryHistory")
async def stream_queryHistory_endpoint(
    request: QueryRequestStreamٌwithConversionId,
    background_tasks: BackgroundTasks
):
   
    user_key='9a6b7ba9-abfe-4207-97fe-02a1da750cb7'
    history,c_id= get_recent_history( conversation_id= request.conversation_id,
                query=request.query,
                user_key=user_key,
                limit = 3)
    print (history,flush=True)
    print (request.conversation_id,flush=True)
    chunks: Queue[Any] = Queue()
    
    def on_chunk(chunk: Any) -> None:    
        chunks.put(chunk)

    def produce() -> None:
        try:
            router_agent.handle_stream(
                query=request.query,
                user_key=user_key,
                on_chunk=on_chunk,
                history=history,
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
        answer_parts = []
        final_usage = {}
        final_response_id = None
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
            if isinstance(chunk, dict) and chunk.get("type") == "source_chunks":
                continue       
            # ۲. تفکیک متادیتا و Usage (ارسالی از openai_provider)
            if isinstance(chunk, dict) and chunk.get("type") == "meta":
                print("ssssssssssssssssss",flush=True)
                final_response_id = chunk.get("response_id")
                final_usage = chunk.get("usage", {}) # دریافت دیکشنری usage
                meta_payload = {
                    **chunk,
                    "conversation_id": c_id,}
                yield ("event: meta\n"f"data: {json.dumps(meta_payload, ensure_ascii=False)}\n\n")
                continue
            # ۳. مدیریت توکن‌های متنی (Tokens)
            if isinstance(chunk, dict) and chunk.get("type") == "token":
                text = chunk.get("content", "")
                answer_parts.append(text)
                payload = {"text": chunk.get("content", "")}
                # تبدیل ساختار OpenAIProvider به ساختار مورد انتظار فرانت (text)            
            elif isinstance(chunk, dict): 
                text = chunk.get("text")
                if text:
                    answer_parts.append(str(text))
                    payload = chunk
           
            else:
              
                text = str(chunk)
                answer_parts.append(text)
                payload = {"text": text}

            yield (
                "event: token\n"
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            )
        final_answer = "".join(answer_parts).strip()
        print(f"append_qa_to_file", flush=True)   
        with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
                save_message(
                    cursor=cursor,
                    conversation_id=c_id,
                    role="assistant",
                    content=final_answer,
                    provider_response_id="1111"
                )
    # ذخیره سؤال و جواب در فایل
        try:
           
            append_qa_to_file(
            question=request.query,
            answer=final_answer         
        )
        except Exception as e:
            print(f"Failed to save QA log: {e}", flush=True)    
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
@router.post("/QueryHistory")
async def query_history_endpoint(
    request: QueryRequestStreamٌwithConversionId,
    background_tasks: BackgroundTasks
):
    user_key = '9a6b7ba9-abfe-4207-97fe-02a1da750cb7'

    history, c_id = get_recent_history(
        conversation_id=request.conversation_id,
        query=request.query,
        user_key=user_key,
        limit=3
    )

    answer_parts = []
    final_usage = {}
    final_response_id = "1111"
    source_payload = {"chunks": [], "conversation_id": c_id}

    def on_chunk(chunk: Any) -> None:
        nonlocal final_usage, final_response_id, source_payload

        if not isinstance(chunk, dict):
            return

        if chunk.get("type") == "token":
            content = chunk.get("content", "")
            if content:
                answer_parts.append(content)

        elif chunk.get("type") == "meta":
            final_usage = chunk.get("usage", {})
            final_response_id = chunk.get("response_id", final_response_id)

        elif chunk.get("type") == "source_chunks":
            source_payload = {
                "chunks": chunk.get("chunks", []),
                "conversation_id": c_id,
            }

    try:
        router_agent.handle_stream(
            query=request.query,
            user_key=user_key,
            on_chunk=on_chunk,
            history=history,
            temperature=request.temperature,
        )
    except Exception as error:
        return {"status": "error", "message": str(error)}

    final_answer = "".join(answer_parts).strip()

    try:
        with DatabaseConnection(SQL_SERVER_CONNECTION_STRING) as cursor:
            save_message(
                cursor=cursor,
                conversation_id=c_id,
                role="assistant",
                content=final_answer,
                provider_response_id=final_response_id
            )
    except Exception as e:
        print(f"Database save error: {e}", flush=True)

    try:
        append_qa_to_file(
            question=request.query,
            answer=final_answer
        )
    except Exception as e:
        print(f"Failed to save QA log: {e}", flush=True)

    if final_usage:
        background_tasks.add_task(
            InsertIntoWallet,
            final_usage.get("total_tokens", 0) * -1,
            final_usage.get("output_tokens", 0),
            final_usage.get("input_tokens", 0),
            user_key,
            final_response_id
        )

    return {
        "status": "success",
        "conversation_id": c_id,
        "answer": final_answer,
        "usage": final_usage,
        "response_id": final_response_id,
        "source": source_payload
    }
