# services/rag_service.py
import json
from fastapi import BackgroundTasks, HTTPException
from LLM.OpenAIManagment import detect_intent, embed_query, rerank_results
from RAG_Management.QdrantManagment import search
from RAG_Management.answerWithRAG import answer_general_stream, answer_with_rag_stream

class RAGService:
    def __init__(self):
        pass

    def _token_event(self, text: str) -> str:
        return f"event: token\ndata: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"

    def _done_event(self) -> str:
        return "event: done\ndata: [DONE]\n\n"

    def _error_event(self, error: Exception) -> str:
        return f"event: error\ndata: {json.dumps({'error': str(error)}, ensure_ascii=False)}\n\n"

    def generate_response_stream(self, query: str, user_key: str, background_tasks: BackgroundTasks):
        """
        تمام منطق بیزینس RAG شامل تشخیص نوع، جستجو، رتبه‌بندی و فرمت‌دهی SSE 
        در این لایه انجام می‌شود تا لایه API کاملاً خالی از منطق بیزینس باشد.
        """
        intent = detect_intent(query)

        # ۱. مسیر سوالات عمومی (General)
        if intent == "general":
            # این تابع خودش خروجی فرمت‌شده به صورت SSE تولید می‌کند
            yield from answer_general_stream(query, user_key, background_tasks)
            return

        # ۲. مسیر سوالات فنی (RAG)
        embedding = embed_query(query)
        results = search(embedding, limit=5, filters=None)
        
        if not results:
            # در صورتی که سندی یافت نشد، خطا در جریان SSE ارسال می‌شود
            yield self._error_event(Exception("هیچ سند مرتبطی یافت نشد"))
            return

        reranked = rerank_results(query, results)

        try:
            # ایجاد استریم پاسخ و تبدیل آن به رویدادهای استاندارد SSE
            for chunk in answer_with_rag_stream(
                query=query,
                userKey=user_key,
                background_tasks=background_tasks,
                results=reranked,
                temperature=0.1
            ):
                if chunk:
                    yield self._token_event(chunk)
            yield self._done_event()
        except Exception as e:
            yield self._error_event(e)
