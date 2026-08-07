from __future__ import annotations

import json
from typing import Any

from providers.base import LLMProvider, StreamCallback


class DocumentAgent:
    def __init__(self, llm_provider: LLMProvider, rag_service):
        self.llm_provider = llm_provider
        self.rag_service = rag_service

    def analyze(self, message: str) -> dict[str, Any]:
        system_prompt = (
            "You are a query analyzer for a Retrieval-Augmented Generation (RAG) system.\n"
            "Your task is to decide if the user's query is too ambiguous to search or process.\n\n"
            "CRITERIA for 'need_clarification':\n"
            "- Set to TRUE ONLY if the query is completely nonsensical, extremely vague (e.g., just saying 'help' or 'it doesn't work' without context), or has multiple conflicting meanings that make retrieval impossible.\n"
            "- Set to FALSE if the query is a standard question, even if it is brief, broad, or has minor typos. If you can formulate a reasonable search intent from it, do NOT clarify.\n\n"
            "Return JSON only in this schema:\n"
            "{\n"
            '  "need_clarification": boolean,\n'
            '  "question": "clarification question string or null"\n'
            "}"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]

        # خروجی llm.chat یک ChatResponse است
        response = self.llm_provider.chat(
            messages=messages,
            temperature=0,
        )

        raw_content = (response.content or "").strip()

        try:
            return json.loads(raw_content)
        except Exception:
            return {
                "need_clarification": False,
                "question": None,
            }

    def handle_stream(
        self,
        message: str,
        on_chunk: StreamCallback,
        temperature: float = 0.1,
    ) -> None:
        # analysis = self.analyze(message)

        # if analysis.get("need_clarification"):
        #     on_chunk(
        #         analysis.get(
        #             "question",
        #             "لطفاً سوالت را دقیق‌تر بپرس.",
        #         )
        #     )
        #     return

        query_vector = self.llm_provider.embed_query(message)
        results = self.rag_service.search(
            query_vector=query_vector,
            limit=5,
            filters=None,
        )

        if not results:
            on_chunk("هیچ سند مرتبطی یافت نشد.")
            return

        reranked_results = self.rag_service.rerank_results(
            message,
            results,
        )

        # فراخوانی با الگوی استاندارد callback-based بدون پارامترهای اضافه
        self.rag_service.answer_with_rag_stream(
            query=message,
            results=reranked_results,
            temperature=temperature,
            on_chunk=on_chunk,
        )
