from __future__ import annotations

import json
import re
from typing import Any

from providers.base import LLMProvider, StreamCallback


class DocumentAgent:
    def __init__(self, llm_provider: LLMProvider, rag_service):
        self.llm_provider = llm_provider
        self.rag_service = rag_service

    # -------------------



    

    def analyze(
        self,
        message: str,
        chunks: list[dict[str, Any]],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        system_prompt = """
    You are a technical support decision maker for banking equipment.
    
    You analyze:
    1. Current user question
    2. Previous conversation history
    3. Retrieved documents from RAG system
    
    
    Previous conversation:
    {history}
    
    
    User question:
    {query}
    
    
    Retrieved documents:
    {chunks}
    
    
    Your tasks:
    
    1. Decide whether the retrieved documents contain enough information to safely answer the user.
    2. Decide whether the user needs clarification before answering.
    3. If clarification is needed, ask only the minimum technical questions required.
    
    
    Decision rules:
    
    - Return "answer" when retrieved documents clearly explain the issue.
    - Return "clarify" when documents are related but important technical information is missing.
    - Return "insufficient" when retrieved documents are irrelevant or do not contain useful information.
    
    Clarification examples:
    - Missing ATM/POS model
    - Missing error code
    - Missing device type
    - Missing important environment details
    
    Important rules:
    
    - Do NOT ask clarification only because the user question is short.
    - If previous conversation contains the missing information, use it.
    - If retrieved documents strongly match the issue, answer even if the question is brief.
    - Never invent technical solutions without evidence.
    - Prefer asking a technical question instead of guessing.
    
    
    Return JSON only:
    
    {{
      "decision": "answer | clarify | insufficient",
      "confidence": 0-100,
      "missing_information": [
          "missing technical information"
      ],
      "clarification_question": "question for user or null"
    }}
    """

        messages = [
            {
                "role": "system",
                "content": system_prompt.format(
                    query=message,
                    history=json.dumps(
                        history or [],
                        ensure_ascii=False,
                        indent=2
                    ),
                    chunks=json.dumps(
                        chunks,
                        ensure_ascii=False,
                        indent=2
                    ),
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ]

        response = self.llm_provider.chat(
            messages=messages,
            temperature=0,
        )
        print(f"response: {response}", flush=True)
        raw_content = (response.content or "").strip()
        usage = getattr(response, "usage", {}) or {}
        # تلاش برای استخراج دقیق بلاک JSON
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(0)
        else:
            print(f"Failed to extract JSON from response: {raw_content}", flush=True)
            raw_content = ""

        try:
            result = json.loads(raw_content)
            
            valid_decisions = {"answer", "clarify", "insufficient"}
            decision = result.get("decision")
            print(f"Parsed decision: {decision}", flush=True)
            
            if decision not in valid_decisions:
                decision = "insufficient"

            return {
                "decision": decision,
                "confidence": result.get("confidence", 0),
                "missing_information": result.get("missing_information", []),
                "clarification_question": result.get("clarification_question"),
                "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
    },
            }

        except (json.JSONDecodeError, Exception) as e:
            print(f"Parsing error: {e}", flush=True)
            return {
                "decision": "insufficient",
                "confidence": 0,
                "missing_information": [],
                "clarification_question": None,
                "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
    },
            }

        # =====================


    def prepare_chunks(
        self,
        chunks: list[Any],
        ) -> list[dict[str, Any]]:

        output = []

        for chunk in chunks:
            output.append(
                {
                    "id": str(chunk.id),
                    "score": chunk.score,
                    "text": chunk.payload.get(
                        "text",
                        ""
                    ),
                    "title": chunk.payload.get(
                        "title",
                        ""
                    ),
                    "heading": chunk.payload.get(
                        "heading",
                        ""
                    ),
                    "source_file": chunk.payload.get(
                        "source_file",
                        ""
                    ),
                }
            )

        return output



    def handle_stream(
        self,
        message: str,
        on_chunk: StreamCallback,
        history: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
    ) -> None:

        query_vector = self.llm_provider.embed_query(
            message
        )

        results = self.rag_service.search(
            query_vector=query_vector,
            limit=20,
            filters=None,
        )
        print("HH0",flush=True)
        if not results:
            on_chunk(
                "هیچ سند مرتبطی یافت نشد."
            )
            return
        print("HH1",flush=True)

        # # reranked_results
        # #= self.rag_service.rerank_results(
        # #     message,
        # #     results,
        # # )


        analysis = self.analyze(
            message=message,
            chunks=self.prepare_chunks(results),
            history=history,
        )
        print("HH2",flush=True)

        decision = analysis.get(
            "decision"
        )
        print("HH3",flush=True)

        if decision == "answer":

            self.rag_service.answer_with_rag_stream(
                query=message,
                results=results,
                temperature=temperature,
                on_chunk=on_chunk,
            )
            print("HH4",flush=True)
            return

        usage_data = analysis.get("usage")
        if usage_data:
            on_chunk({
                "type": "meta",
                "response_id": "1111",
                "usage": usage_data
            })
        if decision == "clarify":

            question = analysis.get(
                "clarification_question"
            )

            if question:
                on_chunk({"type": "token", "content": question })
            else:
                on_chunk({"type": "token", "content":  "لطفاً اطلاعات بیشتری درباره مشکل دستگاه ارسال کنید."
                })
            

            print("HH4")


            return


        on_chunk({"type": "token", "content":  "اطلاعات کافی برای پاسخ دقیق پیدا نشد."
                })

        return
