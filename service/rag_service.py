from __future__ import annotations

import json
from typing import Any, Optional

from qdrant_client import models

from Models.mainModels import SearchFilters
from config import COLLECTION_NAME
from prompts_config import SYSTEM_PROMPT, USER_PROMPT
from providers.base import LLMProvider, StreamCallback


class RAGService:
    def __init__(
        self,
        llm: LLMProvider,
        qdrant_client: Any,
        collection_name: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name or COLLECTION_NAME

    def embed_query(self, text: str) -> list[float]:
        return self.llm.embed_query(text)

    @staticmethod
    def build_filter(
        filters: Optional[SearchFilters],
    ) -> Optional[models.Filter]:
        if not filters:
            return None

        conditions: list[models.FieldCondition] = []

        if getattr(filters, "doc_ids", None):
            conditions.append(
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchAny(any=filters.doc_ids),
                )
            )

        if getattr(filters, "tags", None):
            conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=filters.tags),
                )
            )

        if getattr(filters, "date_from", None):
            conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.Range(gte=filters.date_from),
                )
            )

        if getattr(filters, "date_to", None):
            conditions.append(
                models.FieldCondition(
                    key="date",
                    range=models.Range(lte=filters.date_to),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        filters: Optional[SearchFilters] = None,
    ) -> list[Any]:
        query_filter = self.build_filter(filters)

        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            query_filter=query_filter,
        )

        return getattr(hits, "points", []) or []

  
    def rerank_results(
        self,
        query: str,
        results: list[Any],
        history: str | None = None,
    ) -> list[Any]:
        if not results:
            return results

        candidates: list[dict[str, Any]] = []
        for result in results:
            payload = self._get_payload(result)
            candidates.append(
                {
                    "id": str(self._get_result_id(result)), # تبدیل به رشته برای تطابق با JSON
                    "text": payload.get("maintext", ""),
                    "title": payload.get("title", ""),
                    "heading": payload.get("heading", ""),
                    "source_file": payload.get("source_file", ""),
                }
            )

        system_prompt = """
You are a technical document reranker. Your job is to score relevance on a scale of 0.0 to 1.0.

Instructions:
- 1.0: The chunk contains the exact answer, error code explanation, or step-by-step solution.
- 0.8-0.9: Highly relevant. Provides critical context or strong supporting evidence for the answer.
- 0.5: Marginally relevant. Contains the right topic but lacks specific actionable details.
- 0.0-0.3: Irrelevant. Wrong device, wrong topic, or gibberish.

Examples:
Query: "How to fix ATM error 404?"
Chunk: "Error 404 indicates a network timeout in the X-500 module. Reset the router." -> Score: 1.0
Chunk: "The X-500 module operates at 24V power supply." -> Score: 0.3

Rules:
- You MUST score every candidate.
- Return ONLY valid JSON array: [{"id": "...", "score": ...}]
- Do not add explanations.
""".strip()

        history_text = history.strip() if history else "No previous conversation."
        user_prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"Query:\n{query}\n\n"
            f"Results to Score:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            f"Return exactly {len(candidates)} JSON items. One score for each ID."
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            content = (response.content or "").strip()
            scored_items = self._parse_json_array(content)

            # استخراج امتیازها در یک دیکشنری
            score_map: dict[str, float] = {}
            for item in scored_items:
                if isinstance(item, dict) and "id" in item:
                    try:
                        score_map[str(item["id"])] = float(item.get("score", 0.0))
                    except (TypeError, ValueError):
                        continue

            # --- بخش مهم: آپدیت کردن امتیاز واقعی روی آبجکت‌ها ---
            processed_results = []
            for result in results:
                res_id = str(self._get_result_id(result))
                # گرفتن امتیاز از LLM، اگر نبود از امتیاز اولیه استفاده شود
                llm_score = score_map.get(res_id, 0.0)
                
                # تزریق امتیاز ریرنک به فیلد اسکور (بسته به ساختار آبجکت Qdrant شما)
                if hasattr(result, 'score'):
                    result.score = llm_score
                elif isinstance(result, dict):
                    result['score'] = llm_score
                
                # همچنین ذخیره در payload برای اطمینان در مراحل بعدی
                payload = self._get_payload(result)
                payload["rerank_score"] = llm_score
                payload["retrieval_score"] = self._get_result_score(result) # ذخیره امتیاز اولیه برای دیباگ
                
                processed_results.append(result)

            # ۱. مرتب‌سازی بر اساس امتیاز جدید (LLM Score)
            processed_results.sort(key=lambda r: score_map.get(str(self._get_result_id(r)), 0.0), reverse=True)

            # ۲. اعمال فیلتر 0.7 و محدودیت 5 عدد
            filtered_results = [r for r in processed_results if score_map.get(str(self._get_result_id(r)), 0.0) >= 0.7]
            final_output = filtered_results[:5]

            # فال‌بک در صورتی که هیچکدام بالای 0.7 نبودند (برای خالی نماندن پاسخ)
            if not final_output and processed_results:
                final_output = processed_results[:3]

            print(f"[RERANK] Expected: {len(results)}, Received: {len(score_map)}, Filtered (>=0.7): {len(filtered_results)}")
            return final_output

        except Exception as exc:
            print(f"Rerank failed: {exc}")
            # در صورت خطا، همان لیست اولیه را بر اساس امتیاز اولیه مرتب و برگردان
            return sorted(results, key=self._get_result_score, reverse=True)[:5]
    
    ###########################################
    
    # def rerank_results(
    #         self,
    #         query: str,
    #         results: list[Any],
    #         history: str | None = None,
    #             ) -> list[Any]:
        
    #         print(f"[RERANK] received candidate count: {len(results)}")
    #         print(
    #             "[RERANK] received candidates:",
    #             [
    #                 {
    #                     "id": self._get_result_id(result),
    #                     "qdrant_score": self._get_result_score(result),
    #                 }
    #                 for result in results
    #             ],
    #         )
    #         if not results:
    #             return results

    #         candidates: list[dict[str, Any]] = []
    #         for result in results:
    #             payload = self._get_payload(result)
    #             candidates.append(
    #                 {
    #                     "id": self._get_result_id(result),
    #                     "text": payload.get("text", ""),
    #                     "title": payload.get("title", ""),
    #                     "heading": payload.get("heading", ""),
    #                     "keywords": payload.get("keywords", []),
    #                     "source_file": payload.get("source_file", ""),
    #                     "score": self._get_result_score(result),
    #                 }
    #             )

    #         system_prompt = (
    #             "You are a reranking assistant.\n\n"
    #         "Given a user query and a list of retrieved results, return ONLY a valid JSON array.\n\n"
    #         "You MUST return one item for EVERY provided result.\n"
    #         "Do NOT omit any result.\n"
    #         "Use the exact same id values from the input.\n"
    #         "The output array length MUST equal the input results length.\n\n"
    #         "Each item must contain exactly:\n"
    #         "- id\n"
    #         "- score\n\n"
    #         "Scoring rules:\n"
    #         "- score must be a float between 0.0 and 1.0\n"
    #         "- higher score means more relevant\n"
    #         "- prefer chunks that directly answer the user's intent\n"
    #         "- penalize irrelevant or weakly related chunks\n"
    #         "- assign low scores such as 0.0, 0.1, or 0.2 to irrelevant results\n"
    #         "- do NOT filter results yourself; only assign scores\n\n"
    #         "Return only valid JSON."
    #         )
    #         history_text = history.strip() if history else "No previous conversation."
    #         user_prompt = (
    #             f"Conversation History:\n{history_text}\n\n"
    #             f"Query:\n{query}\n\n"
    #             f"Results:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
    #             f"Return exactly {len(candidates)} JSON items, one per input result.\n"
    #             "Return only JSON in this format:\n"
    #             '[{"id": "26200006", "score": 0.95}, {"id": "26400015", "score": 0.40}]'
    #             )

    #         try:
    #             response = self.llm.chat(
    #                 messages=[
    #                     {"role": "system", "content": system_prompt},
    #                     {"role": "user", "content": user_prompt},
    #                 ],
    #                 temperature=0,
    #             )

    #             content = (response.content or "").strip()
    #             scored_items = self._parse_json_array(content)

    #             score_map: dict[str, float] = {}
    #             for item in scored_items:
    #                 if not isinstance(item, dict):
    #                     continue
    #                 result_id = item.get("id")
    #                 if result_id is None:
    #                     continue
    #                 try:
    #                     score = float(item.get("score", 0.0))
    #                 except (TypeError, ValueError):
    #                     score = 0.0
    #                 score_map[str(result_id)] = max(0.0, min(1.0, score))

    #             sorted_results = sorted(
    #                 results,
    #                 key=lambda r: score_map.get(self._get_result_id(r), 0.0),
    #                 reverse=True,
    #             )
    #             print(f"[RERANK] llm score_map: {score_map}",flush=True)
    #             print(f"[RERANK] expected scores: {len(candidates)}, received scores: {len(score_map)}")

    #         # فیلتر بر اساس امتیاز LLM (آستانه سخت‌گیرانه 0.7)
    #             filtered_results = [
    #                 r for r in sorted_results 
    #                 if score_map.get(self._get_result_id(r), 0.0) >= 0.7
    #             ][:5]
            
    #         # اگر هیچ چانکی بالای 0.7 نبود، برای خالی نماندن دست مدل،
    #         # تا حداکثر 3 چانک برتر از خروجی‌های LLM (حتی با امتیاز کمتر) را برمی‌گردانیم.
    #             if not filtered_results and sorted_results:
    #                 filtered_results = sorted_results[:3]
    #             print(f"[RERANK] filtered count: {len(filtered_results)}",flush=True)
    #             print(f"[RERANK] filtered original scores: {[self._get_result_score(r) for r in filtered_results]}",flush=True)

    #             return filtered_results

    #         except Exception as exc:
    #         # لاگ کردن خطا برای اینکه متوجه شوید چرا Rerank خطا می‌دهد
    #             print(f"[RERANK ERROR] Fallback triggered. Error: {exc}")
            
    #             sorted_default = sorted(
    #                 results,
    #                 key=self._get_result_score,
    #                 reverse=True,
    #             )
            
    #         # در حالت خطا و Fallback، چانک‌های با امتیاز بالای 0.4 را برمی‌گردانیم تا سیستم از کار نیفتد
    #             filtered_default = []
    #             for r in sorted_default:
    #                 if self._get_result_score(r) >= 0.4:
    #                     filtered_default.append(r)
    #                 if len(filtered_default) == 5:
    #                     break
                        
    #             # اگر باز هم خالی بود، حداقل اولین چانک برتر را بفرست
    #             if not filtered_default and sorted_default:
    #                 return [sorted_default[0]]
                    
    #             return filtered_default

    
    
    
    
    ###############################################
    def answer_with_rag_stream(
        self,
        query: str,
        results: list[Any],
        on_chunk: StreamCallback,
        temperature: float = 0.1,
        history: str | None = None,
    ) -> None:
        context = self._build_context(results)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context=context,
                    query=query,
                    history=history,
                ),
            },
        ]

        self.llm.chat_stream(
            messages=messages,
            on_chunk=on_chunk,
            temperature=temperature,
        )

    def _build_context(self, results: list[Any]) -> str:
        chunks = []   
        for i, r in enumerate(results, start=1):
            if isinstance(r, dict):
                payload = r["payload"]
            else:
                payload = r.payload
            # text = payload.get("text", "")
            maintext=payload.get("maintext", "")
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
            chunks.append(f"{header}\n{maintext}")
        return "\n\n".join(chunks)
    @staticmethod
    def _get_payload(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            payload = result.get("payload")
            return payload if isinstance(payload, dict) else {}
        payload = getattr(result, "payload", None)
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _get_result_id(result: Any) -> str:
        if isinstance(result, dict):
            return str(result.get("id", ""))
        return str(getattr(result, "id", ""))

    @staticmethod
    def _get_result_score(result: Any) -> float:
        if isinstance(result, dict):
            value = result.get("score", 0.0)
        else:
            value = getattr(result, "score", 0.0)

        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
    @staticmethod
    def _parse_json_array(content: str) -> list[dict[str, Any]]:
        content = (content or "").strip()

        if not content:
            raise ValueError("Empty reranker response")

        code_fence = "```"

        if content.startswith(code_fence):
            lines = content.splitlines()

            if lines and lines[0].strip().startswith(code_fence):
                lines = lines[1:]

            if lines and lines[-1].strip() == code_fence:
                lines = lines[:-1]

            content = "\n".join(lines).strip()

        parsed = json.loads(content)

        if not isinstance(parsed, list):
            raise ValueError("Reranker response must be a JSON array")

        return parsed
