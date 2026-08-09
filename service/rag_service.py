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
                    "id": self._get_result_id(result),
                    "text": payload.get("text", ""),
                    "title": payload.get("title", ""),
                    "heading": payload.get("heading", ""),
                    "keywords": payload.get("keywords", []),
                    "source_file": payload.get("source_file", ""),
                    "score": self._get_result_score(result),
                }
            )

        system_prompt = (
            "You are a reranking assistant.\n\n"
            "Given a user query and a list of retrieved results, return ONLY a valid JSON array.\n\n"
            "Each item must contain exactly:\n"
            "- id\n"
            "- score\n\n"
            "Scoring rules:\n"
            "- score must be a float between 0.0 and 1.0\n"
            "- higher score means more relevant\n"
            "- identify the most relevant result(s) for answering the query\n"
            "- prefer chunks that directly answer the user's intent\n"
            "- penalize irrelevant or weakly related chunks\n"
            "- do not prefer a result only because it contains generic technical words\n"
            "- if multiple chunks are from the same source and are all relevant, you may score them similarly\n"
            "- do not return explanations\n\n"
            "- return score higher than 0.7\n\n"
            "Return only valid JSON."
        )
        history_text = history.strip() if history else "No previous conversation."
        user_prompt = (
            f"Conversation History:\n{history_text}\n\n"
            f"Query:\n{query}\n\n"
            f"Results:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}\n\n"
            "Return only JSON in this format:\n"
            '[{"id": "1", "score": 0.95}, {"id": "2", "score": 0.40}]'
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

            score_map: dict[str, float] = {}
            for item in scored_items:
                if not isinstance(item, dict):
                    continue

                result_id = item.get("id")
                if result_id is None:
                    continue

                try:
                    score = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    score = 0.0

                score_map[str(result_id)] = max(0.0, min(1.0, score))

            return sorted(
                results,
                key=lambda result: score_map.get(
                    self._get_result_id(result),
                    self._get_result_score(result),
                ),
                reverse=True,
            )

        except Exception as exc:
            print(f"Rerank failed: {exc}")
            return sorted(
                results,
                key=self._get_result_score,
                reverse=True,
            )

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
        parts: list[str] = []

        for index, result in enumerate(results, start=1):
            payload = self._get_payload(result)

            title = payload.get("title", "")
            heading = payload.get("heading", "")
            text = payload.get("text", "")
            source_file = payload.get("source_file", "")

            block = [
                f"[Chunk {index}]",
                f"Source: {source_file}",
                f"Title: {title}",
                f"Heading: {heading}",
                "Text:",
                text,
            ]
            parts.append("\n".join(block).strip())

        return "\n\n---\n\n".join(parts)

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
