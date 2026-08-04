import json

class DocumentAgent:
    def __init__(self, llm, rag_service):
        self.llm = llm
        self.rag_service = rag_service

    def analyze(self, message: str) -> dict:
        system_prompt = """
You are a query analyzer.
Return JSON only:
{
  "need_clarification": true or false,
  "question": "clarification question or null"
}
"""
        result = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=message,
            temperature=0
        )

        try:
            return json.loads(result)
        except Exception:
            return {
                "need_clarification": False,
                "question": None
            }

    def handle_stream(self, message: str, user_key: str, background_tasks, on_chunk):
        analysis = self.analyze(message)

        if analysis.get("need_clarification"):
            on_chunk(analysis.get("question", "لطفاً سوالت را دقیق‌تر بپرس."))
            return

        query_vector = self.rag_service.embed_query(message)
        results = self.rag_service.search(query_vector=query_vector, limit=5, filters=None)

        if not results:
            on_chunk("هیچ سند مرتبطی یافت نشد.")
            return

        reranked_results = self.rag_service.rerank_results(message, results)

        for chunk in self.rag_service.answer_with_rag_stream(
            query=message,
            userKey=user_key,
            background_tasks=background_tasks,
            results=reranked_results,
            temperature=0.1
        ):
            if chunk:
                on_chunk(chunk)
