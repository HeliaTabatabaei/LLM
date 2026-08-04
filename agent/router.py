from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse


class RouterAgent:
    STREAM_HEADERS = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    def __init__(
        self,
        llm,
        chat_agent,
        document_agent,
    ):
        self.llm = llm
        self.chat_agent = chat_agent
        self.document_agent = document_agent

    def classify(self, query: str) -> str:
        system_prompt = """
You are a classifier.

Classify the user's query as exactly one of these two labels:

1. technical
   Use this label only if the query is related to banking equipment
   or its software/services, including:

   ATM, kiosk, self-service banking machines, card reader, pinpad,
   printer, dispenser, recycler, cash handling, device errors,
   troubleshooting, installation, setup, configuration, maintenance,
   monitoring, or operation.

2. general
   Use this label for anything else, including:
   greetings, small talk, thanks, goodbyes, and any query not related
   to banking equipment.

Reply with one word only:
technical
or
general
"""

        result = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=query,
            temperature=0,
        ).strip().lower()

        if result not in {"technical", "general"}:
            return "technical"

        return result

    def handle_stream(
        self,
        query: str,
        user_key: str,
        background_tasks: BackgroundTasks,
        temperature: float = 0.1,
    ) -> StreamingResponse:
        stream = self._route_stream(
            query=query,
            user_key=user_key,
            background_tasks=background_tasks,
            temperature=temperature,
        )

        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers=self.STREAM_HEADERS,
        )

    def _route_stream(
        self,
        query: str,
        user_key: str,
        background_tasks: BackgroundTasks,
        temperature: float,
    ):
        intent = self.classify(query)

        if intent == "general":
            return self.chat_agent.handle_stream(
                query=query,
                user_key=user_key,
                background_tasks=background_tasks,
                temperature=temperature,
            )

        return self.document_agent.handle_stream(
            query=query,
            user_key=user_key,
            background_tasks=background_tasks,
            temperature=temperature,
        )
