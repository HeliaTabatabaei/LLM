from __future__ import annotations

from typing import Any, Callable

from providers.base import LLMProvider
from .chat_agent import ChatAgent
from .document_agent import DocumentAgent


ChunkCallback = Callable[[Any], None]


class RouterAgent:
    def __init__(
        self,
        llm: LLMProvider,
        chat_agent: ChatAgent,
        document_agent: DocumentAgent,
    ):
        self.llm = llm
        self.chat_agent = chat_agent
        self.document_agent = document_agent

    def classify(self, query: str) -> str:
        """
        تشخیص می‌دهد پرسش عمومی است یا فنی.
        خروجی فقط یکی از دو مقدار زیر است:

        technical
        general
        """

        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the user query as exactly one label.\n\n"

                    "technical: ATM, banking equipment, device errors, "
                    "troubleshooting, installation, configuration, maintenance, "
                    "printer, pinpad, dispenser, card reader, cash handling, "
                    "or operation.\n\n"

                    "general: greetings, small talk, unrelated questions, "
                    "or anything not technical.\n\n"

                    "Return exactly one word: technical or general."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        response = self.llm.chat(
            messages=messages,
            temperature=0,
        )

        result = (response.content or "").strip().lower()
        print("Resuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuut1111112323",flush=True)
        print(result,flush=True)
        if result not in {"technical", "general"}:
            return "technical"

        return result

    def handle_stream(
        self,
        query: str,
        on_chunk: ChunkCallback,
        temperature: float = 0.1,
    ) -> None:
        """
        پرسش را به Agent مناسب هدایت می‌کند.

        این متد دیگر StreamingResponse نمی‌سازد.
        تمام chunkها از طریق on_chunk به endpoint ارسال می‌شوند.
        """

        intent = self.classify(query)

        if intent == "general":
            self.chat_agent.answer_stream(
                message=query,
                on_chunk=on_chunk,
                temperature=temperature,
            )
            return

        self.document_agent.handle_stream(
            message=query,
            on_chunk=on_chunk,
            temperature=temperature,
        )
