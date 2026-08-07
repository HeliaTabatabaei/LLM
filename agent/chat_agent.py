from __future__ import annotations

from providers.base import (
    ChatResponse,
    LLMProvider,
    StreamCallback,
)


class ChatAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @staticmethod
    def _build_messages(
        message: str,
    ) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant."
                ),
            },
            {
                "role": "user",
                "content": message,
            },
        ]

    def answer(
        self,
        message: str,
        temperature: float = 0.3,
    ) -> ChatResponse:
        """
        پاسخ معمولی و غیر streaming.

        خروجی:
            ChatResponse
        """

        return self.llm.chat(
            messages=self._build_messages(message),
            temperature=temperature,
        )

    def answer_stream(
        self,
        message: str,
        on_chunk: StreamCallback,
        temperature: float = 0.3,
    ) -> None:
        """
        پاسخ streaming از طریق callback.

        این متد خودش str یا generator برنمی‌گرداند.
        هر chunk از طریق on_chunk ارسال می‌شود.
        """

        self.llm.chat_stream(
            messages=self._build_messages(message),
            on_chunk=on_chunk,
            temperature=temperature,
        )
