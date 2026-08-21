from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import ChatResponse, LLMProvider, StreamCallback


class OpenAIPlatform(LLMProvider):
    """
    Provider مربوط به OpenAI Platform.

    این کلاس با قرارداد تعریف‌شده در base.py سازگار است و
    می‌تواند به‌عنوان جایگزین OpenAIProvider در factory.py استفاده شود.
    """

    def __init__(
        self,
        client: OpenAI,
        chat_model: str,
        embedding_model: str,
    ) -> None:
        self.client = client
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def embed_query(self, text: str) -> list[float]:
        """
        تولید embedding برای متن ورودی.
        """

        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        return response.data[0].embedding

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.3,
    ) -> ChatResponse:
        """
        ارسال درخواست چت به OpenAI Platform.
        """

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
        )

        content = ""

        if response.choices:
            content = response.choices[0].message.content or ""

        usage: dict[str, int] = {}

        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }

        meta: dict[str, Any] = {
            "response_id": response.id,
            "model": response.model,
        }

        return ChatResponse(
            content=content,
            usage=usage,
            meta=meta,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        on_chunk: StreamCallback,
        temperature: float = 0.3,
    ) -> None:
        """
        ارسال درخواست چت به‌صورت Streaming.

        خروجی callback با همان ساختار مورد انتظار پروژه ارسال می‌شود:

        {
            "type": "token",
            "content": "..."
        }

        و در پایان، در صورت وجود usage:

        {
            "type": "meta",
            "response_id": "...",
            "usage": {...}
        }
        """

        stream = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=temperature,
            stream=True,
            stream_options={
                "include_usage": True,
            },
        )

        response_id: str | None = None

        for chunk in stream:
            response_id = getattr(chunk, "id", None) or response_id

            if chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)

                if content:
                    on_chunk(
                        {
                            "type": "token",
                            "content": content,
                        }
                    )

            usage = getattr(chunk, "usage", None)

            if usage:
                on_chunk(
                    {
                        "type": "meta",
                        "response_id": response_id,
                        "usage": {
                            "prompt_tokens": getattr(
                                usage,
                                "prompt_tokens",
                                0,
                            )
                            or 0,
                            "completion_tokens": getattr(
                                usage,
                                "completion_tokens",
                                0,
                            )
                            or 0,
                            "total_tokens": getattr(
                                usage,
                                "total_tokens",
                                0,
                            )
                            or 0,
                        },
                    }
                )
