from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = self.base_uri or "https://api.openai.com/v1"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        return (completion.choices[0].message.content or "").strip()

    def chat_stream(
            self,
            system_prompt: str,
            user_prompt: str,
            temperature: float,
            on_chunk
    ):

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
            stream=True,
        )

        for chunk in stream:
            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                on_chunk(content)