from functools import lru_cache
from openai import OpenAI

from .openai_provider import OpenAIProvider



PROVIDER_MAP = {
    "openai": OpenAIProvider,
}

@lru_cache(maxsize=10)
def _get_cached_openai_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

def create_provider(
    provider_name: str,
    base_uri: str,
    api_key: str,
    model: str,
    embed_model: str,
):
    provider_cls = PROVIDER_MAP.get(
        provider_name.lower(),
        OpenAIProvider,
    )

    if provider_cls is OpenAIProvider:
        client = _get_cached_openai_client(base_uri, api_key)

        return provider_cls(
            client=client,
            chat_model=model,
            embedding_model=embed_model,
        )

    return provider_cls(
        base_uri=base_uri,
        api_key=api_key,
        model=model,
        embed_model=embed_model,
    )