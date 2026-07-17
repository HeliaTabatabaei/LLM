from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from  providers.factory import create_provider

router = APIRouter()


class ChatRequest(BaseModel):
    provider_name: str
    base_uri: str | None = None
    api_key: str
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7


@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    provider = create_provider(
        provider_name=request.provider_name,
        base_uri=request.base_uri,
        api_key=request.api_key,
        model=request.model,
        auth_header_name="Authorization",
        auth_token_prefix="Bearer",
        api_path=""
    )

    def generate():
        chunks = []

        def on_chunk(text):
            chunks.append(text)

        provider.chat_stream(
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            temperature=request.temperature,
            on_chunk=on_chunk
        )

        for chunk in chunks:
            yield chunk


    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )