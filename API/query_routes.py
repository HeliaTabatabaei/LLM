from fastapi import (
    APIRouter,
    BackgroundTasks,
    HTTPException,
    Security,
)
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)
from jose import ExpiredSignatureError, JWTError

from Models.mainModels import QueryRequestStream
from Utility.utiliy import get_current_user_payload
from agent.container import router_agent


api_router = APIRouter()
security = HTTPBearer()


@api_router.post("/api/query/stream1")
async def query_stream1_endpoint(
    request: QueryRequestStream,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    token = credentials.credentials

    try:
        is_valid, message, user_key = get_current_user_payload(token)

        if not is_valid:
            raise HTTPException(
                status_code=401,
                detail=message,
            )

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired",
        )

    except JWTError as error:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {error}",
        )

    return router_agent.handle_stream(
        query=request.query,
        user_key=user_key,
        background_tasks=background_tasks,
        temperature=request.temperature,
    )
