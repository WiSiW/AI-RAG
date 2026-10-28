from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import SessionLocal, get_db
from app.schemas import ChatRequest, ChatResponse, SearchRequest, SearchResponse
from app.services.chat_service import ChatService

router = APIRouter(tags=["chat"])
chat_service = ChatService()


@router.post("/chat", response_model=ChatResponse | None)
async def chat(
    request: ChatRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    if not request.stream:
        return await chat_service.chat(db, request)

    async def event_stream():
        stream_db = SessionLocal()
        try:
            async for event in chat_service.stream_chat(stream_db, request):
                yield event
        finally:
            stream_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> SearchResponse:
    return await chat_service.search(db, request)
