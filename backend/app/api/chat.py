from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.security import get_current_admin
from app.db.session import SessionLocal, get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChatTaskCreated,
    ChatTaskCreateRequest,
    GenerationTaskRead,
    SearchRequest,
    SearchResponse,
)
from app.services.chat_service import ChatService
from app.services.chat_task_service import ChatTaskService

router = APIRouter(tags=["chat"])
chat_service = ChatService()
chat_task_service = ChatTaskService(chat_service)


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


@router.post("/chat/tasks", response_model=ChatTaskCreated, status_code=202)
def create_chat_task(
    request: ChatTaskCreateRequest,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ChatTaskCreated:
    task = chat_task_service.create_task(db, request)
    return ChatTaskCreated(
        task_id=task.id,
        conversation_id=task.conversation_id,
        user_message_id=task.user_message_id,
        assistant_message_id=task.assistant_message_id,
        status=task.status,
        last_event_id=task.last_event_id,
    )


@router.get("/chat/tasks/{task_id}", response_model=GenerationTaskRead)
def get_chat_task(
    task_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> GenerationTaskRead:
    task = chat_task_service.get_task(db, task_id)
    return chat_task_service.task_read(db, task)


@router.get("/chat/tasks/{task_id}/events")
def stream_chat_task_events(
    task_id: str,
    after: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    chat_task_service.get_task(db, task_id)

    async def event_stream():
        stream_db = SessionLocal()
        try:
            async for event in chat_task_service.subscribe(
                stream_db, task_id, after or last_event_id
            ):
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


@router.post("/chat/tasks/{task_id}/cancel", response_model=GenerationTaskRead, status_code=202)
def cancel_chat_task(
    task_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> GenerationTaskRead:
    task = chat_task_service.cancel_task(db, task_id)
    return chat_task_service.task_read(db, task)
