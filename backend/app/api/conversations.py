from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import NotFoundError
from app.core.security import get_current_admin
from app.db.models import Conversation, GenerationTask, KnowledgeBase, Message
from app.db.session import get_db
from app.memory.session_manager import ConversationMemory
from app.schemas import ConversationDetail, ConversationRead, GenerationTaskRead, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])
memory = ConversationMemory()
SUMMARY_STATUSES = {"idle", "running", "succeeded", "degraded", "failed"}


def _knowledge_base_ids(conversation: Conversation) -> list[str]:
    ids = [link.knowledge_base_id for link in conversation.knowledge_base_links]
    if ids:
        return ids
    return [conversation.knowledge_base_id] if conversation.knowledge_base_id else []


def _summary_status(conversation: Conversation) -> str:
    status = str(conversation.summary_status or "idle")
    return status if status in SUMMARY_STATUSES else "idle"


def _active_task_read(db: Session, conversation_id: str) -> GenerationTaskRead | None:
    task = db.scalar(
        select(GenerationTask)
        .where(
            GenerationTask.conversation_id == conversation_id,
            GenerationTask.status.in_(("queued", "running")),
        )
        .order_by(GenerationTask.created_at.desc())
    )
    if task is None:
        return None
    heartbeat = task.heartbeat_at
    if heartbeat is not None and heartbeat.tzinfo is None:
        heartbeat = heartbeat.replace(tzinfo=UTC)
    if (
        task.status == "running"
        and heartbeat is not None
        and datetime.now(UTC) - heartbeat
        >= timedelta(seconds=settings.chat_task_stale_seconds)
    ):
        task.status = "failed"
        task.error = "生成任务心跳超时"
        task.finished_at = datetime.now(UTC)
        assistant = db.get(Message, task.assistant_message_id)
        if assistant is not None:
            assistant.status = "failed"
            assistant.usage = {**(assistant.usage or {}), "error": task.error}
        db.commit()
        return None
    assistant = db.get(Message, task.assistant_message_id)
    return GenerationTaskRead(
        id=task.id,
        conversation_id=task.conversation_id,
        user_message_id=task.user_message_id,
        assistant_message_id=task.assistant_message_id,
        status=task.status,
        phase=task.phase or "retrieving",
        cancel_requested=bool(task.cancel_requested),
        last_event_id=task.last_event_id,
        content=assistant.content if assistant is not None else "",
        citations=assistant.citations if assistant is not None else [],
        error=task.error,
        created_at=task.created_at,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


@router.get("", response_model=list[ConversationRead])
def list_conversations(
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> list[ConversationRead]:
    rows = db.execute(
        select(Conversation, func.count(Message.id))
        .outerjoin(Message, Message.conversation_id == Conversation.id)
        .group_by(Conversation.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    all_kb_ids = {
        kb_id
        for conversation, _ in rows
        for kb_id in _knowledge_base_ids(conversation)
    }
    kb_names = {
        knowledge_base.id: knowledge_base.name
        for knowledge_base in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(all_kb_ids))
        )
    }
    return [
        ConversationRead(
            id=conversation.id,
            knowledge_base_id=conversation.knowledge_base_id,
            knowledge_base_ids=_knowledge_base_ids(conversation),
            knowledge_base_names=[
                kb_names[kb_id]
                for kb_id in _knowledge_base_ids(conversation)
                if kb_id in kb_names
            ],
            document_id=conversation.document_id,
            title=conversation.title,
            llm_profile_id=conversation.llm_profile_id,
            summary=conversation.summary,
            summary_status=_summary_status(conversation),
            summary_version=conversation.summary_version or 0,
            summary_message_count=conversation.summary_message_count or 0,
            summary_updated_at=conversation.summary_updated_at,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=count,
            active_task=_active_task_read(db, conversation.id),
        )
        for conversation, count in rows
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> ConversationDetail:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("对话不存在")
    messages = memory.messages(db, conversation_id)
    knowledge_base_ids = _knowledge_base_ids(conversation)
    knowledge_bases = list(
        db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(knowledge_base_ids))
        )
    )
    kb_names = {knowledge_base.id: knowledge_base.name for knowledge_base in knowledge_bases}
    return ConversationDetail(
        id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        knowledge_base_ids=knowledge_base_ids,
        knowledge_base_names=[
            kb_names[kb_id] for kb_id in knowledge_base_ids if kb_id in kb_names
        ],
        document_id=conversation.document_id,
        title=conversation.title,
        llm_profile_id=conversation.llm_profile_id,
        summary=conversation.summary,
        summary_status=_summary_status(conversation),
        summary_version=conversation.summary_version or 0,
        summary_message_count=conversation.summary_message_count or 0,
        summary_updated_at=conversation.summary_updated_at,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        messages=[MessageRead.model_validate(message) for message in messages],
        active_task=_active_task_read(db, conversation.id),
    )


@router.delete("/{conversation_id}", status_code=204)
def delete_conversation(
    conversation_id: str,
    _: str = Depends(get_current_admin),
    db: Session = Depends(get_db),
) -> Response:
    conversation = db.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("对话不存在")
    active_task = db.scalar(
        select(GenerationTask).where(
            GenerationTask.conversation_id == conversation_id,
            GenerationTask.status.in_(("queued", "running")),
        )
    )
    if active_task is not None:
        from app.services.chat_task_service import ChatTaskService

        ChatTaskService().cancel_task(db, active_task.id)
    db.delete(conversation)
    db.commit()
    memory.invalidate(conversation_id)
    return Response(status_code=204)



