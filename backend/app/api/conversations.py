from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.security import get_current_admin
from app.db.models import Conversation, Message
from app.db.session import get_db
from app.memory.session_manager import ConversationMemory
from app.schemas import ConversationDetail, ConversationRead, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])
memory = ConversationMemory()


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
    return [
        ConversationRead(
            id=conversation.id,
            knowledge_base_id=conversation.knowledge_base_id,
            document_id=conversation.document_id,
            title=conversation.title,
            llm_profile_id=conversation.llm_profile_id,
            summary=conversation.summary,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            message_count=count,
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
    return ConversationDetail(
        id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        document_id=conversation.document_id,
        title=conversation.title,
        llm_profile_id=conversation.llm_profile_id,
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=len(messages),
        messages=[MessageRead.model_validate(message) for message in messages],
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
    db.delete(conversation)
    db.commit()
    memory.invalidate(conversation_id)
    return Response(status_code=204)



