from fastapi import APIRouter, Depends, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.security import get_current_admin
from app.db.models import Conversation, KnowledgeBase, Message
from app.db.session import get_db
from app.memory.session_manager import ConversationMemory
from app.schemas import ConversationDetail, ConversationRead, MessageRead

router = APIRouter(prefix="/conversations", tags=["conversations"])
memory = ConversationMemory()


def _knowledge_base_ids(conversation: Conversation) -> list[str]:
    ids = [link.knowledge_base_id for link in conversation.knowledge_base_links]
    if ids:
        return ids
    return [conversation.knowledge_base_id] if conversation.knowledge_base_id else []


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



