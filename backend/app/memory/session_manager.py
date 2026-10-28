import logging
import uuid
from datetime import UTC, datetime

import redis
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.db.models import (
    Conversation,
    ConversationKnowledgeBase,
    Message,
    ModelProfile,
)
from app.generation.prompts import SUMMARY_PROMPT
from app.indexing.model_gateway import ModelGateway
from app.schemas import Citation

logger = logging.getLogger(__name__)


class ConversationMemory:
    def __init__(self, gateway: ModelGateway | None = None):
        self.gateway = gateway or ModelGateway()
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            self.redis = None

    def _cache_key(self, conversation_id: str) -> str:
        return f"zhiyuan:history:{conversation_id}"

    def invalidate(self, conversation_id: str) -> None:
        if self.redis is None:
            return
        try:
            self.redis.delete(self._cache_key(conversation_id))
        except Exception:
            logger.warning("failed to invalidate conversation cache", exc_info=True)

    def get_or_create(
        self,
        db: Session,
        *,
        kb_ids: list[str],
        conversation_id: str | None,
        llm_profile_id: str,
        first_query: str,
        document_id: str | None = None,
    ) -> Conversation:
        if not kb_ids:
            raise ValidationError("至少选择一个知识库")
        unique_kb_ids = list(dict.fromkeys(kb_ids))
        if conversation_id:
            conversation = db.get(Conversation, conversation_id)
            if conversation is None:
                raise NotFoundError("对话不存在")
            saved_kb_ids = self.knowledge_base_ids(conversation)
            if saved_kb_ids != unique_kb_ids:
                raise ValidationError("对话知识库范围不匹配，请新建对话")
            if document_id and conversation.document_id and conversation.document_id != document_id:
                raise NotFoundError("对话所属文档不匹配")
            if conversation.document_id is None and document_id:
                conversation.document_id = document_id
            conversation.llm_profile_id = llm_profile_id
            if not conversation.knowledge_base_links:
                self._add_scope_links(conversation, unique_kb_ids)
            db.commit()
            return conversation
        conversation = Conversation(
            id=str(uuid.uuid4()),
            knowledge_base_id=unique_kb_ids[0],
            document_id=document_id,
            title=first_query.strip()[:80] or "新对话",
            llm_profile_id=llm_profile_id,
        )
        db.add(conversation)
        db.flush()
        self._add_scope_links(conversation, unique_kb_ids)
        db.commit()
        db.refresh(conversation)
        return conversation

    @staticmethod
    def knowledge_base_ids(conversation: Conversation) -> list[str]:
        ids = [link.knowledge_base_id for link in conversation.knowledge_base_links]
        if ids:
            return ids
        return [conversation.knowledge_base_id] if conversation.knowledge_base_id else []

    @staticmethod
    def _add_scope_links(conversation: Conversation, kb_ids: list[str]) -> None:
        for position, kb_id in enumerate(kb_ids):
            conversation.knowledge_base_links.append(
                ConversationKnowledgeBase(
                    conversation_id=conversation.id,
                    knowledge_base_id=kb_id,
                    position=position,
                )
            )

    def messages(self, db: Session, conversation_id: str) -> list[Message]:
        return list(
            db.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
        )

    def history_text(self, db: Session, conversation: Conversation) -> str:
        cache_key = self._cache_key(conversation.id)
        if self.redis is not None:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    return cached
            except Exception:
                logger.warning("failed to read conversation cache", exc_info=True)

        messages = self.messages(db, conversation.id)
        recent = messages[conversation.summary_message_count :][
            -(settings.session_window_size * 2) :
        ]
        lines: list[str] = []
        if conversation.summary:
            lines.append(f"【较早对话摘要】\n{conversation.summary}")
        for message in recent:
            role = "用户" if message.role == "user" else "助手"
            lines.append(f"{role}：{message.content}")
        text = "\n".join(lines)
        if self.redis is not None:
            try:
                self.redis.setex(cache_key, 3600, text)
            except Exception:
                logger.warning("failed to write conversation cache", exc_info=True)
        return text

    def add_exchange(
        self,
        db: Session,
        conversation: Conversation,
        *,
        question: str,
        answer: str,
        rewritten_query: str,
        citations: list[Citation],
        context: list[dict],
        usage: dict,
    ) -> Message:
        user_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="user",
            content=question,
            rewritten_query=rewritten_query,
        )
        assistant_message = Message(
            id=str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            rewritten_query=rewritten_query,
            citations=[citation.model_dump() for citation in citations],
            context=context,
            usage=usage,
        )
        db.add_all([user_message, assistant_message])
        conversation.updated_at = datetime.now(UTC)
        db.commit()
        self.invalidate(conversation.id)
        return assistant_message

    async def maybe_summarize(
        self,
        db: Session,
        conversation: Conversation,
        llm_profile: ModelProfile,
    ) -> None:
        total = db.scalar(
            select(func.count(Message.id)).where(Message.conversation_id == conversation.id)
        ) or 0
        if total < settings.summary_trigger_messages:
            return
        cutoff = total - settings.summary_keep_messages
        if cutoff <= conversation.summary_message_count:
            return
        messages = self.messages(db, conversation.id)
        to_summarize = messages[conversation.summary_message_count : cutoff]
        if not to_summarize:
            return
        new_text = "\n".join(
            f"{'用户' if message.role == 'user' else '助手'}：{message.content}"
            for message in to_summarize
        )
        prompt = SUMMARY_PROMPT.format(
            existing_summary=conversation.summary or "无",
            new_messages=new_text,
        )
        try:
            summary = await self.gateway.complete(
                llm_profile,
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=settings.summary_max_tokens,
            )
        except Exception:
            logger.exception("conversation summary generation failed")
            return
        conversation.summary = summary.strip()
        conversation.summary_message_count = cutoff
        db.commit()
        self.invalidate(conversation.id)

    @staticmethod
    def json_messages(messages: list[Message]) -> list[dict]:
        return [
            {
                "id": message.id,
                "conversation_id": message.conversation_id,
                "role": message.role,
                "content": message.content,
                "rewritten_query": message.rewritten_query,
                "citations": message.citations,
                "context": message.context,
                "usage": message.usage,
                "created_at": message.created_at.isoformat(),
            }
            for message in messages
        ]

