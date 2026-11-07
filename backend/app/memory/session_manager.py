import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import redis
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.core.errors import NotFoundError, ValidationError
from app.db.models import (
    Conversation,
    ConversationKnowledgeBase,
    Message,
)
from app.indexing.model_gateway import ModelGateway
from app.retrieval.context_window import clip_head_tokens, clip_tail_tokens
from app.schemas import Citation

logger = logging.getLogger(__name__)


@dataclass
class HistoryContext:
    text: str
    token_count: int
    included_message_count: int
    truncated: bool
    summary_tokens: int = 0


@dataclass(frozen=True)
class _HistoryBlock:
    text: str
    message_count: int
    token_count: int
    truncated: bool


class ConversationMemory:
    def __init__(self, gateway: ModelGateway | None = None):
        self.gateway = gateway or ModelGateway()
        try:
            self.redis = redis.from_url(settings.redis_url, decode_responses=True)
        except Exception:
            self.redis = None

    def _cache_key(self, conversation_id: str) -> str:
        return f"zhiyuan:history:v2:{conversation_id}"

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

    @staticmethod
    def _is_stopped_assistant(message: Message) -> bool:
        usage = message.usage if isinstance(message.usage, dict) else {}
        return message.role == "assistant" and bool(usage.get("stopped"))

    @staticmethod
    def _is_incomplete_assistant(message: Message) -> bool:
        return message.role == "assistant" and str(getattr(message, "status", None) or "completed") in {
            "pending",
            "streaming",
            "failed",
        }

    @staticmethod
    def _truncate_message(content: str, max_tokens: int) -> tuple[str, bool]:
        if max_tokens <= 0 or count_tokens(content) <= max_tokens:
            return content, False
        marker = " [中间内容已截断] "
        marker_tokens = count_tokens(marker)
        available = max(0, max_tokens - marker_tokens)
        if available <= 0:
            return clip_head_tokens(content, max_tokens), True
        head_tokens = max(1, available * 2 // 3)
        tail_tokens = max(0, available - head_tokens)
        head = clip_head_tokens(content, head_tokens)
        tail = clip_tail_tokens(content, tail_tokens) if tail_tokens else ""
        return f"{head}{marker}{tail}".strip(), True

    @classmethod
    def _history_blocks(cls, messages: list[Message]) -> list[_HistoryBlock]:
        blocks: list[_HistoryBlock] = []
        index = 0
        max_message_tokens = max(1, settings.history_message_max_tokens)
        while index < len(messages):
            message = messages[index]
            if message.role == "user":
                next_message = messages[index + 1] if index + 1 < len(messages) else None
                user_text, user_truncated = cls._truncate_message(
                    message.content, max_message_tokens
                )
                if next_message is not None and next_message.role == "assistant":
                    if cls._is_incomplete_assistant(next_message):
                        index += 2
                        continue
                    if cls._is_stopped_assistant(next_message):
                        block_text = f"[未完成提问]\n用户：{user_text}"
                        message_count = 2
                        truncated = user_truncated
                    else:
                        assistant_text, assistant_truncated = cls._truncate_message(
                            next_message.content, max_message_tokens
                        )
                        block_text = f"用户：{user_text}\n助手：{assistant_text}"
                        message_count = 2
                        truncated = user_truncated or assistant_truncated
                    index += 2
                else:
                    block_text = f"[未完成提问]\n用户：{user_text}"
                    message_count = 1
                    truncated = user_truncated
                    index += 1
            else:
                if cls._is_stopped_assistant(message) or cls._is_incomplete_assistant(message):
                    index += 1
                    continue
                assistant_text, assistant_truncated = cls._truncate_message(
                    message.content, max_message_tokens
                )
                block_text = f"助手：{assistant_text}"
                message_count = 1
                truncated = assistant_truncated
                index += 1
            blocks.append(
                _HistoryBlock(
                    text=block_text,
                    message_count=message_count,
                    token_count=count_tokens(block_text),
                    truncated=truncated,
                )
            )
        return blocks

    def history_context(
        self,
        db: Session,
        conversation: Conversation,
        *,
        max_tokens: int | None = None,
    ) -> HistoryContext:
        cache_key = self._cache_key(conversation.id)
        if self.redis is not None:
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    payload = json.loads(cached)
                    return HistoryContext(**payload)
            except Exception:
                logger.warning("failed to read conversation cache", exc_info=True)

        budget = max(0, max_tokens if max_tokens is not None else settings.history_max_tokens)
        if budget == 0:
            return HistoryContext(text="", token_count=0, included_message_count=0, truncated=False)

        messages = self.messages(db, conversation.id)
        lines: list[str] = []
        summary_tokens = 0
        summary_truncated = False
        if conversation.summary:
            summary_label = "【较早对话摘要】\n"
            label_tokens = count_tokens(summary_label)
            summary_budget = min(
                settings.summary_max_tokens,
                max(0, budget - label_tokens),
            )
            summary_text = clip_head_tokens(conversation.summary, summary_budget)
            if summary_text:
                summary_block = f"{summary_label}{summary_text}"
                lines.append(summary_block)
                summary_tokens = count_tokens(summary_block)
                summary_truncated = count_tokens(conversation.summary) > summary_budget

        recent_messages = messages[conversation.summary_message_count :]
        blocks = self._history_blocks(recent_messages)
        selected: list[_HistoryBlock] = []
        remaining = max(0, budget - summary_tokens)
        included_messages = 0
        truncated = summary_truncated
        for block in reversed(blocks):
            if included_messages + block.message_count > settings.session_window_size * 2:
                truncated = True
                break
            if block.token_count > remaining:
                truncated = True
                break
            selected.append(block)
            included_messages += block.message_count
            remaining -= block.token_count
            truncated = truncated or block.truncated

        for block in reversed(selected):
            lines.append(block.text)
        text = "\n".join(lines)
        context = HistoryContext(
            text=text,
            token_count=count_tokens(text),
            included_message_count=included_messages,
            truncated=truncated,
            summary_tokens=summary_tokens,
        )
        if self.redis is not None:
            try:
                self.redis.setex(
                    cache_key,
                    3600,
                    json.dumps(
                        {
                            "text": context.text,
                            "token_count": context.token_count,
                            "included_message_count": context.included_message_count,
                            "truncated": context.truncated,
                            "summary_tokens": context.summary_tokens,
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception:
                logger.warning("failed to write conversation cache", exc_info=True)
        return context

    def history_text(self, db: Session, conversation: Conversation) -> str:
        return self.history_context(db, conversation).text

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
        user_message, assistant_message = self.start_exchange(
            db,
            conversation,
            question=question,
            rewritten_query=rewritten_query,
        )
        return self.finish_exchange(
            db,
            conversation,
            assistant_message,
            answer=answer,
            rewritten_query=rewritten_query,
            citations=citations,
            context=context,
            usage=usage,
        )

    def start_exchange(
        self,
        db: Session,
        conversation: Conversation,
        *,
        question: str,
        rewritten_query: str = "",
        user_message_id: str | None = None,
        assistant_message_id: str | None = None,
        generation_task_id: str | None = None,
        commit: bool = True,
    ) -> tuple[Message, Message]:
        user_message = Message(
            id=user_message_id or str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="user",
            content=question,
            rewritten_query=rewritten_query,
            status="completed",
            generation_task_id=generation_task_id,
        )
        assistant_message = Message(
            id=assistant_message_id or str(uuid.uuid4()),
            conversation_id=conversation.id,
            role="assistant",
            content="",
            rewritten_query=rewritten_query,
            status="pending",
            generation_task_id=generation_task_id,
        )
        db.add_all([user_message, assistant_message])
        conversation.updated_at = datetime.now(UTC)
        if commit:
            db.commit()
            db.refresh(user_message)
            db.refresh(assistant_message)
            self.invalidate(conversation.id)
        else:
            db.flush()
        return user_message, assistant_message

    def finish_exchange(
        self,
        db: Session,
        conversation: Conversation,
        assistant_message: Message,
        *,
        answer: str,
        rewritten_query: str,
        citations: list[Citation],
        context: list[dict],
        usage: dict,
        status: str = "completed",
    ) -> Message:
        assistant_message.content = answer
        assistant_message.rewritten_query = rewritten_query
        assistant_message.citations = [citation.model_dump() for citation in citations]
        assistant_message.context = context
        assistant_message.usage = usage
        assistant_message.status = status
        if assistant_message.generation_task_id:
            user_message = db.scalar(
                select(Message).where(
                    Message.generation_task_id == assistant_message.generation_task_id,
                    Message.role == "user",
                )
            )
            if user_message is not None:
                user_message.rewritten_query = rewritten_query
        conversation.updated_at = datetime.now(UTC)
        db.commit()
        self.invalidate(conversation.id)
        return assistant_message

    def mark_task_message_status(
        self,
        db: Session,
        assistant_message: Message,
        *,
        status: str,
        error: str = "",
    ) -> None:
        assistant_message.status = status
        usage = dict(assistant_message.usage or {})
        if error:
            usage["error"] = error
        assistant_message.usage = usage
        db.commit()

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

