import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, or_, update
from sqlalchemy.orm import Session

from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.db.models import Conversation, Message, ModelProfile
from app.generation.prompts import SUMMARY_PROMPT
from app.indexing.model_gateway import ModelGateway
from app.memory.session_manager import ConversationMemory
from app.retrieval.context_window import clip_head_tokens

logger = logging.getLogger(__name__)

SUMMARY_LIST_LIMITS = {
    "user_goals": 8,
    "constraints": 12,
    "decisions": 12,
    "open_questions": 12,
    "key_entities": 20,
}
SUMMARY_FACT_LIMIT = 20
SUMMARY_TEXT_LIMIT = 240
SUMMARY_FACT_LIMIT_CHARS = 300
SUMMARY_NARRATIVE_LIMIT = 1600


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _clean_text(value: object, limit: int = SUMMARY_TEXT_LIMIT) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].strip()


def _clean_text_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _clean_text(item)
        normalized = _normalize(text)
        if not text or normalized in seen:
            continue
        seen.add(normalized)
        output.append(text)
        if len(output) >= limit:
            break
    return output


def _extract_json(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


@dataclass
class SummaryFact:
    text: str
    source_message_ids: list[str] = field(default_factory=list)


@dataclass
class ConversationSummaryData:
    narrative: str = ""
    user_goals: list[str] = field(default_factory=list)
    confirmed_facts: list[SummaryFact] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: object) -> "ConversationSummaryData":
        if not isinstance(payload, dict):
            return cls()
        facts: list[SummaryFact] = []
        raw_facts = payload.get("confirmed_facts")
        if isinstance(raw_facts, list):
            for item in raw_facts:
                if not isinstance(item, dict):
                    continue
                text = _clean_text(item.get("text"), SUMMARY_FACT_LIMIT_CHARS)
                ids = _clean_text_list(item.get("source_message_ids"), 20)
                if text:
                    facts.append(SummaryFact(text=text, source_message_ids=ids))
        return cls(
            narrative=_clean_text(payload.get("narrative"), SUMMARY_NARRATIVE_LIMIT),
            user_goals=_clean_text_list(
                payload.get("user_goals"), SUMMARY_LIST_LIMITS["user_goals"]
            ),
            confirmed_facts=facts[:SUMMARY_FACT_LIMIT],
            constraints=_clean_text_list(
                payload.get("constraints"), SUMMARY_LIST_LIMITS["constraints"]
            ),
            decisions=_clean_text_list(
                payload.get("decisions"), SUMMARY_LIST_LIMITS["decisions"]
            ),
            open_questions=_clean_text_list(
                payload.get("open_questions"), SUMMARY_LIST_LIMITS["open_questions"]
            ),
            key_entities=_clean_text_list(
                payload.get("key_entities"), SUMMARY_LIST_LIMITS["key_entities"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "narrative": self.narrative,
            "user_goals": list(self.user_goals),
            "confirmed_facts": [
                {"text": item.text, "source_message_ids": list(item.source_message_ids)}
                for item in self.confirmed_facts
            ],
            "constraints": list(self.constraints),
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "key_entities": list(self.key_entities),
        }

    def is_empty(self) -> bool:
        return not any(
            (
                self.narrative,
                self.user_goals,
                self.confirmed_facts,
                self.constraints,
                self.decisions,
                self.open_questions,
                self.key_entities,
            )
        )


@dataclass(frozen=True)
class _SummaryMessage:
    message_id: str
    role: str
    content: str


class ConversationSummaryService:
    def __init__(
        self,
        gateway: ModelGateway | None = None,
        memory: ConversationMemory | None = None,
    ):
        self.gateway = gateway or ModelGateway()
        self.memory = memory or ConversationMemory(self.gateway)

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

    @classmethod
    def _summary_messages(cls, messages: list[Message]) -> list[_SummaryMessage]:
        prepared: list[_SummaryMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            content = re.sub(r"\s+", " ", message.content or "").strip()
            if message.role == "user":
                next_message = messages[index + 1] if index + 1 < len(messages) else None
                if next_message is not None and next_message.role == "assistant":
                    if cls._is_incomplete_assistant(next_message):
                        index += 2
                        continue
                    if cls._is_stopped_assistant(next_message):
                        content = f"[未完成提问] {content}"
                        prepared.append(_SummaryMessage(message.id, "user", content))
                    else:
                        prepared.append(_SummaryMessage(message.id, "user", content))
                        assistant_content = re.sub(
                            r"\s+", " ", next_message.content or ""
                        ).strip()
                        if assistant_content:
                            prepared.append(
                                _SummaryMessage(
                                    next_message.id,
                                    "assistant",
                                    assistant_content,
                                )
                            )
                    index += 2
                    continue
                prepared.append(_SummaryMessage(message.id, "user", content))
            elif not cls._is_stopped_assistant(message) and not cls._is_incomplete_assistant(message):
                prepared.append(_SummaryMessage(message.id, "assistant", content))
            index += 1
        return [item for item in prepared if item.content]

    @staticmethod
    def _keep_cutoff(messages: list[Message], start: int) -> int:
        cutoff = len(messages)
        kept_count = 0
        kept_tokens = 0
        for message in reversed(messages[start:]):
            message_tokens = count_tokens(message.content or "")
            if kept_count >= settings.summary_keep_messages:
                break
            if kept_count >= 2 and kept_tokens + message_tokens > settings.summary_keep_tokens:
                break
            cutoff -= 1
            kept_count += 1
            kept_tokens += message_tokens
        max_cutoff = max(start, len(messages) - 2)
        cutoff = min(cutoff, max_cutoff)
        if (cutoff - start) % 2:
            cutoff -= 1
        return cutoff

    @classmethod
    def summary_plan(
        cls,
        conversation: Conversation,
        messages: list[Message],
    ) -> tuple[int, list[Message], list[_SummaryMessage]] | None:
        start = max(0, int(getattr(conversation, "summary_message_count", 0) or 0))
        unsummarized = messages[start:]
        if not unsummarized:
            return None
        prepared_all = cls._summary_messages(unsummarized)
        unsummarized_tokens = sum(count_tokens(item.content) for item in prepared_all)
        if (
            len(unsummarized) < settings.summary_trigger_messages
            and unsummarized_tokens < settings.summary_trigger_tokens
        ):
            return None
        cutoff = cls._keep_cutoff(messages, start)
        if cutoff <= start:
            return None
        to_summarize = messages[start:cutoff]
        prepared = cls._summary_messages(to_summarize)
        if not prepared:
            return None
        return cutoff, to_summarize, prepared

    def maybe_schedule(self, db: Session, conversation: Conversation) -> bool:
        try:
            return self._maybe_schedule(db, conversation)
        except Exception:
            db.rollback()
            logger.exception(
                "failed to schedule conversation summary conversation_id=%s",
                conversation.id,
            )
            return False

    def _maybe_schedule(self, db: Session, conversation: Conversation) -> bool:
        messages = self.memory.messages(db, conversation.id)
        plan = self.summary_plan(conversation, messages)
        if plan is None:
            return False

        now = datetime.now(UTC)
        stale_before = now - timedelta(seconds=settings.summary_task_stale_seconds)
        task_id = str(uuid.uuid4())
        result = db.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation.id,
                or_(
                    Conversation.summary_status.is_(None),
                    Conversation.summary_status != "running",
                    Conversation.summary_started_at.is_(None),
                    Conversation.summary_started_at < stale_before,
                ),
            )
            .values(
                summary_status="running",
                summary_task_id=task_id,
                summary_started_at=now,
                summary_error=None,
            )
        )
        db.commit()
        if result.rowcount != 1:
            return False

        try:
            from app.workers.celery_app import celery_app

            celery_app.send_task(
                "zhiyuan.summarize_conversation",
                args=[conversation.id, task_id],
                task_id=task_id,
                queue="celery",
            )
            return True
        except Exception as exc:
            logger.exception("failed to enqueue conversation summary task")
            self.mark_failed(db, conversation.id, task_id, f"任务队列不可用：{exc}")
            return False

    def _profile(self, db: Session, conversation: Conversation) -> ModelProfile | None:
        if not conversation.llm_profile_id:
            return None
        profile = db.get(ModelProfile, conversation.llm_profile_id)
        if profile is None or not profile.is_active or profile.kind not in {"llm", "enrichment"}:
            return None
        return profile

    @staticmethod
    def _batches(messages: list[_SummaryMessage]) -> list[list[_SummaryMessage]]:
        max_tokens = max(1, settings.summary_batch_input_tokens)
        batches: list[list[_SummaryMessage]] = []
        current: list[_SummaryMessage] = []
        current_tokens = 0
        for message in messages:
            prefix = f"{message.message_id} {message.role} "
            text = f"{prefix}{message.content}"
            tokens = count_tokens(text)
            if tokens > max_tokens:
                content_budget = max(1, max_tokens - count_tokens(prefix))
                content = clip_head_tokens(message.content, content_budget)
                message = _SummaryMessage(message.message_id, message.role, content)
                tokens = count_tokens(f"{prefix}{message.content}")
            if current and current_tokens + tokens > max_tokens:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(message)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _fallback_patch(self, messages: list[_SummaryMessage]) -> ConversationSummaryData:
        user_goals = []
        open_questions = []
        snippets = []
        for message in messages:
            text = _clean_text(message.content, SUMMARY_TEXT_LIMIT)
            if not text:
                continue
            snippets.append(f"{'用户' if message.role == 'user' else '助手'}：{text}")
            if message.role == "user":
                user_goals.append(text)
                if text.endswith(("?", "？")):
                    open_questions.append(text)
        narrative = "；".join(snippets)[:SUMMARY_NARRATIVE_LIMIT]
        return ConversationSummaryData(
            narrative=narrative,
            user_goals=_clean_text_list(user_goals, SUMMARY_LIST_LIMITS["user_goals"]),
            open_questions=_clean_text_list(
                open_questions, SUMMARY_LIST_LIMITS["open_questions"]
            ),
        )

    async def _generate_patch(
        self,
        profile: ModelProfile | None,
        batch: list[_SummaryMessage],
    ) -> tuple[ConversationSummaryData, bool]:
        if profile is None:
            return self._fallback_patch(batch), True
        payload = [
            {"id": item.message_id, "role": item.role, "content": item.content}
            for item in batch
        ]
        prompt = SUMMARY_PROMPT.format(messages=json.dumps(payload, ensure_ascii=False))
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                response = await self.gateway.complete(
                    profile,
                    [{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=settings.summary_max_tokens,
                )
                parsed = _extract_json(response)
                data = self._sanitize_patch(parsed, {item.message_id for item in batch})
                if data.is_empty():
                    raise ValueError("摘要模型返回空结构")
                return data, False
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "conversation summary attempt %s failed: %s",
                    attempt + 1,
                    last_error,
                    exc_info=True,
                )
        return self._fallback_patch(batch), True

    @staticmethod
    def _sanitize_patch(payload: object, batch_ids: set[str]) -> ConversationSummaryData:
        if not isinstance(payload, dict):
            raise ValueError("摘要模型未返回 JSON 对象")
        raw_facts = payload.get("confirmed_facts")
        facts: list[SummaryFact] = []
        if isinstance(raw_facts, list):
            for item in raw_facts:
                if not isinstance(item, dict):
                    continue
                text = _clean_text(item.get("text"), SUMMARY_FACT_LIMIT_CHARS)
                raw_ids = _clean_text_list(item.get("source_message_ids"), 20)
                source_ids = [message_id for message_id in raw_ids if message_id in batch_ids]
                if not source_ids:
                    source_ids = sorted(batch_ids)
                if text:
                    facts.append(SummaryFact(text=text, source_message_ids=source_ids))
        return ConversationSummaryData(
            narrative=_clean_text(payload.get("narrative"), SUMMARY_NARRATIVE_LIMIT),
            user_goals=_clean_text_list(
                payload.get("user_goals"), SUMMARY_LIST_LIMITS["user_goals"]
            ),
            confirmed_facts=facts[:SUMMARY_FACT_LIMIT],
            constraints=_clean_text_list(
                payload.get("constraints"), SUMMARY_LIST_LIMITS["constraints"]
            ),
            decisions=_clean_text_list(
                payload.get("decisions"), SUMMARY_LIST_LIMITS["decisions"]
            ),
            open_questions=_clean_text_list(
                payload.get("open_questions"), SUMMARY_LIST_LIMITS["open_questions"]
            ),
            key_entities=_clean_text_list(
                payload.get("key_entities"), SUMMARY_LIST_LIMITS["key_entities"]
            ),
        )

    @staticmethod
    def _merge_text(existing: list[str], incoming: list[str], limit: int) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for value in [*existing, *incoming]:
            normalized = _normalize(value)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(value)
            if len(merged) >= limit:
                break
        return merged

    @classmethod
    def merge(
        cls,
        existing: ConversationSummaryData,
        incoming: ConversationSummaryData,
    ) -> ConversationSummaryData:
        narrative_parts = [
            value
            for value in (existing.narrative.strip(), incoming.narrative.strip())
            if value
        ]
        narrative = " ".join(dict.fromkeys(narrative_parts))[:SUMMARY_NARRATIVE_LIMIT]
        facts: list[SummaryFact] = []
        fact_index: dict[str, SummaryFact] = {}
        for fact in [*existing.confirmed_facts, *incoming.confirmed_facts]:
            normalized = _normalize(fact.text)
            target = fact_index.get(normalized)
            if target is None:
                target = SummaryFact(fact.text, list(dict.fromkeys(fact.source_message_ids)))
                fact_index[normalized] = target
                facts.append(target)
            else:
                target.source_message_ids = list(
                    dict.fromkeys([*target.source_message_ids, *fact.source_message_ids])
                )
        return ConversationSummaryData(
            narrative=narrative,
            user_goals=cls._merge_text(
                existing.user_goals, incoming.user_goals, SUMMARY_LIST_LIMITS["user_goals"]
            ),
            confirmed_facts=facts[:SUMMARY_FACT_LIMIT],
            constraints=cls._merge_text(
                existing.constraints,
                incoming.constraints,
                SUMMARY_LIST_LIMITS["constraints"],
            ),
            decisions=cls._merge_text(
                existing.decisions, incoming.decisions, SUMMARY_LIST_LIMITS["decisions"]
            ),
            open_questions=cls._merge_text(
                existing.open_questions,
                incoming.open_questions,
                SUMMARY_LIST_LIMITS["open_questions"],
            ),
            key_entities=cls._merge_text(
                existing.key_entities,
                incoming.key_entities,
                SUMMARY_LIST_LIMITS["key_entities"],
            ),
        )

    @staticmethod
    def render(data: ConversationSummaryData) -> str:
        parts: list[str] = []
        if data.narrative:
            parts.append(data.narrative)
        sections = (
            ("用户目标", data.user_goals),
            ("已确认事实", [item.text for item in data.confirmed_facts]),
            ("约束条件", data.constraints),
            ("已做决策", data.decisions),
            ("未解决问题", data.open_questions),
            ("关键实体", data.key_entities),
        )
        for label, values in sections:
            if not values:
                continue
            parts.append(f"{label}：\n" + "\n".join(f"- {value}" for value in values))
        rendered = "\n\n".join(parts).strip()
        return clip_head_tokens(rendered, settings.summary_max_tokens).strip()

    async def summarize(self, db: Session, conversation_id: str, task_id: str) -> bool:
        conversation = db.get(Conversation, conversation_id)
        if conversation is None:
            return False
        if conversation.summary_task_id != task_id or conversation.summary_status != "running":
            return False
        messages = self.memory.messages(db, conversation.id)
        plan = self.summary_plan(conversation, messages)
        if plan is None:
            self._finish_without_update(db, conversation, task_id)
            return False
        cutoff, _, prepared = plan
        profile = self._profile(db, conversation)
        existing_payload = getattr(conversation, "summary_json", None)
        if isinstance(existing_payload, dict) and existing_payload:
            data = ConversationSummaryData.from_dict(existing_payload)
        elif conversation.summary:
            data = ConversationSummaryData(narrative=conversation.summary)
        else:
            data = ConversationSummaryData()

        degraded = profile is None
        for batch in self._batches(prepared):
            patch, used_fallback = await self._generate_patch(profile, batch)
            degraded = degraded or used_fallback
            data = self.merge(data, patch)

        summary_text = self.render(data)
        if not summary_text:
            raise ValueError("摘要结果为空")
        now = datetime.now(UTC)
        result = db.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation.id,
                Conversation.summary_task_id == task_id,
                Conversation.summary_status == "running",
            )
            .values(
                summary=summary_text,
                summary_json=data.to_dict(),
                summary_message_count=cutoff,
                summary_status="degraded" if degraded else "succeeded",
                summary_version=func.coalesce(Conversation.summary_version, 0) + 1,
                summary_updated_at=now,
                summary_error=None,
            )
        )
        db.commit()
        if result.rowcount != 1:
            return False
        self.memory.invalidate(conversation.id)
        db.refresh(conversation)
        self.maybe_schedule(db, conversation)
        return True

    def _finish_without_update(self, db: Session, conversation: Conversation, task_id: str) -> None:
        result = db.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation.id,
                Conversation.summary_task_id == task_id,
                Conversation.summary_status == "running",
            )
            .values(
                summary_status="succeeded" if conversation.summary else "idle",
                summary_updated_at=datetime.now(UTC),
                summary_error=None,
            )
        )
        db.commit()
        if result.rowcount == 1:
            self.memory.invalidate(conversation.id)

    def mark_failed(self, db: Session, conversation_id: str, task_id: str, error: str) -> None:
        db.rollback()
        db.execute(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.summary_task_id == task_id,
                Conversation.summary_status == "running",
            )
            .values(
                summary_status="failed",
                summary_error=error[:4000],
                summary_updated_at=datetime.now(UTC),
            )
        )
        db.commit()
        self.memory.invalidate(conversation_id)

