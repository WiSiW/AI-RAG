from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.models import Base, Conversation, KnowledgeBase, Message, ModelProfile
from app.memory.session_manager import ConversationMemory
from app.schemas import ConversationRead
from app.services.conversation_summary_service import ConversationSummaryService


def message(index: int, *, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"m{index}",
        role="user" if index % 2 == 0 else "assistant",
        content=content or f"第 {index} 条消息",
        usage={},
    )


def test_summary_plan_uses_unsummarized_count_instead_of_total(monkeypatch):
    monkeypatch.setattr(settings, "summary_trigger_tokens", 100000)
    conversation = SimpleNamespace(summary_message_count=0)
    messages = [message(index) for index in range(16)]

    plan = ConversationSummaryService.summary_plan(conversation, messages)

    assert plan is not None
    cutoff, _, prepared = plan
    assert cutoff == 8
    assert len(prepared) == 8

    conversation.summary_message_count = 8
    messages.extend([message(16), message(17)])
    assert ConversationSummaryService.summary_plan(conversation, messages) is None

    messages.extend([message(index) for index in range(18, 24)])
    plan = ConversationSummaryService.summary_plan(conversation, messages)
    assert plan is not None
    assert plan[0] == 16


def test_summary_plan_can_trigger_by_unsummarized_tokens(monkeypatch):
    monkeypatch.setattr(settings, "summary_trigger_messages", 100)
    monkeypatch.setattr(settings, "summary_trigger_tokens", 3000)
    conversation = SimpleNamespace(summary_message_count=0)
    messages = [message(index, content="长" * 600) for index in range(6)]

    plan = ConversationSummaryService.summary_plan(conversation, messages)

    assert plan is not None
    assert plan[0] == 4


def test_history_context_respects_budget_and_marks_incomplete_turn(monkeypatch):
    monkeypatch.setattr(settings, "history_max_tokens", 100)
    monkeypatch.setattr(settings, "history_message_max_tokens", 30)
    monkeypatch.setattr(settings, "summary_max_tokens", 30)
    memory = ConversationMemory(gateway=SimpleNamespace())
    memory.redis = None
    messages = [
        SimpleNamespace(id="m1", role="user", content="第一部分" * 100, usage={}),
        SimpleNamespace(
            id="m2",
            role="assistant",
            content="未完成回答" * 100,
            usage={"stopped": True},
        ),
        SimpleNamespace(id="m3", role="user", content="第二部分", usage={}),
        SimpleNamespace(id="m4", role="assistant", content="完整回答", usage={}),
    ]
    monkeypatch.setattr(memory, "messages", lambda db, conversation_id: messages)
    conversation = SimpleNamespace(
        id="conversation-1",
        summary="较早摘要" * 100,
        summary_message_count=0,
    )

    context = memory.history_context(object(), conversation)

    assert context.token_count <= 100
    assert context.truncated is True
    assert "较早对话摘要" in context.text
    assert "未完成回答" not in context.text
    assert "完整回答" in context.text


def test_structured_summary_merge_and_render_are_bounded():
    existing = ConversationSummaryService._sanitize_patch(
        {
            "narrative": "已有摘要",
            "user_goals": ["排查端口问题"],
            "confirmed_facts": [{"text": "端口是 5432", "source_message_ids": ["m1"]}],
            "constraints": [],
            "decisions": [],
            "open_questions": [],
            "key_entities": ["PostgreSQL"],
        },
        {"m1"},
    )
    incoming = ConversationSummaryService._sanitize_patch(
        {
            "narrative": "新增摘要",
            "user_goals": ["排查端口问题"],
            "confirmed_facts": [{"text": "端口是 5432", "source_message_ids": ["m2"]}],
            "constraints": ["只读副本"],
            "decisions": ["先检查连接池"],
            "open_questions": ["是否只在高峰期出现？"],
            "key_entities": ["连接池"],
        },
        {"m2"},
    )

    merged = ConversationSummaryService.merge(existing, incoming)
    rendered = ConversationSummaryService.render(merged)

    assert merged.user_goals == ["排查端口问题"]
    assert merged.confirmed_facts[0].source_message_ids == ["m1", "m2"]
    assert "约束条件" in rendered
    assert "连接池" in rendered


@pytest.mark.asyncio
async def test_invalid_model_output_uses_degraded_fallback(monkeypatch):
    class BadGateway:
        async def complete(self, *args, **kwargs):
            return "not-json"

    monkeypatch.setattr(settings, "summary_trigger_tokens", 100000)
    service = ConversationSummaryService(gateway=BadGateway(), memory=SimpleNamespace())
    batch = [
        SimpleNamespace(message_id="m1", role="user", content="如何排查端口？"),
        SimpleNamespace(message_id="m2", role="assistant", content="先检查连接池。"),
    ]

    data, degraded = await service._generate_patch(SimpleNamespace(), batch)

    assert degraded is True
    assert data.narrative
    assert "如何排查端口" in data.narrative


def _database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    return engine, session_factory


def _conversation_fixture(db):
    embedding = ModelProfile(
        id="embedding-1",
        name="embedding",
        kind="embedding",
        base_url="https://example.test/v1",
        model_name="embedding-test",
        dimensions=3,
        is_active=True,
    )
    llm = ModelProfile(
        id="llm-1",
        name="llm",
        kind="llm",
        base_url="https://example.test/v1",
        model_name="chat-test",
        is_active=True,
    )
    kb = KnowledgeBase(
        id="kb-1",
        name="测试知识库",
        embedding_profile_id=embedding.id,
    )
    db.add_all([embedding, llm, kb])
    db.flush()
    conversation = Conversation(
        id="conversation-1",
        knowledge_base_id=kb.id,
        llm_profile_id=llm.id,
        title="测试会话",
        summary="",
        summary_json={},
        summary_message_count=0,
        summary_status="idle",
        summary_version=0,
    )
    db.add(conversation)
    db.commit()
    return conversation


def test_summary_scheduling_is_idempotent(monkeypatch):
    engine, session_factory = _database()
    db = session_factory()
    try:
        conversation = _conversation_fixture(db)
        response_model = ConversationRead.model_validate(conversation)
        assert response_model.summary_status == "idle"
        assert response_model.summary_version == 0
        assert response_model.summary_message_count == 0
        start = datetime.now(UTC)
        for index in range(16):
            db.add(
                Message(
                    id=f"m{index}",
                    conversation_id=conversation.id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"消息 {index}",
                    created_at=start + timedelta(seconds=index),
                )
            )
        db.commit()
        memory = ConversationMemory(gateway=SimpleNamespace())
        memory.redis = None
        service = ConversationSummaryService(gateway=SimpleNamespace(), memory=memory)
        calls: list[dict] = []

        import app.workers.celery_app as celery_module

        monkeypatch.setattr(
            celery_module.celery_app,
            "send_task",
            lambda *args, **kwargs: calls.append(kwargs),
        )

        assert service.maybe_schedule(db, conversation) is True
        db.refresh(conversation)
        assert conversation.summary_status == "running"
        assert service.maybe_schedule(db, conversation) is False
        assert len(calls) == 1
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_summary_task_updates_structure_and_version_atomically(monkeypatch):
    engine, session_factory = _database()
    db = session_factory()
    try:
        conversation = _conversation_fixture(db)
        conversation.summary_status = "running"
        conversation.summary_task_id = "task-1"
        conversation.summary_started_at = datetime.now(UTC)
        start = datetime.now(UTC)
        for index in range(16):
            db.add(
                Message(
                    id=f"m{index}",
                    conversation_id=conversation.id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"消息 {index}",
                    created_at=start + timedelta(seconds=index),
                )
            )
        db.commit()

        class Gateway:
            async def complete(self, *args, **kwargs):
                return (
                    '{"narrative":"会话概述","user_goals":["排查故障"],'
                    '"confirmed_facts":[],"constraints":["只读"],"decisions":[],'
                    '"open_questions":[],"key_entities":["连接池"]}'
                )

        memory = ConversationMemory(gateway=Gateway())
        memory.redis = None
        service = ConversationSummaryService(gateway=memory.gateway, memory=memory)
        monkeypatch.setattr(service, "maybe_schedule", lambda db, conversation: False)

        assert await service.summarize(db, conversation.id, "task-1") is True
        db.refresh(conversation)
        assert conversation.summary_status == "succeeded"
        assert conversation.summary_version == 1
        assert conversation.summary_message_count == 8
        assert conversation.summary_json["user_goals"] == ["排查故障"]
        assert "连接池" in conversation.summary

        assert await service.summarize(db, conversation.id, "stale-task") is False
        db.refresh(conversation)
        assert conversation.summary_version == 1
    finally:
        db.close()
        engine.dispose()

