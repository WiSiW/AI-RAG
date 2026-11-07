import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.errors import ConflictError
from app.db.models import KnowledgeBase, Message, ModelProfile
from app.db.session import Base, SessionLocal, engine
from app.memory.session_manager import ConversationMemory
from app.schemas import ChatRequest, ChatTaskCreateRequest
from app.services.chat_task_service import ChatTaskService
from app.workers.celery_app import celery_app


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.streams: dict[str, list[tuple[str, dict[str, str]]]] = {}
        self.counters: dict[str, int] = {}
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    def ping(self) -> bool:
        return True

    def incr(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def xadd(self, key: str, fields: dict[str, str], **_: object) -> str:
        rows = self.streams.setdefault(key, [])
        stream_id = f"{len(rows) + 1}-0"
        rows.append((stream_id, fields))
        return stream_id

    def hset(self, key: str, mapping: dict[str, str]) -> int:
        self.hashes.setdefault(key, {}).update(mapping)
        return len(mapping)

    def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    def expire(self, key: str, seconds: int) -> bool:
        self.expirations[key] = seconds
        return True

    def set(self, key: str, value: str, **_: object) -> bool:
        self.values[key] = value
        return True

    def exists(self, key: str) -> int:
        return int(key in self.values)


def create_fixtures(db) -> tuple[str, str]:
    suffix = uuid.uuid4().hex[:8]
    embedding = ModelProfile(
        id=str(uuid.uuid4()),
        name=f"embedding-{suffix}",
        kind="embedding",
        base_url="http://localhost/v1",
        model_name="embedding-test",
        dimensions=3,
    )
    llm = ModelProfile(
        id=str(uuid.uuid4()),
        name=f"llm-{suffix}",
        kind="llm",
        base_url="http://localhost/v1",
        model_name="chat-test",
    )
    knowledge_base = KnowledgeBase(
        id=str(uuid.uuid4()),
        name=f"kb-{suffix}",
        embedding_profile_id=embedding.id,
        default_llm_profile_id=llm.id,
    )
    db.add_all([embedding, llm, knowledge_base])
    db.commit()
    return knowledge_base.id, llm.id


def make_request(kb_id: str, llm_id: str, client_request_id: str) -> ChatTaskCreateRequest:
    return ChatTaskCreateRequest(
        knowledge_base_ids=[kb_id],
        query="测试多会话任务",
        llm_profile_id=llm_id,
        client_request_id=client_request_id,
    )


def parse_sse_event(raw_event: str) -> tuple[str, dict]:
    event_name = "message"
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    return event_name, json.loads("\n".join(data_lines))


@pytest.fixture
def task_service(monkeypatch):
    Base.metadata.create_all(bind=engine)
    redis = FakeRedis()
    service = ChatTaskService(redis_client=redis)
    service.chat_service.memory.redis = None
    monkeypatch.setattr(
        celery_app,
        "send_task",
        lambda *args, **kwargs: SimpleNamespace(id="celery-task-1"),
    )
    return service, redis


def test_create_task_is_idempotent_and_blocks_same_conversation(task_service):
    service, redis = task_service
    db = SessionLocal()
    try:
        kb_id, llm_id = create_fixtures(db)
        request = make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")

        created = service.create_task(db, request)
        duplicate = service.create_task(db, request)

        assert duplicate.id == created.id
        assert created.status == "queued"
        assert db.get(Message, created.user_message_id).status == "completed"
        assert db.get(Message, created.assistant_message_id).status == "pending"
        queued_event = redis.streams[service.events_key(created.id)][0][1]
        assert queued_event["event"] == "queued"

        with pytest.raises(ConflictError):
            conflicting = make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")
            conflicting.conversation_id = created.conversation_id
            service.create_task(db, conflicting)
    finally:
        db.close()


def test_cancel_queued_task_releases_conversation(task_service):
    service, redis = task_service
    db = SessionLocal()
    try:
        kb_id, llm_id = create_fixtures(db)
        created = service.create_task(
            db, make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")
        )

        cancelled = service.cancel_task(db, created.id)

        assert cancelled.status == "cancelled"
        assert redis.values[service.cancel_key(created.id)] == "1"
        assistant = db.get(Message, created.assistant_message_id)
        assert assistant.status == "stopped"
        assert assistant.usage["stopped"] is True
        assert redis.streams[service.events_key(created.id)][-1][1]["event"] == "cancelled"

        next_task = service.create_task(
            db, make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")
        )
        assert next_task.status == "queued"
    finally:
        db.close()


def test_task_snapshot_reads_running_redis_content(task_service):
    service, _ = task_service
    db = SessionLocal()
    try:
        kb_id, llm_id = create_fixtures(db)
        created = service.create_task(
            db, make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")
        )
        service.publish_event(created, "token", {"content": "流式", "status": "running"})
        service.publish_event(
            created,
            "citations",
            {
                "citations": [
                    {
                        "index": 1,
                        "chunk_id": "chunk-1",
                        "source": "a.md",
                        "section": "一",
                        "snippet": "内容",
                    }
                ],
                "status": "running",
            },
        )

        snapshot = service.task_read(db, created)

        assert snapshot.content == "流式"
        assert snapshot.citations[0].chunk_id == "chunk-1"
        assert snapshot.last_event_id == "3-0"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_shared_stream_cancellation_marks_task_message_stopped(task_service, monkeypatch):
    service, _ = task_service
    db = SessionLocal()
    try:
        kb_id, llm_id = create_fixtures(db)
        created = service.create_task(
            db, make_request(kb_id, llm_id, f"request-{uuid.uuid4()}")
        )
        knowledge_base = SimpleNamespace(id=kb_id, name="task-test-kb")
        profile = SimpleNamespace(id=llm_id, model_name="chat-test")
        monkeypatch.setattr(service.chat_service, "_get_kbs", lambda db, ids: [knowledge_base])
        monkeypatch.setattr(
            service.chat_service, "_validate_document", lambda db, kbs, document_id: None
        )
        monkeypatch.setattr(
            service.chat_service, "_llm_profile", lambda db, kbs, profile_id: profile
        )
        monkeypatch.setattr(
            service.chat_service,
            "_rewrite_query",
            AsyncMock(return_value="不应执行"),
        )
        request = ChatRequest(
            knowledge_base_ids=[kb_id],
            query="取消中的问题",
            llm_profile_id=llm_id,
            conversation_id=created.conversation_id,
        )

        events = [
            parse_sse_event(event)
            async for event in service.chat_service.stream_chat(
                db,
                request,
                task_id=created.id,
                assistant_message_id=created.assistant_message_id,
                cancel_requested=lambda: True,
            )
        ]

        cancelled = [data for name, data in events if name == "cancelled"]
        assistant = db.get(Message, created.assistant_message_id)
        assert cancelled[0]["stopped_stage"] == "retrieving"
        assert assistant.status == "stopped"
        assert assistant.usage["stopped_stage"] == "retrieving"
    finally:
        db.close()


def test_history_excludes_streaming_and_failed_turns():
    messages = [
        SimpleNamespace(id="u1", role="user", content="未完成任务", usage={}, status="completed"),
        SimpleNamespace(id="a1", role="assistant", content="", usage={}, status="streaming"),
        SimpleNamespace(id="u2", role="user", content="失败任务", usage={}, status="completed"),
        SimpleNamespace(id="a2", role="assistant", content="失败", usage={}, status="failed"),
        SimpleNamespace(id="u3", role="user", content="已完成", usage={}, status="completed"),
        SimpleNamespace(id="a3", role="assistant", content="正常回答", usage={}, status="completed"),
    ]

    blocks = ConversationMemory._history_blocks(messages)
    text = "\n".join(block.text for block in blocks)

    assert "未完成任务" not in text
    assert "失败任务" not in text
    assert "已完成" in text
    assert "正常回答" in text
