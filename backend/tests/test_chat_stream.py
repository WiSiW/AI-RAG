import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.indexing.types import RetrievedChunk
from app.memory.session_manager import ConversationMemory, HistoryContext
from app.schemas import ChatRequest
from app.services.chat_service import ChatService, RetrievalOutcome
from app.services.conversation_summary_service import ConversationSummaryService


def parse_sse_event(raw_event: str) -> tuple[str, dict]:
    event_name = "message"
    data_lines: list[str] = []
    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    return event_name, json.loads("\n".join(data_lines))


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        document_id="document-1",
        kb_id="kb-1",
        content="制度原文",
        section="第一章",
        source="制度.md",
    )


def test_context_token_budget_clips_single_oversized_chunk():
    chunk = make_chunk()
    chunk.context_text = "超长上下文内容"

    selected = ChatService._limit_context_tokens([chunk], max_tokens=2)

    assert len(selected) == 1
    assert selected[0].context_text == "超长"


def test_usage_reports_history_query_and_prompt_estimates():
    history = HistoryContext(
        text="用户：上一轮问题",
        token_count=8,
        included_message_count=1,
        truncated=True,
        summary_tokens=3,
    )

    usage = ChatService._usage(
        "回答",
        [make_chunk()],
        SimpleNamespace(model_name="chat-test"),
        history,
        "当前问题",
    )

    assert usage["history_tokens_estimate"] == 8
    assert usage["summary_tokens_estimate"] == 3
    assert usage["history_truncated"] is True
    assert usage["query_tokens_estimate"] == 4
    assert usage["prompt_tokens_estimate"] > usage["context_tokens_estimate"]


class FakeMemory:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.conversation = SimpleNamespace(id="conversation-1", document_id=None)

    def get_or_create(self, db, **kwargs):
        return self.conversation

    def history_context(self, db, conversation) -> HistoryContext:
        return HistoryContext(text="", token_count=0, included_message_count=0, truncated=False)

    def add_exchange(self, db, conversation, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(id=f"message-{len(self.calls)}")


class FakeGateway:
    def __init__(self, tokens: list[str]) -> None:
        self.tokens = tokens

    async def stream(self, profile, messages):
        for token in self.tokens:
            yield token


def make_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    outcome: RetrievalOutcome,
    tokens: list[str],
) -> tuple[ChatService, FakeMemory, AsyncMock]:
    service = ChatService()
    memory = FakeMemory()
    service.memory = memory
    service.gateway = FakeGateway(tokens)
    service.summary_service = SimpleNamespace(maybe_schedule=lambda db, conversation: True)
    knowledge_base = SimpleNamespace(id="kb-1", name="测试知识库")
    profile = SimpleNamespace(id="llm-1", model_name="chat-test")
    monkeypatch.setattr(service, "_get_kbs", lambda db, ids: [knowledge_base])
    monkeypatch.setattr(service, "_validate_document", lambda db, kbs, document_id: None)
    monkeypatch.setattr(service, "_llm_profile", lambda db, kbs, profile_id: profile)
    rewrite = AsyncMock(return_value="改写后的问题")
    monkeypatch.setattr(service, "_rewrite_query", rewrite)
    retrieve = AsyncMock(return_value=outcome)
    monkeypatch.setattr(service, "_retrieve", retrieve)

    return service, memory, rewrite


@pytest.mark.asyncio
async def test_stream_chat_emits_stage_events_before_matching_work(monkeypatch):
    outcome = RetrievalOutcome(chunks=[make_chunk()])
    service, memory, _ = make_service(monkeypatch, outcome=outcome, tokens=["答", "案"])
    request = ChatRequest(knowledge_base_ids=["kb-1"], query="问题")

    events = [parse_sse_event(event) async for event in service.stream_chat(object(), request)]
    names = [name for name, _ in events]

    assert names[:4] == ["stage", "stage", "start", "token"]
    assert events[0][1]["stage"] == "retrieving"
    assert events[0][1]["context_count"] == 0
    assert events[1][1]["stage"] == "generating"
    assert events[1][1]["context_count"] == 1
    assert names.index("stage") < names.index("start") < names.index("token")
    assert memory.calls[0]["answer"] == "答案"
    assert "stopped" not in memory.calls[0]["usage"]


@pytest.mark.asyncio
async def test_closing_during_retrieval_persists_stopped_exchange(monkeypatch):
    service, memory, rewrite = make_service(
        monkeypatch,
        outcome=RetrievalOutcome(chunks=[]),
        tokens=[],
    )
    request = ChatRequest(knowledge_base_ids=["kb-1"], query="需要停止的问题")
    stream = service.stream_chat(object(), request)

    first_event = parse_sse_event(await anext(stream))
    assert first_event[0] == "stage"
    assert first_event[1]["stage"] == "retrieving"

    await stream.aclose()

    rewrite.assert_not_awaited()
    assert len(memory.calls) == 1
    assert memory.calls[0]["question"] == "需要停止的问题"
    assert memory.calls[0]["answer"] == ""
    assert memory.calls[0]["usage"]["stopped"] is True
    assert memory.calls[0]["usage"]["stopped_stage"] == "retrieving"


@pytest.mark.asyncio
async def test_closing_during_generation_persists_partial_answer(monkeypatch):
    outcome = RetrievalOutcome(chunks=[make_chunk()])
    service, memory, _ = make_service(monkeypatch, outcome=outcome, tokens=["部分回答", "不应出现"])
    request = ChatRequest(knowledge_base_ids=["kb-1"], query="需要停止的问题")
    stream = service.stream_chat(object(), request)

    assert parse_sse_event(await anext(stream))[1]["stage"] == "retrieving"
    assert parse_sse_event(await anext(stream))[1]["stage"] == "generating"
    assert parse_sse_event(await anext(stream))[0] == "start"
    token_event = parse_sse_event(await anext(stream))
    assert token_event == ("token", {"content": "部分回答"})

    await stream.aclose()

    assert len(memory.calls) == 1
    assert memory.calls[0]["answer"] == "部分回答"
    assert memory.calls[0]["usage"]["stopped"] is True
    assert memory.calls[0]["usage"]["stopped_stage"] == "generating"


@pytest.mark.asyncio
async def test_closing_after_exchange_is_saved_does_not_duplicate(monkeypatch):
    outcome = RetrievalOutcome(chunks=[make_chunk()])
    service, memory, _ = make_service(monkeypatch, outcome=outcome, tokens=["完整回答"])
    request = ChatRequest(knowledge_base_ids=["kb-1"], query="问题")
    stream = service.stream_chat(object(), request)

    while True:
        event = parse_sse_event(await anext(stream))
        if event[0] == "citations":
            break
    await stream.aclose()

    assert len(memory.calls) == 1
    assert "stopped" not in memory.calls[0]["usage"]


def test_stopped_assistant_is_excluded_from_history_and_summary(monkeypatch):
    memory = ConversationMemory(gateway=SimpleNamespace())
    memory.redis = None
    messages = [
        SimpleNamespace(id="m1", role="user", content="问题一", usage={}),
        SimpleNamespace(
            id="m2",
            role="assistant",
            content="未完成内容",
            usage={"stopped": True, "stopped_stage": "generating"},
        ),
        SimpleNamespace(id="m3", role="user", content="问题二", usage={}),
        SimpleNamespace(id="m4", role="assistant", content="正常回答", usage={}),
    ]
    monkeypatch.setattr(memory, "messages", lambda db, conversation_id: messages)
    conversation = SimpleNamespace(id="conversation-1", summary="", summary_message_count=0)

    history = memory.history_context(object(), conversation).text
    assert "[未完成提问]" in history
    assert "问题一" in history
    assert "未完成内容" not in history
    assert "正常回答" in history

    prepared = ConversationSummaryService._summary_messages(messages)
    summary_text = "\n".join(item.content for item in prepared)
    assert "[未完成提问]" in summary_text
    assert "未完成内容" not in summary_text
    assert "正常回答" in summary_text
