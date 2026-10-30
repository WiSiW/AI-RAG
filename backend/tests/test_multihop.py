import json
from types import SimpleNamespace

import pytest

from app.indexing.types import RetrievedChunk
from app.retrieval.complexity import classify_complexity, extract_complexity_features
from app.retrieval.multihop_planner import MultiHopPlan
from app.schemas import RetrievalTrace, SearchResponse
from app.services.chat_service import ChatService, RetrievalOutcome, sse_event


class FakeClassifier:
    def __init__(self, probability: float):
        self.probability = probability

    async def predict(self, question, features):
        return self.probability, "fake-v1"


class FakeGateway:
    def __init__(self, answer="否"):
        self.answer = answer
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.mark.asyncio
async def test_complexity_features_and_rule_routing():
    features = extract_complexity_features("相比设备A，设备B在安装后更稳定吗？")
    assert features.comparison_flag is True
    assert features.entity_count >= 2
    assert features.bridge_entity_count >= 1

    decision = await classify_complexity(
        "先根据设备A的配置找到设备B，然后比较两者差异。",
        retrieval_mode="auto",
        llm_profile=None,
        gateway=FakeGateway(),
    )
    assert decision.is_multi_hop is True
    assert decision.decision_source == "rules"


@pytest.mark.asyncio
async def test_bert_high_probability_routes_to_multi_hop():
    decision = await classify_complexity(
        "设备A在系统中对应哪些配置参数？",
        retrieval_mode="auto",
        llm_profile=SimpleNamespace(id="llm"),
        gateway=FakeGateway("否"),
        classifier_client=FakeClassifier(0.91),
    )
    assert decision.is_multi_hop is True
    assert decision.decision_source == "bert"


@pytest.mark.asyncio
async def test_uncertain_bert_falls_back_to_llm_yes():
    gateway = FakeGateway("是")
    decision = await classify_complexity(
        "设备A在系统中对应哪些配置参数？",
        retrieval_mode="auto",
        llm_profile=SimpleNamespace(id="llm"),
        gateway=gateway,
        classifier_client=FakeClassifier(0.5),
    )
    assert decision.is_multi_hop is True
    assert decision.decision_source == "llm"
    assert gateway.calls == 1


@pytest.mark.asyncio
async def test_llm_failure_uses_rule_and_bert_fallback():
    decision = await classify_complexity(
        "如果设备A发生故障，那么设备B的运行状态是否会受到影响，具体原因是什么？",
        retrieval_mode="auto",
        llm_profile=SimpleNamespace(id="llm"),
        gateway=FakeGateway(RuntimeError("down")),
        classifier_client=FakeClassifier(0.45),
    )
    assert decision.is_multi_hop is True
    assert decision.decision_source == "fallback"


def chunk(
    chunk_id: str,
    score: float = 0.5,
    *,
    content: str | None = None,
    granularity_level: str = "L2",
    granule_type: str = "paragraph",
    matched_representations: list[str] | None = None,
    vector_score: float | None = None,
    bm25_score: float | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        kb_id="kb",
        content=content or f"content-{chunk_id}",
        section=f"{chunk_id} 章节",
        source=f"{chunk_id}.md",
        metadata={
            "granularity_level": granularity_level,
            "granule_type": granule_type,
            "matched_representations": matched_representations or [],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
        },
        rrf_score=score,
    )


@pytest.mark.asyncio
async def test_recursive_retrieval_merges_rounds_and_reranks_once(monkeypatch):
    service = ChatService()
    kb = SimpleNamespace(
        id="kb",
        settings={"max_multi_hop_steps": 3},
        rerank_profile_id=None,
    )
    calls: list[str] = []

    async def fake_retrieve_single(db, kbs, query, **kwargs):
        calls.append(query)
        if query == "原问题":
            return RetrievalOutcome(
                chunks=[
                    chunk(
                        "a",
                        0.8,
                        matched_representations=["bm25", "original", "original"],
                        vector_score=0.88,
                        bm25_score=2.4,
                    )
                ]
            )
        return RetrievalOutcome(
            chunks=[
                chunk(
                    "a",
                    0.8,
                    matched_representations=["summary", "original"],
                    vector_score=0.81,
                ),
                chunk(
                    "b",
                    0.7,
                    granularity_level="L3",
                    granule_type="error_line",
                    matched_representations=["questions", "original"],
                    vector_score=0.75,
                ),
            ]
        )

    plans = iter(
        [
            MultiHopPlan(True, "第二问", ["设备A"], 0.8, "需要补充实体"),
            MultiHopPlan(False, "", ["设备B"], 0.9, "信息已足够"),
        ]
    )

    class Planner:
        async def plan(self, **kwargs):
            return next(plans)

    rerank_calls = []

    async def fake_rerank(*, profile, query, chunks, top_k, threshold):
        rerank_calls.append((query, [item.chunk_id for item in chunks]))
        return chunks[:top_k]

    monkeypatch.setattr(service, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(service, "multihop_planner", Planner())
    monkeypatch.setattr(service.reranker, "rerank", fake_rerank)

    outcome = await service._retrieve(
        None,
        [kb],
        "原问题",
        top_k=2,
        rerank=True,
        enforce_context_budget=False,
        retrieval_mode="multi",
    )

    assert calls == ["原问题", "第二问"]
    assert [item.chunk_id for item in outcome.chunks] == ["a", "b"]
    assert rerank_calls == [("原问题", ["a", "b"])]
    assert outcome.trace is not None
    assert outcome.trace.total_rounds == 2
    assert outcome.trace.stop_reason == "planner_sufficient"
    assert outcome.trace.uncertain is False
    first, second = outcome.trace.steps
    assert [item.chunk_id for item in first.chunks] == ["a"]
    assert [item.chunk_id for item in second.chunks] == ["a", "b"]
    assert first.chunks[0].matched_representations == ["bm25", "original"]
    assert first.route_counts == {"bm25": 1, "original": 1}
    assert first.granularity_counts == {"L1": 0, "L2": 1, "L3": 0}
    assert second.route_counts == {"summary": 1, "original": 2, "questions": 1}
    assert second.granularity_counts == {"L1": 0, "L2": 1, "L3": 1}
    assert [item.chunk_id for item in second.chunks] == second.chunk_ids
    assert second.chunks[0].source == "a.md"
    assert second.chunks[0].section == "a 章节"
    assert second.chunks[0].rrf_score == pytest.approx(0.8)
    assert second.chunks[0].vector_score == pytest.approx(0.81)
    assert second.chunks[1].granule_type == "error_line"


@pytest.mark.asyncio
async def test_max_steps_marks_trace_uncertain(monkeypatch):
    service = ChatService()
    kb = SimpleNamespace(
        id="kb",
        settings={"max_multi_hop_steps": 2},
        rerank_profile_id=None,
    )
    counter = 0

    async def fake_retrieve_single(db, kbs, query, **kwargs):
        nonlocal counter
        counter += 1
        return RetrievalOutcome(chunks=[chunk(f"c{counter}")])

    class Planner:
        async def plan(self, **kwargs):
            return MultiHopPlan(True, f"后续问题{kwargs['round_number']}", [], 0.4, "仍需信息")

    monkeypatch.setattr(service, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(service, "multihop_planner", Planner())

    outcome = await service._retrieve(
        None,
        [kb],
        "原问题",
        rerank=False,
        enforce_context_budget=False,
        retrieval_mode="multi",
    )

    assert outcome.trace is not None
    assert outcome.trace.stop_reason == "max_steps"
    assert outcome.trace.uncertain is True
    assert "注意：" in outcome.trace.uncertainty_notice
    assert [step.chunks[0].chunk_id for step in outcome.trace.steps] == ["c1", "c2"]
    assert all(step.route_counts == {} for step in outcome.trace.steps)
    assert all(step.granularity_counts["L2"] == 1 for step in outcome.trace.steps)


@pytest.mark.asyncio
async def test_planner_failure_keeps_round_evidence(monkeypatch):
    service = ChatService()
    kb = SimpleNamespace(
        id="kb",
        settings={"max_multi_hop_steps": 3},
        rerank_profile_id=None,
    )

    async def fake_retrieve_single(db, kbs, query, **kwargs):
        return RetrievalOutcome(
            chunks=[
                chunk(
                    "evidence",
                    matched_representations=["questions", "original"],
                    vector_score=0.76,
                )
            ]
        )

    class Planner:
        async def plan(self, **kwargs):
            return None

    monkeypatch.setattr(service, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(service, "multihop_planner", Planner())

    outcome = await service._retrieve(
        None,
        [kb],
        "原问题",
        rerank=False,
        enforce_context_budget=False,
        retrieval_mode="multi",
    )

    assert outcome.trace is not None
    assert outcome.trace.stop_reason == "planner_unavailable"
    assert outcome.trace.steps[0].chunks[0].chunk_id == "evidence"
    assert outcome.trace.steps[0].route_counts == {"questions": 1, "original": 1}
    assert outcome.trace.steps[0].reason == "规划模型调用失败"


@pytest.mark.asyncio
async def test_no_new_evidence_keeps_round_evidence(monkeypatch):
    service = ChatService()
    kb = SimpleNamespace(
        id="kb",
        settings={"max_multi_hop_steps": 3},
        rerank_profile_id=None,
    )

    async def fake_retrieve_single(db, kbs, query, **kwargs):
        return RetrievalOutcome(
            chunks=[chunk("same", matched_representations=["summary"], vector_score=0.7)]
        )

    class Planner:
        async def plan(self, **kwargs):
            return MultiHopPlan(True, f"追问-{kwargs['round_number']}", [], 0.5, "仍需信息")

    monkeypatch.setattr(service, "_retrieve_single", fake_retrieve_single)
    monkeypatch.setattr(service, "multihop_planner", Planner())

    outcome = await service._retrieve(
        None,
        [kb],
        "原问题",
        rerank=False,
        enforce_context_budget=False,
        retrieval_mode="multi",
    )

    assert outcome.trace is not None
    assert outcome.trace.stop_reason == "no_new_evidence"
    assert [step.new_chunk_count for step in outcome.trace.steps] == [1, 0]
    assert [step.chunks[0].chunk_id for step in outcome.trace.steps] == ["same", "same"]


def test_trace_serialization_caps_and_truncates_round_snapshots():
    chunks = [
        chunk(
            f"chunk-{index}",
            content=f"内容-{index}-" + ("长文本" * 180),
            matched_representations=["original", "original"],
        )
        for index in range(3)
    ]
    step = ChatService._trace_step(
        round_number=1,
        query="多跳问题",
        chunks=chunks,
        new_chunk_count=3,
        limit=2,
        key_entities=["设备A"],
        need_more=True,
        confidence=0.8,
        reason="需要补充证据",
    )
    trace = RetrievalTrace(
        requested_mode="multi",
        steps=[step],
        stop_reason="max_steps",
        uncertain=True,
        total_rounds=1,
        total_chunks=2,
    )

    assert step.chunk_ids == ["chunk-0", "chunk-1", "chunk-2"]
    assert [item.chunk_id for item in step.chunks] == ["chunk-0", "chunk-1"]
    assert len(step.chunks[0].content_snippet) == 240
    assert step.route_counts == {"original": 2}

    response_payload = SearchResponse(
        query="多跳问题",
        results=[],
        retrieval_trace=trace,
    ).model_dump()
    serialized_step = response_payload["retrieval_trace"]["steps"][0]
    assert serialized_step["route_counts"] == {"original": 2}
    assert serialized_step["granularity_counts"] == {"L1": 0, "L2": 2, "L3": 0}
    assert serialized_step["chunks"][0]["content_snippet"].startswith("内容-0-")

    event = sse_event("retrieval_step", step.model_dump())
    data_line = next(line for line in event.splitlines() if line.startswith("data: "))
    event_payload = json.loads(data_line.removeprefix("data: "))
    assert event_payload["chunks"][0]["source"] == "chunk-0.md"
    assert event_payload["route_counts"] == {"original": 2}

    usage_payload = {"retrieval_trace": trace.model_dump()}
    assert usage_payload["retrieval_trace"]["steps"][0]["chunks"][1]["chunk_id"] == "chunk-1"
