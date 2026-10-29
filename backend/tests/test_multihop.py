from types import SimpleNamespace

import pytest

from app.indexing.types import RetrievedChunk
from app.retrieval.complexity import classify_complexity, extract_complexity_features
from app.retrieval.multihop_planner import MultiHopPlan
from app.services.chat_service import ChatService, RetrievalOutcome


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


def chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        kb_id="kb",
        content=f"content-{chunk_id}",
        section="第一章",
        source="demo.md",
        metadata={},
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
            return RetrievalOutcome(chunks=[chunk("a", 0.8)])
        return RetrievalOutcome(chunks=[chunk("a", 0.8), chunk("b", 0.7)])

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
