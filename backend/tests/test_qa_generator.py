import json
from types import SimpleNamespace

import pytest

from app.indexing.qa_generator import QAGenerator, QASource
from app.services.ingestion_service import IngestionService


class SequenceGateway:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.embedding_texts = []

    async def complete(self, *args, **kwargs):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response

    async def embed_texts(self, profile, texts):
        self.embedding_texts.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]


def source() -> QASource:
    return QASource(
        chunk_id="chunk-1",
        content="糖尿病患者应控制饮食并定期监测血糖。血糖过高时需要及时就医。",
        contextual_text="[文档: 手册 / 章节: 血糖管理]\n糖尿病患者应控制饮食并定期监测血糖。",
        section_path="血糖管理",
        keywords=["糖尿病", "血糖", "饮食"],
        summary="糖尿病患者的血糖管理方式。",
    )


@pytest.mark.asyncio
async def test_qa_generator_filters_and_keeps_three_types():
    payload = [
        {
            "chunk_id": "chunk-1",
            "qa_pairs": [
                {
                    "question": "糖尿病患者应该如何控制饮食？",
                    "answer": "应控制饮食并定期监测血糖。",
                    "question_type": "detail",
                },
                {
                    "question": "控制饮食与监测血糖之间有什么关系？",
                    "answer": "两者共同用于糖尿病管理。",
                    "question_type": "comparison",
                },
                {
                    "question": "血糖过高时应该怎么办？",
                    "answer": "需要及时就医。",
                    "question_type": "summary",
                },
                {
                    "question": "糖尿病患者应该如何控制饮食？",
                    "answer": "重复问题。",
                    "question_type": "detail",
                },
            ],
        }
    ]
    gateway = SequenceGateway([json.dumps(payload, ensure_ascii=False)])
    result = await QAGenerator(gateway).generate(
        SimpleNamespace(id="model", model_name="judge"), [source()]
    )

    pairs = result["chunk-1"].pairs
    assert result["chunk-1"].status == "ready"
    assert [pair.question_type for pair in pairs] == ["detail", "comparison", "summary"]
    assert all(pair.answer for pair in pairs)
    assert gateway.calls == 1


@pytest.mark.asyncio
async def test_qa_generator_retries_missing_types_before_fallback():
    first = [
        {
            "chunk_id": "chunk-1",
            "qa_pairs": [
                {
                    "question": "糖尿病患者应如何监测血糖？",
                    "answer": "应定期监测血糖。",
                    "question_type": "detail",
                }
            ],
        }
    ]
    second = [
        {
            "chunk_id": "chunk-1",
            "qa_pairs": [
                {
                    "question": "控制饮食与监测血糖有什么关系？",
                    "answer": "两者共同用于血糖管理。",
                    "question_type": "comparison",
                },
                {
                    "question": "糖尿病患者的血糖管理重点是什么？",
                    "answer": "控制饮食并监测血糖。",
                    "question_type": "summary",
                },
            ],
        }
    ]
    gateway = SequenceGateway(
        [json.dumps(first, ensure_ascii=False), json.dumps(second, ensure_ascii=False)]
    )

    result = await QAGenerator(gateway).generate(
        SimpleNamespace(id="model", model_name="judge"), [source()]
    )

    assert gateway.calls == 2
    assert result["chunk-1"].status == "ready"
    assert len(result["chunk-1"].pairs) == 3


@pytest.mark.asyncio
async def test_qa_generator_falls_back_to_three_degraded_pairs():
    gateway = SequenceGateway([RuntimeError("model unavailable")])

    result = await QAGenerator(gateway).generate(
        SimpleNamespace(id="model", model_name="judge"), [source()]
    )

    assert result["chunk-1"].status == "degraded"
    assert len(result["chunk-1"].pairs) == 3
    assert {pair.question_type for pair in result["chunk-1"].pairs} == {
        "detail",
        "comparison",
        "summary",
    }
    assert "qa_fallback" in result["chunk-1"].meta["degraded_reasons"]


@pytest.mark.asyncio
async def test_all_granularities_index_three_question_vectors():
    gateway = SequenceGateway(["{}"])
    service = IngestionService()
    service.gateway = gateway
    qa_pairs = [
        {"question": "具体参数是什么？", "answer": "参数答案。", "question_type": "detail"},
        {"question": "两个对象有什么关系？", "answer": "关系答案。", "question_type": "comparison"},
        {"question": "主要结论是什么？", "answer": "概括答案。", "question_type": "summary"},
    ]

    def row(chunk_id: str, level: str) -> SimpleNamespace:
        return SimpleNamespace(
            id=chunk_id,
            content=f"{level} 原文",
            section_path="第一章",
            summary=f"{level} 摘要" if level != "L3" else "",
            hypothetical_questions=[item["question"] for item in qa_pairs],
            qa_pairs=qa_pairs,
            keywords=["参数"],
            enrichment_meta={},
            enrichment_status="ready",
            granularity_level=level,
        )

    rows = [row("l1", "L1"), row("l2", "L2"), row("l3", "L3")]
    document = SimpleNamespace(id="doc", kb_id="kb", filename="demo.md")

    representations, vectors = await service._build_representations(
        SimpleNamespace(dimensions=3), document, rows, [[0.1, 0.2, 0.3]]
    )

    question_reps = [
        item for item in representations if item.representation_type == "question"
    ]
    assert len(question_reps) == 9
    assert len(vectors) == 13
    assert "参数答案。" not in gateway.embedding_texts
    assert "关系答案。" not in gateway.embedding_texts
    assert "概括答案。" not in gateway.embedding_texts
    for chunk_id in ("l1", "l2", "l3"):
        assert sorted(
            item.question_index for item in question_reps if item.chunk_id == chunk_id
        ) == [1, 2, 3]
