import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.db.session import Base
from app.indexing.chunk_enricher import (
    ChunkEnricher,
    EnrichmentResult,
    fallback_keywords,
    rule_based_enrichment,
)
from app.indexing.qa_generator import QAPair, QAResult
from app.services.ingestion_service import IngestionService


class FakeGateway:
    def __init__(self, response: str | Exception):
        self.response = response
        self.calls = 0

    async def complete(self, *args, **kwargs):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def embed_texts(self, profile, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_enricher_parses_model_output_and_bounds_fields():
    payload = [
        {
            "chunk_id": "chunk-1",
            "keywords": ["糖尿病", "血糖", "糖尿病", "胰岛素"],
            "summary": "文档介绍糖尿病患者通过饮食、运动和药物控制血糖，并强调定期监测。",
        }
    ]
    enricher = ChunkEnricher(FakeGateway(json.dumps(payload, ensure_ascii=False)))

    result = await enricher.enrich(
        SimpleNamespace(id="profile-1", model_name="small-model"),
        [("chunk-1", "糖尿病患者应控制血糖。")],
    )

    item = result["chunk-1"]
    assert item.status == "ready"
    assert item.summary.startswith("文档介绍")
    assert item.keywords[:3] == ["糖尿病", "血糖", "胰岛素"]
    assert 3 <= len(item.keywords) <= 10
    assert item.hypothetical_questions == []
    assert item.meta["prompt_version"]


@pytest.mark.asyncio
async def test_enricher_falls_back_to_deterministic_rules():
    enricher = ChunkEnricher(FakeGateway(RuntimeError("model unavailable")))
    content = "糖尿病是一种慢性代谢疾病。患者需要监测血糖并控制饮食。"

    result = await enricher.enrich(
        SimpleNamespace(id="profile-1", model_name="small-model"),
        [("chunk-1", content)],
    )

    item = result["chunk-1"]
    assert item.status == "degraded"
    assert item.meta["generated_by"] == "rules"
    assert item.summary
    assert item.hypothetical_questions == []


def test_rule_based_summary_has_hard_limit_and_keywords_are_bm25_tokens():
    content = "糖尿病管理" * 30
    result = rule_based_enrichment(content)

    assert len(result.summary) <= 80
    assert 1 <= len(result.keywords) <= 10
    assert all(" " not in keyword for keyword in result.keywords)
    keywords = fallback_keywords("RAG 检索增强生成 RAG")
    assert keywords[0] == "RAG"
    assert len(keywords) <= 10


@pytest.mark.asyncio
async def test_representation_builder_links_original_summary_and_questions_to_chunk():
    service = IngestionService()
    service.gateway = FakeGateway("{}")
    document = SimpleNamespace(id="doc-1", kb_id="kb-1", filename="demo.md", chunk_mode="structure")
    row = SimpleNamespace(
        id="chunk-1",
        content="原始分块内容",
        section_path="第一章",
        summary="这是摘要",
        hypothetical_questions=["问题一？", "问题二？"],
        enrichment_meta={},
        enrichment_status="ready",
    )

    representations, vectors = await service._build_representations(
        SimpleNamespace(dimensions=3), document, [row], [[0.1, 0.2, 0.3]]
    )

    assert [item.representation_type for item in representations] == [
        "original",
        "summary",
        "question",
        "question",
    ]
    assert {item.chunk_id for item in representations} == {"chunk-1"}
    assert len(vectors) == 4
    assert row.enrichment_meta["representation_count"] == 4


@pytest.mark.asyncio
async def test_representation_builder_adds_l1_and_l3_vector_routes():
    service = IngestionService()
    service.gateway = FakeGateway("{}")
    document = SimpleNamespace(id="doc-1", kb_id="kb-1", filename="demo.md")
    l2 = SimpleNamespace(
        id="l2-1",
        content="L2 段落",
        section_path="第一章",
        summary="L2 摘要",
        hypothetical_questions=[],
        keywords=["L2"],
        enrichment_meta={},
        enrichment_status="ready",
        granularity_level="L2",
    )
    l1 = SimpleNamespace(
        id="l1-1",
        content="第一章\n章节摘要",
        section_path="第一章",
        summary="章节摘要",
        hypothetical_questions=[],
        keywords=["章节"],
        enrichment_meta={},
        enrichment_status="ready",
        granularity_level="L1",
    )
    l3 = SimpleNamespace(
        id="l3-1",
        content="ERROR 500",
        section_path="第一章",
        summary="",
        hypothetical_questions=[],
        keywords=[],
        enrichment_meta={},
        enrichment_status="ready",
        granularity_level="L3",
    )

    representations, vectors = await service._build_representations(
        SimpleNamespace(dimensions=3), document, [l2, l1, l3], [[0.1, 0.2, 0.3]]
    )

    assert [item.representation_type for item in representations] == [
        "original",
        "summary",
        "l1_summary",
        "l3_atomic",
    ]
    assert len(vectors) == 4


@pytest.mark.asyncio
async def test_representation_embedding_failure_degrades_without_removing_original():
    class FailingGateway:
        async def embed_texts(self, profile, texts):
            raise RuntimeError("embedding down")

    service = IngestionService()
    service.gateway = FailingGateway()
    document = SimpleNamespace(id="doc-1", kb_id="kb-1", filename="demo.md", chunk_mode="structure")
    row = SimpleNamespace(
        id="chunk-1",
        content="原始分块内容",
        section_path="第一章",
        summary="这是摘要",
        hypothetical_questions=["问题一？"],
        enrichment_meta={},
        enrichment_status="ready",
    )

    representations, vectors = await service._build_representations(
        SimpleNamespace(dimensions=3), document, [row], [[0.1, 0.2, 0.3]]
    )

    assert [item.representation_type for item in representations] == ["original"]
    assert len(vectors) == 1
    assert row.enrichment_status == "degraded"
    assert "representation_embedding_failed" in row.enrichment_meta["degraded_reasons"]


@pytest.mark.asyncio
async def test_backfill_preserves_chunks_and_switches_to_v2():
    class BackfillEnricher:
        def __init__(self):
            self.calls = 0
            self.section_calls = 0

        async def enrich(self, profile, chunks):
            self.calls += 1
            return {
                chunk_id: EnrichmentResult(
                    summary="固定摘要",
                    keywords=["固定关键词"],
                    hypothetical_questions=[],
                    status="ready",
                    meta={"generated_by": "model", "prompt_version": "chunk-enrichment-v2"},
                )
                for chunk_id, _ in chunks
            }

        async def summarize_sections(self, profile, sections):
            self.section_calls += 1
            return {
                section.section_path: SimpleNamespace(
                    summary="章节摘要",
                    status="ready",
                    meta={"generated_by": "model", "prompt_version": "section-summary-v1"},
                )
                for section in sections
            }

    class BackfillQA:
        async def generate(self, profile, sources):
            return {
                source.chunk_id: QAResult(
                    pairs=[
                        QAPair("具体参数是什么？", "参数来自原文。", "detail"),
                        QAPair("这两个概念有什么关系？", "二者存在关联。", "comparison"),
                        QAPair("这段内容主要概括了什么？", "核心结论来自原文。", "summary"),
                    ],
                    status="ready",
                    meta={
                        "generated_by": "model",
                        "prompt_version": "qa-generation-v1",
                        "qa_types": ["detail", "comparison", "summary"],
                        "degraded_reasons": [],
                    },
                )
                for source in sources
            }

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        embedding = ModelProfile(
            id="embedding-1",
            name="embedding",
            kind=ModelKind.EMBEDDING,
            base_url="https://example.com/v1",
            model_name="embedding",
            dimensions=3,
        )
        enrichment = ModelProfile(
            id="enrichment-1",
            name="enrichment",
            kind=ModelKind.ENRICHMENT,
            base_url="https://example.com/v1",
            model_name="enrichment",
        )
        kb = KnowledgeBase(
            id="kb-1",
            name="kb",
            description="",
            embedding_profile_id=embedding.id,
            enrichment_profile_id=enrichment.id,
            settings={},
        )
        document = Document(
            id="doc-1",
            kb_id=kb.id,
            filename="demo.md",
            object_key="objects/demo.md",
            status=DocumentStatus.READY,
        )
        chunk = DocumentChunk(
            id="chunk-1",
            document_id=document.id,
            kb_id=kb.id,
            chunk_index=0,
            content="必须逐字保留的原始分块。",
            section_path="第一章",
            token_count=8,
        )
        db.add_all([embedding, enrichment, kb, document, chunk])
        db.commit()

        service = IngestionService()
        service.gateway = FakeGateway("{}")
        service.enricher = BackfillEnricher()
        service.qa_generator = BackfillQA()
        writes = []
        service._write_v2_indexes = lambda document, profile, indexed, representations, vectors: writes.append(
            (document.id, [item.chunk_id for item in representations])
        )

        first = await service.rebuild_knowledge_base_index(db, kb.id)
        second = await service.rebuild_knowledge_base_index(db, kb.id)

        db.refresh(chunk)
        db.refresh(kb)
        assert first["status"] == "completed"
        assert second["status"] == "completed"
        assert chunk.id == "chunk-1"
        assert chunk.content == "必须逐字保留的原始分块。"
        assert chunk.summary == "固定摘要"
        assert kb.settings["index_version"] == "v2"
        assert writes[0][0] == "doc-1"
        assert writes[0][1][:5] == ["chunk-1"] * 5
        assert len(writes[0][1]) == 9
        assert service.enricher.calls == 1
        assert service.enricher.section_calls == 2
    finally:
        db.close()
        engine.dispose()
