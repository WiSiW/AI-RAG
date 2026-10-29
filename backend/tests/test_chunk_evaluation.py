import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    ChunkEvaluationResult,
    ChunkEvaluationRun,
    Document,
    DocumentChunk,
    DocumentStatus,
    EvaluationStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.db.session import Base, SessionLocal
from app.db.session import engine as global_engine
from app.evaluation.chunk_evaluator import (
    EvaluationSample,
    LLMChunkJudge,
    combine_scores,
    evaluate_rules,
    source_units,
)
from app.evaluation.service import ChunkEvaluationService
from app.schemas import ChunkEvaluationCreate, ChunkEvaluationReviewUpdate


def chunk(
    chunk_id: str,
    content: str,
    *,
    level: str = "L2",
    section: str = "安装",
    tokens: int = 512,
    parent_id: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=chunk_id,
        content=content,
        section_path=section,
        granularity_level=level,
        granule_type="paragraph" if level == "L2" else "section_summary" if level == "L1" else "error_line",
        token_count=tokens,
        context_prefix="[文档: 手册 / 章节: 安装]",
        parent_chunk_id=parent_id,
        metadata_json={"pages": [1]},
    )


def evaluate_target(target, source: str, all_l2: list[str]):
    return evaluate_rules(
        target,
        source_text=source,
        source_unit_list=source_units(source),
        all_l2_content=all_l2,
        duplicate_units=set(),
        has_pages=True,
    )


def test_l2_source_coverage_uses_all_l2_chunks():
    source = "第一段。第二段。"
    metrics = evaluate_target(chunk("c1", "第一段。"), source, ["第一段。", "第二段。"])
    assert metrics["source_coverage"] == 100.0
    assert "source_content_missing" not in metrics["critical"]


def test_missing_source_and_broken_code_are_critical_and_capped():
    source = "第一段。第二段。"
    metrics = evaluate_target(
        chunk("c1", "```python\nprint(1)", tokens=512),
        source,
        ["```python\nprint(1)"],
    )
    assert metrics["source_coverage"] < 98.0
    assert {"source_content_missing", "broken_code"}.issubset(metrics["critical"])
    score, grade, *_ = combine_scores(
        "L2", metrics, {}, llm_available=False
    )
    assert score <= 59
    assert grade == "fail"


def test_l3_missing_parent_is_capped():
    metrics = evaluate_target(chunk("l3", "ERROR 500", level="L3"), "", [])
    score, grade, issues, *_ = combine_scores(
        "L3", metrics, {}, llm_available=False
    )
    assert "missing_parent" in issues
    assert score <= 69
    assert grade == "warn"


@pytest.mark.asyncio
async def test_llm_judge_parses_structured_scores():
    payload = [
        {
            "chunk_id": "c1",
            "scores": {"self_contained": 5, "coherence": 4},
            "issue_codes": [],
            "reason": "完整",
            "suggestions": ["保持"],
        }
    ]

    class Gateway:
        async def complete(self, *args, **kwargs):
            return json.dumps(payload, ensure_ascii=False)

    judge = LLMChunkJudge(Gateway())
    chunk_obj = chunk("c1", "完整段落。")
    results, errors = await judge.evaluate(
        SimpleNamespace(id="judge"), [EvaluationSample(chunk_obj, {}, [])]
    )

    assert not errors
    assert results["c1"]["scores"]["self_contained"] == 5.0
    assert results["c1"]["reason"] == "完整"


@pytest.mark.asyncio
async def test_evaluation_run_persists_snapshots_review_and_comparison():
    class Judge:
        async def evaluate(self, profile, samples):
            results = {}
            for sample in samples:
                scores = {
                    "self_contained": 5,
                    "coherence": 5,
                    "key_fact_coverage": 5,
                    "faithfulness": 5,
                    "atomic_clarity": 5,
                    "fact_completeness": 5,
                    "parent_restorability": 5,
                }
                results[sample.chunk.id] = {
                    "scores": scores,
                    "issue_codes": [],
                    "reason": "完整",
                    "suggestions": [],
                }
            return results, []

    class FailingJudge:
        async def evaluate(self, profile, samples):
            return {}, ["judge unavailable"]

    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        embedding = ModelProfile(
            id="embedding",
            name="embedding",
            kind=ModelKind.EMBEDDING,
            base_url="https://example.com/v1",
            model_name="embedding",
            dimensions=3,
        )
        enrichment = ModelProfile(
            id="judge",
            name="judge",
            kind=ModelKind.ENRICHMENT,
            base_url="https://example.com/v1",
            model_name="judge",
        )
        kb = KnowledgeBase(
            id="kb",
            name="kb",
            description="",
            embedding_profile_id=embedding.id,
            enrichment_profile_id=enrichment.id,
            settings={},
        )
        document = Document(
            id="doc",
            kb_id=kb.id,
            filename="manual.md",
            object_key="objects/manual.md",
            status=DocumentStatus.READY,
            extracted_text="安装前必须关闭电源。",
            metadata_json={"title": "手册", "pages": 1},
        )
        l2 = DocumentChunk(
            id="l2",
            document_id=document.id,
            kb_id=kb.id,
            chunk_index=0,
            content="安装前必须关闭电源。",
            section_path="安装",
            token_count=512,
            granularity_level="L2",
            context_prefix="[文档: 手册 / 章节: 安装]",
        )
        l1 = DocumentChunk(
            id="l1",
            document_id=document.id,
            kb_id=kb.id,
            chunk_index=1,
            content="安装\n安装前关闭电源。",
            section_path="安装",
            token_count=20,
            granularity_level="L1",
            granule_type="section_summary",
            summary="安装前关闭电源。",
            context_prefix="[文档: 手册 / 章节: 安装]",
        )
        l3 = DocumentChunk(
            id="l3",
            document_id=document.id,
            kb_id=kb.id,
            chunk_index=2,
            content="ERROR 500",
            section_path="安装",
            token_count=3,
            granularity_level="L3",
            granule_type="error_line",
            parent_chunk_id=l2.id,
            context_prefix="[文档: 手册 / 章节: 安装 / 父段摘要: 安装]",
        )
        db.add_all([embedding, enrichment, kb, document, l2, l1, l3])
        db.commit()

        service = ChunkEvaluationService()
        service.judge = Judge()
        request = ChunkEvaluationCreate(document_id=document.id)
        first = service.create_run(db, request)
        await service.run(db, first.id)
        db.refresh(first)

        assert first.status == EvaluationStatus.COMPLETED
        assert first.aggregate["result_count"] == 3
        assert set(first.aggregate["level_scores"]) == {"L1", "L2", "L3"}
        results = list(
            db.query(ChunkEvaluationResult).filter(ChunkEvaluationResult.run_id == first.id)
        )
        assert all(item.review_sample for item in results)
        reviewed = service.update_review(
            db,
            first.id,
            results[0].id,
            ChunkEvaluationReviewUpdate(human_score=5, human_comment="通过"),
        )
        assert reviewed.aggregate["reviewed_count"] == 1
        assert reviewed.aggregate["calibration"]["mae"] is not None

        second = service.create_run(db, request)
        await service.run(db, second.id)
        comparison = service.compare(db, [first.id, second.id])
        assert comparison["overall_delta"] == 0

        service.judge = FailingJudge()
        third = service.create_run(db, request)
        await service.run(db, third.id)
        db.refresh(third)
        assert third.status == EvaluationStatus.PARTIAL
        assert third.aggregate["result_count"] == 3

        snapshot = results[0].content_snapshot
        db.delete(l3)
        db.delete(l2)
        db.delete(l1)
        db.commit()
        persisted = db.get(ChunkEvaluationResult, results[0].id)
        run_persisted = db.get(ChunkEvaluationRun, first.id)
        assert persisted is not None
        assert persisted.content_snapshot == snapshot
        assert run_persisted is not None
    finally:
        db.close()
        engine.dispose()


def test_evaluation_api_creates_and_lists_run(monkeypatch):
    from app.main import app
    from app.workers.celery_app import celery_app

    Base.metadata.drop_all(bind=global_engine)
    monkeypatch.setattr(celery_app, "send_task", lambda *args, **kwargs: None)
    try:
        with TestClient(app) as client:
            login = client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "test-password"},
            )
            headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
            embedding = client.post(
                "/api/v1/model-profiles",
                headers=headers,
                json={
                    "name": "embedding",
                    "kind": "embedding",
                    "base_url": "https://example.com/v1",
                    "model_name": "embedding",
                    "dimensions": 3,
                },
            ).json()
            enrichment = client.post(
                "/api/v1/model-profiles",
                headers=headers,
                json={
                    "name": "enrichment",
                    "kind": "enrichment",
                    "base_url": "https://example.com/v1",
                    "model_name": "enrichment",
                },
            ).json()
            knowledge_base = client.post(
                "/api/v1/knowledge-bases",
                headers=headers,
                json={
                    "name": "评估知识库",
                    "embedding_profile_id": embedding["id"],
                    "enrichment_profile_id": enrichment["id"],
                },
            ).json()
            db = SessionLocal()
            try:
                document = Document(
                    id="eval-doc",
                    kb_id=knowledge_base["id"],
                    filename="eval.md",
                    object_key="objects/eval.md",
                    status=DocumentStatus.READY,
                    extracted_text="完整句子。",
                )
                db.add(document)
                db.add(
                    DocumentChunk(
                        id="eval-chunk",
                        document_id=document.id,
                        kb_id=knowledge_base["id"],
                        chunk_index=0,
                        content="完整句子。",
                        section_path="第一章",
                        token_count=512,
                        granularity_level="L2",
                        context_prefix="[文档: eval / 章节: 第一章]",
                    )
                )
                db.commit()
            finally:
                db.close()

            created = client.post(
                "/api/v1/chunk-evaluations",
                headers=headers,
                json={
                    "document_id": "eval-doc",
                    "granularities": ["L2"],
                    "llm_enabled": False,
                },
            )
            assert created.status_code == 201, created.text
            run = created.json()
            assert run["status"] == "queued"
            listing = client.get("/api/v1/chunk-evaluations", headers=headers)
            assert listing.status_code == 200
            assert listing.json()[0]["id"] == run["id"]
            detail = client.get(f"/api/v1/chunk-evaluations/{run['id']}", headers=headers)
            assert detail.status_code == 200
    finally:
        Base.metadata.drop_all(bind=global_engine)
