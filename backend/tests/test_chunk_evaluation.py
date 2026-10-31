import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    ChunkEvaluationItem,
    ChunkEvaluationResult,
    ChunkEvaluationRun,
    Document,
    DocumentChunk,
    DocumentStatus,
    EvaluationItemStatus,
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
from app.schemas import (
    ChunkEvaluationCreate,
    ChunkEvaluationRetryFailed,
    ChunkEvaluationReviewUpdate,
)


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
        assert first.config_snapshot["knowledge_base_name"] == "kb"
        assert first.config_snapshot["document_name"] == "manual.md"
        assert first.config_snapshot["judge_profile_name"] == "judge"
        assert first.config_snapshot["judge_model_name"] == "judge"
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
            assert run["config_snapshot"]["knowledge_base_name"] == "评估知识库"
            assert run["config_snapshot"]["document_name"] == "eval.md"
            assert run["config_snapshot"]["judge_profile_name"] is None
            listing = client.get("/api/v1/chunk-evaluations", headers=headers)
            assert listing.status_code == 200
            assert listing.json()[0]["id"] == run["id"]
            detail = client.get(f"/api/v1/chunk-evaluations/{run['id']}", headers=headers)
            assert detail.status_code == 200
            assert run["stage"] == "queued"
            assert run["can_delete"] is False
            blocked_delete = client.delete(
                f"/api/v1/chunk-evaluations/{run['id']}", headers=headers
            )
            assert blocked_delete.status_code == 409
            db = SessionLocal()
            try:
                stored = db.get(ChunkEvaluationRun, run["id"])
                assert stored is not None
                stored.status = EvaluationStatus.RUNNING
                stored.stage = "rules_running"
                stored.celery_task_id = None
                db.commit()
            finally:
                db.close()
            paused = client.post(
                f"/api/v1/chunk-evaluations/{run['id']}/pause", headers=headers
            )
            assert paused.status_code == 200
            assert paused.json()["status"] == "paused"
            deleted = client.delete(f"/api/v1/chunk-evaluations/{run['id']}", headers=headers)
            assert deleted.status_code == 204
    finally:
        Base.metadata.drop_all(bind=global_engine)


def make_evaluation_db(path: str | None = None):
    engine = create_engine(f"sqlite:///{path}" if path else "sqlite://")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)


def seed_evaluation(db, count: int) -> tuple[Document, ChunkEvaluationCreate]:
    embedding = ModelProfile(
        id="embedding",
        name=f"embedding-{uuid.uuid4()}",
        kind=ModelKind.EMBEDDING,
        base_url="https://example.com/v1",
        model_name="embedding",
        dimensions=3,
    )
    enrichment = ModelProfile(
        id="judge",
        name=f"judge-{uuid.uuid4()}",
        kind=ModelKind.ENRICHMENT,
        base_url="https://example.com/v1",
        model_name="judge",
        request_timeout_seconds=1200,
        request_max_retries=2,
    )
    kb = KnowledgeBase(
        id=f"kb-{uuid.uuid4()}",
        name=f"kb-{uuid.uuid4()}",
        description="",
        embedding_profile_id=embedding.id,
        enrichment_profile_id=enrichment.id,
        settings={},
    )
    document = Document(
        id=f"doc-{uuid.uuid4()}",
        kb_id=kb.id,
        filename="manual.md",
        object_key=f"objects/{uuid.uuid4()}.md",
        status=DocumentStatus.READY,
        extracted_text="".join(f"第{i}段完整句子。" for i in range(count)),
        metadata_json={"pages": 1},
    )
    db.add_all([embedding, enrichment, kb, document])
    db.flush()
    for index in range(count):
        db.add(
            DocumentChunk(
                id=f"chunk-{uuid.uuid4()}",
                document_id=document.id,
                kb_id=kb.id,
                chunk_index=index,
                content=f"第{index}段完整句子。",
                section_path="安装",
                token_count=512,
                granularity_level="L2",
                context_prefix="[文档: 手册 / 章节: 安装]",
            )
        )
    db.commit()
    return document, ChunkEvaluationCreate(document_id=document.id, granularities=["L2"])


def test_initial_selection_is_deterministic_and_capped():
    engine, factory = make_evaluation_db()
    db = factory()
    try:
        document, request = seed_evaluation(db, 250)
        request.initial_chunk_limit = 100
        service = ChunkEvaluationService()
        first = service.create_run(db, request)
        service._ensure_items(db, first)
        db.commit()
        first_ids = list(
            db.scalars(
                select(ChunkEvaluationItem.chunk_id)
                .where(ChunkEvaluationItem.run_id == first.id)
                .order_by(ChunkEvaluationItem.chunk_id)
            )
        )
        second = service.create_run(db, request)
        service._ensure_items(db, second)
        db.commit()
        second_ids = list(
            db.scalars(
                select(ChunkEvaluationItem.chunk_id)
                .where(ChunkEvaluationItem.run_id == second.id)
                .order_by(ChunkEvaluationItem.chunk_id)
            )
        )
        assert len(first_ids) == 100
        assert first_ids == second_ids
        assert first.chunks_not_selected == 150
        assert document.id == first.document_id

        service.pause_run(db, first.id)
        extra_chunk_id = db.scalar(
            select(DocumentChunk.id).where(
                DocumentChunk.document_id == document.id,
                DocumentChunk.id.not_in(first_ids),
            )
        )
        assert extra_chunk_id is not None
        first = service.select_chunks(db, first.id, [extra_chunk_id])
        assert first.chunks_total == 101
        assert first.chunks_not_selected == 149
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_llm_chunk_requests_are_bounded_and_each_result_is_persisted():
    engine, factory = make_evaluation_db()
    db = factory()
    try:
        _, request = seed_evaluation(db, 5)
        state = {"active": 0, "max_active": 0, "calls": 0}

        class Judge:
            async def evaluate(self, profile, samples):
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
                state["calls"] += 1
                await asyncio.sleep(0.03)
                state["active"] -= 1
                sample = samples[0]
                return {
                    sample.chunk.id: {
                        "scores": {"self_contained": 5, "coherence": 5},
                        "issue_codes": [],
                        "reason": "完整",
                        "suggestions": [],
                    }
                }, []

        service = ChunkEvaluationService()
        service.judge = Judge()
        run = service.create_run(db, request.model_copy(update={"llm_concurrency": 2}))
        await service.run(db, run.id)
        db.refresh(run)
        items = list(
            db.scalars(select(ChunkEvaluationItem).where(ChunkEvaluationItem.run_id == run.id))
        )
        results = list(
            db.scalars(select(ChunkEvaluationResult).where(ChunkEvaluationResult.run_id == run.id))
        )
        assert run.status == EvaluationStatus.COMPLETED
        assert state["calls"] == 5
        assert state["max_active"] == 2
        assert len(items) == 5
        assert len(results) == 5
        assert all(item.status == EvaluationItemStatus.SUCCEEDED for item in items)
        assert run.progress["batch_completed"] == 5
        assert run.progress["chunks_succeeded"] == 5
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_two_consecutive_llm_failures_break_circuit_and_retry_updates_same_results():
    engine, factory = make_evaluation_db()
    db = factory()
    try:
        _, request = seed_evaluation(db, 6)

        class FailingJudge:
            def __init__(self):
                self.calls = 0

            async def evaluate(self, profile, samples):
                self.calls += 1
                await asyncio.sleep(0.01)
                return {}, ["judge unavailable"]

        failing = FailingJudge()
        service = ChunkEvaluationService()
        service.judge = failing
        run = service.create_run(db, request.model_copy(update={"llm_concurrency": 2}))
        await service.run(db, run.id)
        db.refresh(run)
        items = list(
            db.scalars(select(ChunkEvaluationItem).where(ChunkEvaluationItem.run_id == run.id))
        )
        results = list(
            db.scalars(select(ChunkEvaluationResult).where(ChunkEvaluationResult.run_id == run.id))
        )
        assert run.status == EvaluationStatus.PARTIAL
        assert failing.calls == 2
        assert len(results) == 6
        assert all(item.status == EvaluationItemStatus.FAILED for item in items)
        assert all(item.fallback_applied for item in items)

        class SuccessJudge:
            async def evaluate(self, profile, samples):
                sample = samples[0]
                return {
                    sample.chunk.id: {
                        "scores": {"self_contained": 5, "coherence": 5},
                        "issue_codes": [],
                        "reason": "完整",
                        "suggestions": [],
                    }
                }, []

        result_ids = {item.id for item in results}
        service.judge = SuccessJudge()
        run = service.retry_failed(db, run.id, ChunkEvaluationRetryFailed())
        await service.run(db, run.id)
        db.refresh(run)
        retried_results = list(
            db.scalars(select(ChunkEvaluationResult).where(ChunkEvaluationResult.run_id == run.id))
        )
        assert run.status == EvaluationStatus.COMPLETED
        assert {item.id for item in retried_results} == result_ids
        assert all(item.llm_scores for item in retried_results)
    finally:
        db.close()
        engine.dispose()

@pytest.mark.asyncio
async def test_pause_cancels_inflight_requests_and_resume_finishes_queue():
    database_path = Path(__file__).with_name(".chunk_evaluation_pause.db")
    database_path.unlink(missing_ok=True)
    engine, factory = make_evaluation_db(str(database_path))
    db = factory()
    control_db = factory()
    try:
        _, request = seed_evaluation(db, 5)
        entered = asyncio.Event()

        class BlockingJudge:
            def __init__(self):
                self.cancelled = 0

            async def evaluate(self, profile, samples):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled += 1
                    raise

        blocking = BlockingJudge()
        service = ChunkEvaluationService()
        service.judge = blocking
        run = service.create_run(db, request.model_copy(update={"llm_concurrency": 2}))
        worker = asyncio.create_task(service.run(db, run.id))
        await asyncio.wait_for(entered.wait(), timeout=5)
        control_run = control_db.get(ChunkEvaluationRun, run.id)
        assert control_run is not None
        control_run.status = EvaluationStatus.PAUSING
        control_run.pause_requested_at = datetime.now(UTC)
        control_db.commit()
        await asyncio.wait_for(worker, timeout=5)
        db.expire_all()
        run = db.get(ChunkEvaluationRun, run.id)
        assert run is not None
        assert run.status == EvaluationStatus.PAUSED
        items = list(
            db.scalars(select(ChunkEvaluationItem).where(ChunkEvaluationItem.run_id == run.id))
        )
        assert blocking.cancelled >= 1
        assert any(item.status == EvaluationItemStatus.INTERRUPTED for item in items)
        assert any(item.status == EvaluationItemStatus.LLM_QUEUED for item in items)

        class SuccessJudge:
            async def evaluate(self, profile, samples):
                sample = samples[0]
                return {
                    sample.chunk.id: {
                        "scores": {"self_contained": 5, "coherence": 5},
                        "issue_codes": [],
                        "reason": "完整",
                        "suggestions": [],
                    }
                }, []

        service.judge = SuccessJudge()
        resumed = service.resume_run(db, run.id)
        await service.run(db, resumed.id)
        db.refresh(resumed)
        assert resumed.status == EvaluationStatus.COMPLETED
        assert resumed.chunks_succeeded == 5
    finally:
        control_db.close()
        db.close()
        engine.dispose()
        database_path.unlink(missing_ok=True)


def test_model_profile_timeout_and_retries_are_applied(monkeypatch):
    from app.indexing.model_gateway import ModelGateway

    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("app.indexing.model_gateway.AsyncOpenAI", FakeAsyncOpenAI)
    profile = SimpleNamespace(
        kind=ModelKind.LLM,
        base_url="https://example.com/v1",
        model_name="judge",
        api_key_encrypted=None,
        request_timeout_seconds=1200,
        request_max_retries=2,
    )
    ModelGateway._openai_client(profile)
    assert captured["timeout"] == 1200
    assert captured["max_retries"] == 2