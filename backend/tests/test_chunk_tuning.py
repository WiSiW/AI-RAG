from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    ChunkTuningCaseResult,
    ChunkTuningRun,
    ChunkTuningStatus,
    ChunkTuningTrial,
    Document,
    DocumentStatus,
    KnowledgeBase,
    ModelKind,
    ModelProfile,
)
from app.db.session import Base, SessionLocal
from app.db.session import engine as global_engine
from app.evaluation.retrieval_metrics import (
    build_trial_parameters,
    evidence_span,
    normalize_evidence_text,
    score_retrieval_case,
)
from app.indexing.types import IndexedChunk, IndexedRepresentation, RetrievedChunk
from app.main import app
from app.parsing.base import ParsedDocument, ParsedElement
from app.schemas import (
    ChunkTuningRunCreate,
    ChunkTuningVerificationCreate,
    RetrievalEvaluationCaseCreate,
    RetrievalEvaluationDatasetCreate,
    RetrievalEvaluationEvidenceInput,
)
from app.services.chunk_tuning_service import ChunkTuningService, PreparedDocument


class FakeGateway:
    def __init__(self):
        self.embedding_inputs: list[str] = []

    async def embed_texts(self, profile, texts):
        self.embedding_inputs.extend(texts)
        return [[1.0, float(index + 1)] for index, _ in enumerate(texts)]

    async def complete(self, *args, **kwargs):
        return ""


class FakeVectorStore:
    def __init__(self):
        self.dropped: list[str] = []

    def drop_knowledge_base(self, namespace):
        self.dropped.append(namespace)


class FakeBM25Store:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.cleaned: list[str] = []

    def delete_all_versions(self, namespace):
        self.cleaned.append(namespace)


class MemoryVectorStore:
    def __init__(self):
        self.rows: dict[str, list[tuple[IndexedRepresentation, list[float]]]] = {}

    def replace_document(
        self,
        namespace,
        document_id,
        source,
        representations,
        vectors,
        dimensions,
        *,
        index_version="v2",
        chunks=None,
    ):
        self.rows[namespace] = list(zip(representations or [], vectors, strict=True))

    def search(
        self,
        namespace,
        query_vector,
        top_k,
        dimensions,
        document_id=None,
        *,
        representation_type=None,
        index_version="v2",
    ):
        candidates = [
            (item, vector)
            for item, vector in self.rows.get(namespace, [])
            if representation_type is None or item.representation_type == representation_type
        ]
        candidates.sort(
            key=lambda pair: sum(left * right for left, right in zip(query_vector, pair[1], strict=False)),
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                kb_id=namespace,
                content=item.indexed_text,
                section=item.section,
                source="memory",
                metadata={
                    "vector_score": 1.0 / (index + 1),
                    "vector_rank": index + 1,
                    "representation_type": item.representation_type,
                    "question_index": 0,
                },
            )
            for index, (item, _) in enumerate(candidates[:top_k])
        ]

    def drop_knowledge_base(self, namespace):
        self.rows.pop(namespace, None)


class MemoryBM25Store(FakeBM25Store):
    def __init__(self, root_dir: Path):
        super().__init__(root_dir)
        self.rows: dict[str, list[IndexedChunk]] = {}

    def replace_document(self, namespace, document_id, chunks, *, index_version="v2"):
        current = self.rows.setdefault(namespace, [])
        self.rows[namespace] = [item for item in current if item.document_id != document_id] + list(chunks)

    def delete_all_versions(self, namespace):
        super().delete_all_versions(namespace)
        self.rows.pop(namespace, None)

    def search(self, namespace, query, top_k, document_id=None, *, index_version="v2"):
        query_terms = set(query)
        ranked = sorted(
            self.rows.get(namespace, []),
            key=lambda chunk: len(query_terms & set(chunk.content)),
            reverse=True,
        )
        return [
            RetrievedChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                kb_id=namespace,
                content=item.content,
                section=item.section,
                source="memory",
                metadata={
                    "bm25_score": float(len(query_terms & set(item.content))),
                    "matched_representations": ["bm25"],
                    "representation_scores": {"bm25": float(len(query_terms & set(item.content)))},
                },
            )
            for item in ranked[:top_k]
        ]


def _profile(profile_id: str, kind: str, *, dimensions: int | None = None) -> ModelProfile:
    return ModelProfile(
        id=profile_id,
        name=profile_id,
        kind=kind,
        provider="openai-compatible",
        base_url="https://example.com/v1",
        model_name=profile_id,
        dimensions=dimensions,
        is_active=True,
    )


def _database():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine)()


def test_trial_matrix_and_evidence_metrics():
    trials = build_trial_parameters(
        [256, 512, 1024, 1536],
        [0, 10, 20, 25],
        baseline_size=512,
        baseline_overlap=64,
    )
    assert len(trials) == 17
    assert sum(1 for item in trials if item.is_baseline) == 1
    assert next(item for item in trials if item.is_baseline).chunk_overlap == 64
    assert any(item.chunk_size == 1024 and item.overlap_ratio == 20 and item.chunk_overlap == 205 for item in trials)

    source = "第一段有\n空格。第二段是证据。"
    assert evidence_span(source, "第一段有 空格") == (0, 7)
    assert normalize_evidence_text(" A\n B ") == "ab"

    evidence = [
        {"id": "e1", "quote": "第一段证据。", "normalized_quote": "第一段证据。"},
        {"id": "e2", "quote": "第二段证据。", "normalized_quote": "第二段证据。"},
    ]
    hits = [
        {"content": "第一段证据。噪声", "chunk_id": "a"},
        {"content": "无关", "chunk_id": "b"},
        {"content": "无关", "chunk_id": "c"},
        {"content": "无关", "chunk_id": "d"},
        {"content": "无关", "chunk_id": "e"},
        {"content": "第二段证据。", "chunk_id": "f"},
    ]
    result = score_retrieval_case(evidence=evidence, hits=hits, latency_ms=12.5)
    assert result["recall_at_5"] == 0.5
    assert result["recall_at_10"] == 1.0
    assert result["all_evidence_hit_at_5"] is False
    assert result["all_evidence_hit_at_10"] is True
    assert result["reciprocal_rank"] == 1.0
    assert result["evidence_density"] > 0


@pytest.mark.asyncio
async def test_query_embeddings_are_generated_once_per_question():
    gateway = FakeGateway()
    service = ChunkTuningService(gateway=gateway)
    profile = _profile("embedding", ModelKind.EMBEDDING, dimensions=2)
    cases = [{"id": str(index), "question": f"问题 {index}"} for index in range(3)]
    vectors = await service._embed_queries(profile, cases)
    assert set(vectors) == {"0", "1", "2"}
    assert gateway.embedding_inputs == ["问题 0", "问题 1", "问题 2"]


def test_screening_run_requires_dataset_and_creates_isolated_trials():
    engine, db = _database()
    try:
        embedding = _profile("embedding", ModelKind.EMBEDDING, dimensions=2)
        enrichment = _profile("enrichment", ModelKind.ENRICHMENT)
        llm = _profile("llm", ModelKind.LLM)
        kb = KnowledgeBase(
            id="kb",
            name="kb",
            description="",
            embedding_profile_id=embedding.id,
            enrichment_profile_id=enrichment.id,
            default_llm_profile_id=llm.id,
            settings={"chunk_mode": "structure", "chunk_size": 512, "chunk_overlap": 64},
        )
        evidence_text = "".join(f"证据段落 {index}。" for index in range(20))
        document = Document(
            id="doc",
            kb_id=kb.id,
            filename="manual.md",
            object_key="objects/manual.md",
            status=DocumentStatus.READY,
            extracted_text=evidence_text,
            metadata_json={"title": "手册", "sha256": "abc"},
        )
        db.add_all([embedding, enrichment, llm, kb, document])
        db.commit()

        service = ChunkTuningService(gateway=FakeGateway())
        dataset = service.create_dataset(
            db,
            RetrievalEvaluationDatasetCreate(
                knowledge_base_id=kb.id,
                name="调优集",
                document_ids=[document.id],
            ),
        )
        for index, quote in enumerate(f"证据段落 {item}。" for item in range(20)):
            service.add_case(
                db,
                dataset.id,
                RetrievalEvaluationCaseCreate(
                    question=f"第 {index} 个问题是什么？",
                    reference_answer=quote,
                    question_type="detail",
                    evidence=[
                        RetrievalEvaluationEvidenceInput(
                            document_id=document.id,
                            quote=quote,
                            section_path="正文",
                        )
                    ],
                ),
            )
        db.expire_all()
        run = service.create_screening_run(
            db,
            ChunkTuningRunCreate(knowledge_base_id=kb.id, dataset_id=dataset.id),
        )
        assert run.status == ChunkTuningStatus.QUEUED
        assert len(run.trials) == 17
        assert all(trial.index_namespace.startswith("eval-") for trial in run.trials)

        parent = db.get(type(run), run.id)
        parent.status = ChunkTuningStatus.AWAITING_SELECTION
        first_trial = run.trials[0]
        first_trial.status = ChunkTuningStatus.COMPLETED
        first_trial.metrics = {"recall_at_5": 0.8}
        db.commit()
        verification = service.create_verification_run(
            db,
            run.id,
            ChunkTuningVerificationCreate(trial_ids=[first_trial.id]),
        )
        assert verification.stage == "verification"
        assert any(trial.is_baseline for trial in verification.trials)

        apply_trial = verification.trials[0]
        apply_trial.status = ChunkTuningStatus.COMPLETED
        apply_trial.metrics = {"recall_at_5": 0.81, "recall_at_10": 0.9}
        db.commit()
        updated = service.apply_trial(db, verification.id, apply_trial.id)
        assert updated.settings["chunk_size"] == apply_trial.chunk_size
        assert updated.settings["last_chunk_tuning"]["run_id"] == verification.id
    finally:
        db.close()
        engine.dispose()


def test_cleanup_only_removes_eval_namespaces(tmp_path):
    vector_store = FakeVectorStore()
    bm25_store = FakeBM25Store(tmp_path)
    service = ChunkTuningService(
        gateway=FakeGateway(), vector_store=vector_store, bm25_store=bm25_store
    )
    service.cleanup_namespace("production-kb")
    assert not vector_store.dropped
    assert not bm25_store.cleaned
    service.cleanup_namespace("eval-abc")
    assert vector_store.dropped == ["eval-abc"]
    assert bm25_store.cleaned == ["eval-abc"]


@pytest.mark.asyncio
async def test_screening_trial_uses_isolated_index_and_scores_retrieval(tmp_path):
    engine, db = _database()
    try:
        embedding = _profile("memory-embedding", ModelKind.EMBEDDING, dimensions=2)
        kb = KnowledgeBase(
            id="kb",
            name="kb",
            description="",
            embedding_profile_id=embedding.id,
            settings={
                "chunk_mode": "structure",
                "chunk_size": 32,
                "chunk_overlap": 4,
                "retrieval_weights": {"bm25": 1, "original": 1},
            },
        )
        document = Document(
            id="doc",
            kb_id=kb.id,
            filename="manual.md",
            object_key="objects/manual.md",
            status=DocumentStatus.READY,
            extracted_text="",
            metadata_json={"title": "手册"},
        )
        run = ChunkTuningRun(
            id="run",
            kb_id=kb.id,
            dataset_id="dataset",
            dataset_version=1,
            stage="screening",
            status=ChunkTuningStatus.RUNNING,
            config_snapshot={},
            dataset_snapshot={},
            progress={},
        )
        trial = ChunkTuningTrial(
            id="trial",
            run_id=run.id,
            label="32 / 4",
            index_namespace="eval-memory",
            chunk_mode="structure",
            chunk_size=32,
            chunk_overlap=4,
            overlap_ratio=12.5,
            status=ChunkTuningStatus.RUNNING,
        )
        db.add_all([embedding, kb, document, run, trial])
        db.commit()

        text = "安装前必须关闭电源。设备必须接地。完成安装后检查指示灯。"
        parsed = ParsedDocument(
            source="manual.md",
            title="手册",
            elements=[ParsedElement(type="title", text="安装", level=1), ParsedElement(type="text", text=text)],
            full_text=text,
            metadata={"format": "markdown"},
        )
        gateway = FakeGateway()
        vector_store = MemoryVectorStore()
        bm25_store = MemoryBM25Store(tmp_path)
        service = ChunkTuningService(
            gateway=gateway, vector_store=vector_store, bm25_store=bm25_store
        )
        cases = [
            {
                "id": "case-1",
                "question": "安装前需要做什么？",
                "question_type": "detail",
                "answerable": True,
                "reference_answer": "关闭电源。",
                "evidence": [
                    {
                        "id": "evidence-1",
                        "document_id": document.id,
                        "quote": "安装前必须关闭电源。",
                        "normalized_quote": "安装前必须关闭电源。",
                    }
                ],
            }
        ]
        vectors = await service._embed_queries(embedding, cases)
        await service._execute_screening_trial(
            db,
            kb,
            run,
            trial,
            embedding,
            [PreparedDocument(document=document, parsed=parsed)],
            cases,
            vectors,
        )
        assert "eval-memory" in vector_store.rows
        assert "eval-memory" in bm25_store.rows
        assert trial.status == ChunkTuningStatus.COMPLETED
        saved_results = db.query(ChunkTuningCaseResult).filter_by(trial_id=trial.id).all()
        assert len(saved_results) == 1
        assert saved_results[0].recall_at_10 == 1.0
        assert gateway.embedding_inputs[0] == "安装前需要做什么？"
        service.cleanup_namespace(trial.index_namespace)
        assert "eval-memory" not in vector_store.rows
        assert "eval-memory" not in bm25_store.rows
    finally:
        db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_verification_trial_builds_full_v2_index_and_marks_degradation(tmp_path):
    engine, db = _database()
    try:
        embedding = _profile("full-embedding", ModelKind.EMBEDDING, dimensions=2)
        llm = _profile("full-llm", ModelKind.LLM)
        kb = KnowledgeBase(
            id="kb-full",
            name="kb-full",
            description="",
            embedding_profile_id=embedding.id,
            default_llm_profile_id=llm.id,
            settings={
                "chunk_mode": "structure",
                "chunk_size": 32,
                "chunk_overlap": 4,
                "retrieval_weights": {
                    "bm25": 1,
                    "l1": 0.8,
                    "original": 1,
                    "summary": 1,
                    "questions": 1,
                    "l3": 0.6,
                },
            },
        )
        document = Document(
            id="doc-full",
            kb_id=kb.id,
            filename="manual.md",
            object_key="objects/manual-full.md",
            status=DocumentStatus.READY,
            extracted_text="",
            metadata_json={"title": "完整手册"},
        )
        run = ChunkTuningRun(
            id="run-full",
            kb_id=kb.id,
            dataset_id="dataset-full",
            dataset_version=1,
            stage="verification",
            status=ChunkTuningStatus.RUNNING,
            config_snapshot={},
            dataset_snapshot={},
            progress={},
        )
        trial = ChunkTuningTrial(
            id="trial-full",
            run_id=run.id,
            label="32 / 4 full",
            index_namespace="eval-full",
            chunk_mode="structure",
            chunk_size=32,
            chunk_overlap=4,
            overlap_ratio=12.5,
            status=ChunkTuningStatus.RUNNING,
        )
        db.add_all([embedding, llm, kb, document, run, trial])
        db.commit()

        text = "安装前必须关闭电源。设备必须接地。完成安装后检查指示灯。错误码 E100 表示电源异常。"
        parsed = ParsedDocument(
            source="manual.md",
            title="完整手册",
            elements=[
                ParsedElement(type="title", text="安装", level=1),
                ParsedElement(type="text", text=text),
            ],
            full_text=text,
            metadata={"format": "markdown"},
        )
        gateway = FakeGateway()
        vector_store = MemoryVectorStore()
        bm25_store = MemoryBM25Store(tmp_path)
        service = ChunkTuningService(
            gateway=gateway, vector_store=vector_store, bm25_store=bm25_store
        )
        cases = [
            {
                "id": "case-full",
                "question": "安装前需要做什么？",
                "question_type": "detail",
                "answerable": True,
                "reference_answer": "关闭电源。",
                "evidence": [
                    {
                        "id": "evidence-full",
                        "document_id": document.id,
                        "quote": "安装前必须关闭电源。",
                        "normalized_quote": "安装前必须关闭电源。",
                    }
                ],
            }
        ]
        vectors = await service._embed_queries(embedding, cases)
        await service._execute_verification_trial(
            db,
            kb,
            run,
            trial,
            embedding,
            None,
            llm,
            [PreparedDocument(document=document, parsed=parsed)],
            cases,
            vectors,
        )
        assert trial.status == ChunkTuningStatus.PARTIAL
        assert trial.cost["candidate_chunk_count"] >= 1
        assert trial.cost["representation_count"] >= 1
        saved = db.query(ChunkTuningCaseResult).filter_by(trial_id=trial.id).one()
        assert saved.recall_at_10 == 1.0
        assert saved.quality_scores == {}
        service.cleanup_namespace(trial.index_namespace)
    finally:
        db.close()
        engine.dispose()


def test_chunk_tuning_api_dataset_case_and_atomic_jsonl_import():
    Base.metadata.drop_all(bind=global_engine)
    with TestClient(app) as client:
        login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "test-password"},
        )
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        db = SessionLocal()
        try:
            embedding = _profile("api-embedding", ModelKind.EMBEDDING, dimensions=2)
            enrichment = _profile("api-enrichment", ModelKind.ENRICHMENT)
            kb = KnowledgeBase(
                id="api-kb",
                name="API KB",
                description="",
                embedding_profile_id=embedding.id,
                enrichment_profile_id=enrichment.id,
                settings={},
            )
            document = Document(
                id="api-doc",
                kb_id=kb.id,
                filename="api.md",
                object_key="objects/api.md",
                status=DocumentStatus.READY,
                extracted_text="安装前必须关闭电源。",
            )
            db.add_all([embedding, enrichment, kb, document])
            db.commit()
        finally:
            db.close()

        created = client.post(
            "/api/v1/retrieval-evaluation-datasets",
            headers=headers,
            json={
                "knowledge_base_id": "api-kb",
                "name": "API 调优集",
                "document_ids": ["api-doc"],
            },
        )
        assert created.status_code == 201, created.text
        dataset_id = created.json()["id"]
        case = client.post(
            f"/api/v1/retrieval-evaluation-datasets/{dataset_id}/cases",
            headers=headers,
            json={
                "question": "安装前应该做什么？",
                "reference_answer": "关闭电源。",
                "question_type": "detail",
                "evidence": [
                    {
                        "document_id": "api-doc",
                        "quote": "安装前必须关闭电源。",
                        "section_path": "安装",
                    }
                ],
            },
        )
        assert case.status_code == 201, case.text

        invalid_import = client.post(
            f"/api/v1/retrieval-evaluation-datasets/{dataset_id}/import",
            headers=headers,
            files={
                "file": (
                    "cases.jsonl",
                    '{"type":"case","schema_version":1,"question":"缺证据？","reference_answer":"答案","question_type":"detail","evidence":[]}',
                    "application/x-ndjson",
                )
            },
        )
        assert invalid_import.status_code == 422
        refreshed = client.get(
            f"/api/v1/retrieval-evaluation-datasets/{dataset_id}",
            headers=headers,
        )
        assert refreshed.json()["case_count"] == 1

        too_small = client.post(
            "/api/v1/chunk-tuning-runs",
            headers=headers,
            json={"knowledge_base_id": "api-kb", "dataset_id": dataset_id},
        )
        assert too_small.status_code == 422
    Base.metadata.drop_all(bind=global_engine)
