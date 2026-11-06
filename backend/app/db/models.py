from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class ModelKind(StrEnum):
    EMBEDDING = "embedding"
    RERANK = "rerank"
    LLM = "llm"
    ENRICHMENT = "enrichment"


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvaluationStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationStage(StrEnum):
    QUEUED = "queued"
    RULES_RUNNING = "rules_running"
    LLM_RUNNING = "llm_running"
    AGGREGATING = "aggregating"
    PAUSING = "pausing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationItemStatus(StrEnum):
    QUEUED = "queued"
    RULES_RUNNING = "rules_running"
    LLM_QUEUED = "llm_queued"
    LLM_RUNNING = "llm_running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ChunkTuningStage(StrEnum):
    SCREENING = "screening"
    VERIFICATION = "verification"


class ChunkTuningStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    AWAITING_SELECTION = "awaiting_selection"
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TransferJobKind(StrEnum):
    EXPORT = "export"
    IMPORT = "import"


class TransferStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelProfile(Base):
    __tablename__ = "model_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    provider: Mapped[str] = mapped_column(String(50), default="openai-compatible")
    base_url: Mapped[str] = mapped_column(String(500))
    model_name: Mapped[str] = mapped_column(String(200))
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_timeout_seconds: Mapped[int] = mapped_column(Integer, default=1200)
    request_max_retries: Mapped[int] = mapped_column(Integer, default=2)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    embedding_profile_id: Mapped[str] = mapped_column(ForeignKey("model_profiles.id"))
    rerank_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id"), nullable=True
    )
    enrichment_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id"), nullable=True
    )
    default_llm_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id"), nullable=True
    )
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    embedding_profile = relationship("ModelProfile", foreign_keys=[embedding_profile_id])
    rerank_profile = relationship("ModelProfile", foreign_keys=[rerank_profile_id])
    enrichment_profile = relationship("ModelProfile", foreign_keys=[enrichment_profile_id])
    default_llm_profile = relationship("ModelProfile", foreign_keys=[default_llm_profile_id])


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kb_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    object_key: Mapped[str] = mapped_column(String(1000), unique=True)
    content_type: Mapped[str] = mapped_column(String(200), default="application/octet-stream")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default=DocumentStatus.PENDING, index=True)
    extracted_text: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    references_json: Mapped[list] = mapped_column(JSON, default=list)
    chunk_mode: Mapped[str] = mapped_column(String(40), default="structure")
    chunk_size: Mapped[int] = mapped_column(Integer, default=512)
    chunk_overlap: Mapped[int] = mapped_column(Integer, default=64)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    knowledge_base = relationship("KnowledgeBase")
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )
    jobs = relationship("IngestionJob", back_populates="document", cascade="all, delete-orphan")
    tables = relationship("DocumentTable", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    kb_id: Mapped[str] = mapped_column(ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(String(1000), default="正文")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    granularity_level: Mapped[str] = mapped_column(String(2), default="L2", index=True)
    granule_type: Mapped[str] = mapped_column(String(40), default="paragraph")
    parent_chunk_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    context_prefix: Mapped[str] = mapped_column(Text, default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    hypothetical_questions: Mapped[list] = mapped_column(JSON, default=list)
    qa_pairs: Mapped[list] = mapped_column(JSON, default=list)
    enrichment_status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    enrichment_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="chunks")


class DocumentTable(Base):
    __tablename__ = "document_tables"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    page_start: Mapped[int] = mapped_column(Integer, default=1)
    page_end: Mapped[int] = mapped_column(Integer, default=1)
    bbox: Mapped[list] = mapped_column(JSON, default=list)
    layout_block_id: Mapped[str] = mapped_column(String(120), default="")
    caption: Mapped[str] = mapped_column(Text, default="")
    cells: Mapped[list] = mapped_column(JSON, default=list)
    headers: Mapped[list] = mapped_column(JSON, default=list)
    rows: Mapped[list] = mapped_column(JSON, default=list)
    markdown: Mapped[str] = mapped_column(Text, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    description_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    extraction_engine: Mapped[str] = mapped_column(String(80), default="")
    quality_score: Mapped[float] = mapped_column(Float, default=0)
    quality_json: Mapped[dict] = mapped_column(JSON, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document = relationship("Document", back_populates="tables")


class ParsingEngineConfig(Base):
    __tablename__ = "parsing_engine_configs"
    __table_args__ = (UniqueConstraint("category", "display_name", name="uq_parsing_engine_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    category: Mapped[str] = mapped_column(String(20), index=True)
    adapter: Mapped[str] = mapped_column(String(80))
    display_name: Mapped[str] = mapped_column(String(120))
    priority: Mapped[int] = mapped_column(Integer, default=100, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(20), default=JobStatus.PENDING, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document = relationship("Document", back_populates="jobs")


class ChunkEvaluationRun(Base):
    __tablename__ = "chunk_evaluation_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(20), index=True)
    knowledge_base_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    granularities: Mapped[list] = mapped_column(JSON, default=list)
    judge_profile_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    llm_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(
        String(20), default=EvaluationStatus.QUEUED, index=True
    )
    stage: Mapped[str] = mapped_column(
        String(30), default=EvaluationStage.QUEUED, index=True
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    pause_requested_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paused_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    aggregate: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    results = relationship(
        "ChunkEvaluationResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    items = relationship(
        "ChunkEvaluationItem",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ChunkEvaluationItem.queue_position",
    )

    def _progress_int(self, key: str) -> int:
        return int((self.progress or {}).get(key, 0) or 0)

    @property
    def batch_total(self) -> int:
        return self._progress_int("batch_total")

    @property
    def batch_completed(self) -> int:
        return self._progress_int("batch_completed")

    @property
    def batch_running(self) -> int:
        return self._progress_int("batch_running")

    @property
    def chunks_total(self) -> int:
        return self._progress_int("chunks_total")

    @property
    def chunks_succeeded(self) -> int:
        return self._progress_int("chunks_succeeded")

    @property
    def chunks_failed(self) -> int:
        return self._progress_int("chunks_failed")

    @property
    def chunks_pending(self) -> int:
        return self._progress_int("chunks_pending")

    @property
    def chunks_interrupted(self) -> int:
        return self._progress_int("chunks_interrupted")

    @property
    def chunks_not_selected(self) -> int:
        return self._progress_int("chunks_not_selected")

    @property
    def can_pause(self) -> bool:
        return self.status in {EvaluationStatus.QUEUED, EvaluationStatus.RUNNING}

    @property
    def can_resume(self) -> bool:
        return self.status == EvaluationStatus.PAUSED

    @property
    def can_delete(self) -> bool:
        return self.status in {
            EvaluationStatus.PAUSED,
            EvaluationStatus.PARTIAL,
            EvaluationStatus.COMPLETED,
            EvaluationStatus.FAILED,
        }


class ChunkEvaluationItem(Base):
    __tablename__ = "chunk_evaluation_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk_id: Mapped[str] = mapped_column(String(36), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    granularity_level: Mapped[str] = mapped_column(String(2), index=True)
    section_path: Mapped[str] = mapped_column(String(1000), default="正文")
    content_snapshot: Mapped[str] = mapped_column(Text, default="")
    selection_source: Mapped[str] = mapped_column(String(20), default="automatic")
    queue_position: Mapped[int] = mapped_column(Integer, default=0, index=True)
    status: Mapped[str] = mapped_column(
        String(20), default=EvaluationItemStatus.QUEUED, index=True
    )
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    rule_succeeded: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_succeeded: Mapped[bool] = mapped_column(Boolean, default=False)
    fallback_applied: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    run = relationship("ChunkEvaluationRun", back_populates="items")


class ChunkEvaluationResult(Base):
    __tablename__ = "chunk_evaluation_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_evaluation_runs.id", ondelete="CASCADE"), index=True
    )
    chunk_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    kb_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    granularity_level: Mapped[str] = mapped_column(String(2), index=True)
    granule_type: Mapped[str] = mapped_column(String(40), default="paragraph")
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    content_snapshot: Mapped[str] = mapped_column(Text)
    section_path: Mapped[str] = mapped_column(String(1000), default="正文")
    context_prefix: Mapped[str] = mapped_column(Text, default="")
    parent_chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    context_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    rule_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    llm_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    llm_reason: Mapped[str] = mapped_column(Text, default="")
    issue_codes: Mapped[list] = mapped_column(JSON, default=list)
    suggestions: Mapped[list] = mapped_column(JSON, default=list)
    composite_score: Mapped[float] = mapped_column(Float, default=0)
    grade: Mapped[str] = mapped_column(String(20), default="fail", index=True)
    review_sample: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    human_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    human_issue_codes: Mapped[list] = mapped_column(JSON, default=list)
    human_comment: Mapped[str] = mapped_column(Text, default="")
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run = relationship("ChunkEvaluationRun", back_populates="results")


class RetrievalEvaluationDataset(Base):
    __tablename__ = "retrieval_evaluation_datasets"
    __table_args__ = (
        UniqueConstraint("kb_id", "name", name="uq_retrieval_eval_dataset_kb_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kb_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    document_ids: Mapped[list] = mapped_column(JSON, default=list)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    cases = relationship(
        "RetrievalEvaluationCase",
        back_populates="dataset",
        cascade="all, delete-orphan",
        order_by="RetrievalEvaluationCase.sort_order",
    )


class RetrievalEvaluationCase(Base):
    __tablename__ = "retrieval_evaluation_cases"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_evaluation_datasets.id", ondelete="CASCADE"),
        index=True,
    )
    question: Mapped[str] = mapped_column(Text)
    normalized_question: Mapped[str] = mapped_column(Text, index=True)
    reference_answer: Mapped[str] = mapped_column(Text, default="")
    question_type: Mapped[str] = mapped_column(String(30), default="detail", index=True)
    answerable: Mapped[bool] = mapped_column(Boolean, default=True)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    dataset = relationship("RetrievalEvaluationDataset", back_populates="cases")
    evidence = relationship(
        "RetrievalEvaluationEvidence",
        back_populates="case",
        cascade="all, delete-orphan",
        order_by="RetrievalEvaluationEvidence.created_at",
    )


class RetrievalEvaluationEvidence(Base):
    __tablename__ = "retrieval_evaluation_evidence"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    case_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_evaluation_cases.id", ondelete="CASCADE"),
        index=True,
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    quote: Mapped[str] = mapped_column(Text)
    normalized_quote: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_path: Mapped[str] = mapped_column(String(1000), default="正文")
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    case = relationship("RetrievalEvaluationCase", back_populates="evidence")


class ChunkTuningRun(Base):
    __tablename__ = "chunk_tuning_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_run_id: Mapped[str | None] = mapped_column(String(36), index=True, nullable=True)
    kb_id: Mapped[str] = mapped_column(String(36), index=True)
    dataset_id: Mapped[str] = mapped_column(String(36), index=True)
    dataset_version: Mapped[int] = mapped_column(Integer, default=1)
    stage: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(
        String(30), default=ChunkTuningStatus.QUEUED, index=True
    )
    config_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    dataset_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    aggregate: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    trials = relationship(
        "ChunkTuningTrial",
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ChunkTuningTrial.created_at",
    )


class ChunkTuningTrial(Base):
    __tablename__ = "chunk_tuning_trials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_tuning_runs.id", ondelete="CASCADE"), index=True
    )
    label: Mapped[str] = mapped_column(String(120), default="")
    index_namespace: Mapped[str] = mapped_column(String(80), unique=True)
    chunk_mode: Mapped[str] = mapped_column(String(40), default="structure")
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)
    overlap_ratio: Mapped[float] = mapped_column(Float, default=0)
    is_baseline: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    status: Mapped[str] = mapped_column(
        String(30), default=ChunkTuningStatus.QUEUED, index=True
    )
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    cost: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run = relationship("ChunkTuningRun", back_populates="trials")
    case_results = relationship(
        "ChunkTuningCaseResult",
        back_populates="trial",
        cascade="all, delete-orphan",
    )


class ChunkTuningCaseResult(Base):
    __tablename__ = "chunk_tuning_case_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trial_id: Mapped[str] = mapped_column(
        ForeignKey("chunk_tuning_trials.id", ondelete="CASCADE"), index=True
    )
    case_id: Mapped[str] = mapped_column(String(36), index=True)
    question: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(30), default="detail", index=True)
    answerable: Mapped[bool] = mapped_column(Boolean, default=True)
    reference_answer: Mapped[str] = mapped_column(Text, default="")
    recall_at_5: Mapped[float] = mapped_column(Float, default=0)
    recall_at_10: Mapped[float] = mapped_column(Float, default=0)
    all_evidence_hit_at_5: Mapped[bool] = mapped_column(Boolean, default=False)
    all_evidence_hit_at_10: Mapped[bool] = mapped_column(Boolean, default=False)
    reciprocal_rank: Mapped[float] = mapped_column(Float, default=0)
    evidence_density: Mapped[float] = mapped_column(Float, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0)
    evidence_results: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_hits: Mapped[list] = mapped_column(JSON, default=list)
    answer: Mapped[str] = mapped_column(Text, default="")
    quality_scores: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    trial = relationship("ChunkTuningTrial", back_populates="case_results")


class KnowledgeBaseTransferJob(Base):
    __tablename__ = "knowledge_base_transfer_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    mode: Mapped[str] = mapped_column(String(20), default="documents")
    index_policy: Mapped[str] = mapped_column(String(20), default="strict")
    status: Mapped[str] = mapped_column(
        String(20), default=TransferStatus.QUEUED, index=True
    )
    source_knowledge_base_id: Mapped[str | None] = mapped_column(
        String(36), index=True, nullable=True
    )
    result_knowledge_base_id: Mapped[str | None] = mapped_column(
        String(36), index=True, nullable=True
    )
    package_name: Mapped[str] = mapped_column(String(500), default="")
    object_key: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)
    manifest: Mapped[dict] = mapped_column(JSON, default=dict)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    created_by: Mapped[str] = mapped_column(String(120), default="admin")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="新对话")
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    llm_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id"), nullable=True
    )
    summary: Mapped[str] = mapped_column(Text, default="")
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    summary_message_count: Mapped[int] = mapped_column(Integer, default=0)
    summary_status: Mapped[str] = mapped_column(String(20), default="idle")
    summary_version: Mapped[int] = mapped_column(Integer, default=0)
    summary_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    summary_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    summary_task_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    summary_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )
    knowledge_base_links = relationship(
        "ConversationKnowledgeBase",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationKnowledgeBase.position",
    )


class ConversationKnowledgeBase(Base):
    __tablename__ = "conversation_knowledge_bases"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    knowledge_base_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation = relationship("Conversation", back_populates="knowledge_base_links")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    rewritten_query: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    context: Mapped[list] = mapped_column(JSON, default=list)
    usage: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), index=True
    )
    rating: Mapped[int] = mapped_column(Integer)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class PublishedApi(Base):
    __tablename__ = "published_apis"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    kind: Mapped[str] = mapped_column(String(20), default="qa", index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    knowledge_base_ids: Mapped[list] = mapped_column(JSON, default=list)
    llm_profile_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_profiles.id"), nullable=True
    )
    settings: Mapped[dict] = mapped_column(JSON, default=dict)
    api_key_hash: Mapped[str] = mapped_column(String(64), index=True)
    api_key_prefix: Mapped[str] = mapped_column(String(24), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
