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
    PARTIAL = "partial"
    COMPLETED = "completed"
    FAILED = "failed"


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
    summary_message_count: Mapped[int] = mapped_column(Integer, default=0)
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


