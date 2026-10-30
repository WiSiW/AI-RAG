from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ModelKindValue = Literal["embedding", "rerank", "llm", "enrichment"]
ChunkModeValue = Literal[
    "structure",
    "recursive",
    "fixed",
    "paragraph",
    "sentence",
    "semantic",
    "parent_child",
]
PublicationKindValue = Literal["qa", "search"]
EvaluationGranularityValue = Literal["L1", "L2", "L3"]
KnowledgeBaseExportMode = Literal["documents", "full"]
KnowledgeBaseIndexPolicy = Literal["strict", "rebuild"]
RetrievalModeValue = Literal["auto", "single", "multi"]


def _normalize_scope(
    knowledge_base_id: str | None,
    knowledge_base_ids: list[str],
    *,
    required: bool,
) -> list[str]:
    if knowledge_base_id and knowledge_base_ids:
        raise ValueError("knowledge_base_id 与 knowledge_base_ids 不能同时使用")
    ids = list(dict.fromkeys(knowledge_base_ids or ([knowledge_base_id] if knowledge_base_id else [])))
    if required and not ids:
        raise ValueError("至少选择一个知识库")
    return ids


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class ModelProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: ModelKindValue
    provider: str = Field(default="openai-compatible", max_length=50)
    base_url: str = Field(min_length=1, max_length=500)
    model_name: str = Field(min_length=1, max_length=200)
    api_key: str | None = None
    dimensions: int | None = Field(default=None, ge=1, le=65536)
    config: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class ModelProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    provider: str | None = Field(default=None, max_length=50)
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model_name: str | None = Field(default=None, min_length=1, max_length=200)
    api_key: str | None = None
    clear_api_key: bool = False
    dimensions: int | None = Field(default=None, ge=1, le=65536)
    config: dict[str, Any] | None = None
    is_active: bool | None = None


class ModelProfileRead(ORMModel):
    id: str
    name: str
    kind: str
    provider: str
    base_url: str
    model_name: str
    dimensions: int | None
    config: dict[str, Any]
    is_active: bool
    has_api_key: bool = False
    created_at: datetime
    updated_at: datetime


class ModelTestResult(BaseModel):
    ok: bool
    latency_ms: float
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    embedding_profile_id: str
    rerank_profile_id: str | None = None
    enrichment_profile_id: str
    default_llm_profile_id: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = None
    rerank_profile_id: str | None = None
    enrichment_profile_id: str | None = None
    default_llm_profile_id: str | None = None
    settings: dict[str, Any] | None = None


class KnowledgeBaseRead(ORMModel):
    id: str
    name: str
    description: str
    embedding_profile_id: str
    rerank_profile_id: str | None
    enrichment_profile_id: str | None
    default_llm_profile_id: str | None
    settings: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    document_count: int = 0
    embedding_profile_name: str | None = None


class DocumentRead(ORMModel):
    id: str
    kb_id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error: str | None
    tags: list[str]
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    chunk_mode: str = "structure"
    chunk_size: int
    chunk_overlap: int
    created_at: datetime
    updated_at: datetime
    chunk_count: int = 0


class DocumentDetail(DocumentRead):
    extracted_text: str


class QAPairRead(BaseModel):
    question: str
    answer: str
    question_type: Literal["detail", "comparison", "summary"]


class ChunkRead(ORMModel):
    id: str
    document_id: str
    kb_id: str
    chunk_index: int
    content: str
    section_path: str
    token_count: int
    metadata_json: dict[str, Any]
    granularity_level: str = "L2"
    granule_type: str = "paragraph"
    parent_chunk_id: str | None = None
    context_prefix: str = ""
    summary: str = ""
    keywords: list[str] = Field(default_factory=list)
    hypothetical_questions: list[str] = Field(default_factory=list)
    qa_pairs: list[QAPairRead] = Field(default_factory=list)
    enrichment_status: str = "pending"
    enrichment_meta: dict[str, Any] = Field(default_factory=dict)


class IngestionJobRead(ORMModel):
    id: str
    document_id: str
    status: str
    attempts: int
    error: str | None
    result: dict[str, Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ChunkEvaluationCreate(BaseModel):
    document_id: str | None = None
    knowledge_base_id: str | None = None
    granularities: list[EvaluationGranularityValue] = Field(
        default_factory=lambda: ["L1", "L2", "L3"]
    )
    judge_profile_id: str | None = None
    llm_enabled: bool = True

    @model_validator(mode="after")
    def validate_scope(self) -> "ChunkEvaluationCreate":
        if bool(self.document_id) == bool(self.knowledge_base_id):
            raise ValueError("document_id 与 knowledge_base_id 必须且只能提供一个")
        self.granularities = list(dict.fromkeys(self.granularities)) or ["L1", "L2", "L3"]
        return self


class ChunkEvaluationRunRead(ORMModel):
    id: str
    scope_type: str
    knowledge_base_id: str | None
    document_id: str | None
    granularities: list[str]
    judge_profile_id: str | None
    llm_enabled: bool
    status: str
    progress: dict[str, Any]
    config_snapshot: dict[str, Any]
    aggregate: dict[str, Any]
    errors: list[Any]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class ChunkEvaluationResultRead(ORMModel):
    id: str
    run_id: str
    chunk_id: str | None
    document_id: str | None
    kb_id: str | None
    granularity_level: str
    granule_type: str
    chunk_index: int
    content_snapshot: str
    section_path: str
    context_prefix: str
    parent_chunk_id: str | None
    metadata_snapshot: dict[str, Any]
    context_snapshot: dict[str, Any]
    rule_metrics: dict[str, Any]
    llm_scores: dict[str, Any]
    llm_reason: str
    issue_codes: list[str]
    suggestions: list[str]
    composite_score: float
    grade: str
    review_sample: bool
    human_score: int | None
    human_issue_codes: list[str]
    human_comment: str
    reviewed_at: datetime | None
    created_at: datetime


class ChunkEvaluationResultsPage(BaseModel):
    items: list[ChunkEvaluationResultRead]
    total: int
    page: int
    page_size: int


class ChunkEvaluationReviewUpdate(BaseModel):
    human_score: int = Field(ge=1, le=5)
    human_issue_codes: list[str] = Field(default_factory=list)
    human_comment: str = Field(default="", max_length=4000)


class KnowledgeBaseExportRequest(BaseModel):
    mode: KnowledgeBaseExportMode = "documents"


class KnowledgeBaseTransferJobRead(ORMModel):
    id: str
    kind: str
    mode: str
    index_policy: str
    status: str
    source_knowledge_base_id: str | None
    result_knowledge_base_id: str | None
    package_name: str
    object_key: str | None
    file_size: int
    progress: dict[str, Any]
    manifest: dict[str, Any]
    errors: list[Any]
    created_by: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class KnowledgeBasePackageInspection(BaseModel):
    valid: bool
    package_name: str
    manifest: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
    document: DocumentRead
    job: IngestionJobRead


class BatchUploadItem(BaseModel):
    filename: str
    ok: bool
    document: DocumentRead | None = None
    job: IngestionJobRead | None = None
    error: str | None = None


class BatchUploadResponse(BaseModel):
    items: list[BatchUploadItem]
    accepted: int
    failed: int


class ServerArchiveRequest(BaseModel):
    path: str = Field(min_length=1, max_length=2000)
    chunk_mode: ChunkModeValue | None = None
    chunk_size: int | None = Field(default=None, ge=32, le=8192)
    chunk_overlap: int | None = Field(default=None, ge=0, le=8191)
    tags: str = ""


class Citation(BaseModel):
    index: int
    document_id: str = ""
    chunk_id: str
    knowledge_base_id: str = ""
    knowledge_base_name: str = ""
    source: str
    section: str
    score: float = 0
    snippet: str
    file_url: str | None = None
    preview_url: str | None = None


class SearchRequest(BaseModel):
    knowledge_base_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=20)
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = True
    rerank_profile_id: str | None = None
    rewrite: bool = False
    llm_profile_id: str | None = None
    conversation_id: str | None = None
    document_id: str | None = None
    retrieval_mode: RetrievalModeValue = "auto"

    @model_validator(mode="after")
    def normalize_scope(self) -> "SearchRequest":
        self.knowledge_base_ids = _normalize_scope(
            self.knowledge_base_id, self.knowledge_base_ids, required=True
        )
        self.knowledge_base_id = (
            self.knowledge_base_ids[0] if len(self.knowledge_base_ids) == 1 else None
        )
        return self


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    knowledge_base_id: str = ""
    knowledge_base_name: str = ""
    source: str
    section: str
    content: str
    rrf_score: float = 0
    rerank_score: float | None = None
    relevance_score: float = 0
    relevance_source: str = "rrf"
    vector_score: float | None = None
    bm25_score: float | None = None
    vector_rank: int | None = None
    bm25_rank: int | None = None
    matched_representations: list[str] = Field(default_factory=list)
    representation_scores: dict[str, float] = Field(default_factory=dict)
    representation_ranks: dict[str, int] = Field(default_factory=dict)
    granularity_level: str = "L2"
    granule_type: str = "paragraph"
    parent_chunk_id: str | None = None
    context_prefix: str = ""
    context_text: str = ""
    matched_questions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    file_url: str | None = None
    preview_url: str | None = None


class ComplexityDecision(BaseModel):
    is_multi_hop: bool
    decision_source: str
    rule_score: float = 0
    bert_probability: float | None = None
    llm_decision: bool | None = None
    reason: str = ""
    classifier_model_version: str = ""


class TraceChunkSnapshot(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    section: str
    granularity_level: str = "L2"
    granule_type: str = "paragraph"
    rrf_score: float = 0
    vector_score: float | None = None
    bm25_score: float | None = None
    matched_representations: list[str] = Field(default_factory=list)
    content_snippet: str = ""


class MultiHopStep(BaseModel):
    round: int
    query: str
    chunk_ids: list[str] = Field(default_factory=list)
    chunks: list[TraceChunkSnapshot] = Field(default_factory=list)
    route_counts: dict[str, int] = Field(default_factory=dict)
    granularity_counts: dict[str, int] = Field(default_factory=dict)
    new_chunk_count: int = 0
    key_entities: list[str] = Field(default_factory=list)
    need_more: bool = False
    confidence: float | None = None
    reason: str = ""


class RetrievalTrace(BaseModel):
    requested_mode: str = "auto"
    complexity: ComplexityDecision | None = None
    steps: list[MultiHopStep] = Field(default_factory=list)
    stop_reason: str = ""
    uncertain: bool = False
    uncertainty_notice: str = ""
    total_rounds: int = 0
    total_chunks: int = 0


class SearchQualitySummary(BaseModel):
    result_count: int = 0
    avg_rrf_score: float = 0
    best_score: float = 0
    reranked_count: int = 0
    vector_hits: int = 0
    bm25_hits: int = 0
    route_hits: dict[str, int] = Field(default_factory=dict)
    queried_knowledge_base_ids: list[str] = Field(default_factory=list)
    failed_knowledge_base_ids: list[str] = Field(default_factory=list)
    hits_by_knowledge_base: dict[str, int] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    rewritten_query: str | None = None
    results: list[SearchResult]
    quality: SearchQualitySummary = Field(default_factory=SearchQualitySummary)
    retrieval_trace: RetrievalTrace | None = None


class ChatRequest(BaseModel):
    knowledge_base_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=20)
    query: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None
    llm_profile_id: str | None = None
    rerank_profile_id: str | None = None
    document_id: str | None = None
    stream: bool = True
    retrieval_mode: RetrievalModeValue = "auto"

    @model_validator(mode="after")
    def normalize_scope(self) -> "ChatRequest":
        self.knowledge_base_ids = _normalize_scope(
            self.knowledge_base_id, self.knowledge_base_ids, required=True
        )
        self.knowledge_base_id = (
            self.knowledge_base_ids[0] if len(self.knowledge_base_ids) == 1 else None
        )
        return self


class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    knowledge_base_ids: list[str] = Field(default_factory=list)
    answer: str
    citations: list[Citation]
    rewritten_query: str
    usage: dict[str, Any] = Field(default_factory=dict)
    retrieval_trace: RetrievalTrace | None = None
    uncertainty_notice: str = ""


class MessageRead(ORMModel):
    id: str
    conversation_id: str
    role: str
    content: str
    rewritten_query: str
    citations: list[Citation]
    context: list[dict[str, Any]]
    usage: dict[str, Any]
    created_at: datetime


class ConversationRead(ORMModel):
    id: str
    knowledge_base_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list)
    knowledge_base_names: list[str] = Field(default_factory=list)
    document_id: str | None = None
    title: str
    llm_profile_id: str | None
    summary: str
    created_at: datetime
    updated_at: datetime
    message_count: int = 0


class ConversationDetail(ConversationRead):
    messages: list[MessageRead]


class FeedbackCreate(BaseModel):
    message_id: str
    rating: int
    comment: str = Field(default="", max_length=2000)

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, value: int) -> int:
        if value not in (-1, 1):
            raise ValueError("rating 只能为 1 或 -1")
        return value


class FeedbackRead(ORMModel):
    id: str
    message_id: str
    rating: int
    comment: str
    created_at: datetime


class PublishedApiCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: PublicationKindValue = "qa"
    description: str = ""
    knowledge_base_ids: list[str] = Field(min_length=1)
    llm_profile_id: str | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class PublishedApiUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    slug: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"^[a-z0-9][a-z0-9-]*$")
    kind: PublicationKindValue | None = None
    description: str | None = None
    knowledge_base_ids: list[str] | None = None
    llm_profile_id: str | None = None
    settings: dict[str, Any] | None = None
    is_active: bool | None = None
    rotate_api_key: bool = False


class PublishedApiRead(ORMModel):
    id: str
    name: str
    slug: str
    kind: str
    description: str
    knowledge_base_ids: list[str]
    llm_profile_id: str | None
    settings: dict[str, Any]
    api_key_prefix: str
    is_active: bool
    last_used_at: datetime | None
    created_at: datetime
    updated_at: datetime
    endpoint: str | None = None


class PublishedApiCreated(PublishedApiRead):
    api_key: str


class PublicChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    knowledge_base_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=20)
    conversation_id: str | None = None
    document_id: str | None = None
    stream: bool = True

    @model_validator(mode="after")
    def normalize_scope(self) -> "PublicChatRequest":
        self.knowledge_base_ids = _normalize_scope(
            self.knowledge_base_id, self.knowledge_base_ids, required=False
        )
        self.knowledge_base_id = (
            self.knowledge_base_ids[0] if len(self.knowledge_base_ids) == 1 else None
        )
        return self


class PublicSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    knowledge_base_id: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list, max_length=20)
    document_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    rerank: bool | None = None

    @model_validator(mode="after")
    def normalize_scope(self) -> "PublicSearchRequest":
        self.knowledge_base_ids = _normalize_scope(
            self.knowledge_base_id, self.knowledge_base_ids, required=False
        )
        self.knowledge_base_id = (
            self.knowledge_base_ids[0] if len(self.knowledge_base_ids) == 1 else None
        )
        return self


class PublicCapabilities(BaseModel):
    publication: str
    kind: str
    knowledge_bases: list[dict[str, Any]]
    models: list[dict[str, Any]] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)


class CapabilitiesResponse(BaseModel):
    upload_max_mb: int
    archive_max_mb: int
    chunk_modes: list[dict[str, str]]
    supported_extensions: list[str]


class HealthComponent(BaseModel):
    status: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    components: dict[str, HealthComponent] = Field(default_factory=dict)

