export type ModelKind = "embedding" | "rerank" | "llm" | "enrichment";
export type ChunkMode =
  | "structure"
  | "recursive"
  | "fixed"
  | "paragraph"
  | "sentence"
  | "semantic"
  | "parent_child";
export type PublicationKind = "qa" | "search";

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface ModelProfile {
  id: string;
  name: string;
  kind: ModelKind;
  provider: string;
  base_url: string;
  model_name: string;
  dimensions: number | null;
  request_timeout_seconds: number;
  request_max_retries: number;
  config: Record<string, unknown>;
  is_active: boolean;
  has_api_key: boolean;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string;
  embedding_profile_id: string;
  rerank_profile_id: string | null;
  enrichment_profile_id: string | null;
  default_llm_profile_id: string | null;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  document_count: number;
  embedding_profile_name: string | null;
}

export interface DocumentReference {
  label: string;
  text: string;
  source_type: "bibliography" | "footnote" | "endnote" | string;
  page_start: number | null;
  page_end: number | null;
  section_path: string;
  metadata: Record<string, unknown>;
}

export interface ReferenceExtraction {
  status: "ready" | "degraded" | string;
  generated_by: string;
  prompt_version: string;
  reference_count: number;
  bibliography_count: number;
  footnote_count: number;
  endnote_count: number;
  inline_markers_removed: number;
  inline_marker_samples: string[];
  reference_like_tail_count: number;
  body_elements: number;
  body_characters: number;
  model_profile_id: string;
  model_name: string;
  warnings: string[];
}

export interface DocumentItem {
  id: string;
  kb_id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: "pending" | "processing" | "ready" | "failed";
  error: string | null;
  tags: string[];
  metadata_json: Record<string, unknown>;
  chunk_mode: ChunkMode;
  chunk_size: number;
  chunk_overlap: number;
  created_at: string;
  updated_at: string;
  chunk_count: number;
  reference_count: number;
}

export interface DocumentDetail extends DocumentItem {
  extracted_text: string;
  references: DocumentReference[];
  reference_extraction: ReferenceExtraction;
}

export interface Chunk {
  id: string;
  document_id: string;
  kb_id: string;
  chunk_index: number;
  content: string;
  section_path: string;
  token_count: number;
  metadata_json: Record<string, unknown>;
  granularity_level: "L1" | "L2" | "L3" | string;
  granule_type: string;
  parent_chunk_id: string | null;
  context_prefix: string;
  summary: string;
  keywords: string[];
  hypothetical_questions: string[];
  qa_pairs: QAPair[];
  enrichment_status: "pending" | "ready" | "degraded" | string;
  enrichment_meta: Record<string, unknown>;
}

export interface DocumentTable {
  id: string;
  document_id: string;
  kb_id: string;
  page_start: number;
  page_end: number;
  bbox: number[];
  layout_block_id: string;
  caption: string;
  cells: Array<Record<string, unknown>>;
  headers: unknown[];
  rows: unknown[];
  markdown: string;
  description: string;
  description_meta: Record<string, unknown>;
  extraction_engine: string;
  quality_score: number;
  quality_json: Record<string, unknown>;
  metadata_json: Record<string, unknown>;
  created_at: string;
}

export interface TableContext {
  id: string;
  caption: string;
  page_start: number;
  page_end: number;
  markdown: string;
  description: string;
  extraction_engine: string;
  quality_score: number;
  injected_mode: string;
}

export type ParsingEngineCategory = "ocr" | "table" | "layout";

export interface ParsingEngineAdapter {
  category: ParsingEngineCategory;
  adapter: string;
  label: string;
  description: string;
  config_schema: Record<string, unknown>;
}

export interface ParsingEngine {
  id: string;
  category: ParsingEngineCategory;
  adapter: string;
  display_name: string;
  priority: number;
  enabled: boolean;
  config: Record<string, unknown>;
  health: { available?: boolean; detail?: string };
  created_at: string;
  updated_at: string;
}

export interface ParsingEngineTestResult {
  ok: boolean;
  message: string;
  latency_ms: number;
  details: Record<string, unknown>;
}

export interface QAPair {
  question: string;
  answer: string;
  question_type: "detail" | "comparison" | "summary";
}

export interface IngestionJob {
  id: string;
  document_id: string;
  status: "pending" | "running" | "succeeded" | "failed";
  attempts: number;
  error: string | null;
  result: Record<string, unknown>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface UploadResponse {
  document: DocumentItem;
  job: IngestionJob;
}

export interface BatchUploadItem {
  filename: string;
  ok: boolean;
  document: DocumentItem | null;
  job: IngestionJob | null;
  error: string | null;
}

export interface ReparsePdfResponse {
  queued: number;
  skipped: number;
  job_ids: string[];
}

export interface BatchUploadResponse {
  items: BatchUploadItem[];
  accepted: number;
  failed: number;
}

export interface Citation {
  index: number;
  document_id: string;
  chunk_id: string;
  knowledge_base_id: string;
  knowledge_base_name: string;
  source: string;
  section: string;
  score: number;
  snippet: string;
  file_url: string | null;
  preview_url: string | null;
}

export interface Message {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  rewritten_query: string;
  citations: Citation[];
  context: Array<Record<string, unknown>>;
  usage: Record<string, unknown>;
  created_at: string;
}

export interface Conversation {
  id: string;
  knowledge_base_id: string | null;
  knowledge_base_ids: string[];
  knowledge_base_names: string[];
  document_id: string | null;
  title: string;
  llm_profile_id: string | null;
  summary: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export type ChatStage = "retrieving" | "generating";

export interface ChatStageEvent {
  stage: ChatStage;
  conversation_id: string;
  context_count: number;
}

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  knowledge_base_ids: string[];
  answer: string;
  citations: Citation[];
  rewritten_query: string;
  usage: Record<string, unknown>;
  retrieval_trace: RetrievalTrace | null;
  uncertainty_notice: string;
}

export interface ModelTestResult {
  ok: boolean;
  latency_ms: number;
  message: string;
  details: Record<string, unknown>;
}

export interface SearchResult {
  chunk_id: string;
  document_id: string;
  knowledge_base_id: string;
  knowledge_base_name: string;
  source: string;
  section: string;
  content: string;
  rrf_score: number;
  rerank_score: number | null;
  relevance_score?: number | null;
  relevance_source?: "rerank" | "rrf" | null;
  vector_score: number | null;
  bm25_score: number | null;
  vector_rank: number | null;
  bm25_rank: number | null;
  matched_representations: string[];
  representation_scores: Record<string, number>;
  representation_ranks: Record<string, number>;
  granularity_level: "L1" | "L2" | "L3" | string;
  granule_type: string;
  parent_chunk_id: string | null;
  context_prefix: string;
  context_text: string;
  matched_questions: string[];
  tables?: TableContext[];
  metadata: Record<string, unknown>;
  file_url: string | null;
  preview_url: string | null;
}

export interface SearchQuality {
  result_count: number;
  avg_rrf_score: number;
  best_score: number;
  reranked_count: number;
  vector_hits: number;
  bm25_hits: number;
  route_hits: Record<string, number>;
  queried_knowledge_base_ids: string[];
  failed_knowledge_base_ids: string[];
  hits_by_knowledge_base: Record<string, number>;
}

export interface SearchResponse {
  query: string;
  rewritten_query: string | null;
  results: SearchResult[];
  quality: SearchQuality;
  retrieval_trace: RetrievalTrace | null;
}

export interface ComplexityDecision {
  is_multi_hop: boolean;
  decision_source: string;
  rule_score: number;
  bert_probability: number | null;
  llm_decision: boolean | null;
  reason: string;
  classifier_model_version: string;
}

export interface TraceChunkSnapshot {
  chunk_id: string;
  document_id: string;
  source: string;
  section: string;
  granularity_level: string;
  granule_type: string;
  rrf_score: number;
  vector_score: number | null;
  bm25_score: number | null;
  matched_representations: string[];
  content_snippet: string;
}

export interface MultiHopStep {
  round: number;
  query: string;
  chunk_ids: string[];
  chunks?: TraceChunkSnapshot[];
  route_counts?: Record<string, number>;
  granularity_counts?: Record<string, number>;
  new_chunk_count: number;
  key_entities: string[];
  need_more: boolean;
  confidence: number | null;
  reason: string;
}

export interface RetrievalTrace {
  requested_mode: "auto" | "single" | "multi" | string;
  complexity: ComplexityDecision | null;
  steps: MultiHopStep[];
  stop_reason: string;
  uncertain: boolean;
  uncertainty_notice: string;
  total_rounds: number;
  total_chunks: number;
}

export type EvaluationStatus = "queued" | "running" | "pausing" | "paused" | "partial" | "completed" | "failed";

export interface ChunkEvaluationRun {
  id: string;
  scope_type: "document" | "knowledge_base";
  knowledge_base_id: string | null;
  document_id: string | null;
  granularities: string[];
  judge_profile_id: string | null;
  llm_enabled: boolean;
  status: EvaluationStatus;
  stage: string;
  celery_task_id: string | null;
  pause_requested_at: string | null;
  paused_at: string | null;
  heartbeat_at: string | null;
  batch_total: number;
  batch_completed: number;
  batch_running: number;
  chunks_total: number;
  chunks_succeeded: number;
  chunks_failed: number;
  chunks_pending: number;
  chunks_interrupted: number;
  chunks_not_selected: number;
  can_pause: boolean;
  can_resume: boolean;
  can_delete: boolean;
  progress: Record<string, unknown>;
  config_snapshot: Record<string, unknown>;
  aggregate: Record<string, unknown>;
  errors: Array<Record<string, unknown>>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ChunkEvaluationChunk {
  id: string | null;
  run_id: string;
  document_id: string;
  chunk_id: string;
  chunk_index: number;
  granularity_level: string;
  section_path: string;
  content_snapshot: string;
  status: string;
  selection_source: string;
  queue_position: number | null;
  attempt_count: number;
  rule_succeeded: boolean;
  llm_succeeded: boolean;
  fallback_applied: boolean;
  error: string | null;
  result_id: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface ChunkEvaluationChunksPage {
  items: ChunkEvaluationChunk[];
  total: number;
  page: number;
  page_size: number;
}

export interface ChunkEvaluationResult {
  id: string;
  run_id: string;
  chunk_id: string | null;
  document_id: string | null;
  kb_id: string | null;
  granularity_level: string;
  granule_type: string;
  chunk_index: number;
  content_snapshot: string;
  section_path: string;
  context_prefix: string;
  parent_chunk_id: string | null;
  metadata_snapshot: Record<string, unknown>;
  context_snapshot: Record<string, unknown>;
  rule_metrics: Record<string, unknown>;
  llm_scores: Record<string, unknown>;
  llm_reason: string;
  issue_codes: string[];
  suggestions: string[];
  composite_score: number;
  grade: "pass" | "warn" | "fail" | string;
  review_sample: boolean;
  human_score: number | null;
  human_issue_codes: string[];
  human_comment: string;
  reviewed_at: string | null;
  created_at: string;
}

export interface ChunkEvaluationResultsPage {
  items: ChunkEvaluationResult[];
  total: number;
  page: number;
  page_size: number;
}

export type EvaluationQuestionType =
  | "detail"
  | "comparison"
  | "summary"
  | "reasoning"
  | "table"
  | "code"
  | "no_answer";

export interface RetrievalEvaluationEvidence {
  id: string;
  case_id: string;
  document_id: string;
  quote: string;
  normalized_quote: string;
  char_start: number | null;
  char_end: number | null;
  section_path: string;
  page_start: number | null;
  page_end: number | null;
  created_at: string;
}

export interface RetrievalEvaluationCase {
  id: string;
  dataset_id: string;
  question: string;
  normalized_question: string;
  reference_answer: string;
  question_type: EvaluationQuestionType;
  answerable: boolean;
  sort_order: number;
  evidence: RetrievalEvaluationEvidence[];
  created_at: string;
}

export interface RetrievalEvaluationDataset {
  id: string;
  kb_id: string;
  name: string;
  description: string;
  document_ids: string[];
  version: number;
  cases: RetrievalEvaluationCase[];
  case_count: number;
  answerable_count: number;
  created_at: string;
  updated_at: string;
}

export type ChunkTuningStage = "screening" | "verification";
export type ChunkTuningStatus =
  | "queued"
  | "running"
  | "awaiting_selection"
  | "partial"
  | "completed"
  | "failed"
  | "cancelled";

export interface ChunkTuningMetrics {
  answerable_cases?: number;
  no_answer_cases?: number;
  recall_at_5?: number;
  recall_at_10?: number;
  all_evidence_hit_at_5?: number;
  all_evidence_hit_at_10?: number;
  mrr_at_10?: number;
  evidence_density?: number;
  latency_p50_ms?: number;
  latency_p95_ms?: number;
  recall_at_5_ci?: [number, number];
  recall_at_10_ci?: [number, number];
  baseline_delta?: {
    recall_at_5?: number;
    recall_at_10?: number;
    evidence_density?: number;
    latency_p95_ms?: number;
    recall_at_5_ci?: [number, number];
    recall_at_10_ci?: [number, number];
  };
  question_types?: Record<string, { case_count: number; recall_at_5: number; recall_at_10: number }>;
  quality?: {
    case_count?: number;
    faithfulness?: number;
    completeness?: number;
    cited_gold_chunk_ratio?: number;
  };
}

export interface ChunkTuningCost {
  candidate_chunk_count?: number;
  representation_count?: number;
  index_tokens?: number;
  representation_tokens?: number;
  estimated_vector_bytes?: number;
  build_time_ms?: number;
  degraded_rows?: number;
}

export interface ChunkTuningTrial {
  id: string;
  run_id: string;
  label: string;
  index_namespace: string;
  chunk_mode: string;
  chunk_size: number;
  chunk_overlap: number;
  overlap_ratio: number;
  is_baseline: boolean;
  status: ChunkTuningStatus;
  metrics: ChunkTuningMetrics;
  cost: ChunkTuningCost;
  errors: Array<Record<string, unknown>>;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ChunkTuningRun {
  id: string;
  parent_run_id: string | null;
  kb_id: string;
  dataset_id: string;
  dataset_version: number;
  stage: ChunkTuningStage;
  status: ChunkTuningStatus;
  config_snapshot: Record<string, unknown>;
  dataset_snapshot: Record<string, unknown>;
  progress: Record<string, unknown>;
  aggregate: {
    trial_count?: number;
    completed_trial_count?: number;
    partial_trial_count?: number;
    failed_trial_count?: number;
    pareto_trial_ids?: string[];
    recommended_trial_ids?: string[];
    baseline_trial_id?: string | null;
    answerable_case_count?: number;
  };
  errors: Array<Record<string, unknown>>;
  trials: ChunkTuningTrial[];
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface ChunkTuningCaseResult {
  id: string;
  trial_id: string;
  case_id: string;
  question: string;
  question_type: EvaluationQuestionType;
  answerable: boolean;
  reference_answer: string;
  recall_at_5: number;
  recall_at_10: number;
  all_evidence_hit_at_5: boolean;
  all_evidence_hit_at_10: boolean;
  reciprocal_rank: number;
  evidence_density: number;
  latency_ms: number;
  evidence_results: Array<Record<string, unknown>>;
  retrieved_hits: Array<Record<string, unknown>>;
  answer: string;
  quality_scores: Record<string, unknown>;
  errors: Array<Record<string, unknown>>;
  created_at: string;
}

export interface ChunkTuningCaseResultsPage {
  items: ChunkTuningCaseResult[];
  total: number;
  page: number;
  page_size: number;
}

export interface ChunkSettingsReindexResponse {
  queued: number;
  skipped: number;
  job_ids: string[];
}

export interface KnowledgeBaseTransferJob {
  id: string;
  kind: "export" | "import";
  mode: "documents" | "full";
  index_policy: "strict" | "rebuild";
  status: "queued" | "running" | "completed" | "failed";
  source_knowledge_base_id: string | null;
  result_knowledge_base_id: string | null;
  package_name: string;
  object_key: string | null;
  file_size: number;
  progress: Record<string, unknown>;
  manifest: Record<string, unknown>;
  errors: Array<Record<string, unknown>>;
  created_by: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

export interface KnowledgeBasePackageInspection {
  valid: boolean;
  package_name: string;
  manifest: Record<string, unknown>;
  warnings: string[];
}

export interface ChunkModeCapability {
  value: ChunkMode;
  label: string;
  description: string;
}

export interface Capabilities {
  upload_max_mb: number;
  archive_max_mb: number;
  chunk_modes: ChunkModeCapability[];
  supported_extensions: string[];
}

export interface PublishedApi {
  id: string;
  name: string;
  slug: string;
  kind: PublicationKind;
  description: string;
  knowledge_base_ids: string[];
  llm_profile_id: string | null;
  settings: Record<string, unknown>;
  api_key_prefix: string;
  is_active: boolean;
  last_used_at: string | null;
  created_at: string;
  updated_at: string;
  endpoint: string | null;
}

export interface PublishedApiCreated extends PublishedApi {
  api_key: string;
}

export interface DocumentContent {
  document_id: string;
  filename: string;
  content_type: string;
  text: string;
  status: string;
  metadata: Record<string, unknown>;
  references: DocumentReference[];
  reference_extraction: ReferenceExtraction;
}
