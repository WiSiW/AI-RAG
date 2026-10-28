export type ModelKind = "embedding" | "rerank" | "llm";
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
  default_llm_profile_id: string | null;
  settings: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  document_count: number;
  embedding_profile_name: string | null;
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
}

export interface DocumentDetail extends DocumentItem {
  extracted_text: string;
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

export interface ChatResponse {
  conversation_id: string;
  message_id: string;
  knowledge_base_ids: string[];
  answer: string;
  citations: Citation[];
  rewritten_query: string;
  usage: Record<string, unknown>;
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
  queried_knowledge_base_ids: string[];
  failed_knowledge_base_ids: string[];
  hits_by_knowledge_base: Record<string, number>;
}

export interface SearchResponse {
  query: string;
  rewritten_query: string | null;
  results: SearchResult[];
  quality: SearchQuality;
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
}

