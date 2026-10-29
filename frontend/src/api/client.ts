import type {
  BatchUploadResponse,
  Capabilities,
  ChatResponse,
  Chunk,
  ChunkEvaluationResultsPage,
  ChunkEvaluationRun,
  Conversation,
  ConversationDetail,
  DocumentContent,
  DocumentDetail,
  DocumentItem,
  IngestionJob,
  KnowledgeBase,
  KnowledgeBasePackageInspection,
  KnowledgeBaseTransferJob,
  ModelProfile,
  ModelTestResult,
  PublishedApi,
  PublishedApiCreated,
  SearchResponse,
  TokenResponse,
  UploadResponse,
} from "../types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";
const TOKEN_KEY = "zhiyuan_access_token";

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(message: string, status: number, code = "api_error") {
    super(message);
    this.status = status;
    this.code = code;
  }
}

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? "";
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    let code = "api_error";
    try {
      const payload = await response.json();
      message = payload?.error?.message ?? payload?.detail ?? message;
      code = payload?.error?.code ?? code;
    } catch {
      // Keep the HTTP fallback message.
    }
    if (response.status === 401) clearToken();
    throw new ApiError(message, response.status, code);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export async function login(username: string, password: string): Promise<TokenResponse> {
  const result = await request<TokenResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
  setToken(result.access_token);
  return result;
}

export const api = {
  me: () => request<{ username: string }>("/auth/me"),
  listModels: (kind?: string) =>
    request<ModelProfile[]>(`/model-profiles${kind ? `?kind=${kind}` : ""}`),
  createModel: (payload: Record<string, unknown>) =>
    request<ModelProfile>("/model-profiles", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateModel: (id: string, payload: Record<string, unknown>) =>
    request<ModelProfile>(`/model-profiles/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteModel: (id: string) =>
    request<void>(`/model-profiles/${id}`, { method: "DELETE" }),
  testModel: (id: string) =>
    request<ModelTestResult>(`/model-profiles/${id}/test`, { method: "POST" }),

  listKnowledgeBases: () => request<KnowledgeBase[]>("/knowledge-bases"),
  getKnowledgeBase: (id: string) => request<KnowledgeBase>(`/knowledge-bases/${id}`),
  createKnowledgeBase: (payload: Record<string, unknown>) =>
    request<KnowledgeBase>("/knowledge-bases", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updateKnowledgeBase: (id: string, payload: Record<string, unknown>) =>
    request<KnowledgeBase>(`/knowledge-bases/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  deleteKnowledgeBase: (id: string) =>
    request<void>(`/knowledge-bases/${id}`, { method: "DELETE" }),
  rebuildKnowledgeBaseIndex: (id: string) =>
    request<{ task_id: string; status: string }>(`/knowledge-bases/${id}/rebuild-index`, {
      method: "POST",
    }),
  exportKnowledgeBase: (id: string, mode: "documents" | "full") =>
    request<KnowledgeBaseTransferJob>(`/knowledge-bases/${id}/exports`, {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  getKnowledgeBaseTransfer: (id: string) =>
    request<KnowledgeBaseTransferJob>(`/knowledge-base-transfers/${id}`),
  deleteKnowledgeBaseTransfer: (id: string) =>
    request<void>(`/knowledge-base-transfers/${id}`, { method: "DELETE" }),
  inspectKnowledgeBasePackage: async (file: File) => {
    const form = new FormData();
    form.append("package", file);
    return request<KnowledgeBasePackageInspection>("/knowledge-base-imports/inspect", {
      method: "POST",
      body: form,
    });
  },
  importKnowledgeBase: async (file: File, payload: Record<string, unknown>) => {
    const form = new FormData();
    form.append("package", file);
    Object.entries(payload).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") form.append(key, String(value));
    });
    return request<KnowledgeBaseTransferJob>("/knowledge-base-imports", {
      method: "POST",
      body: form,
    });
  },

  getCapabilities: () => request<Capabilities>("/system/capabilities"),

  listDocuments: (kbId: string) =>
    request<DocumentItem[]>(`/knowledge-bases/${kbId}/documents`),
  getDocument: (id: string) => request<DocumentDetail>(`/documents/${id}`),
  getDocumentContent: (id: string) =>
    request<DocumentContent>(`/documents/${id}/content`),
  listChunks: (id: string) => request<Chunk[]>(`/documents/${id}/chunks`),
  getChunk: (documentId: string, chunkId: string) =>
    request<Chunk>(`/documents/${documentId}/chunks/${chunkId}`),
  deleteDocument: (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" }),
  reindexDocument: (id: string) =>
    request<IngestionJob>(`/documents/${id}/reindex`, { method: "POST" }),
  getJob: (id: string) => request<IngestionJob>(`/ingestion-jobs/${id}`),
  createChunkEvaluation: (payload: Record<string, unknown>) =>
    request<ChunkEvaluationRun>("/chunk-evaluations", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  listChunkEvaluations: (params: {
    documentId?: string;
    knowledgeBaseId?: string;
    status?: string;
  } = {}) => {
    const query = new URLSearchParams();
    if (params.documentId) query.set("document_id", params.documentId);
    if (params.knowledgeBaseId) query.set("knowledge_base_id", params.knowledgeBaseId);
    if (params.status) query.set("status", params.status);
    return request<ChunkEvaluationRun[]>(`/chunk-evaluations?${query.toString()}`);
  },
  getChunkEvaluation: (id: string) =>
    request<ChunkEvaluationRun>(`/chunk-evaluations/${id}`),
  listChunkEvaluationResults: (
    id: string,
    params: {
      granularity?: string;
      grade?: string;
      issue?: string;
      reviewSample?: boolean;
      page?: number;
      pageSize?: number;
    } = {},
  ) => {
    const query = new URLSearchParams();
    if (params.granularity) query.set("granularity", params.granularity);
    if (params.grade) query.set("grade", params.grade);
    if (params.issue) query.set("issue", params.issue);
    if (params.reviewSample !== undefined) query.set("review_sample", String(params.reviewSample));
    query.set("page", String(params.page ?? 1));
    query.set("page_size", String(params.pageSize ?? 20));
    return request<ChunkEvaluationResultsPage>(`/chunk-evaluations/${id}/results?${query.toString()}`);
  },
  reviewChunkEvaluationResult: (
    runId: string,
    resultId: string,
    payload: Record<string, unknown>,
  ) =>
    request<ChunkEvaluationRun>(`/chunk-evaluations/${runId}/results/${resultId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  compareChunkEvaluations: (runIds: string[]) =>
    request<Record<string, unknown>>(
      `/chunk-evaluations/compare?run_ids=${encodeURIComponent(runIds.join(","))}`,
    ),
  deleteChunkEvaluation: (id: string) =>
    request<void>(`/chunk-evaluations/${id}`, { method: "DELETE" }),
  uploadDocument: async (
    kbId: string,
    file: File,
    options: UploadOptions,
  ): Promise<UploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    appendUploadOptions(form, options);
    return request<UploadResponse>(`/knowledge-bases/${kbId}/documents`, {
      method: "POST",
      body: form,
    });
  },
  uploadDocuments: async (kbId: string, files: File[], options: UploadOptions) => {
    const form = buildUploadForm(files, options);
    return request<BatchUploadResponse>(`/knowledge-bases/${kbId}/documents/batch`, {
      method: "POST",
      body: form,
    });
  },
  uploadArchive: async (kbId: string, file: File, options: UploadOptions) => {
    const form = new FormData();
    form.append("file", file);
    appendUploadOptions(form, options);
    return request<BatchUploadResponse>(`/knowledge-bases/${kbId}/documents/archive`, {
      method: "POST",
      body: form,
    });
  },
  importServerArchive: (kbId: string, payload: Record<string, unknown>) =>
    request<BatchUploadResponse>(`/knowledge-bases/${kbId}/documents/server-archive`, {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  search: (payload: Record<string, unknown>) =>
    request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  listPublications: () => request<PublishedApi[]>("/publications"),
  createPublication: (payload: Record<string, unknown>) =>
    request<PublishedApiCreated>("/publications", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  updatePublication: (id: string, payload: Record<string, unknown>) =>
    request<PublishedApi>(`/publications/${id}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    }),
  rotatePublicationKey: (id: string) =>
    request<PublishedApiCreated>(`/publications/${id}/rotate-key`, { method: "POST" }),
  deletePublication: (id: string) =>
    request<void>(`/publications/${id}`, { method: "DELETE" }),

  listConversations: () => request<Conversation[]>("/conversations"),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${id}`),
  deleteConversation: (id: string) =>
    request<void>(`/conversations/${id}`, { method: "DELETE" }),

  sendFeedback: (messageId: string, rating: 1 | -1, comment = "") =>
    request("/feedback", {
      method: "POST",
      body: JSON.stringify({ message_id: messageId, rating, comment }),
    }),
};

export interface UploadOptions {
  chunkSize?: number;
  chunkOverlap?: number;
  chunkMode?: string;
  tags?: string;
}

function appendUploadOptions(form: FormData, options: UploadOptions): void {
  if (options.chunkSize) form.append("chunk_size", String(options.chunkSize));
  if (options.chunkOverlap !== undefined) {
    form.append("chunk_overlap", String(options.chunkOverlap));
  }
  if (options.chunkMode) form.append("chunk_mode", options.chunkMode);
  if (options.tags) form.append("tags", options.tags);
}

function buildUploadForm(files: File[], options: UploadOptions): FormData {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  appendUploadOptions(form, options);
  return form;
}

export interface ChatPayload {
  knowledge_base_ids: string[];
  query: string;
  conversation_id?: string;
  llm_profile_id?: string;
  rerank_profile_id?: string;
  document_id?: string;
  stream: true;
  retrieval_mode?: "auto" | "single" | "multi";
}

export interface StreamCallbacks {
  onEvent: (event: string, data: Record<string, unknown>) => void;
  onError: (error: Error) => void;
}

export async function streamChat(
  payload: ChatPayload,
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify(payload),
    signal,
  });
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try {
      const payload = await response.json();
      message = payload?.error?.message ?? payload?.detail ?? message;
    } catch {
      // Ignore malformed error bodies.
    }
    if (response.status === 401) clearToken();
    throw new ApiError(message, response.status);
  }
  if (!response.body) throw new ApiError("浏览器不支持流式响应", 0);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        parseEventBlock(block, callbacks);
        boundary = buffer.indexOf("\n\n");
      }
    }
  } catch (error) {
    if ((error as Error).name !== "AbortError") callbacks.onError(error as Error);
    throw error;
  }
}

function parseEventBlock(block: string, callbacks: StreamCallbacks): void {
  if (!block.trim()) return;
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  if (!dataLines.length) return;
  try {
    callbacks.onEvent(event, JSON.parse(dataLines.join("\n")) as Record<string, unknown>);
  } catch (error) {
    callbacks.onError(new Error(`无法解析流式事件：${(error as Error).message}`));
  }
}

export async function sendNonStreamingChat(payload: Omit<ChatPayload, "stream">): Promise<ChatResponse> {
  return request<ChatResponse>("/chat", {
    method: "POST",
    body: JSON.stringify({ ...payload, stream: false }),
  });
}

export async function downloadKnowledgeBasePackage(job: KnowledgeBaseTransferJob): Promise<void> {
  const response = await fetch(`${API_BASE}/knowledge-base-transfers/${job.id}/download`, {
    headers: { Authorization: `Bearer ${getToken()}` },
  });
  if (!response.ok) throw new ApiError(`下载失败 (${response.status})`, response.status);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = job.package_name || `knowledge-base-${job.id}.zip`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}




