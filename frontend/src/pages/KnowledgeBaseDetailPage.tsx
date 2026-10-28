import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Bot, FileText, Pencil, RefreshCw, Search, Trash2, UploadCloud } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader, StatusBadge, formatBytes, formatDate } from "../components/UI";
import type { KnowledgeBase } from "../types";

export default function KnowledgeBaseDetailPage() {
  const { kbId = "" } = useParams();
  const queryClient = useQueryClient();
  const [editOpen, setEditOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [rerankId, setRerankId] = useState("");
  const [llmId, setLlmId] = useState("");
  const [candidateTopK, setCandidateTopK] = useState("20");
  const [finalTopK, setFinalTopK] = useState("5");
  const [threshold, setThreshold] = useState("0.3");
  const [chunkMode, setChunkMode] = useState("structure");
  const [chunkSize, setChunkSize] = useState("512");
  const [chunkOverlap, setChunkOverlap] = useState("64");
  const [semanticThreshold, setSemanticThreshold] = useState("0.72");

  const kbQuery = useQuery({ queryKey: ["knowledge-base", kbId], queryFn: () => api.getKnowledgeBase(kbId), enabled: !!kbId });
  const docsQuery = useQuery({
    queryKey: ["documents", kbId],
    queryFn: () => api.listDocuments(kbId),
    enabled: !!kbId,
    refetchInterval: (query) => {
      const docs = query.state.data ?? [];
      return docs.some((doc) => doc.status === "pending" || doc.status === "processing") ? 2500 : false;
    },
  });
  const modelQuery = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });
  const capabilitiesQuery = useQuery({ queryKey: ["capabilities"], queryFn: api.getCapabilities });
  const models = modelQuery.data ?? [];
  const rerankModels = useMemo(() => models.filter((item) => item.kind === "rerank"), [models]);
  const llmModels = useMemo(() => models.filter((item) => item.kind === "llm"), [models]);

  const updateMutation = useMutation({
    mutationFn: () =>
      api.updateKnowledgeBase(kbId, {
        name: name.trim(),
        description,
        rerank_profile_id: rerankId || null,
        default_llm_profile_id: llmId || null,
        settings: {
          candidate_top_k: Number(candidateTopK),
          final_top_k: Number(finalTopK),
          rerank_threshold: Number(threshold),
          chunk_mode: chunkMode,
          chunk_size: Number(chunkSize),
          chunk_overlap: Number(chunkOverlap),
          semantic_threshold: Number(semanticThreshold),
        },
      }),
    onSuccess: async () => {
      setEditOpen(false);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["knowledge-base", kbId] }),
        queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
      ]);
    },
  });

  const deleteDocMutation = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["documents", kbId] }),
  });
  const reindexMutation = useMutation({
    mutationFn: api.reindexDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["documents", kbId] }),
  });

  function openEdit(kb: KnowledgeBase) {
    setName(kb.name);
    setDescription(kb.description);
    setRerankId(kb.rerank_profile_id ?? "");
    setLlmId(kb.default_llm_profile_id ?? "");
    setCandidateTopK(String(kb.settings.candidate_top_k ?? 20));
    setFinalTopK(String(kb.settings.final_top_k ?? 5));
    setThreshold(String(kb.settings.rerank_threshold ?? 0.3));
    setChunkMode(String(kb.settings.chunk_mode ?? "structure"));
    setChunkSize(String(kb.settings.chunk_size ?? 512));
    setChunkOverlap(String(kb.settings.chunk_overlap ?? 64));
    setSemanticThreshold(String(kb.settings.semantic_threshold ?? 0.72));
    setEditOpen(true);
  }

  function submitEdit(event: FormEvent) {
    event.preventDefault();
    updateMutation.mutate();
  }

  if (kbQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;
  const kb = kbQuery.data!;
  const documents = docsQuery.data ?? [];
  const actionError = deleteDocMutation.error ?? reindexMutation.error;

  return (
    <div>
      <Link to="/knowledge-bases" className="mb-4 inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800">
        <ArrowLeft className="h-4 w-4" />返回知识库
      </Link>
      <PageHeader
        title={kb.name}
        description={kb.description || "暂无描述"}
        action={
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => openEdit(kb)} className="btn-secondary"><Pencil className="h-4 w-4" />设置</button>
            <Link to={`/upload?kb=${kb.id}`} className="btn-primary"><UploadCloud className="h-4 w-4" />上传文档</Link>
            <Link to={`/search-quality?kb=${kb.id}`} className="btn-secondary"><Search className="h-4 w-4" />检索质检</Link>
            <Link to={`/chat?kb=${kb.id}`} className="btn-secondary"><Bot className="h-4 w-4" />开始问答</Link>
          </div>
        }
      />

      <div className="mb-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="文档数量" value={String(kb.document_count)} />
        <Metric label="Embedding 模型" value={kb.embedding_profile_name ?? "未配置"} />
        <Metric label="召回 / 上下文" value={`${String(kb.settings.candidate_top_k ?? 20)} / ${String(kb.settings.final_top_k ?? 5)}`} />
        <Metric label="重排阈值" value={String(kb.settings.rerank_threshold ?? 0.3)} />
        <Metric label="默认分块模式" value={String(kb.settings.chunk_mode ?? "structure")} />
      </div>

      {actionError && <div className="mb-4"><ErrorState message={actionError.message} /></div>}

      <div className="panel overflow-hidden">
        <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
          <div className="font-semibold text-slate-900">文档列表</div>
          <Link to={`/upload?kb=${kb.id}`} className="text-sm font-medium text-brand-700 hover:text-brand-800">上传新文档</Link>
        </div>
        {docsQuery.isLoading ? (
          <LoadingState />
        ) : !documents.length ? (
          <EmptyState
            title="知识库暂时没有文档"
            description="支持 PDF、DOCX、Markdown 与 EPUB，上传后会自动解析、分块和建立索引。"
            action={<Link to={`/upload?kb=${kb.id}`} className="btn-primary"><UploadCloud className="h-4 w-4" />上传文档</Link>}
          />
        ) : (
          <div className="divide-y divide-slate-100">
            {documents.map((doc) => (
              <div key={doc.id} className="flex flex-col gap-3 px-5 py-4 sm:flex-row sm:items-center">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-slate-100 text-slate-500">
                  <FileText className="h-5 w-5" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Link to={`/documents/${doc.id}`} className="truncate font-medium text-slate-900 hover:text-brand-700">{doc.filename}</Link>
                    <StatusBadge status={doc.status} />
                  </div>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                    <span>{formatBytes(doc.size_bytes)}</span>
                    <span>{doc.chunk_count} 个分块</span>
                    <span>{formatDate(doc.created_at)}</span>
                    {doc.tags.map((tag) => <span key={tag} className="chip bg-slate-100 text-slate-600">#{tag}</span>)}
                  </div>
                  {doc.error && <div className="mt-2 text-xs text-red-600">{doc.error}</div>}
                </div>
                <div className="flex shrink-0 gap-1">
                  <Link to={`/documents/${doc.id}`} className="btn-ghost px-3 py-2 text-xs">预览</Link>
                  <Link to={`/chat?kb=${kb.id}&document=${doc.id}`} className="btn-ghost px-3 py-2 text-xs"><Bot className="h-4 w-4" />问答</Link>
                  <button
                    type="button"
                    onClick={() => reindexMutation.mutate(doc.id)}
                    disabled={doc.status === "pending" || doc.status === "processing"}
                    className="btn-ghost px-3 py-2 text-xs"
                  >
                    <RefreshCw className="h-4 w-4" />重索引
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm(`确认删除“${doc.filename}”？`)) deleteDocMutation.mutate(doc.id);
                    }}
                    disabled={doc.status === "pending" || doc.status === "processing"}
                    className="btn-ghost px-3 py-2 text-xs text-red-600 hover:bg-red-50"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <Modal open={editOpen} onClose={() => setEditOpen(false)} title="知识库设置" wide>
        <form onSubmit={submitEdit} className="space-y-5">
          <div>
            <label className="label">知识库名称</label>
            <input
              className="input"
              value={name}
              onChange={(event) => setName(event.target.value)}
              minLength={1}
              maxLength={120}
              placeholder="请输入知识库名称"
              required
            />
            <div className="mt-1 text-right text-xs text-slate-400">{name.length} / 120</div>
          </div>
          <div>
            <label className="label">描述</label>
            <textarea className="input min-h-24" value={description} onChange={(event) => setDescription(event.target.value)} />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">重排序模型</label>
              <select className="input" value={rerankId} onChange={(event) => setRerankId(event.target.value)}>
                <option value="">不使用</option>
                {rerankModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
            <div>
              <label className="label">默认问答模型</label>
              <select className="input" value={llmId} onChange={(event) => setLlmId(event.target.value)}>
                <option value="">请选择</option>
                {llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">默认分块模式</label>
              <select className="input" value={chunkMode} onChange={(event) => setChunkMode(event.target.value)}>
                {(capabilitiesQuery.data?.chunk_modes ?? []).map((mode) => <option key={mode.value} value={mode.value}>{mode.label}</option>)}
              </select>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="label">分块大小</label><input className="input" type="number" min={32} max={8192} value={chunkSize} onChange={(event) => setChunkSize(event.target.value)} /></div>
              <div><label className="label">重叠</label><input className="input" type="number" min={0} value={chunkOverlap} onChange={(event) => setChunkOverlap(event.target.value)} /></div>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-4">
            <div><label className="label">召回数量</label><input className="input" type="number" value={candidateTopK} onChange={(event) => setCandidateTopK(event.target.value)} /></div>
            <div><label className="label">最终上下文数</label><input className="input" type="number" value={finalTopK} onChange={(event) => setFinalTopK(event.target.value)} /></div>
            <div><label className="label">重排阈值</label><input className="input" type="number" step="0.01" value={threshold} onChange={(event) => setThreshold(event.target.value)} /></div>
            <div><label className="label">语义切分阈值</label><input className="input" type="number" step="0.01" min={0} max={1} value={semanticThreshold} onChange={(event) => setSemanticThreshold(event.target.value)} /></div>
          </div>
          {updateMutation.error && <ErrorState message={updateMutation.error.message} />}
          <div className="rounded-xl bg-amber-50 p-3 text-xs leading-5 text-amber-700">
            Embedding 模型决定向量维度，创建后不可在此处更换。如需更换，请新建知识库并重新入库。
          </div>
          <div className="flex justify-end gap-3">
            <button type="button" className="btn-secondary" onClick={() => setEditOpen(false)}>取消</button>
            <button type="submit" className="btn-primary" disabled={updateMutation.isPending || !name.trim()}>{updateMutation.isPending ? "保存中..." : "保存设置"}</button>
          </div>
        </form>
      </Modal>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel p-4">
      <div className="text-xs text-slate-400">{label}</div>
      <div className="mt-2 truncate text-base font-semibold text-slate-800">{value}</div>
    </div>
  );
}

