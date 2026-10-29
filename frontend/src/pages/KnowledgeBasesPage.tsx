import { FormEvent, useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Database, FileText, FileUp, Plus, Trash2, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";
import type { KnowledgeBasePackageInspection, KnowledgeBaseTransferJob } from "../types";

interface FormState {
  name: string;
  description: string;
  embeddingProfileId: string;
  rerankProfileId: string;
  enrichmentProfileId: string;
  llmProfileId: string;
  candidateTopK: string;
  finalTopK: string;
  threshold: string;
  chunkMode: string;
  chunkSize: string;
  chunkOverlap: string;
  semanticThreshold: string;
}

const initialForm: FormState = {
  name: "",
  description: "",
  embeddingProfileId: "",
  rerankProfileId: "",
  enrichmentProfileId: "",
  llmProfileId: "",
  candidateTopK: "20",
  finalTopK: "5",
  threshold: "0.3",
  chunkMode: "structure",
  chunkSize: "512",
  chunkOverlap: "64",
  semanticThreshold: "0.72",
};

export default function KnowledgeBasesPage() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);
  const [importFile, setImportFile] = useState<File | null>(null);
  const [inspection, setInspection] = useState<KnowledgeBasePackageInspection | null>(null);
  const [importName, setImportName] = useState("");
  const [importEmbeddingId, setImportEmbeddingId] = useState("");
  const [importEnrichmentId, setImportEnrichmentId] = useState("");
  const [importRerankId, setImportRerankId] = useState("");
  const [importLlmId, setImportLlmId] = useState("");
  const [indexPolicy, setIndexPolicy] = useState<"strict" | "rebuild">("strict");
  const [importJob, setImportJob] = useState<KnowledgeBaseTransferJob | null>(null);
  const [form, setForm] = useState<FormState>(initialForm);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const modelQuery = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });
  const capabilitiesQuery = useQuery({ queryKey: ["capabilities"], queryFn: api.getCapabilities });
  const models = modelQuery.data ?? [];
  const embeddingModels = useMemo(() => models.filter((item) => item.kind === "embedding" && item.is_active), [models]);
  const rerankModels = useMemo(() => models.filter((item) => item.kind === "rerank" && item.is_active), [models]);
  const enrichmentModels = useMemo(() => models.filter((item) => item.kind === "enrichment" && item.is_active), [models]);
  const llmModels = useMemo(() => models.filter((item) => item.kind === "llm" && item.is_active), [models]);

  const createMutation = useMutation({
    mutationFn: () =>
      api.createKnowledgeBase({
        name: form.name.trim(),
        description: form.description.trim(),
        embedding_profile_id: form.embeddingProfileId,
        rerank_profile_id: form.rerankProfileId || null,
        enrichment_profile_id: form.enrichmentProfileId,
        default_llm_profile_id: form.llmProfileId || null,
        settings: {
          candidate_top_k: Number(form.candidateTopK),
          final_top_k: Number(form.finalTopK),
          rerank_threshold: Number(form.threshold),
          chunk_mode: form.chunkMode,
          chunk_size: Number(form.chunkSize),
          chunk_overlap: Number(form.chunkOverlap),
          semantic_threshold: Number(form.semanticThreshold),
        },
      }),
    onSuccess: async () => {
      setOpen(false);
      setForm(initialForm);
      await queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteKnowledgeBase,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] }),
  });

  const inspectMutation = useMutation({
    mutationFn: (file: File) => api.inspectKnowledgeBasePackage(file),
    onSuccess: (result) => {
      setInspection(result);
      const kbMetadata = (result.manifest.knowledge_base as Record<string, unknown> | undefined) ?? {};
      setImportName(String(kbMetadata.name ?? ""));
      setIndexPolicy("strict");
    },
  });

  const importMutation = useMutation({
    mutationFn: () => api.importKnowledgeBase(importFile!, {
      embedding_profile_id: importEmbeddingId,
      enrichment_profile_id: importEnrichmentId,
      rerank_profile_id: importRerankId || undefined,
      llm_profile_id: importLlmId || undefined,
      name: importName.trim(),
      index_policy: inspection?.manifest.mode === "documents" ? "rebuild" : indexPolicy,
    }),
    onSuccess: setImportJob,
  });

  const importJobQuery = useQuery({
    queryKey: ["kb-transfer", importJob?.id],
    queryFn: () => api.getKnowledgeBaseTransfer(importJob!.id),
    enabled: !!importJob?.id && (importJob?.status === "queued" || importJob?.status === "running"),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running" ? 2000 : false;
    },
  });
  const activeImportJob = importJobQuery.data ?? importJob;

  useEffect(() => {
    if (activeImportJob?.status === "completed") {
      queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
    }
  }, [activeImportJob?.status, queryClient]);

  function openCreate() {
    setForm({
      ...initialForm,
      embeddingProfileId: embeddingModels[0]?.id ?? "",
      rerankProfileId: rerankModels[0]?.id ?? "",
      enrichmentProfileId: enrichmentModels[0]?.id ?? "",
      llmProfileId: llmModels[0]?.id ?? "",
    });
    setOpen(true);
  }

  function openImport() {
    setImportOpen(true);
    setImportFile(null);
    setInspection(null);
    setImportJob(null);
    setImportName("");
    setImportEmbeddingId(embeddingModels[0]?.id ?? "");
    setImportEnrichmentId(enrichmentModels[0]?.id ?? "");
    setImportRerankId(rerankModels[0]?.id ?? "");
    setImportLlmId(llmModels[0]?.id ?? "");
    setIndexPolicy("strict");
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!form.embeddingProfileId || !form.enrichmentProfileId) return;
    createMutation.mutate();
  }

  if (kbQuery.isLoading || modelQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;

  const knowledgeBases = kbQuery.data ?? [];
  const mutationError = createMutation.error ?? deleteMutation.error;

  return (
    <div>
      <PageHeader
        title="知识库"
        description="知识库绑定固定的 Embedding、增强与默认问答模型，并统一管理文档和检索参数。"
        action={<div className="flex gap-2"><button type="button" onClick={openImport} className="btn-secondary"><FileUp className="h-4 w-4" />导入知识库</button><button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />新建知识库</button></div>}
      />

      {mutationError && <div className="mb-4"><ErrorState message={mutationError.message} /></div>}

      {!knowledgeBases.length ? (
        <div className="panel">
          <EmptyState
            title="还没有知识库"
            description="先配置模型，再创建知识库并上传文档。"
            action={<button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />新建知识库</button>}
          />
        </div>
      ) : (
        <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {knowledgeBases.map((kb) => (
            <div key={kb.id} className="panel flex flex-col p-5 transition hover:-translate-y-0.5 hover:shadow-xl">
              <div className="flex items-start justify-between gap-3">
                <div className="grid h-11 w-11 place-items-center rounded-2xl bg-brand-50 text-brand-700">
                  <Database className="h-5 w-5" />
                </div>
                <button
                  type="button"
                  onClick={() => {
                    if (window.confirm(`确认删除知识库“${kb.name}”及其全部文档和历史对话？`)) {
                      deleteMutation.mutate(kb.id);
                    }
                  }}
                  className="grid h-9 w-9 place-items-center rounded-full text-slate-400 hover:bg-red-50 hover:text-red-600"
                  aria-label="删除知识库"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
              <Link to={`/knowledge-bases/${kb.id}`} className="mt-4 text-lg font-semibold text-slate-900 hover:text-brand-700">
                {kb.name}
              </Link>
              <p className="mt-2 line-clamp-3 min-h-[4.5rem] text-sm leading-6 text-slate-500">
                {kb.description || "暂无描述"}
              </p>
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-xs text-slate-400">文档数量</div>
                  <div className="mt-1 font-semibold text-slate-800">{kb.document_count}</div>
                </div>
                <div className="rounded-xl bg-slate-50 p-3">
                  <div className="text-xs text-slate-400">Embedding</div>
                  <div className="mt-1 truncate font-medium text-slate-700">{kb.embedding_profile_name ?? "未配置"}</div>
                </div>
              </div>
              <div className="mt-5 flex gap-2 border-t border-slate-100 pt-4">
                <Link to={`/upload?kb=${kb.id}`} className="btn-secondary flex-1 px-3 text-xs"><UploadCloud className="h-4 w-4" />上传</Link>
                <Link to={`/chat?kb=${kb.id}`} className="btn-secondary flex-1 px-3 text-xs"><Bot className="h-4 w-4" />问答</Link>
                <Link to={`/knowledge-bases/${kb.id}`} className="btn-secondary flex-1 px-3 text-xs"><FileText className="h-4 w-4" />文档</Link>
              </div>
            </div>
          ))}
        </div>
      )}

      <Modal open={open} onClose={() => setOpen(false)} title="新建知识库" wide>
        <form onSubmit={submit} className="space-y-5">
          {!embeddingModels.length && (
            <ErrorState message="尚未配置可用的 Embedding 模型，请先前往“模型配置”创建。" />
          )}
          {!enrichmentModels.length && (
            <ErrorState message="尚未配置可用的增强模型，请先前往“模型配置”创建。" />
          )}
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">名称</label>
              <input className="input" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required />
            </div>
            <div>
              <label className="label">Embedding 模型</label>
              <select className="input" value={form.embeddingProfileId} onChange={(event) => setForm({ ...form, embeddingProfileId: event.target.value })} required>
                <option value="">请选择</option>
                {embeddingModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.dimensions} 维</option>)}
              </select>
            </div>
          </div>
          <div>
            <label className="label">描述</label>
            <textarea className="input min-h-24 resize-y" value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} />
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label className="label">重排序模型</label>
              <select className="input" value={form.rerankProfileId} onChange={(event) => setForm({ ...form, rerankProfileId: event.target.value })}>
                <option value="">不使用</option>
                {rerankModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
            <div>
              <label className="label">默认问答模型</label>
              <select className="input" value={form.llmProfileId} onChange={(event) => setForm({ ...form, llmProfileId: event.target.value })}>
                <option value="">请选择</option>
                {llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
            <div>
              <label className="label">分块增强模型</label>
              <select className="input" value={form.enrichmentProfileId} onChange={(event) => setForm({ ...form, enrichmentProfileId: event.target.value })} required>
                <option value="">请选择</option>
                {enrichmentModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">默认分块模式</label>
              <select className="input" value={form.chunkMode} onChange={(event) => setForm({ ...form, chunkMode: event.target.value })}>
                {(capabilitiesQuery.data?.chunk_modes ?? [{ value: "structure", label: "结构感知", description: "" }]).map((mode) => (
                  <option key={mode.value} value={mode.value}>{mode.label}</option>
                ))}
              </select>
              <p className="mt-1 text-xs text-slate-400">{capabilitiesQuery.data?.chunk_modes.find((mode) => mode.value === form.chunkMode)?.description}</p>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div><label className="label">分块大小</label><input className="input" type="number" min={32} max={8192} value={form.chunkSize} onChange={(event) => setForm({ ...form, chunkSize: event.target.value })} /></div>
              <div><label className="label">重叠</label><input className="input" type="number" min={0} value={form.chunkOverlap} onChange={(event) => setForm({ ...form, chunkOverlap: event.target.value })} /></div>
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <div>
              <label className="label">召回数量</label>
              <input className="input" type="number" min={1} max={100} value={form.candidateTopK} onChange={(event) => setForm({ ...form, candidateTopK: event.target.value })} />
            </div>
            <div>
              <label className="label">最终上下文数</label>
              <input className="input" type="number" min={1} max={20} value={form.finalTopK} onChange={(event) => setForm({ ...form, finalTopK: event.target.value })} />
            </div>
            <div>
              <label className="label">重排阈值</label>
              <input className="input" type="number" step="0.01" min={0} max={1} value={form.threshold} onChange={(event) => setForm({ ...form, threshold: event.target.value })} />
            </div>
          </div>
          <div className="flex justify-end gap-3">
            <button type="button" onClick={() => setOpen(false)} className="btn-secondary">取消</button>
            <button type="submit" disabled={createMutation.isPending || !embeddingModels.length || !enrichmentModels.length} className="btn-primary">
              {createMutation.isPending ? "创建中..." : "创建知识库"}
            </button>
          </div>
        </form>
      </Modal>

      <Modal open={importOpen} onClose={() => setImportOpen(false)} title="导入知识库" wide>
        <div className="space-y-4">
          {!activeImportJob || activeImportJob.status === "failed" ? (
            <>
              <div><label className="label">知识库包 ZIP</label><input className="input" type="file" accept=".zip,application/zip" onChange={(event) => { const file = event.target.files?.[0] ?? null; setImportFile(file); setInspection(null); setImportJob(null); }} /></div>
              {activeImportJob?.status === "failed" && <div className="rounded-xl bg-red-50 p-3 text-xs text-red-700">{activeImportJob.errors.map((error) => String(error.error ?? JSON.stringify(error))).join("；")}</div>}
              {importFile && <button type="button" className="btn-secondary" disabled={inspectMutation.isPending} onClick={() => inspectMutation.mutate(importFile)}>{inspectMutation.isPending ? "预检中..." : "预检知识库包"}</button>}
              {inspectMutation.error && <ErrorState message={inspectMutation.error.message} />}
              {importJobQuery.error && <ErrorState message={importJobQuery.error.message} />}
              {inspection && (
                <div className="space-y-4">
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-800">包校验通过 · {inspection.package_name} · {String(inspection.manifest.mode === "full" ? "完整包" : "文档包")}</div>
                  {inspection.warnings.length > 0 && <div className="rounded-xl bg-amber-50 p-3 text-xs text-amber-700">{inspection.warnings.join("；")}</div>}
                  <div><label className="label">新知识库名称</label><input className="input" value={importName} onChange={(event) => setImportName(event.target.value)} placeholder="留空则使用包内名称" /></div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div><label className="label">Embedding 模型</label><select className="input" value={importEmbeddingId} onChange={(event) => setImportEmbeddingId(event.target.value)}><option value="">请选择</option>{embeddingModels.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.dimensions} 维</option>)}</select></div>
                    <div><label className="label">增强模型</label><select className="input" value={importEnrichmentId} onChange={(event) => setImportEnrichmentId(event.target.value)}><option value="">请选择</option>{enrichmentModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></div>
                    <div><label className="label">重排序模型</label><select className="input" value={importRerankId} onChange={(event) => setImportRerankId(event.target.value)}><option value="">不使用</option>{rerankModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></div>
                    <div><label className="label">默认问答模型</label><select className="input" value={importLlmId} onChange={(event) => setImportLlmId(event.target.value)}><option value="">请选择</option>{llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}</select></div>
                  </div>
                  {inspection.manifest.mode === "full" ? (
                    <div><label className="label">索引恢复策略</label><div className="flex gap-2"><button type="button" onClick={() => setIndexPolicy("strict")} className={`chip ${indexPolicy === "strict" ? "bg-brand-100 text-brand-800" : "bg-slate-100"}`}>严格恢复</button><button type="button" onClick={() => setIndexPolicy("rebuild")} className={`chip ${indexPolicy === "rebuild" ? "bg-brand-100 text-brand-800" : "bg-slate-100"}`}>重新嵌入</button></div><p className="mt-1 text-xs text-slate-500">严格恢复要求目标 Embedding 的 provider/model/dimensions 与包一致；不匹配时请选择重新嵌入。</p></div>
                  ) : <div className="rounded-xl bg-amber-50 p-3 text-xs text-amber-700">文档包不携带向量，将基于包内 chunk 和 QA 重新生成 Embedding 与索引。</div>}
                  {importMutation.error && <ErrorState message={importMutation.error.message} />}
                  <button type="button" className="btn-primary" disabled={importMutation.isPending || !importEmbeddingId || !importEnrichmentId} onClick={() => importMutation.mutate()}>{importMutation.isPending ? "创建导入任务..." : "开始导入"}</button>
                </div>
              )}
            </>
          ) : (
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              {importJobQuery.error && <ErrorState message={importJobQuery.error.message} />}
              <div className="flex items-center justify-between"><span className="font-medium text-slate-800">导入任务</span><span className="chip bg-white text-xs">{activeImportJob.status}</span></div>
              <div className="mt-2 text-xs text-slate-500">阶段：{String(activeImportJob.progress.phase ?? "-")} · 文档 {String(activeImportJob.progress.documents ?? 0)} · Chunk {String(activeImportJob.progress.chunks ?? 0)}</div>
              {activeImportJob.status === "completed" && activeImportJob.result_knowledge_base_id && <Link to={`/knowledge-bases/${activeImportJob.result_knowledge_base_id}`} className="btn-primary mt-3" onClick={() => queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] })}>打开新知识库</Link>}
            </div>
          )}
        </div>
      </Modal>
    </div>
  );
}

