import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, Database, FileText, Plus, Trash2, UploadCloud } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";

interface FormState {
  name: string;
  description: string;
  embeddingProfileId: string;
  rerankProfileId: string;
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
  const [form, setForm] = useState<FormState>(initialForm);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const modelQuery = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });
  const capabilitiesQuery = useQuery({ queryKey: ["capabilities"], queryFn: api.getCapabilities });
  const models = modelQuery.data ?? [];
  const embeddingModels = useMemo(() => models.filter((item) => item.kind === "embedding" && item.is_active), [models]);
  const rerankModels = useMemo(() => models.filter((item) => item.kind === "rerank" && item.is_active), [models]);
  const llmModels = useMemo(() => models.filter((item) => item.kind === "llm" && item.is_active), [models]);

  const createMutation = useMutation({
    mutationFn: () =>
      api.createKnowledgeBase({
        name: form.name.trim(),
        description: form.description.trim(),
        embedding_profile_id: form.embeddingProfileId,
        rerank_profile_id: form.rerankProfileId || null,
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

  function openCreate() {
    setForm({
      ...initialForm,
      embeddingProfileId: embeddingModels[0]?.id ?? "",
      rerankProfileId: rerankModels[0]?.id ?? "",
      llmProfileId: llmModels[0]?.id ?? "",
    });
    setOpen(true);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    if (!form.embeddingProfileId) return;
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
        description="知识库绑定固定的 Embedding 模型，并统一管理文档、检索参数与默认问答模型。"
        action={
          <button type="button" onClick={openCreate} className="btn-primary">
            <Plus className="h-4 w-4" />
            新建知识库
          </button>
        }
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
          <div className="grid gap-4 sm:grid-cols-2">
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
            <button type="submit" disabled={createMutation.isPending || !embeddingModels.length} className="btn-primary">
              {createMutation.isPending ? "创建中..." : "创建知识库"}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}

