import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, BrainCircuit, Braces, Cpu, Database, LoaderCircle, Pencil, Plus, Sparkles, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";
import type { ModelKind, ModelProfile, ModelTestResult } from "../types";

interface ModelForm {
  kind: ModelKind;
  name: string;
  provider: string;
  baseUrl: string;
  modelName: string;
  apiKey: string;
  dimensions: string;
  config: string;
  isActive: boolean;
}

const initialForm: ModelForm = {
  kind: "llm",
  name: "",
  provider: "openai-compatible",
  baseUrl: "https://api.openai.com/v1",
  modelName: "",
  apiKey: "",
  dimensions: "",
  config: "{}",
  isActive: true,
};

const kindMeta: Record<ModelKind, { label: string; description: string; icon: typeof Cpu; className: string }> = {
  embedding: { label: "Embedding", description: "文档与查询向量化", icon: Database, className: "bg-cyan-50 text-cyan-700" },
  rerank: { label: "Rerank", description: "候选分块语义重排", icon: Activity, className: "bg-amber-50 text-amber-700" },
  llm: { label: "问答模型", description: "查询改写、摘要与回答生成", icon: BrainCircuit, className: "bg-violet-50 text-violet-700" },
  enrichment: { label: "增强模型", description: "分块关键词、摘要与假设问题生成", icon: Sparkles, className: "bg-emerald-50 text-emerald-700" },
};

export default function ModelsPage() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<ModelForm>(initialForm);
  const [testResults, setTestResults] = useState<Record<string, ModelTestResult>>({});

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });
  const models = modelsQuery.data ?? [];
  const grouped = useMemo(() => ({
    embedding: models.filter((item) => item.kind === "embedding"),
    rerank: models.filter((item) => item.kind === "rerank"),
    llm: models.filter((item) => item.kind === "llm"),
    enrichment: models.filter((item) => item.kind === "enrichment"),
  }), [models]);

  const saveMutation = useMutation({
    mutationFn: () => {
      let config: Record<string, unknown> = {};
      try {
        config = JSON.parse(form.config || "{}") as Record<string, unknown>;
      } catch {
        throw new Error("高级配置必须是合法 JSON");
      }
      const payload: Record<string, unknown> = {
        name: form.name.trim(),
        provider: form.provider.trim(),
        base_url: form.baseUrl.trim(),
        model_name: form.modelName.trim(),
        dimensions: form.kind === "embedding" ? Number(form.dimensions) : undefined,
        config,
        is_active: form.isActive,
      };
      if (form.apiKey) payload.api_key = form.apiKey;
      if (editingId) return api.updateModel(editingId, payload);
      return api.createModel({ ...payload, kind: form.kind, api_key: form.apiKey || undefined });
    },
    onSuccess: async () => {
      setOpen(false);
      setEditingId(null);
      setForm(initialForm);
      await queryClient.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteModel,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["models"] }),
  });

  const testMutation = useMutation({
    mutationFn: api.testModel,
    onSuccess: (result, id) => setTestResults((current) => ({ ...current, [id]: result })),
    onError: (error, id) => {
      setTestResults((current) => ({
        ...current,
        [id]: { ok: false, latency_ms: 0, message: error.message, details: {} },
      }));
    },
  });

  function openCreate() {
    setEditingId(null);
    setForm(initialForm);
    setOpen(true);
  }

  function openEdit(model: ModelProfile) {
    setEditingId(model.id);
    setForm({
      kind: model.kind,
      name: model.name,
      provider: model.provider,
      baseUrl: model.base_url,
      modelName: model.model_name,
      apiKey: "",
      dimensions: model.dimensions ? String(model.dimensions) : "",
      config: JSON.stringify(model.config ?? {}, null, 2),
      isActive: model.is_active,
    });
    setOpen(true);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    saveMutation.mutate();
  }

  if (modelsQuery.isLoading) return <LoadingState />;
  if (modelsQuery.error) return <ErrorState message={modelsQuery.error.message} />;

  return (
    <div>
      <PageHeader
        title="模型配置"
        description="统一配置 OpenAI-compatible 问答、增强与 Embedding API、Rerank HTTP API；已保存模型可随时编辑。"
        action={<button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />添加模型</button>}
      />

      {(saveMutation.error || deleteMutation.error) && (
        <div className="mb-4"><ErrorState message={(saveMutation.error ?? deleteMutation.error)!.message} /></div>
      )}

      {!models.length ? (
        <div className="panel">
          <EmptyState title="尚未配置模型" description="至少需要 Embedding、增强和问答模型；重排序模型可按知识库选用。" action={<button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />添加模型</button>} />
        </div>
      ) : (
        <div className="space-y-8">
          {(Object.keys(grouped) as ModelKind[]).map((kind) => {
            const meta = kindMeta[kind];
            const Icon = meta.icon;
            const items = grouped[kind];
            if (!items.length) return null;
            return (
              <section key={kind}>
                <div className="mb-3 flex items-center gap-3">
                  <div className={`grid h-9 w-9 place-items-center rounded-xl ${meta.className}`}><Icon className="h-4 w-4" /></div>
                  <div>
                    <h2 className="font-semibold text-slate-900">{meta.label}</h2>
                    <p className="text-xs text-slate-500">{meta.description}</p>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {items.map((model) => (
                    <ModelCard
                      key={model.id}
                      model={model}
                      testResult={testResults[model.id]}
                      testing={testMutation.isPending && testMutation.variables === model.id}
                      onTest={() => testMutation.mutate(model.id)}
                      onEdit={() => openEdit(model)}
                      onDelete={() => {
                        if (window.confirm(`确认删除模型“${model.name}”？`)) deleteMutation.mutate(model.id);
                      }}
                    />
                  ))}
                </div>
              </section>
            );
          })}
        </div>
      )}

      <Modal open={open} onClose={() => setOpen(false)} title={editingId ? "编辑模型" : "添加模型"} wide>
        <form onSubmit={submit} className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">模型类型</label>
              <select className="input" value={form.kind} onChange={(event) => setForm({ ...form, kind: event.target.value as ModelKind })} disabled={!!editingId} required>
                <option value="embedding">Embedding</option>
                <option value="rerank">Rerank</option>
                <option value="llm">问答模型</option>
                <option value="enrichment">增强模型</option>
              </select>
            </div>
            <div><label className="label">显示名称</label><input className="input" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div><label className="label">Base URL</label><input className="input" value={form.baseUrl} onChange={(event) => setForm({ ...form, baseUrl: event.target.value })} placeholder="https://api.example.com/v1" required /></div>
            <div><label className="label">模型名称</label><input className="input" value={form.modelName} onChange={(event) => setForm({ ...form, modelName: event.target.value })} placeholder="gpt-4.1-mini / bge-m3 / bge-reranker-v2-m3" required /></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div><label className="label">API Key</label><input className="input" type="password" value={form.apiKey} onChange={(event) => setForm({ ...form, apiKey: event.target.value })} placeholder={editingId ? "留空保留现有密钥" : "留空则使用 EMPTY"} /></div>
            <div><label className="label">Provider</label><input className="input" value={form.provider} onChange={(event) => setForm({ ...form, provider: event.target.value })} /></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            {form.kind === "embedding" && (
              <div><label className="label">向量维度</label><input className="input" type="number" min={1} value={form.dimensions} onChange={(event) => setForm({ ...form, dimensions: event.target.value })} placeholder="例如 1024" required /></div>
            )}
            <label className="flex items-center gap-2 self-end rounded-xl border border-slate-200 px-3 py-3 text-sm text-slate-700">
              <input type="checkbox" checked={form.isActive} onChange={(event) => setForm({ ...form, isActive: event.target.checked })} />
              启用该模型
            </label>
          </div>
          <div>
            <label className="label">高级配置 JSON</label>
            <div className="relative">
              <Braces className="pointer-events-none absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
              <textarea className="input min-h-28 pl-10 font-mono text-xs" value={form.config} onChange={(event) => setForm({ ...form, config: event.target.value })} />
            </div>
            <p className="mt-1.5 text-xs leading-5 text-slate-400">可配置 temperature、rerank_path、Ollama 的 num_gpu 等。Rerank 默认请求 {"{base_url}"}/rerank。</p>
          </div>
          <div className="flex justify-end gap-3">
            <button type="button" className="btn-secondary" onClick={() => setOpen(false)}>取消</button>
            <button type="submit" className="btn-primary" disabled={saveMutation.isPending}>{saveMutation.isPending ? "保存中..." : editingId ? "保存修改" : "创建模型"}</button>
          </div>
        </form>
      </Modal>
    </div>
  );
}

function ModelCard({
  model,
  testResult,
  testing,
  onTest,
  onEdit,
  onDelete,
}: {
  model: ModelProfile;
  testResult?: ModelTestResult;
  testing: boolean;
  onTest: () => void;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="panel flex flex-col p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-slate-100 text-slate-600"><Cpu className="h-5 w-5" /></div>
        <span className={`chip ${model.is_active ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>{model.is_active ? "启用" : "停用"}</span>
      </div>
      <div className="mt-4 font-semibold text-slate-900">{model.name}</div>
      <div className="mt-1 truncate text-xs text-slate-500">{model.model_name}</div>
      <div className="mt-4 space-y-2 rounded-xl bg-slate-50 p-3 text-xs">
        <div className="flex justify-between gap-3"><span className="text-slate-400">Base URL</span><span className="truncate text-right text-slate-600">{model.base_url}</span></div>
        {model.dimensions && <div className="flex justify-between"><span className="text-slate-400">维度</span><span className="text-slate-600">{model.dimensions}</span></div>}
        <div className="flex justify-between"><span className="text-slate-400">API Key</span><span className="text-slate-600">{model.has_api_key ? "已保存" : "EMPTY"}</span></div>
      </div>
      {testResult && (
        <div className={`mt-3 rounded-xl p-3 text-xs leading-5 ${testResult.ok ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
          {testResult.message}{testResult.latency_ms > 0 ? ` · ${Math.round(testResult.latency_ms)} ms` : ""}
        </div>
      )}
      <div className="mt-4 flex gap-2 border-t border-slate-100 pt-4">
        <button type="button" onClick={onTest} className="btn-secondary flex-1 px-3 text-xs" disabled={testing}>
          {testing ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <Activity className="h-4 w-4" />}测试
        </button>
        <button type="button" onClick={onEdit} className="btn-secondary px-3 text-xs"><Pencil className="h-4 w-4" />编辑</button>
        <button type="button" onClick={onDelete} className="grid h-10 w-10 place-items-center rounded-xl text-slate-400 hover:bg-red-50 hover:text-red-600" aria-label="删除模型"><Trash2 className="h-4 w-4" /></button>
      </div>
    </div>
  );
}
