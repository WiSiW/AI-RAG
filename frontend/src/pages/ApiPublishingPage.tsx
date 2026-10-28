import { FormEvent, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Code2, Copy, KeyRound, Pencil, Plus, RefreshCw, ShieldCheck, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";
import type { PublicationKind, PublishedApi } from "../types";

interface PublicationForm {
  name: string;
  slug: string;
  kind: PublicationKind;
  description: string;
  knowledgeBaseIds: string[];
  llmProfileId: string;
  topK: string;
  rerank: boolean;
  streamEnabled: boolean;
  isActive: boolean;
}

const initialForm: PublicationForm = {
  name: "",
  slug: "",
  kind: "qa",
  description: "",
  knowledgeBaseIds: [],
  llmProfileId: "",
  topK: "5",
  rerank: true,
  streamEnabled: true,
  isActive: true,
};

export default function ApiPublishingPage() {
  const queryClient = useQueryClient();
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [form, setForm] = useState<PublicationForm>(initialForm);
  const [secretKey, setSecretKey] = useState("");
  const [copied, setCopied] = useState(false);

  const publicationsQuery = useQuery({ queryKey: ["publications"], queryFn: api.listPublications });
  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const modelsQuery = useQuery({ queryKey: ["models", "llm"], queryFn: () => api.listModels("llm") });
  const publications = publicationsQuery.data ?? [];
  const knowledgeBases = kbQuery.data ?? [];
  const llmModels = modelsQuery.data ?? [];

  const saveMutation = useMutation({
    mutationFn: () => {
      const payload = {
        name: form.name.trim(),
        slug: form.slug.trim(),
        kind: form.kind,
        description: form.description.trim(),
        knowledge_base_ids: form.knowledgeBaseIds,
        llm_profile_id: form.llmProfileId || null,
        settings: {
          top_k: Number(form.topK),
          rerank: form.rerank,
          stream_enabled: form.streamEnabled,
        },
        is_active: form.isActive,
      };
      return editingId ? api.updatePublication(editingId, payload) : api.createPublication(payload);
    },
    onSuccess: async (data) => {
      setOpen(false);
      setEditingId(null);
      if ("api_key" in data && typeof data.api_key === "string") setSecretKey(data.api_key);
      await queryClient.invalidateQueries({ queryKey: ["publications"] });
    },
  });

  const rotateMutation = useMutation({
    mutationFn: api.rotatePublicationKey,
    onSuccess: async (data) => {
      setSecretKey(data.api_key);
      await queryClient.invalidateQueries({ queryKey: ["publications"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deletePublication,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["publications"] }),
  });

  function openCreate() {
    setEditingId(null);
    setForm({
      ...initialForm,
      knowledgeBaseIds: knowledgeBases[0] ? [knowledgeBases[0].id] : [],
      llmProfileId: llmModels[0]?.id ?? "",
    });
    setOpen(true);
  }

  function openEdit(publication: PublishedApi) {
    setEditingId(publication.id);
    setForm({
      name: publication.name,
      slug: publication.slug,
      kind: publication.kind,
      description: publication.description,
      knowledgeBaseIds: publication.knowledge_base_ids,
      llmProfileId: publication.llm_profile_id ?? "",
      topK: String(publication.settings.top_k ?? 5),
      rerank: Boolean(publication.settings.rerank ?? true),
      streamEnabled: Boolean(publication.settings.stream_enabled ?? true),
      isActive: publication.is_active,
    });
    setOpen(true);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    saveMutation.mutate();
  }

  function toggleKb(id: string) {
    setForm((current) => ({
      ...current,
      knowledgeBaseIds: current.knowledgeBaseIds.includes(id)
        ? current.knowledgeBaseIds.filter((item) => item !== id)
        : [...current.knowledgeBaseIds, id],
    }));
  }

  async function copySecret() {
    await navigator.clipboard.writeText(secretKey);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  }

  const endpointFor = (publication: PublishedApi) => {
    const path = publication.endpoint ?? `/api/v1/public/${publication.kind === "qa" ? "qa" : "search"}/${publication.slug}`;
    return `${window.location.origin}${path}`;
  };

  if (publicationsQuery.isLoading || kbQuery.isLoading || modelsQuery.isLoading) return <LoadingState />;
  if (publicationsQuery.error) return <ErrorState message={publicationsQuery.error.message} />;

  return (
    <div>
      <PageHeader
        title="API 发布"
        description="将问答或检索能力发布为带独立 API Key 的接口，可限定知识库、问答模型、Top K、重排与流式输出。"
        action={<button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />新建发布接口</button>}
      />

      {(saveMutation.error || rotateMutation.error || deleteMutation.error) && (
        <div className="mb-4"><ErrorState message={(saveMutation.error ?? rotateMutation.error ?? deleteMutation.error)!.message} /></div>
      )}

      {!!secretKey && (
        <div className="mb-5 rounded-2xl border border-amber-200 bg-amber-50 p-4">
          <div className="flex items-center gap-2 font-semibold text-amber-900"><KeyRound className="h-4 w-4" />API Key 仅显示一次，请立即保存</div>
          <div className="mt-3 flex items-center gap-2 rounded-xl bg-white p-3 font-mono text-xs text-slate-700">
            <span className="min-w-0 flex-1 break-all">{secretKey}</span>
            <button type="button" onClick={copySecret} className="btn-secondary shrink-0 px-3 text-xs"><Copy className="h-4 w-4" />{copied ? "已复制" : "复制"}</button>
          </div>
        </div>
      )}

      {!publications.length ? (
        <div className="panel"><EmptyState title="尚未发布接口" description="可分别发布问答接口和检索接口，并为每个接口设置独立访问范围。" action={<button type="button" onClick={openCreate} className="btn-primary"><Plus className="h-4 w-4" />新建发布接口</button>} /></div>
      ) : (
        <div className="grid gap-5 xl:grid-cols-2">
          {publications.map((publication) => (
            <article key={publication.id} className="panel p-5">
              <div className="flex items-start justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className={`grid h-10 w-10 place-items-center rounded-xl ${publication.kind === "qa" ? "bg-brand-50 text-brand-700" : "bg-cyan-50 text-cyan-700"}`}><Code2 className="h-5 w-5" /></div>
                  <div>
                    <div className="font-semibold text-slate-900">{publication.name}</div>
                    <div className="text-xs text-slate-500">{publication.kind === "qa" ? "问答接口" : "检索接口"} · /{publication.slug}</div>
                  </div>
                </div>
                <span className={`chip ${publication.is_active ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"}`}>{publication.is_active ? "已启用" : "已停用"}</span>
              </div>
              <p className="mt-3 min-h-10 text-sm leading-6 text-slate-500">{publication.description || "暂无描述"}</p>
              <div className="mt-4 space-y-2 rounded-xl bg-slate-50 p-3 text-xs text-slate-600">
                <div><span className="text-slate-400">接口地址：</span><code className="break-all">{endpointFor(publication)}</code></div>
                <div><span className="text-slate-400">知识库：</span>{publication.knowledge_base_ids.map((id) => knowledgeBases.find((kb) => kb.id === id)?.name ?? id).join("、")}</div>
                <div><span className="text-slate-400">问答模型：</span>{publication.llm_profile_id ? llmModels.find((model) => model.id === publication.llm_profile_id)?.name ?? publication.llm_profile_id : "使用知识库默认模型"}</div>
                <div><span className="text-slate-400">Top K / 重排 / 流式：</span>{String(publication.settings.top_k ?? 5)} / {publication.settings.rerank ? "是" : "否"} / {publication.settings.stream_enabled ? "是" : "否"}</div>
                <div><span className="text-slate-400">API Key：</span><span className="font-mono">{publication.api_key_prefix}…</span></div>
              </div>
              <div className="mt-4 flex flex-wrap gap-2 border-t border-slate-100 pt-4">
                <button type="button" onClick={() => openEdit(publication)} className="btn-secondary px-3 text-xs"><Pencil className="h-4 w-4" />编辑</button>
                <button type="button" onClick={() => { if (window.confirm("轮换后旧 API Key 将立即失效，继续？")) rotateMutation.mutate(publication.id); }} className="btn-secondary px-3 text-xs"><RefreshCw className="h-4 w-4" />轮换密钥</button>
                <button type="button" onClick={() => { if (window.confirm(`确认删除发布接口“${publication.name}”？`)) deleteMutation.mutate(publication.id); }} className="btn-ghost px-3 text-xs text-red-600"><Trash2 className="h-4 w-4" />删除</button>
              </div>
            </article>
          ))}
        </div>
      )}

      <div className="mt-6 panel p-5">
        <div className="flex items-center gap-2 font-semibold text-slate-900"><ShieldCheck className="h-4 w-4 text-emerald-600" />调用示例</div>
        <pre className="thin-scrollbar mt-3 overflow-x-auto rounded-xl bg-ink-950 p-4 text-xs leading-6 text-slate-200">{`curl -X POST "${window.location.origin}/api/v1/public/qa/{slug}" \\
  -H "Authorization: Bearer zy_xxx" \\
  -H "Content-Type: application/json" \\
  -d '{"query":"问题","stream":false}'`}</pre>
        <p className="mt-2 text-xs text-slate-500">返回引用中的 <code>file_url</code> 可直接访问原文件，<code>preview_url</code> 返回原文与分块 JSON。文件地址使用短时签名，不暴露 API Key。</p>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title={editingId ? "编辑发布接口" : "新建发布接口"} wide>
        <form onSubmit={submit} className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-2">
            <div><label className="label">名称</label><input className="input" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></div>
            <div><label className="label">接口标识 slug</label><input className="input" value={form.slug} onChange={(event) => setForm({ ...form, slug: event.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-") })} placeholder="customer-qa" required /></div>
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="label">接口类型</label>
              <select className="input" value={form.kind} onChange={(event) => setForm({ ...form, kind: event.target.value as PublicationKind })}>
                <option value="qa">问答接口</option>
                <option value="search">检索接口</option>
              </select>
            </div>
            <div>
              <label className="label">指定问答模型</label>
              <select className="input" value={form.llmProfileId} onChange={(event) => setForm({ ...form, llmProfileId: event.target.value })}>
                <option value="">使用知识库默认模型</option>
                {llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
          </div>
          <div><label className="label">描述</label><textarea className="input min-h-20" value={form.description} onChange={(event) => setForm({ ...form, description: event.target.value })} /></div>
          <div>
            <label className="label">可访问知识库</label>
            <div className="grid gap-2 rounded-xl border border-slate-200 p-3 sm:grid-cols-2">
              {knowledgeBases.map((kb) => (
                <label key={kb.id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-slate-700 hover:bg-slate-50">
                  <input type="checkbox" checked={form.knowledgeBaseIds.includes(kb.id)} onChange={() => toggleKb(kb.id)} />{kb.name}
                </label>
              ))}
            </div>
          </div>
          <div className="grid gap-4 sm:grid-cols-3">
            <div><label className="label">Top K</label><input className="input" type="number" min={1} max={50} value={form.topK} onChange={(event) => setForm({ ...form, topK: event.target.value })} /></div>
            <label className="flex items-center gap-2 self-end rounded-xl border border-slate-200 px-3 py-3 text-sm"><input type="checkbox" checked={form.rerank} onChange={(event) => setForm({ ...form, rerank: event.target.checked })} />启用重排</label>
            <label className="flex items-center gap-2 self-end rounded-xl border border-slate-200 px-3 py-3 text-sm"><input type="checkbox" checked={form.streamEnabled} onChange={(event) => setForm({ ...form, streamEnabled: event.target.checked })} />允许流式</label>
          </div>
          <label className="flex items-center gap-2 text-sm text-slate-700"><input type="checkbox" checked={form.isActive} onChange={(event) => setForm({ ...form, isActive: event.target.checked })} />启用接口</label>
          <div className="flex justify-end gap-3">
            <button type="button" className="btn-secondary" onClick={() => setOpen(false)}>取消</button>
            <button type="submit" className="btn-primary" disabled={saveMutation.isPending || !form.knowledgeBaseIds.length}>{saveMutation.isPending ? "保存中..." : editingId ? "保存修改" : "创建并生成 API Key"}</button>
          </div>
        </form>
      </Modal>
    </div>
  );
}

