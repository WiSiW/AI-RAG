import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, ArrowDown, ArrowUp, Plus, ScanText, Trash2 } from "lucide-react";
import { api } from "../api/client";
import { ErrorState, LoadingState, PageHeader } from "../components/UI";
import type { ParsingEngine, ParsingEngineCategory } from "../types";

const categoryLabels: Record<ParsingEngineCategory, string> = {
  ocr: "OCR 引擎",
  table: "表格引擎",
  layout: "版面分析",
};

const categoryDescriptions: Record<ParsingEngineCategory, string> = {
  ocr: "通过 PDF 处理服务按优先级调用 RapidOCR/Tesseract；文本为空、乱码或置信度不足时自动回退。",
  table: "同一页按顺序尝试可用表格引擎；扫描页仅使用 OCR 坐标重建。",
  layout: "独立版面服务优先；不可用时自动退化为文本与坐标几何规则。",
};

export default function ParsingEnginesPage() {
  const queryClient = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [category, setCategory] = useState<ParsingEngineCategory>("ocr");
  const [adapter, setAdapter] = useState("http_ocr");
  const [displayName, setDisplayName] = useState("");
  const [priority, setPriority] = useState("100");
  const [configText, setConfigText] = useState(defaultConfigText("ocr", "http_ocr"));
  const [testResults, setTestResults] = useState<Record<string, string>>({});

  const enginesQuery = useQuery({ queryKey: ["parsing-engines"], queryFn: () => api.listParsingEngines() });
  const adaptersQuery = useQuery({ queryKey: ["parsing-engine-adapters"], queryFn: api.listParsingEngineAdapters });
  const engines = enginesQuery.data ?? [];
  const adapters = adaptersQuery.data ?? [];
  const availableAdapters = adapters.filter((item) => item.category === category);

  const refresh = () => queryClient.invalidateQueries({ queryKey: ["parsing-engines"] });
  const createMutation = useMutation({
    mutationFn: async () => {
      const config = JSON.parse(configText || "{}") as Record<string, unknown>;
      return api.createParsingEngine({
        category,
        adapter,
        display_name: displayName.trim() || availableAdapters.find((item) => item.adapter === adapter)?.label,
        priority: Number(priority),
        enabled: true,
        config,
      });
    },
    onSuccess: () => {
      setShowCreate(false);
      setDisplayName("");
      setConfigText(defaultConfigText("ocr", "http_ocr"));
      refresh();
    },
  });
  const updateMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Record<string, unknown> }) => api.updateParsingEngine(id, payload),
    onSuccess: refresh,
  });
  const deleteMutation = useMutation({ mutationFn: api.deleteParsingEngine, onSuccess: refresh });
  const testMutation = useMutation({
    mutationFn: api.testParsingEngine,
    onSuccess: (result, id) => setTestResults((current) => ({
      ...current,
      [id]: `${result.ok ? "可用" : "异常"} · ${result.message} · ${result.latency_ms.toFixed(0)}ms`,
    })),
  });
  const reorderMutation = useMutation({
    mutationFn: ({ type, ids }: { type: ParsingEngineCategory; ids: string[] }) => api.reorderParsingEngines(type, ids),
    onSuccess: refresh,
  });

  const grouped = useMemo(() => ({
    ocr: engines.filter((item) => item.category === "ocr"),
    table: engines.filter((item) => item.category === "table"),
    layout: engines.filter((item) => item.category === "layout"),
  }), [engines]);

  function submitCreate(event: FormEvent) {
    event.preventDefault();
    createMutation.mutate();
  }

  function moveEngine(type: ParsingEngineCategory, index: number, offset: number) {
    const items = grouped[type];
    const target = index + offset;
    if (target < 0 || target >= items.length) return;
    const ids = items.map((item) => item.id);
    [ids[index], ids[target]] = [ids[target], ids[index]];
    reorderMutation.mutate({ type, ids });
  }

  if (enginesQuery.isLoading || adaptersQuery.isLoading) return <LoadingState />;
  if (enginesQuery.error) return <ErrorState message={enginesQuery.error.message} />;
  if (adaptersQuery.error) return <ErrorState message={adaptersQuery.error.message} />;

  const actionError = createMutation.error ?? updateMutation.error ?? deleteMutation.error ?? testMutation.error ?? reorderMutation.error;

  return (
    <div>
      <PageHeader
        title="解析引擎"
        description="管理 PDF 的 OCR、表格和版面分析适配器。配置在下一次入库或重新解析时生效，密钥只允许填写环境变量名。"
        action={<button type="button" className="btn-primary" onClick={() => setShowCreate((value) => !value)}><Plus className="h-4 w-4" />新增引擎</button>}
      />
      {actionError && <div className="mb-4"><ErrorState message={actionError.message} /></div>}

      {showCreate && (
        <form onSubmit={submitCreate} className="panel mb-5 grid gap-4 p-5 lg:grid-cols-2">
          <div>
            <label className="label">类别</label>
            <select className="input" value={category} onChange={(event) => {
              const next = event.target.value as ParsingEngineCategory;
              setCategory(next);
              const nextAdapter = adapters.find((item) => item.category === next)?.adapter ?? "";
              setAdapter(nextAdapter);
              setConfigText(defaultConfigText(next, nextAdapter));
            }}>
              {(Object.keys(categoryLabels) as ParsingEngineCategory[]).map((item) => <option key={item} value={item}>{categoryLabels[item]}</option>)}
            </select>
          </div>
          <div>
            <label className="label">适配器</label>
            <select className="input" value={adapter} onChange={(event) => { setAdapter(event.target.value); setConfigText(defaultConfigText(category, event.target.value)); }}>
              {availableAdapters.map((item) => <option key={item.adapter} value={item.adapter}>{item.label}</option>)}
            </select>
            <div className="mt-1 text-xs text-slate-400">{availableAdapters.find((item) => item.adapter === adapter)?.description}</div>
          </div>
          <div><label className="label">显示名称</label><input className="input" value={displayName} onChange={(event) => setDisplayName(event.target.value)} placeholder="例如：RapidOCR 中文" /></div>
          <div><label className="label">优先级（越小越先）</label><input className="input" type="number" min={0} value={priority} onChange={(event) => setPriority(event.target.value)} /></div>
          <div className="lg:col-span-2"><label className="label">配置 JSON</label><textarea className="input min-h-28 font-mono text-xs" value={configText} onChange={(event) => setConfigText(event.target.value)} /></div>
          <div className="flex items-center gap-2 lg:col-span-2"><button type="submit" className="btn-primary" disabled={createMutation.isPending}>创建</button><button type="button" className="btn-secondary" onClick={() => setShowCreate(false)}>取消</button></div>
        </form>
      )}

      <div className="space-y-5">
        {(Object.keys(categoryLabels) as ParsingEngineCategory[]).map((type) => (
          <section key={type} className="panel p-5">
            <div className="flex items-start gap-3">
              <div className="grid h-9 w-9 place-items-center rounded-xl bg-brand-50 text-brand-700"><ScanText className="h-4 w-4" /></div>
              <div><h2 className="font-semibold text-slate-900">{categoryLabels[type]}</h2><p className="mt-1 text-xs leading-5 text-slate-500">{categoryDescriptions[type]}</p></div>
            </div>
            <div className="mt-4 space-y-2">
              {grouped[type].map((engine, index) => (
                <EngineRow
                  key={engine.id}
                  engine={engine}
                  index={index}
                  total={grouped[type].length}
                  testMessage={testResults[engine.id] ?? ""}
                  busy={updateMutation.isPending || deleteMutation.isPending || testMutation.isPending || reorderMutation.isPending}
                  onMove={(offset) => moveEngine(type, index, offset)}
                  onToggle={() => updateMutation.mutate({ id: engine.id, payload: { enabled: !engine.enabled } })}
                  onDelete={() => window.confirm(`确认删除解析引擎“${engine.display_name}”？`) && deleteMutation.mutate(engine.id)}
                  onTest={() => testMutation.mutate(engine.id)}
                  onSaveConfig={(config, nextPriority) => updateMutation.mutate({ id: engine.id, payload: { config, priority: nextPriority } })}
                />
              ))}
              {!grouped[type].length && <div className="rounded-xl border border-dashed border-slate-200 p-6 text-center text-sm text-slate-400">尚未配置该类引擎</div>}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function defaultConfigText(category: ParsingEngineCategory, adapter: string): string {
  if (category === "ocr" && adapter === "http_ocr") {
    return JSON.stringify(
      {
        base_url: "http://pdf-process:8020",
        engine: "rapidocr",
        transport: "multipart",
        recognize_path: "/ocr",
        health_path: "/health",
        language: "chi_sim+eng",
      },
      null,
      2,
    );
  }
  if (category === "layout" && adapter === "layout_http") {
    return JSON.stringify({ base_url: "http://pdf-process:8020", api_key_env: "" }, null, 2);
  }
  return "{}";
}

function EngineRow({
  engine,
  index,
  total,
  testMessage,
  busy,
  onMove,
  onToggle,
  onDelete,
  onTest,
  onSaveConfig,
}: {
  engine: ParsingEngine;
  index: number;
  total: number;
  testMessage: string;
  busy: boolean;
  onMove: (offset: number) => void;
  onToggle: () => void;
  onDelete: () => void;
  onTest: () => void;
  onSaveConfig: (config: Record<string, unknown>, priority: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [configText, setConfigText] = useState(JSON.stringify(engine.config, null, 2));
  const [priority, setPriority] = useState(String(engine.priority));
  const healthy = engine.health.available === true;
  return (
    <div className={`rounded-xl border p-3 ${engine.enabled ? "border-slate-200 bg-white" : "border-slate-200 bg-slate-50 opacity-75"}`}>
      <div className="flex flex-wrap items-center gap-2">
        <span className="grid h-7 w-7 place-items-center rounded-lg bg-slate-100 font-mono text-xs text-slate-600">{index + 1}</span>
        <span className="font-medium text-slate-800">{engine.display_name}</span>
        <span className="chip bg-slate-100 text-[10px]">{engine.adapter}</span>
        {engine.category === "ocr" && !!engine.config.engine && <span className="chip bg-sky-50 text-[10px] text-sky-700">{String(engine.config.engine)}</span>}
        {engine.category === "ocr" && !!engine.config.base_url && <span className="max-w-[260px] truncate chip bg-white text-[10px]" title={String(engine.config.base_url)}>{String(engine.config.base_url)}</span>}
        <span className={`chip text-[10px] ${healthy ? "bg-emerald-50 text-emerald-700" : "bg-amber-50 text-amber-700"}`}>
          {healthy ? "可用" : engine.health.detail || "不可用"}
        </span>
        <span className={`chip text-[10px] ${engine.enabled ? "bg-brand-50 text-brand-700" : "bg-slate-100 text-slate-500"}`}>{engine.enabled ? "已启用" : "已停用"}</span>
        <div className="ml-auto flex items-center gap-1">
          <button type="button" className="btn-secondary px-2 py-1.5" disabled={busy || index === 0} onClick={() => onMove(-1)} aria-label="上移"><ArrowUp className="h-3.5 w-3.5" /></button>
          <button type="button" className="btn-secondary px-2 py-1.5" disabled={busy || index === total - 1} onClick={() => onMove(1)} aria-label="下移"><ArrowDown className="h-3.5 w-3.5" /></button>
          <button type="button" className="btn-secondary px-2 py-1.5" disabled={busy} onClick={onToggle}>{engine.enabled ? "停用" : "启用"}</button>
          <button type="button" className="btn-secondary px-2 py-1.5" disabled={busy} onClick={onTest}><Activity className="h-3.5 w-3.5" />检测</button>
          <button type="button" className="btn-secondary px-2 py-1.5" disabled={busy} onClick={() => setEditing((value) => !value)}>参数</button>
          <button type="button" className="btn-secondary px-2 py-1.5 text-red-600" disabled={busy} onClick={onDelete}><Trash2 className="h-3.5 w-3.5" /></button>
        </div>
      </div>
      {testMessage && <div className="mt-2 text-xs text-slate-500">{testMessage}</div>}
      {editing && (
        <div className="mt-3 grid gap-3 border-t border-slate-100 pt-3 lg:grid-cols-[160px_1fr]">
          <div><label className="label">优先级</label><input className="input" type="number" min={0} value={priority} onChange={(event) => setPriority(event.target.value)} /></div>
          <div><label className="label">配置 JSON</label><textarea className="input min-h-24 font-mono text-xs" value={configText} onChange={(event) => setConfigText(event.target.value)} /></div>
          <div className="lg:col-span-2"><button type="button" className="btn-primary" disabled={busy} onClick={() => onSaveConfig(JSON.parse(configText || "{}"), Number(priority))}>保存参数</button></div>
        </div>
      )}
    </div>
  );
}