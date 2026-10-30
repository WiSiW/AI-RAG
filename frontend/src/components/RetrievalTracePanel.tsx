import { ChevronDown } from "lucide-react";
import type { RetrievalTrace, TraceChunkSnapshot } from "../types";

const sourceLabels: Record<string, string> = {
  rules: "规则",
  bert: "BERT 分类器",
  llm: "LLM 判定",
  fallback: "降级判定",
  user_override: "用户指定",
};

const routeLabels: Record<string, string> = {
  bm25: "BM25",
  l1: "L1 章节摘要",
  original: "原块向量",
  summary: "摘要向量",
  questions: "问题向量",
  l3: "L3 原子块",
};

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(4);
}

function routeLabel(route: string): string {
  return routeLabels[route] ?? route;
}

export default function RetrievalTracePanel({ trace }: { trace: RetrievalTrace }) {
  const complexity = trace.complexity;
  if (!complexity) return null;
  return (
    <details className="mt-3 rounded-xl border border-slate-200 bg-slate-50/80 px-3 py-2 text-xs text-slate-600">
      <summary className="flex cursor-pointer list-none items-center gap-2 font-medium text-slate-700">
        <ChevronDown className="h-3.5 w-3.5" />
        <span className={`chip text-[10px] ${complexity.is_multi_hop ? "bg-amber-50 text-amber-700" : "bg-emerald-50 text-emerald-700"}`}>
          {complexity.is_multi_hop ? "多跳检索" : "单跳检索"}
        </span>
        <span>{sourceLabels[complexity.decision_source] ?? complexity.decision_source}</span>
        {trace.total_rounds > 0 && <span>· {trace.total_rounds} 轮 / {trace.total_chunks} 个证据块</span>}
      </summary>
      <div className="mt-3 space-y-2 border-t border-slate-200 pt-3">
        <div className="flex flex-wrap gap-2">
          <span className="chip bg-white">规则分 {complexity.rule_score.toFixed(2)}</span>
          {complexity.bert_probability != null && <span className="chip bg-white">BERT {complexity.bert_probability.toFixed(3)}</span>}
          {complexity.llm_decision != null && <span className="chip bg-white">LLM：{complexity.llm_decision ? "多跳" : "单跳"}</span>}
          {complexity.classifier_model_version && <span className="chip bg-white">{complexity.classifier_model_version}</span>}
        </div>
        <div className="leading-5 text-slate-500">{complexity.reason || "-"}</div>
        {trace.steps.length > 0 && (
          <div className="space-y-2">
            {trace.steps.map((step) => {
              const chunks = step.chunks ?? [];
              const routeCounts = Object.entries(step.route_counts ?? {}).filter(([, count]) => count > 0);
              const granularityCounts = Object.entries(step.granularity_counts ?? {}).filter(([, count]) => count > 0);
              return (
                <details key={step.round} className="group rounded-lg border border-slate-200 bg-white">
                  <summary className="flex cursor-pointer list-none flex-wrap items-center gap-2 p-2.5 font-medium text-slate-700">
                    <ChevronDown className="h-3.5 w-3.5 shrink-0 text-slate-400 transition-transform group-open:rotate-180" />
                    <span>第 {step.round} 轮</span>
                    <span className="min-w-0 flex-1 text-slate-500">{step.query}</span>
                    <span className="chip bg-slate-100 text-[10px]">新增 {step.new_chunk_count} 块</span>
                    <span className="chip bg-white text-[10px]">本轮 {chunks.length || step.chunk_ids?.length || 0} 块</span>
                    {step.need_more && <span className="chip bg-amber-50 text-[10px] text-amber-700">继续检索</span>}
                  </summary>
                  <div className="space-y-2 border-t border-slate-100 px-2.5 py-2.5">
                    {step.key_entities.length > 0 && <div className="text-slate-500">线索：{step.key_entities.join("、")}</div>}
                    {step.reason && <div className="text-slate-500">{step.reason}</div>}
                    {(routeCounts.length > 0 || granularityCounts.length > 0) && (
                      <div className="flex flex-wrap gap-1.5">
                        {routeCounts.map(([route, count]) => (
                          <span key={route} className="chip bg-emerald-50 text-[10px] text-emerald-700">{routeLabel(route)} · {count}</span>
                        ))}
                        {granularityCounts.map(([level, count]) => (
                          <span key={level} className="chip bg-violet-50 text-[10px] text-violet-700">{level} · {count}</span>
                        ))}
                      </div>
                    )}
                    {chunks.length > 0 ? (
                      <div className="space-y-1.5">
                        {chunks.map((chunk) => <ChunkEvidence key={chunk.chunk_id} chunk={chunk} />)}
                      </div>
                    ) : step.chunk_ids?.length ? (
                      <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-2.5 text-slate-400">
                        该轮来自旧版轨迹，仅保存块 ID：<span className="ml-1 break-all font-mono text-[10px] text-slate-500">{step.chunk_ids.join("、")}</span>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-2.5 text-slate-400">该轮未命中任何证据块。</div>
                    )}
                  </div>
                </details>
              );
            })}
          </div>
        )}
        <div className="text-slate-400">停止原因：{trace.stop_reason || "-"} {trace.uncertain ? "· 信息可能不完整" : ""}</div>
      </div>
    </details>
  );
}

function ChunkEvidence({ chunk }: { chunk: TraceChunkSnapshot }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-2.5">
      <div className="flex flex-wrap items-center gap-1.5 font-medium text-slate-700">
        <span className="truncate">{chunk.source}</span>
        <span className="font-normal text-slate-400">{chunk.section || "未标注章节"}</span>
        <span className="chip bg-violet-50 text-[10px] text-violet-700">{chunk.granularity_level} · {chunk.granule_type}</span>
      </div>
      <div className="mt-1.5 flex flex-wrap gap-1">
        {chunk.matched_representations.map((route) => (
          <span key={route} className="chip bg-emerald-50 text-[10px] text-emerald-700">{routeLabel(route)}</span>
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-2 font-mono text-[10px] text-slate-500">
        <span>RRF {score(chunk.rrf_score)}</span>
        <span>Vector {score(chunk.vector_score)}</span>
        <span>BM25 {score(chunk.bm25_score)}</span>
      </div>
      <div className="mt-1.5 whitespace-pre-wrap break-words leading-5 text-slate-600">{chunk.content_snippet || "无内容摘要"}</div>
    </div>
  );
}
