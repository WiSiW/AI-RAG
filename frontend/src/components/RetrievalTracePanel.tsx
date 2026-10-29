import { ChevronDown } from "lucide-react";
import type { RetrievalTrace } from "../types";

const sourceLabels: Record<string, string> = {
  rules: "规则",
  bert: "BERT 分类器",
  llm: "LLM 判定",
  fallback: "降级判定",
  user_override: "用户指定",
};

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
            {trace.steps.map((step) => (
              <div key={step.round} className="rounded-lg border border-slate-200 bg-white p-2.5">
                <div className="flex flex-wrap items-center gap-2 font-medium text-slate-700">
                  <span>第 {step.round} 轮</span>
                  <span className="text-slate-500">{step.query}</span>
                  <span className="chip bg-slate-100 text-[10px]">新增 {step.new_chunk_count} 块</span>
                  {step.need_more && <span className="chip bg-amber-50 text-[10px] text-amber-700">继续检索</span>}
                </div>
                {step.key_entities.length > 0 && <div className="mt-1 text-slate-500">线索：{step.key_entities.join("、")}</div>}
                {step.reason && <div className="mt-1 text-slate-500">{step.reason}</div>}
              </div>
            ))}
          </div>
        )}
        <div className="text-slate-400">停止原因：{trace.stop_reason || "-"} {trace.uncertain ? "· 信息可能不完整" : ""}</div>
      </div>
    </details>
  );
}
