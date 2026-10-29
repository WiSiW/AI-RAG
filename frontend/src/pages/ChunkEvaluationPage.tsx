import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ClipboardCheck,
  GitCompare,
  Play,
  Trash2,
  XCircle,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";
import type { ChunkEvaluationResult, ChunkEvaluationRun } from "../types";

const levels = ["L1", "L2", "L3"] as const;
const levelLabels: Record<string, string> = {
  L1: "L1 章节摘要",
  L2: "L2 段落块",
  L3: "L3 原子块",
};

const statusLabels: Record<string, string> = {
  queued: "排队中",
  running: "评估中",
  partial: "部分完成",
  completed: "已完成",
  failed: "失败",
};

export default function ChunkEvaluationPage() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const initialKb = searchParams.get("kb") ?? "";
  const initialDocument = searchParams.get("document") ?? "";
  const [scope, setScope] = useState<"document" | "knowledge_base">(initialDocument ? "document" : "knowledge_base");
  const [kbId, setKbId] = useState(initialKb);
  const [documentId, setDocumentId] = useState(initialDocument);
  const [selectedLevels, setSelectedLevels] = useState<string[]>(["L1", "L2", "L3"]);
  const [llmEnabled, setLlmEnabled] = useState(true);
  const [judgeProfileId, setJudgeProfileId] = useState("");
  const [selectedRunId, setSelectedRunId] = useState("");
  const [granularityFilter, setGranularityFilter] = useState("");
  const [gradeFilter, setGradeFilter] = useState("");
  const [issueFilter, setIssueFilter] = useState("");
  const [reviewSampleOnly, setReviewSampleOnly] = useState(false);
  const [reviewResult, setReviewResult] = useState<ChunkEvaluationResult | null>(null);
  const [humanScore, setHumanScore] = useState("5");
  const [humanIssues, setHumanIssues] = useState("");
  const [humanComment, setHumanComment] = useState("");
  const [compareLeft, setCompareLeft] = useState("");
  const [compareRight, setCompareRight] = useState("");
  const [comparison, setComparison] = useState<Record<string, unknown> | null>(null);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const modelQuery = useQuery({ queryKey: ["models"], queryFn: () => api.listModels() });
  const docsQuery = useQuery({
    queryKey: ["documents", kbId],
    queryFn: () => api.listDocuments(kbId),
    enabled: !!kbId,
  });
  const runsQuery = useQuery({
    queryKey: ["chunk-evaluations", kbId, documentId, scope],
    queryFn: () => api.listChunkEvaluations({
      knowledgeBaseId: kbId || undefined,
      documentId: scope === "document" && documentId ? documentId : undefined,
    }),
    refetchInterval: (query) => {
      const runs = query.state.data ?? [];
      return runs.some((run) => run.status === "queued" || run.status === "running") ? 2500 : false;
    },
  });
  const runQuery = useQuery({
    queryKey: ["chunk-evaluation", selectedRunId],
    queryFn: () => api.getChunkEvaluation(selectedRunId),
    enabled: !!selectedRunId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "running" ? 2000 : false;
    },
  });
  const resultsQuery = useQuery({
    queryKey: ["chunk-evaluation-results", selectedRunId, granularityFilter, gradeFilter, issueFilter, reviewSampleOnly],
    queryFn: () => api.listChunkEvaluationResults(selectedRunId, {
      granularity: granularityFilter || undefined,
      grade: gradeFilter || undefined,
      issue: issueFilter || undefined,
      reviewSample: reviewSampleOnly ? true : undefined,
      pageSize: 100,
    }),
    enabled: !!selectedRunId,
    refetchInterval: () => {
      const status = runQuery.data?.status;
      return status === "queued" || status === "running" ? 2000 : false;
    },
  });

  const knowledgeBases = kbQuery.data ?? [];
  const models = (modelQuery.data ?? []).filter((model) => model.is_active && ["llm", "enrichment"].includes(model.kind));
  const documents = docsQuery.data ?? [];
  const runs = runsQuery.data ?? [];
  const run = runQuery.data;
  const results = resultsQuery.data?.items ?? [];

  useEffect(() => {
    if (!kbId && knowledgeBases[0]) setKbId(knowledgeBases[0].id);
  }, [kbId, knowledgeBases]);

  useEffect(() => {
    if (!selectedRunId && runs[0]) setSelectedRunId(runs[0].id);
  }, [runs, selectedRunId]);

  useEffect(() => {
    setGranularityFilter("");
    setGradeFilter("");
    setIssueFilter("");
    setReviewSampleOnly(false);
  }, [selectedRunId]);

  useEffect(() => {
    if (!judgeProfileId && models[0]) setJudgeProfileId(models[0].id);
  }, [judgeProfileId, models]);

  useEffect(() => {
    if (selectedRunId) {
      queryClient.invalidateQueries({ queryKey: ["chunk-evaluation-results", selectedRunId] });
    }
  }, [queryClient, runQuery.data?.status, selectedRunId]);

  const createMutation = useMutation({
    mutationFn: () => api.createChunkEvaluation({
      document_id: scope === "document" ? documentId || undefined : undefined,
      knowledge_base_id: scope === "knowledge_base" ? kbId : undefined,
      granularities: selectedLevels,
      judge_profile_id: llmEnabled ? judgeProfileId || undefined : undefined,
      llm_enabled: llmEnabled,
    }),
    onSuccess: async (created) => {
      setSelectedRunId(created.id);
      await queryClient.invalidateQueries({ queryKey: ["chunk-evaluations"] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: api.deleteChunkEvaluation,
    onSuccess: async (_, deletedId) => {
      if (selectedRunId === deletedId) setSelectedRunId("");
      await queryClient.invalidateQueries({ queryKey: ["chunk-evaluations"] });
    },
  });

  const reviewMutation = useMutation({
    mutationFn: () => api.reviewChunkEvaluationResult(selectedRunId, reviewResult!.id, {
      human_score: Number(humanScore),
      human_issue_codes: humanIssues.split(",").map((item) => item.trim()).filter(Boolean),
      human_comment: humanComment,
    }),
    onSuccess: async () => {
      setReviewResult(null);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["chunk-evaluation", selectedRunId] }),
        queryClient.invalidateQueries({ queryKey: ["chunk-evaluation-results", selectedRunId] }),
      ]);
    },
  });

  const compareMutation = useMutation({
    mutationFn: () => api.compareChunkEvaluations([compareLeft, compareRight]),
    onSuccess: setComparison,
  });

  const aggregate = run?.aggregate ?? {};
  const levelScores = useMemo(() => (aggregate.level_scores as Record<string, number> | undefined) ?? {}, [aggregate]);
  const issueCounts = useMemo(() => (aggregate.issue_counts as Record<string, number> | undefined) ?? {}, [aggregate]);
  const calibration = useMemo(() => (aggregate.calibration as Record<string, number | null> | undefined) ?? {}, [aggregate]);
  const finishedRuns = runs.filter((item) => item.status === "completed" || item.status === "partial");

  function toggleLevel(level: string) {
    setSelectedLevels((current) => current.includes(level) ? current.filter((item) => item !== level) : [...current, level]);
  }

  function openReview(item: ChunkEvaluationResult) {
    setReviewResult(item);
    setHumanScore(String(item.human_score ?? 5));
    setHumanIssues(item.human_issue_codes.join(", "));
    setHumanComment(item.human_comment);
  }

  if (kbQuery.isLoading || modelQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;

  return (
    <div>
      <PageHeader
        title="分块语义完整性评估"
        description="规则检测、LLM 语义评判与人工复核结合，定位断句、上下文依赖、源内容遗漏和元数据问题。"
      />

      <div className="grid gap-5 xl:grid-cols-[380px_minmax(0,1fr)]">
        <section className="panel h-fit p-5">
          <div className="font-semibold text-slate-900">创建评估</div>
          <div className="mt-4 space-y-4">
            <div>
              <label className="label">评估范围</label>
              <select className="input" value={scope} onChange={(event) => setScope(event.target.value as "document" | "knowledge_base")}>
                <option value="knowledge_base">整个知识库</option>
                <option value="document">单个文档</option>
              </select>
            </div>
            <div>
              <label className="label">知识库</label>
              <select className="input" value={kbId} onChange={(event) => { setKbId(event.target.value); setDocumentId(""); setSelectedRunId(""); }} required>
                <option value="">请选择</option>
                {knowledgeBases.map((kb) => <option key={kb.id} value={kb.id}>{kb.name}</option>)}
              </select>
            </div>
            {scope === "document" && (
              <div>
                <label className="label">文档</label>
                <select className="input" value={documentId} onChange={(event) => setDocumentId(event.target.value)} required>
                  <option value="">请选择</option>
                  {documents.filter((doc) => doc.status === "ready").map((doc) => <option key={doc.id} value={doc.id}>{doc.filename}</option>)}
                </select>
              </div>
            )}
            <div>
              <label className="label">评估粒度</label>
              <div className="flex gap-2">
                {levels.map((level) => (
                  <button key={level} type="button" onClick={() => toggleLevel(level)} className={`chip ${selectedLevels.includes(level) ? "bg-brand-100 text-brand-800" : "bg-slate-100 text-slate-500"}`}>
                    {level}
                  </button>
                ))}
              </div>
            </div>
            <label className="flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-3 text-sm text-slate-700">
              <input type="checkbox" checked={llmEnabled} onChange={(event) => setLlmEnabled(event.target.checked)} />
              启用 LLM 语义评判
            </label>
            {llmEnabled && (
              <div>
                <label className="label">评判模型</label>
                <select className="input" value={judgeProfileId} onChange={(event) => setJudgeProfileId(event.target.value)}>
                  <option value="">使用知识库默认增强模型</option>
                  {models.map((model) => <option key={model.id} value={model.id}>{model.name} · {model.kind}</option>)}
                </select>
              </div>
            )}
            {(createMutation.error || deleteMutation.error) && <ErrorState message={(createMutation.error ?? deleteMutation.error)!.message} />}
            <button
              type="button"
              onClick={() => createMutation.mutate()}
              disabled={createMutation.isPending || !kbId || !selectedLevels.length || (scope === "document" && !documentId)}
              className="btn-primary w-full"
            >
              <Play className="h-4 w-4" />{createMutation.isPending ? "创建中..." : "开始评估"}
            </button>
          </div>
        </section>

        <section className="space-y-5">
          <div className="panel overflow-hidden">
            <div className="border-b border-slate-100 px-5 py-4 font-semibold text-slate-900">评估历史</div>
            {runsQuery.error ? <ErrorState message={runsQuery.error.message} /> : runsQuery.isLoading ? <LoadingState /> : !runs.length ? (
              <EmptyState title="暂无评估记录" description="选择范围后开始第一次分块完整性评估。" />
            ) : (
              <div className="divide-y divide-slate-100">
                {runs.map((item) => (
                  <button key={item.id} type="button" onClick={() => setSelectedRunId(item.id)} className={`flex w-full flex-col gap-2 px-5 py-4 text-left transition hover:bg-slate-50 sm:flex-row sm:items-center ${selectedRunId === item.id ? "bg-brand-50/60" : ""}`}>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium text-slate-900">{new Date(item.created_at).toLocaleString()}</span>
                        <StatusChip status={item.status} />
                        <span className="chip bg-slate-100 text-[11px] text-slate-600">{item.scope_type === "document" ? "文档" : "知识库"}</span>
                      </div>
                      <div className="mt-1 text-xs text-slate-500">粒度 {item.granularities.join(" / ")} · {item.llm_enabled ? "LLM + 规则" : "仅规则"}</div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right"><div className="text-xs text-slate-400">综合分</div><div className="font-mono text-lg font-semibold text-slate-800">{score(item.aggregate.overall_score)}</div></div>
                      <Trash2 className="h-4 w-4 text-slate-300" onClick={(event) => { event.stopPropagation(); if (window.confirm("确认删除该评估记录？")) deleteMutation.mutate(item.id); }} />
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>

          {run && <RunSummary run={run} levelScores={levelScores} issueCounts={issueCounts} calibration={calibration} />}

          {run && (
            <div className="panel overflow-hidden">
              <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="font-semibold text-slate-900">Chunk 结果</div>
                <div className="flex flex-wrap gap-2">
                  <select className="input w-auto" value={granularityFilter} onChange={(event) => setGranularityFilter(event.target.value)}>
                    <option value="">全部粒度</option>{levels.map((level) => <option key={level} value={level}>{level}</option>)}
                  </select>
                  <select className="input w-auto" value={gradeFilter} onChange={(event) => setGradeFilter(event.target.value)}>
                    <option value="">全部等级</option><option value="pass">通过</option><option value="warn">警告</option><option value="fail">失败</option>
                  </select>
                  <select className="input w-auto" value={issueFilter} onChange={(event) => setIssueFilter(event.target.value)}>
                    <option value="">全部问题</option>{Object.keys(issueCounts).map((issue) => <option key={issue} value={issue}>{issue}</option>)}
                  </select>
                  <label className="flex items-center gap-1 text-xs text-slate-600"><input type="checkbox" checked={reviewSampleOnly} onChange={(event) => setReviewSampleOnly(event.target.checked)} />仅人工样本</label>
                </div>
              </div>
              {resultsQuery.error ? <ErrorState message={resultsQuery.error.message} /> : resultsQuery.isLoading ? <LoadingState /> : !results.length ? <EmptyState title="没有匹配结果" description="调整筛选条件或等待评估完成。" /> : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead className="bg-slate-50 text-left text-xs text-slate-500"><tr><th className="px-4 py-3">粒度</th><th className="px-4 py-3">章节 / 内容</th><th className="px-4 py-3">分数</th><th className="px-4 py-3">问题</th><th className="px-4 py-3">复核</th></tr></thead>
                    <tbody className="divide-y divide-slate-100">
                      {results.map((item) => (
                        <tr key={item.id} className="align-top hover:bg-slate-50">
                          <td className="px-4 py-3"><div className="font-semibold text-slate-800">{item.granularity_level}</div><div className="text-[11px] text-slate-400">{item.granule_type}</div></td>
                          <td className="max-w-xl px-4 py-3"><div className="truncate text-xs text-brand-700">{item.section_path}</div><div className="mt-1 line-clamp-2 text-xs leading-5 text-slate-600">{item.content_snapshot}</div></td>
                          <td className="px-4 py-3"><GradeBadge grade={item.grade} score={item.composite_score} /></td>
                          <td className="max-w-md px-4 py-3"><div className="flex flex-wrap gap-1">{item.issue_codes.slice(0, 4).map((issue) => <span key={issue} className="chip bg-amber-50 text-[10px] text-amber-700">{issue}</span>)}</div></td>
                          <td className="px-4 py-3"><button type="button" onClick={() => openReview(item)} className="btn-secondary px-2 py-1 text-xs"><ClipboardCheck className="h-3 w-3" />{item.human_score ? `已评 ${item.human_score}` : "复核"}</button></td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          <div className="panel p-5">
            <div className="flex items-center gap-2 font-semibold text-slate-900"><GitCompare className="h-4 w-4 text-brand-600" />历史运行对比</div>
            <div className="mt-3 grid gap-3 sm:grid-cols-[1fr_1fr_auto]">
              <select className="input" value={compareLeft} onChange={(event) => setCompareLeft(event.target.value)}><option value="">基准运行</option>{finishedRuns.map((item) => <option key={item.id} value={item.id}>{runLabel(item)}</option>)}</select>
              <select className="input" value={compareRight} onChange={(event) => setCompareRight(event.target.value)}><option value="">对比运行</option>{finishedRuns.map((item) => <option key={item.id} value={item.id}>{runLabel(item)}</option>)}</select>
              <button type="button" className="btn-secondary" disabled={!compareLeft || !compareRight || compareLeft === compareRight || compareMutation.isPending} onClick={() => compareMutation.mutate()}>开始对比</button>
            </div>
            {compareMutation.error && <div className="mt-3"><ErrorState message={compareMutation.error.message} /></div>}
            {comparison && <ComparisonView data={comparison} />}
          </div>
        </section>
      </div>

      <Modal open={!!reviewResult} onClose={() => setReviewResult(null)} title="人工复核" wide>
        {reviewResult && (
          <div className="space-y-4">
            <div className="rounded-xl bg-slate-50 p-4"><div className="text-xs text-slate-400">{reviewResult.granularity_level} · {reviewResult.section_path}</div><div className="mt-2 whitespace-pre-wrap text-sm leading-6 text-slate-700">{reviewResult.content_snapshot}</div></div>
            <div className="grid gap-3 lg:grid-cols-2">
              <ContextBlock label="章节源文本" value={String(reviewResult.context_snapshot.section_source ?? "")} />
              <ContextBlock label="父文档局部上下文" value={String(reviewResult.context_snapshot.parent_excerpt ?? "")} />
              <ContextBlock label="前一段" value={String(reviewResult.context_snapshot.previous_excerpt ?? "")} />
              <ContextBlock label="后一段" value={String(reviewResult.context_snapshot.next_excerpt ?? "")} />
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div><label className="label">人工评分</label><select className="input" value={humanScore} onChange={(event) => setHumanScore(event.target.value)}>{[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value} 分</option>)}</select></div>
              <div><label className="label">问题标签（逗号分隔）</label><input className="input" value={humanIssues} onChange={(event) => setHumanIssues(event.target.value)} placeholder="broken_sentence, missing_parent" /></div>
            </div>
            <div><label className="label">备注</label><textarea className="input min-h-24" value={humanComment} onChange={(event) => setHumanComment(event.target.value)} /></div>
            <RuleEvidence metrics={reviewResult.rule_metrics} llmScores={reviewResult.llm_scores} />
            <div className="rounded-xl border border-slate-200 p-3 text-xs leading-5 text-slate-500"><div>自动分：{score(reviewResult.composite_score)} · {reviewResult.grade}</div><div>LLM 理由：{reviewResult.llm_reason || "-"}</div>{reviewResult.suggestions.length > 0 && <div>建议：{reviewResult.suggestions.join("；")}</div>}</div>
            {reviewMutation.error && <ErrorState message={reviewMutation.error.message} />}
            <div className="flex justify-end gap-2"><button type="button" className="btn-secondary" onClick={() => setReviewResult(null)}>取消</button><button type="button" className="btn-primary" disabled={reviewMutation.isPending} onClick={() => reviewMutation.mutate()}>{reviewMutation.isPending ? "保存中..." : "保存复核"}</button></div>
          </div>
        )}
      </Modal>
    </div>
  );
}

function RunSummary({ run, levelScores, issueCounts, calibration }: { run: ChunkEvaluationRun; levelScores: Record<string, number>; issueCounts: Record<string, number>; calibration: Record<string, number | null> }) {
  const aggregate = run.aggregate;
  const progress = run.progress ?? {};
  const gradeCounts = (aggregate.grade_counts as Record<string, number> | undefined) ?? {};
  return (
    <div className="panel p-5">
      <div className="flex flex-wrap items-center justify-between gap-3"><div><div className="font-semibold text-slate-900">运行摘要</div><div className="mt-1 text-xs text-slate-500">{statusLabels[run.status] ?? run.status} · 处理 {String(progress.processed_documents ?? 0)}/{String(progress.total_documents ?? 0)} 文档 · {String(progress.evaluated_chunks ?? aggregate.result_count ?? 0)} 个 chunk</div></div><StatusChip status={run.status} /></div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
        <Metric label="综合分" value={score(aggregate.overall_score)} />
        {levels.map((level) => <Metric key={level} label={levelLabels[level]} value={score(levelScores[level])} />)}
        <Metric label="人工校准 MAE" value={calibration.mae == null ? "-" : String(calibration.mae)} />
        <Metric label="人工相关性" value={calibration.correlation == null ? "-" : String(calibration.correlation)} />
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500"><span>通过 {gradeCounts.pass ?? 0}</span><span>警告 {gradeCounts.warn ?? 0}</span><span>失败 {gradeCounts.fail ?? 0}</span><span>人工样本 {String(aggregate.review_sample_count ?? 0)} 个</span><span>已复核 {String(aggregate.reviewed_count ?? 0)} 个</span></div>
      {!!Object.keys(issueCounts).length && <div className="mt-3 flex flex-wrap gap-1">{Object.entries(issueCounts).slice(0, 12).map(([issue, count]) => <span key={issue} className="chip bg-amber-50 text-[11px] text-amber-700">{issue} · {count}</span>)}</div>}
      {run.errors.length > 0 && <div className="mt-3 rounded-xl bg-red-50 p-3 text-xs text-red-700">{run.errors.map((error) => String(error.error ?? JSON.stringify(error))).join("；")}</div>}
    </div>
  );
}

function ComparisonView({ data }: { data: Record<string, unknown> }) {
  const levelDeltas = (data.level_deltas as Record<string, number> | undefined) ?? {};
  const issueDeltas = (data.issue_deltas as Record<string, number> | undefined) ?? {};
  const regressions = (data.regressions as Array<Record<string, unknown>> | undefined) ?? [];
  return <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4"><div className="flex items-center gap-2 font-medium text-slate-800"><BarChart3 className="h-4 w-4" />综合分变化 <span className={Number(data.overall_delta) >= 0 ? "text-emerald-700" : "text-red-700"}>{Number(data.overall_delta) > 0 ? "+" : ""}{String(data.overall_delta ?? 0)}</span></div><div className="mt-2 flex flex-wrap gap-2">{Object.entries(levelDeltas).map(([level, delta]) => <span key={level} className="chip bg-white text-[11px] text-slate-600">{level}: {delta > 0 ? "+" : ""}{delta}</span>)}</div>{!!Object.keys(issueDeltas).length && <div className="mt-2 flex flex-wrap gap-2">{Object.entries(issueDeltas).filter(([, delta]) => delta !== 0).map(([issue, delta]) => <span key={issue} className={`chip text-[11px] ${delta > 0 ? "bg-red-50 text-red-700" : "bg-emerald-50 text-emerald-700"}`}>{issue}: {delta > 0 ? "+" : ""}{delta}</span>)}</div>}{regressions.length > 0 && <div className="mt-3 space-y-1 text-xs text-red-700">{regressions.slice(0, 5).map((item) => <div key={String(item.key)}>{String(item.section_path)} · {String(item.delta)}</div>)}</div>}</div>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-slate-50 p-3"><div className="text-xs text-slate-400">{label}</div><div className="mt-1 font-mono text-lg font-semibold text-slate-800">{value}</div></div>;
}

function ContextBlock({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl border border-slate-200 p-3"><div className="text-xs font-medium text-slate-500">{label}</div><div className="mt-1 max-h-40 overflow-y-auto whitespace-pre-wrap text-xs leading-5 text-slate-600">{value || "-"}</div></div>;
}

function RuleEvidence({ metrics, llmScores }: { metrics: Record<string, unknown>; llmScores: Record<string, unknown> }) {
  const scores = (llmScores.scores as Record<string, number> | undefined) ?? {};
  return <div className="rounded-xl border border-slate-200 p-3"><div className="text-xs font-medium text-slate-500">规则与语义证据</div><div className="mt-2 grid gap-2 text-xs text-slate-600 sm:grid-cols-3 lg:grid-cols-5"><span>边界 {score(metrics.boundary_score)}</span><span>源覆盖 {score(metrics.source_coverage)}</span><span>尺寸 {score(metrics.size_compliance)}</span><span>元数据 {score(metrics.metadata_score)}</span><span>结构 {score(metrics.structure_score)}</span></div>{!!Object.keys(scores).length && <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-500">{Object.entries(scores).map(([key, value]) => <span key={key} className="chip bg-slate-100">{key}: {value}/5</span>)}</div>}</div>;
}

function GradeBadge({ grade, score: value }: { grade: string; score: number }) {
  const Icon = grade === "pass" ? CheckCircle2 : grade === "warn" ? AlertTriangle : XCircle;
  const className = grade === "pass" ? "bg-emerald-50 text-emerald-700" : grade === "warn" ? "bg-amber-50 text-amber-700" : "bg-red-50 text-red-700";
  return <span className={`chip gap-1 ${className}`}><Icon className="h-3 w-3" />{value.toFixed(1)}</span>;
}

function StatusChip({ status }: { status: string }) {
  const className = status === "completed" ? "bg-emerald-50 text-emerald-700" : status === "failed" ? "bg-red-50 text-red-700" : status === "partial" ? "bg-amber-50 text-amber-700" : "bg-sky-50 text-sky-700";
  return <span className={`chip text-[11px] ${className}`}>{statusLabels[status] ?? status}</span>;
}

function score(value: unknown): string {
  return typeof value === "number" ? value.toFixed(2) : "-";
}

function runLabel(run: ChunkEvaluationRun): string {
  return `${new Date(run.created_at).toLocaleString()} · ${score(run.aggregate.overall_score)} 分`;
}
