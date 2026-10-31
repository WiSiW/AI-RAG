import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Download,
  FlaskConical,
  Play,
  RefreshCcw,
  Save,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, Modal, PageHeader } from "../components/UI";
import type {
  ChunkTuningCaseResult,
  ChunkTuningRun,
  ChunkTuningTrial,
  DocumentItem,
  EvaluationQuestionType,
  RetrievalEvaluationCase,
  RetrievalEvaluationDataset,
  RetrievalEvaluationEvidence,
} from "../types";

type WorkspaceTab = "dataset" | "screening" | "analysis" | "verification";

const TABS: Array<{ id: WorkspaceTab; label: string }> = [
  { id: "dataset", label: "评测集" },
  { id: "screening", label: "快速筛选" },
  { id: "analysis", label: "结果分析" },
  { id: "verification", label: "深度验证" },
];

const QUESTION_TYPES: Array<{ value: EvaluationQuestionType; label: string }> = [
  { value: "detail", label: "细节查询" },
  { value: "comparison", label: "比较" },
  { value: "summary", label: "总结" },
  { value: "reasoning", label: "推理" },
  { value: "table", label: "表格" },
  { value: "code", label: "代码" },
  { value: "no_answer", label: "无答案" },
];

const statusLabels: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  awaiting_selection: "等待选择",
  partial: "部分完成",
  completed: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function parseNumberList(value: string): number[] {
  return value
    .split(/[,，\s]+/)
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
}

function metric(value: number | undefined, digits = 3): string {
  return value === undefined || Number.isNaN(value) ? "-" : value.toFixed(digits);
}

function percent(value: number | undefined): string {
  return value === undefined || Number.isNaN(value) ? "-" : `${(value * 100).toFixed(1)}%`;
}

function runLabel(run: ChunkTuningRun): string {
  const stage = run.stage === "screening" ? "快速筛选" : "深度验证";
  return `${stage} · ${new Date(run.created_at).toLocaleString()} · ${statusLabels[run.status] ?? run.status}`;
}

export default function ChunkTuningPage() {
  const queryClient = useQueryClient();
  const [searchParams] = useSearchParams();
  const initialKb = searchParams.get("kb") ?? "";
  const [kbId, setKbId] = useState(initialKb);
  const [datasetId, setDatasetId] = useState("");
  const [runId, setRunId] = useState("");
  const [trialId, setTrialId] = useState("");
  const [tab, setTab] = useState<WorkspaceTab>("dataset");
  const [showDatasetModal, setShowDatasetModal] = useState(false);
  const [showCaseModal, setShowCaseModal] = useState(false);
  const [editingCase, setEditingCase] = useState<RetrievalEvaluationCase | null>(null);
  const [selectedEvidence, setSelectedEvidence] = useState<RetrievalEvaluationEvidence[]>([]);
  const [selectedCandidates, setSelectedCandidates] = useState<string[]>([]);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const datasetsQuery = useQuery({
    queryKey: ["retrieval-evaluation-datasets", kbId],
    queryFn: () => api.listRetrievalEvaluationDatasets(kbId || undefined),
  });
  const documentsQuery = useQuery({
    queryKey: ["documents", kbId],
    queryFn: () => api.listDocuments(kbId),
    enabled: !!kbId,
  });
  const runsQuery = useQuery({
    queryKey: ["chunk-tuning-runs", kbId, datasetId],
    queryFn: () =>
      api.listChunkTuningRuns({
        knowledgeBaseId: kbId || undefined,
        datasetId: datasetId || undefined,
      }),
    refetchInterval: (query) =>
      (query.state.data ?? []).some((run) => ["queued", "running"].includes(run.status))
        ? 2500
        : false,
  });
  const runQuery = useQuery({
    queryKey: ["chunk-tuning-run", runId],
    queryFn: () => api.getChunkTuningRun(runId),
    enabled: !!runId,
    refetchInterval: (query) =>
      ["queued", "running"].includes(query.state.data?.status ?? "") ? 2000 : false,
  });
  const resultsQuery = useQuery({
    queryKey: ["chunk-tuning-results", runId, trialId],
    queryFn: () => api.listChunkTuningResults(runId, { trialId: trialId || undefined, pageSize: 100 }),
    enabled: !!runId && !!trialId && (tab === "analysis" || tab === "verification"),
  });

  const knowledgeBases = kbQuery.data ?? [];
  const datasets = datasetsQuery.data ?? [];
  const documents = documentsQuery.data ?? [];
  const runs = runsQuery.data ?? [];
  const dataset = datasets.find((item) => item.id === datasetId);
  const run = runQuery.data;
  const currentTrial = run?.trials.find((item) => item.id === trialId);
  const screeningRuns = runs.filter((item) => item.stage === "screening");
  const verificationRuns = runs.filter((item) => item.stage === "verification");

  useEffect(() => {
    if (!kbId && knowledgeBases[0]) setKbId(knowledgeBases[0].id);
  }, [kbId, knowledgeBases]);

  useEffect(() => {
    if (!datasets.some((item) => item.id === datasetId)) {
      setDatasetId(datasets[0]?.id ?? "");
      setRunId("");
    }
  }, [datasetId, datasets]);

  useEffect(() => {
    setSelectedEvidence([]);
  }, [datasetId]);

  useEffect(() => {
    if (!runId && runs[0]) setRunId(runs[0].id);
  }, [runId, runs]);

  useEffect(() => {
    setTrialId(run?.trials[0]?.id ?? "");
    setSelectedCandidates([]);
  }, [run?.id]);

  const invalidateDatasets = () =>
    queryClient.invalidateQueries({ queryKey: ["retrieval-evaluation-datasets", kbId] });

  const createDatasetMutation = useMutation({
    mutationFn: api.createRetrievalEvaluationDataset,
    onSuccess: async (created) => {
      await invalidateDatasets();
      setDatasetId(created.id);
      setShowDatasetModal(false);
      setTab("dataset");
    },
  });
  const deleteDatasetMutation = useMutation({
    mutationFn: api.deleteRetrievalEvaluationDataset,
    onSuccess: async () => {
      setDatasetId("");
      setRunId("");
      await invalidateDatasets();
    },
  });
  const createCaseMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: Record<string, unknown> }) =>
      api.createRetrievalEvaluationCase(id, payload),
    onSuccess: async () => {
      setShowCaseModal(false);
      setSelectedEvidence([]);
      await invalidateDatasets();
    },
  });
  const updateCaseMutation = useMutation({
    mutationFn: ({ datasetId: id, caseId, payload }: { datasetId: string; caseId: string; payload: Record<string, unknown> }) =>
      api.updateRetrievalEvaluationCase(id, caseId, payload),
    onSuccess: async () => {
      setShowCaseModal(false);
      setEditingCase(null);
      setSelectedEvidence([]);
      await invalidateDatasets();
    },
  });
  const deleteCaseMutation = useMutation({
    mutationFn: ({ datasetId: id, caseId }: { datasetId: string; caseId: string }) =>
      api.deleteRetrievalEvaluationCase(id, caseId),
    onSuccess: invalidateDatasets,
  });
  const importMutation = useMutation({
    mutationFn: ({ id, file }: { id: string; file: File }) =>
      api.importRetrievalEvaluationCases(id, file),
    onSuccess: invalidateDatasets,
  });
  const createRunMutation = useMutation({
    mutationFn: api.createChunkTuningRun,
    onSuccess: async (created) => {
      setRunId(created.id);
      setTab("screening");
      await queryClient.invalidateQueries({ queryKey: ["chunk-tuning-runs", kbId, datasetId] });
    },
  });
  const verificationMutation = useMutation({
    mutationFn: ({ id, trialIds }: { id: string; trialIds: string[] }) =>
      api.createChunkTuningVerification(id, trialIds),
    onSuccess: async (created) => {
      setRunId(created.id);
      setTab("verification");
      await queryClient.invalidateQueries({ queryKey: ["chunk-tuning-runs", kbId, datasetId] });
    },
  });
  const cancelMutation = useMutation({
    mutationFn: api.cancelChunkTuningRun,
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["chunk-tuning-runs", kbId, datasetId] });
      await queryClient.invalidateQueries({ queryKey: ["chunk-tuning-run", runId] });
    },
  });
  const deleteRunMutation = useMutation({
    mutationFn: api.deleteChunkTuningRun,
    onSuccess: async () => {
      setRunId("");
      await queryClient.invalidateQueries({ queryKey: ["chunk-tuning-runs", kbId, datasetId] });
    },
  });
  const applyMutation = useMutation({
    mutationFn: ({ id, selectedTrialId }: { id: string; selectedTrialId: string }) =>
      api.applyChunkTuningTrial(id, selectedTrialId),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["knowledge-bases"] });
    },
  });
  const reindexMutation = useMutation({
    mutationFn: ({ id, trial }: { id: string; trial: ChunkTuningTrial }) =>
      api.reindexDocumentsWithChunkSettings(id, {
        document_ids: dataset?.document_ids ?? [],
        chunk_mode: trial.chunk_mode,
        chunk_size: trial.chunk_size,
        chunk_overlap: trial.chunk_overlap,
      }),
  });

  function selectDataset(value: string) {
    setDatasetId(value);
    setRunId("");
    setTrialId("");
  }

  function toggleCandidate(id: string) {
    setSelectedCandidates((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      if (current.length >= 5) return current;
      return [...current, id];
    });
  }

  async function exportDataset() {
    if (!dataset) return;
    const blob = await api.exportRetrievalEvaluationCases(dataset.id);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${dataset.name}.jsonl`;
    link.click();
    URL.revokeObjectURL(url);
  }

  if (kbQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;

  return (
    <div>
      <PageHeader
        title="分块调优"
        description="用固定评测集和原文黄金证据比较分块大小与重叠窗口。实验在隔离索引中运行，不会改动生产文档和索引。"
        action={
          <button type="button" className="btn-primary" onClick={() => setShowDatasetModal(true)} disabled={!kbId}>
            <FlaskConical className="h-4 w-4" />新建评测集
          </button>
        }
      />

      <div className="panel mb-5 grid gap-4 p-4 lg:grid-cols-[minmax(220px,0.7fr)_minmax(260px,1fr)_auto]">
        <div>
          <label className="label">知识库</label>
          <select
            className="input"
            value={kbId}
            onChange={(event) => {
              setKbId(event.target.value);
              setDatasetId("");
              setRunId("");
            }}
          >
            {knowledgeBases.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
          </select>
        </div>
        <div>
          <label className="label">评测集</label>
          <select className="input" value={datasetId} onChange={(event) => selectDataset(event.target.value)}>
            <option value="">请选择评测集</option>
            {datasets.map((item) => (
              <option key={item.id} value={item.id}>
                {item.name} · v{item.version} · {item.case_count} 题
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-end gap-2">
          <button
            type="button"
            className="btn-secondary"
            disabled={!dataset}
            onClick={() => {
              setEditingCase(null);
              setSelectedEvidence([]);
              setShowCaseModal(true);
            }}
          >
            添加问题
          </button>
          {dataset && (
            <button type="button" className="btn-secondary" onClick={exportDataset}>
              <Download className="h-4 w-4" />导出
            </button>
          )}
        </div>
      </div>

      <div className="mb-5 flex flex-wrap gap-2">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={tab === item.id ? "btn-primary" : "btn-secondary"}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </div>

      {!dataset && (
        <div className="panel">
          <EmptyState
            title="先创建一个评测集"
            description="选择代表性文档，录入真实问题、标准答案，并从原文划选最小黄金证据。"
          />
        </div>
      )}

      {dataset && tab === "dataset" && (
        <DatasetWorkspace
          dataset={dataset}
          documents={documents}
          onDeleteDataset={() => {
            if (window.confirm("删除评测集？历史调优运行的指标快照仍会保留。")) {
              deleteDatasetMutation.mutate(dataset.id);
            }
          }}
          onDeleteCase={(caseId) => deleteCaseMutation.mutate({ datasetId: dataset.id, caseId })}
          onEditCase={(item) => {
            setEditingCase(item);
            setSelectedEvidence(item.evidence);
            setShowCaseModal(true);
          }}
          onImport={(file) => importMutation.mutate({ id: dataset.id, file })}
          importPending={importMutation.isPending}
          error={
            deleteDatasetMutation.error?.message ??
            deleteCaseMutation.error?.message ??
            importMutation.error?.message ??
            updateCaseMutation.error?.message
          }
        />
      )}

      {dataset && tab === "screening" && (
        <ScreeningWorkspace
          dataset={dataset}
          runs={screeningRuns}
          run={run}
          selectedRunId={runId}
          onSelectRun={setRunId}
          onStart={(sizes, ratios) => createRunMutation.mutate({
            knowledge_base_id: kbId,
            dataset_id: dataset.id,
            chunk_sizes: sizes,
            overlap_ratios: ratios,
          })}
          startPending={createRunMutation.isPending}
          startError={createRunMutation.error?.message}
          onCancel={() => run && cancelMutation.mutate(run.id)}
        />
      )}

      {dataset && tab === "analysis" && (
        <AnalysisWorkspace
          run={run}
          selectedTrialId={trialId}
          onSelectTrial={setTrialId}
          results={resultsQuery.data?.items ?? []}
          selectedCandidates={selectedCandidates}
          onToggleCandidate={toggleCandidate}
          onStartVerification={() => {
            if (!run || !selectedCandidates.length) return;
            verificationMutation.mutate({ id: run.id, trialIds: selectedCandidates });
          }}
          verificationPending={verificationMutation.isPending}
          verificationError={verificationMutation.error?.message}
          onDeleteRun={() => run && deleteRunMutation.mutate(run.id)}
        />
      )}

      {dataset && tab === "verification" && (
        <VerificationWorkspace
          runs={verificationRuns}
          run={run}
          selectedRunId={runId}
          onSelectRun={setRunId}
          selectedTrialId={trialId}
          onSelectTrial={setTrialId}
          results={resultsQuery.data?.items ?? []}
          currentTrial={currentTrial}
          onApply={() => {
            if (!run || !currentTrial) return;
            if (window.confirm("将该参数保存为知识库默认值？现有文档不会自动重建。")) {
              applyMutation.mutate({ id: run.id, selectedTrialId: currentTrial.id });
            }
          }}
          onReindex={() => {
            if (!currentTrial) return;
            if (window.confirm("按当前 trial 参数重建评测集内的现有文档？这会替换其生产分块和索引。")) {
              reindexMutation.mutate({ id: kbId, trial: currentTrial });
            }
          }}
          applyPending={applyMutation.isPending}
          reindexPending={reindexMutation.isPending}
          applyMessage={applyMutation.isSuccess ? "默认参数已保存。" : undefined}
          reindexMessage={
            reindexMutation.data
              ? `已排队 ${reindexMutation.data.queued} 篇，跳过 ${reindexMutation.data.skipped} 篇。`
              : undefined
          }
        />
      )}

      <DatasetModal
        open={showDatasetModal}
        documents={documents.filter((item) => item.status === "ready")}
        pending={createDatasetMutation.isPending}
        error={createDatasetMutation.error?.message}
        onClose={() => setShowDatasetModal(false)}
        onSubmit={(payload) => createDatasetMutation.mutate({ ...payload, knowledge_base_id: kbId })}
      />

      {dataset && (
        <CaseModal
          open={showCaseModal}
          dataset={dataset}
          documents={documents.filter((item) => dataset.document_ids.includes(item.id))}
          selectedEvidence={selectedEvidence}
          setSelectedEvidence={setSelectedEvidence}
          initialCase={editingCase}
          pending={createCaseMutation.isPending || updateCaseMutation.isPending}
          error={createCaseMutation.error?.message ?? updateCaseMutation.error?.message}
          onClose={() => {
            setShowCaseModal(false);
            setEditingCase(null);
          }}
          onSubmit={(payload) => {
            if (editingCase) {
              updateCaseMutation.mutate({ datasetId: dataset.id, caseId: editingCase.id, payload });
            } else {
              createCaseMutation.mutate({ id: dataset.id, payload });
            }
          }}
        />
      )}
    </div>
  );
}

function statusClass(status: string): string {
  if (status === "completed") return "bg-emerald-50 text-emerald-700";
  if (status === "partial" || status === "awaiting_selection") return "bg-amber-50 text-amber-700";
  if (status === "failed" || status === "cancelled") return "bg-red-50 text-red-700";
  return "bg-sky-50 text-sky-700";
}

function DatasetWorkspace({
  dataset,
  documents,
  onDeleteDataset,
  onDeleteCase,
  onEditCase,
  onImport,
  importPending,
  error,
}: {
  dataset: RetrievalEvaluationDataset;
  documents: DocumentItem[];
  onDeleteDataset: () => void;
  onDeleteCase: (caseId: string) => void;
  onEditCase: (item: RetrievalEvaluationCase) => void;
  onImport: (file: File) => void;
  importPending: boolean;
  error?: string;
}) {
  const documentNames = useMemo(
    () => Object.fromEntries(documents.map((item) => [item.id, item.filename])),
    [documents],
  );
  return (
    <div className="space-y-4">
      {error && <ErrorState message={error} />}
      <div className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-lg font-semibold text-slate-900">
              <ClipboardList className="h-5 w-5 text-brand-600" />
              {dataset.name}
            </div>
            <p className="mt-1 text-sm text-slate-500">{dataset.description || "暂无说明"}</p>
            <div className="mt-3 flex flex-wrap gap-2 text-xs">
              <span className="chip bg-slate-100 text-slate-600">版本 v{dataset.version}</span>
              <span className="chip bg-brand-50 text-brand-700">{dataset.document_ids.length} 篇文档</span>
              <span className="chip bg-emerald-50 text-emerald-700">{dataset.answerable_count} 条可回答</span>
              <span className="chip bg-slate-100 text-slate-600">{dataset.case_count - dataset.answerable_count} 条无答案</span>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <label className="btn-secondary cursor-pointer">
              <Upload className="h-4 w-4" />{importPending ? "导入中" : "导入 JSONL"}
              <input
                className="hidden"
                type="file"
                accept=".jsonl,.ndjson,application/x-ndjson"
                disabled={importPending}
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) onImport(file);
                  event.target.value = "";
                }}
              />
            </label>
            <button type="button" className="btn-secondary text-red-600" onClick={onDeleteDataset}>
              <Trash2 className="h-4 w-4" />删除评测集
            </button>
          </div>
        </div>
        {dataset.answerable_count < 20 && (
          <div className="mt-4 rounded-xl bg-amber-50 p-3 text-sm text-amber-800">
            至少需要 20 条可回答问题才能启动实验；建议准备 50–100 条，并按细节、比较、总结、推理、表格和代码分层。
          </div>
        )}
        {dataset.answerable_count >= 20 && dataset.answerable_count < 50 && (
          <div className="mt-4 rounded-xl bg-sky-50 p-3 text-sm text-sky-800">
            当前已有 {dataset.answerable_count} 条可回答问题，可先试跑；正式决策建议补足到 50 条以上。
          </div>
        )}
        <div className="mt-4 flex flex-wrap gap-2">
          {dataset.document_ids.map((id) => (
            <span key={id} className="chip bg-white text-slate-600">{documentNames[id] ?? id}</span>
          ))}
        </div>
      </div>

      <div className="panel overflow-hidden">
        <div className="border-b border-slate-100 px-5 py-4 font-semibold text-slate-900">评测问题</div>
        {!dataset.cases.length ? (
          <EmptyState title="还没有评测问题" description="添加真实问题、标准答案和原文黄金证据后才能开始调优。" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] text-left text-sm">
              <thead className="bg-slate-50 text-xs text-slate-500">
                <tr>
                  <th className="px-4 py-3">问题</th>
                  <th className="px-4 py-3">类型</th>
                  <th className="px-4 py-3">标准答案</th>
                  <th className="px-4 py-3">黄金证据</th>
                  <th className="px-4 py-3">文档</th>
                  <th className="px-4 py-3 text-right">操作</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {dataset.cases.map((item) => (
                  <tr key={item.id} className="align-top hover:bg-slate-50/70">
                    <td className="max-w-sm px-4 py-3 font-medium text-slate-800">{item.question}</td>
                    <td className="px-4 py-3"><span className="chip bg-slate-100 text-slate-600">{item.question_type}</span></td>
                    <td className="max-w-md px-4 py-3 text-slate-600">{item.reference_answer || "-"}</td>
                    <td className="max-w-sm px-4 py-3">
                      {item.evidence.length ? item.evidence.map((evidence) => (
                        <details key={evidence.id} className="mb-1 rounded-lg bg-amber-50 px-2 py-1 text-xs text-amber-900">
                          <summary className="cursor-pointer">{evidence.section_path} · {evidence.quote.slice(0, 36)}</summary>
                          <div className="mt-1 whitespace-pre-wrap leading-5">{evidence.quote}</div>
                        </details>
                      )) : <span className="text-slate-400">无</span>}
                    </td>
                    <td className="px-4 py-3 text-xs text-slate-500">
                      {item.evidence.map((evidence) => documentNames[evidence.document_id] ?? evidence.document_id).join("、") || "-"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        type="button"
                        className="btn-secondary mr-2 px-3 py-1.5 text-xs"
                        onClick={() => onEditCase(item)}
                      >
                        编辑
                      </button>
                      <button
                        type="button"
                        className="btn-secondary px-3 py-1.5 text-xs text-red-600"
                        onClick={() => {
                          if (window.confirm("删除该评测问题？")) onDeleteCase(item.id);
                        }}
                      >
                        删除
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function ScreeningWorkspace({
  dataset,
  runs,
  run,
  selectedRunId,
  onSelectRun,
  onStart,
  startPending,
  startError,
  onCancel,
}: {
  dataset: RetrievalEvaluationDataset;
  runs: ChunkTuningRun[];
  run?: ChunkTuningRun;
  selectedRunId: string;
  onSelectRun: (id: string) => void;
  onStart: (sizes: number[], ratios: number[]) => void;
  startPending: boolean;
  startError?: string;
  onCancel: () => void;
}) {
  const [sizes, setSizes] = useState("256, 512, 1024, 1536");
  const [ratios, setRatios] = useState("0, 10, 20, 25");
  const parsedSizes = parseNumberList(sizes);
  const parsedRatios = parseNumberList(ratios);
  const trialCount = parsedSizes.length * parsedRatios.length;

  return (
    <div className="space-y-4">
      {startError && <ErrorState message={startError} />}
      <div className="panel p-5">
        <div className="flex items-center gap-2 text-lg font-semibold text-slate-900">
          <Play className="h-5 w-5 text-brand-600" />参数矩阵
        </div>
        <p className="mt-1 text-sm text-slate-500">
          快速筛选只构建隔离的 L2 原文向量和 BM25，从当前 {dataset.answerable_count} 条可回答问题生成 {trialCount} 组组合，并额外保留当前参数作为基线。
        </p>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div>
            <label className="label">分块大小（token）</label>
            <input className="input" value={sizes} onChange={(event) => setSizes(event.target.value)} />
          </div>
          <div>
            <label className="label">重叠比例（%）</label>
            <input className="input" value={ratios} onChange={(event) => setRatios(event.target.value)} />
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <div className="text-xs text-slate-500">
            将生成 {parsedSizes.length} × {parsedRatios.length} 组；每组重叠四舍五入为整数 token，最多 25 组（含基线）。
          </div>
          <button
            type="button"
            className="btn-primary"
            disabled={startPending || !parsedSizes.length || !parsedRatios.length || trialCount > 24}
            onClick={() => onStart(parsedSizes, parsedRatios)}
          >
            <Play className="h-4 w-4" />启动快速筛选
          </button>
        </div>
      </div>

      <div className="panel p-5">
        <div className="grid gap-3 md:grid-cols-[minmax(260px,1fr)_auto]">
          <div>
            <label className="label">历史筛选运行</label>
            <select className="input" value={selectedRunId} onChange={(event) => onSelectRun(event.target.value)}>
              <option value="">选择运行</option>
              {runs.map((item) => <option key={item.id} value={item.id}>{runLabel(item)}</option>)}
            </select>
          </div>
          <div className="flex items-end">
            {run && ["queued", "running"].includes(run.status) && (
              <button type="button" className="btn-secondary text-red-600" onClick={onCancel}>
                <Square className="h-4 w-4" />取消运行
              </button>
            )}
          </div>
        </div>
        {run && <RunProgress run={run} />}
      </div>

      {run && <TrialTable run={run} selectedTrialId="" onSelectTrial={() => undefined} />}
      {!runs.length && (
        <div className="panel">
          <EmptyState title="还没有筛选运行" description="确认评测集至少有 20 条可回答问题和可用 Embedding 模型后启动。" />
        </div>
      )}
    </div>
  );
}

function AnalysisWorkspace({
  run,
  selectedTrialId,
  onSelectTrial,
  results,
  selectedCandidates,
  onToggleCandidate,
  onStartVerification,
  verificationPending,
  verificationError,
  onDeleteRun,
}: {
  run?: ChunkTuningRun;
  selectedTrialId: string;
  onSelectTrial: (id: string) => void;
  results: ChunkTuningCaseResult[];
  selectedCandidates: string[];
  onToggleCandidate: (id: string) => void;
  onStartVerification: () => void;
  verificationPending: boolean;
  verificationError?: string;
  onDeleteRun: () => void;
}) {
  if (!run) {
    return (
      <div className="panel">
        <EmptyState title="请选择快速筛选运行" description="运行完成后可查看排行榜、Pareto 前沿和逐题召回证据。" />
      </div>
    );
  }
  const completedTrials = run.trials.filter((item) => ["completed", "partial"].includes(item.status));
  const pareto = new Set(run.aggregate.pareto_trial_ids ?? []);
  const recommended = new Set(run.aggregate.recommended_trial_ids ?? []);

  return (
    <div className="space-y-4">
      {verificationError && <ErrorState message={verificationError} />}
      <div className="panel p-5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex items-center gap-2 text-lg font-semibold text-slate-900">
              <BarChart3 className="h-5 w-5 text-brand-600" />Pareto 结果
            </div>
            <p className="mt-1 text-sm text-slate-500">
              优先看 Recall 与上下文证据密度，同时约束 p95 延迟和索引 token 成本。绿色为 Pareto，蓝色为推荐 Top 3。
            </p>
          </div>
          <div className="flex gap-2">
            <button type="button" className="btn-secondary text-red-600" onClick={onDeleteRun}>
              <Trash2 className="h-4 w-4" />删除运行
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={!selectedCandidates.length || verificationPending}
              onClick={onStartVerification}
            >
              <FlaskConical className="h-4 w-4" />
              {verificationPending ? "创建中" : `深度验证 (${selectedCandidates.length}/5)`}
            </button>
          </div>
        </div>
        <RunProgress run={run} />
        {!!completedTrials.length && (
          <ParetoScatter trials={completedTrials} pareto={pareto} recommended={recommended} />
        )}
      </div>

      <div className="panel overflow-hidden">
        <div className="border-b border-slate-100 px-5 py-4 text-sm text-slate-500">
          勾选最多 5 个候选进入完整 L1/L2/L3 索引和答案质量验证。
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[1200px] text-left text-sm">
            <thead className="bg-slate-50 text-xs text-slate-500">
              <tr>
                <th className="px-3 py-3">候选</th>
                <th className="px-3 py-3">参数</th>
                <th className="px-3 py-3">状态</th>
                <th className="px-3 py-3">Recall@5</th>
                <th className="px-3 py-3">Recall@10</th>
                <th className="px-3 py-3">基线 Δ / 95% CI</th>
                <th className="px-3 py-3">MRR</th>
                <th className="px-3 py-3">证据密度</th>
                <th className="px-3 py-3">p95 延迟</th>
                <th className="px-3 py-3">Chunk</th>
                <th className="px-3 py-3">索引 token</th>
                <th className="px-3 py-3">标记</th>
                <th className="px-3 py-3">详情</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {run.trials.map((trial) => {
                const canSelect = ["completed", "partial"].includes(trial.status);
                return (
                  <tr key={trial.id} className={selectedTrialId === trial.id ? "bg-brand-50/50" : "hover:bg-slate-50"}>
                    <td className="px-3 py-3">
                      <input
                        type="checkbox"
                        checked={selectedCandidates.includes(trial.id)}
                        disabled={!canSelect}
                        onChange={() => onToggleCandidate(trial.id)}
                      />
                    </td>
                    <td className="px-3 py-3">
                      <div className="font-medium text-slate-800">{trial.label}</div>
                      <div className="text-xs text-slate-400">{trial.chunk_size} / {trial.chunk_overlap} tokens</div>
                    </td>
                    <td className="px-3 py-3"><span className={`chip ${statusClass(trial.status)}`}>{statusLabels[trial.status] ?? trial.status}</span></td>
                    <td className="px-3 py-3 font-mono">{percent(trial.metrics.recall_at_5)}</td>
                    <td className="px-3 py-3 font-mono">{percent(trial.metrics.recall_at_10)}</td>
                    <td className="px-3 py-3 font-mono text-xs">
                      {trial.is_baseline
                        ? "基线"
                        : `${percent(trial.metrics.baseline_delta?.recall_at_5)} [${percent(trial.metrics.baseline_delta?.recall_at_5_ci?.[0])}, ${percent(trial.metrics.baseline_delta?.recall_at_5_ci?.[1])}]`}
                    </td>
                    <td className="px-3 py-3 font-mono">{metric(trial.metrics.mrr_at_10)}</td>
                    <td className="px-3 py-3 font-mono">{metric(trial.metrics.evidence_density)}</td>
                    <td className="px-3 py-3 font-mono">{metric(trial.metrics.latency_p95_ms, 1)} ms</td>
                    <td className="px-3 py-3 font-mono">{trial.cost.candidate_chunk_count ?? "-"}</td>
                    <td className="px-3 py-3 font-mono">{trial.cost.index_tokens ?? "-"}</td>
                    <td className="px-3 py-3">
                      <div className="flex flex-wrap gap-1">
                        {trial.is_baseline && <span className="chip bg-slate-100 text-slate-600">基线</span>}
                        {pareto.has(trial.id) && <span className="chip bg-emerald-50 text-emerald-700">Pareto</span>}
                        {recommended.has(trial.id) && <span className="chip bg-sky-50 text-sky-700">推荐</span>}
                      </div>
                    </td>
                    <td className="px-3 py-3">
                      <button type="button" className="btn-secondary px-3 py-1.5 text-xs" onClick={() => onSelectTrial(trial.id)}>查看</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {selectedTrialId && (
        <CaseResultTable results={results} trial={run.trials.find((item) => item.id === selectedTrialId)} />
      )}
    </div>
  );
}

function ParetoScatter({
  trials,
  pareto,
  recommended,
}: {
  trials: ChunkTuningTrial[];
  pareto: Set<string>;
  recommended: Set<string>;
}) {
  const [xMetric, setXMetric] = useState<"latency" | "tokens">("latency");
  const points = trials.filter(
    (trial) =>
      trial.metrics.recall_at_5 !== undefined &&
      (xMetric === "latency"
        ? trial.metrics.latency_p95_ms !== undefined
        : trial.cost.index_tokens !== undefined),
  );
  if (!points.length) return null;
  const width = 760;
  const height = 320;
  const padding = 44;
  const xValue = (trial: ChunkTuningTrial) =>
    xMetric === "latency"
      ? trial.metrics.latency_p95_ms ?? 0
      : trial.cost.index_tokens ?? 0;
  const maxX = Math.max(...points.map(xValue), 1);
  const minY = Math.min(...points.map((trial) => trial.metrics.recall_at_5 ?? 0), 0);
  const maxY = Math.max(...points.map((trial) => trial.metrics.recall_at_5 ?? 0), 0.01);
  const x = (value: number) => padding + (value / maxX) * (width - padding * 2);
  const y = (value: number) => height - padding - ((value - minY) / Math.max(0.0001, maxY - minY)) * (height - padding * 2);
  return (
    <div className="mt-5 rounded-xl border border-slate-200 bg-slate-50/60 p-3">
      <div className="mb-2 flex justify-end gap-2">
        <button type="button" className={xMetric === "latency" ? "btn-primary px-3 py-1.5 text-xs" : "btn-secondary px-3 py-1.5 text-xs"} onClick={() => setXMetric("latency")}>延迟</button>
        <button type="button" className={xMetric === "tokens" ? "btn-primary px-3 py-1.5 text-xs" : "btn-secondary px-3 py-1.5 text-xs"} onClick={() => setXMetric("tokens")}>索引 token</button>
      </div>
      <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="min-w-[680px]" role="img" aria-label="Recall 与延迟 Pareto 散点图">
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#cbd5e1" />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} stroke="#cbd5e1" />
        <text x={width / 2} y={height - 8} textAnchor="middle" fill="#64748b" fontSize="12">{xMetric === "latency" ? "p95 检索延迟 (ms)" : "索引 token"}</text>
        <text x={14} y={height / 2} transform={`rotate(-90 14 ${height / 2})`} textAnchor="middle" fill="#64748b" fontSize="12">Recall@5</text>
        {points.map((trial) => {
          const isPareto = pareto.has(trial.id);
          const isRecommended = recommended.has(trial.id);
          const cx = x(xValue(trial));
          const cy = y(trial.metrics.recall_at_5 ?? 0);
          return (
            <g key={trial.id}>
              <circle
                cx={cx}
                cy={cy}
                r={isRecommended ? 7 : 5}
                fill={isPareto ? "#10b981" : isRecommended ? "#0ea5e9" : trial.is_baseline ? "#475569" : "#94a3b8"}
                opacity={0.9}
              />
              <title>{trial.label}: Recall@5 {percent(trial.metrics.recall_at_5)}, {xMetric === "latency" ? `p95 ${metric(trial.metrics.latency_p95_ms, 1)} ms` : `索引 ${trial.cost.index_tokens ?? 0} token`}</title>
            </g>
          );
        })}
      </svg>
      </div>
    </div>
  );
}

function RunProgress({ run }: { run: ChunkTuningRun }) {
  const completed = Number(run.progress.completed_trials ?? 0);
  const total = Number(run.progress.total_trials ?? run.trials.length ?? 0);
  const value = total ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className={`chip ${statusClass(run.status)}`}>{statusLabels[run.status] ?? run.status}</span>
        <span className="font-mono text-slate-500">{completed} / {total} trials · {value}%</span>
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${value}%` }} />
      </div>
      {run.errors.length > 0 && (
        <div className="mt-3 rounded-lg bg-red-50 p-2 text-xs text-red-700">
          {run.errors.map((error) => String(error.error ?? JSON.stringify(error))).join("；")}
        </div>
      )}
    </div>
  );
}

function TrialTable({
  run,
  selectedTrialId,
  onSelectTrial,
}: {
  run: ChunkTuningRun;
  selectedTrialId: string;
  onSelectTrial: (id: string) => void;
}) {
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full min-w-[900px] text-left text-sm">
        <thead className="bg-slate-50 text-xs text-slate-500">
          <tr>
            <th className="px-4 py-3">参数</th>
            <th className="px-4 py-3">状态</th>
            <th className="px-4 py-3">Recall@5</th>
            <th className="px-4 py-3">Recall@10</th>
            <th className="px-4 py-3">证据密度</th>
            <th className="px-4 py-3">p95</th>
            <th className="px-4 py-3">索引 token</th>
            <th className="px-4 py-3">详情</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {run.trials.map((trial) => (
            <tr key={trial.id} className={selectedTrialId === trial.id ? "bg-brand-50/50" : ""}>
              <td className="px-4 py-3">{trial.label}</td>
              <td className="px-4 py-3"><span className={`chip ${statusClass(trial.status)}`}>{statusLabels[trial.status] ?? trial.status}</span></td>
              <td className="px-4 py-3 font-mono">{percent(trial.metrics.recall_at_5)}</td>
              <td className="px-4 py-3 font-mono">{percent(trial.metrics.recall_at_10)}</td>
              <td className="px-4 py-3 font-mono">{metric(trial.metrics.evidence_density)}</td>
              <td className="px-4 py-3 font-mono">{metric(trial.metrics.latency_p95_ms, 1)} ms</td>
              <td className="px-4 py-3 font-mono">{trial.cost.index_tokens ?? "-"}</td>
              <td className="px-4 py-3"><button type="button" className="btn-secondary px-3 py-1.5 text-xs" onClick={() => onSelectTrial(trial.id)}>查看</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VerificationWorkspace({
  runs,
  run,
  selectedRunId,
  onSelectRun,
  selectedTrialId,
  onSelectTrial,
  results,
  currentTrial,
  onApply,
  onReindex,
  applyPending,
  reindexPending,
  applyMessage,
  reindexMessage,
}: {
  runs: ChunkTuningRun[];
  run?: ChunkTuningRun;
  selectedRunId: string;
  onSelectRun: (id: string) => void;
  selectedTrialId: string;
  onSelectTrial: (id: string) => void;
  results: ChunkTuningCaseResult[];
  currentTrial?: ChunkTuningTrial;
  onApply: () => void;
  onReindex: () => void;
  applyPending: boolean;
  reindexPending: boolean;
  applyMessage?: string;
  reindexMessage?: string;
}) {
  return (
    <div className="space-y-4">
      <div className="panel p-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-[280px] flex-1">
            <label className="label">深度验证运行</label>
            <select className="input" value={selectedRunId} onChange={(event) => onSelectRun(event.target.value)}>
              <option value="">选择运行</option>
              {runs.map((item) => <option key={item.id} value={item.id}>{runLabel(item)}</option>)}
            </select>
          </div>
          {currentTrial && (
            <div className="flex flex-wrap gap-2">
              <button type="button" className="btn-secondary" disabled={applyPending || currentTrial.status !== "completed"} onClick={onApply}>
                <Save className="h-4 w-4" />{applyPending ? "保存中" : "保存为默认参数"}
              </button>
              <button type="button" className="btn-secondary text-amber-700" disabled={reindexPending || currentTrial.status !== "completed"} onClick={onReindex}>
                <RefreshCcw className="h-4 w-4" />{reindexPending ? "排队中" : "重建现有文档"}
              </button>
            </div>
          )}
        </div>
        {applyMessage && <div className="mt-3 flex items-center gap-2 rounded-xl bg-emerald-50 p-3 text-sm text-emerald-700"><CheckCircle2 className="h-4 w-4" />{applyMessage}</div>}
        {reindexMessage && <div className="mt-3 rounded-xl bg-sky-50 p-3 text-sm text-sky-700">{reindexMessage}</div>}
        {run && <RunProgress run={run} />}
      </div>

      {!run && (
        <div className="panel">
          <EmptyState title="请选择深度验证运行" description="在结果分析页勾选 Pareto 候选后启动，系统会构建完整多表示索引并评估答案质量。" />
        </div>
      )}
      {run && <TrialTable run={run} selectedTrialId={selectedTrialId} onSelectTrial={onSelectTrial} />}
      {run && currentTrial && (
        <div className="panel grid gap-3 p-5 sm:grid-cols-2 lg:grid-cols-4">
          <Metric label="Faithfulness" value={metric(currentTrial.metrics.quality?.faithfulness, 2)} />
          <Metric label="Completeness" value={metric(currentTrial.metrics.quality?.completeness, 2)} />
          <Metric label="黄金引用利用率" value={percent(currentTrial.metrics.quality?.cited_gold_chunk_ratio)} />
          <Metric label="质量样本" value={String(currentTrial.metrics.quality?.case_count ?? 0)} />
        </div>
      )}
      {run && currentTrial && (
        <CaseResultTable results={results} trial={currentTrial} showAnswer />
      )}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-xl bg-slate-50 p-3"><div className="text-xs text-slate-400">{label}</div><div className="mt-1 font-mono text-lg font-semibold text-slate-800">{value}</div></div>;
}

function CaseResultTable({
  results,
  trial,
  showAnswer = false,
}: {
  results: ChunkTuningCaseResult[];
  trial?: ChunkTuningTrial;
  showAnswer?: boolean;
}) {
  const [selected, setSelected] = useState<ChunkTuningCaseResult | null>(null);
  if (!trial) return null;
  return (
    <div className="panel overflow-hidden">
      <div className="border-b border-slate-100 px-5 py-4">
        <div className="font-semibold text-slate-900">逐题结果 · {trial.label}</div>
        <div className="mt-1 text-xs text-slate-400">点击查看黄金证据命中、Top 10 召回块和答案质量。</div>
      </div>
      {!results.length ? (
        <div className="p-8 text-center text-sm text-slate-400">暂无结果</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[980px] text-left text-sm">
            <thead className="bg-slate-50 text-xs text-slate-500">
              <tr>
                <th className="px-4 py-3">问题</th>
                <th className="px-4 py-3">类型</th>
                <th className="px-4 py-3">Recall@5</th>
                <th className="px-4 py-3">Recall@10</th>
                <th className="px-4 py-3">MRR</th>
                <th className="px-4 py-3">证据密度</th>
                <th className="px-4 py-3">延迟</th>
                {showAnswer && <th className="px-4 py-3">质量</th>}
                <th className="px-4 py-3">详情</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {results.map((item) => (
                <tr key={item.id} className="hover:bg-slate-50">
                  <td className="max-w-md px-4 py-3 font-medium text-slate-800">{item.question}</td>
                  <td className="px-4 py-3"><span className="chip bg-slate-100 text-slate-600">{item.question_type}</span></td>
                  <td className="px-4 py-3 font-mono">{percent(item.recall_at_5)}</td>
                  <td className="px-4 py-3 font-mono">{percent(item.recall_at_10)}</td>
                  <td className="px-4 py-3 font-mono">{metric(item.reciprocal_rank)}</td>
                  <td className="px-4 py-3 font-mono">{metric(item.evidence_density)}</td>
                  <td className="px-4 py-3 font-mono">{metric(item.latency_ms, 1)} ms</td>
                  {showAnswer && <td className="px-4 py-3 text-xs text-slate-500">F {String(item.quality_scores.faithfulness ?? "-")} / C {String(item.quality_scores.completeness ?? "-")}</td>}
                  <td className="px-4 py-3"><button type="button" className="btn-secondary px-3 py-1.5 text-xs" onClick={() => setSelected(item)}>查看</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <Modal open={!!selected} title="召回与质量明细" onClose={() => setSelected(null)} wide>
        {selected && <CaseResultDetail result={selected} />}
      </Modal>
    </div>
  );
}

function CaseResultDetail({ result }: { result: ChunkTuningCaseResult }) {
  return (
    <div className="space-y-4 text-sm">
      {result.answer && (
        <div className="rounded-xl border border-slate-200 p-3">
          <div className="text-xs font-medium text-slate-500">生成答案</div>
          <div className="mt-2 whitespace-pre-wrap leading-6 text-slate-700">{result.answer}</div>
        </div>
      )}
      <div className="rounded-xl border border-slate-200 p-3">
        <div className="text-xs font-medium text-slate-500">黄金证据命中</div>
        <div className="mt-2 space-y-2">
          {result.evidence_results.map((evidence, index) => (
            <div key={`${String(evidence.evidence_id)}-${index}`} className={`rounded-lg p-2 text-xs ${evidence.hit_at_10 ? "bg-emerald-50 text-emerald-800" : "bg-red-50 text-red-700"}`}>
              {evidence.hit_at_10 ? "已命中" : "未命中"} · {String(evidence.quote ?? "")}
            </div>
          ))}
        </div>
      </div>
      <div className="space-y-2">
        {result.retrieved_hits.map((hit, index) => (
          <details key={`${String(hit.chunk_id)}-${index}`} className="rounded-xl border border-slate-200 p-3">
            <summary className="cursor-pointer font-medium text-slate-800">#{String(hit.rank ?? index + 1)} · {String(hit.section ?? "")} · {String(hit.granularity_level ?? "")}</summary>
            <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
              <span>RRF {String(hit.rrf_score ?? "-")}</span>
              <span>Rerank {String(hit.rerank_score ?? "-")}</span>
              <span>路由 {(hit.matched_representations as string[] | undefined)?.join(" / ") || "-"}</span>
            </div>
            <p className="mt-2 whitespace-pre-wrap text-xs leading-5 text-slate-600">{String(hit.content ?? "")}</p>
          </details>
        ))}
      </div>
      {!!Object.keys(result.quality_scores).length && (
        <div className="rounded-xl bg-sky-50 p-3 text-xs leading-5 text-sky-900">
          {Object.entries(result.quality_scores).map(([key, value]) => <div key={key}>{key}: {String(value)}</div>)}
        </div>
      )}
    </div>
  );
}

function DatasetModal({
  open,
  documents,
  pending,
  error,
  onClose,
  onSubmit,
}: {
  open: boolean;
  documents: DocumentItem[];
  pending: boolean;
  error?: string;
  onClose: () => void;
  onSubmit: (payload: Record<string, unknown>) => void;
}) {
  const [name, setName] = useState("分块调优评测集");
  const [description, setDescription] = useState("");
  const [documentIds, setDocumentIds] = useState<string[]>([]);

  useEffect(() => {
    if (!open) return;
    setName("分块调优评测集");
    setDescription("");
    setDocumentIds([]);
  }, [open]);

  function toggle(id: string) {
    setDocumentIds((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  return (
    <Modal open={open} title="新建检索评测集" onClose={onClose} wide>
      {error && <ErrorState message={error} />}
      <form
        className="mt-3 space-y-4"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({ name, description, document_ids: documentIds });
        }}
      >
        <div>
          <label className="label">名称</label>
          <input className="input" value={name} onChange={(event) => setName(event.target.value)} />
        </div>
        <div>
          <label className="label">说明</label>
          <textarea className="input min-h-20" value={description} onChange={(event) => setDescription(event.target.value)} />
        </div>
        <div>
          <label className="label">评测文档</label>
          <div className="grid max-h-64 gap-2 overflow-y-auto rounded-xl border border-slate-200 p-3 sm:grid-cols-2">
            {documents.map((document) => (
              <label key={document.id} className="flex items-start gap-2 rounded-lg p-2 text-sm hover:bg-slate-50">
                <input
                  className="mt-1"
                  type="checkbox"
                  checked={documentIds.includes(document.id)}
                  onChange={() => toggle(document.id)}
                />
                <span>
                  <span className="block font-medium text-slate-700">{document.filename}</span>
                  <span className="text-xs text-slate-400">{document.chunk_count} chunks · {document.chunk_size}/{document.chunk_overlap}</span>
                </span>
              </label>
            ))}
          </div>
          {!documents.length && <div className="mt-2 text-xs text-slate-400">当前知识库没有已就绪文档。</div>}
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>取消</button>
          <button type="submit" className="btn-primary" disabled={pending || !name.trim() || !documentIds.length}>
            {pending ? "创建中" : "创建"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function CaseModal({
  open,
  dataset,
  documents,
  selectedEvidence,
  setSelectedEvidence,
  initialCase,
  pending,
  error,
  onClose,
  onSubmit,
}: {
  open: boolean;
  dataset: RetrievalEvaluationDataset;
  documents: DocumentItem[];
  selectedEvidence: RetrievalEvaluationEvidence[];
  setSelectedEvidence: (items: RetrievalEvaluationEvidence[]) => void;
  initialCase: RetrievalEvaluationCase | null;
  pending: boolean;
  error?: string;
  onClose: () => void;
  onSubmit: (payload: Record<string, unknown>) => void;
}) {
  const [question, setQuestion] = useState("");
  const [referenceAnswer, setReferenceAnswer] = useState("");
  const [questionType, setQuestionType] = useState<EvaluationQuestionType>("detail");
  const [showPicker, setShowPicker] = useState(false);

  useEffect(() => {
    if (!open) return;
    setQuestion(initialCase?.question ?? "");
    setReferenceAnswer(initialCase?.reference_answer ?? "");
    setQuestionType(initialCase?.question_type ?? "detail");
    setShowPicker(false);
  }, [initialCase, open]);

  const isNoAnswer = questionType === "no_answer";
  return (
    <Modal open={open} title={`${initialCase ? "编辑" : "添加"}评测问题 · ${dataset.name}`} onClose={onClose} wide>
      {error && <ErrorState message={error} />}
      <form
        className="mt-3 space-y-4"
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit({
            question,
            reference_answer: isNoAnswer ? "" : referenceAnswer,
            question_type: questionType,
            evidence: isNoAnswer
              ? []
              : selectedEvidence.map((item) => ({
                  document_id: item.document_id,
                  quote: item.quote,
                  section_path: item.section_path,
                  page_start: item.page_start,
                  page_end: item.page_end,
                })),
          });
        }}
      >
        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_190px]">
          <div>
            <label className="label">问题</label>
            <textarea className="input min-h-24" value={question} onChange={(event) => setQuestion(event.target.value)} />
          </div>
          <div>
            <label className="label">问题类型</label>
            <select className="input" value={questionType} onChange={(event) => setQuestionType(event.target.value as EvaluationQuestionType)}>
              {QUESTION_TYPES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}
            </select>
          </div>
        </div>
        {!isNoAnswer && (
          <>
            <div>
              <label className="label">标准答案</label>
              <textarea className="input min-h-28" value={referenceAnswer} onChange={(event) => setReferenceAnswer(event.target.value)} />
            </div>
            <div className="rounded-xl border border-slate-200 p-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <div className="text-sm font-medium text-slate-800">黄金证据</div>
                  <div className="text-xs text-slate-400">选择最小充分片段；长内容请拆成多条证据。</div>
                </div>
                <button type="button" className="btn-secondary px-3 py-2 text-xs" onClick={() => setShowPicker(!showPicker)}>
                  {showPicker ? "收起" : "从原文划选"}
                </button>
              </div>
              {!!selectedEvidence.length && (
                <div className="mt-3 space-y-2">
                  {selectedEvidence.map((item, index) => (
                    <div key={`${item.document_id}-${index}`} className="flex items-start gap-2 rounded-lg bg-amber-50 p-2 text-xs text-amber-900">
                      <div className="min-w-0 flex-1">
                        <div className="font-medium">{item.section_path}</div>
                        <div className="mt-1 whitespace-pre-wrap leading-5">{item.quote}</div>
                      </div>
                      <button type="button" className="text-red-600" onClick={() => setSelectedEvidence(selectedEvidence.filter((_, itemIndex) => itemIndex !== index))}>删除</button>
                    </div>
                  ))}
                </div>
              )}
              {showPicker && (
                <EvidencePicker
                  documents={documents}
                  onAdd={(evidence) => {
                    setSelectedEvidence([...selectedEvidence, evidence]);
                    setShowPicker(false);
                  }}
                />
              )}
            </div>
          </>
        )}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-secondary" onClick={onClose}>取消</button>
          <button
            type="submit"
            className="btn-primary"
            disabled={pending || !question.trim() || (!isNoAnswer && (!referenceAnswer.trim() || !selectedEvidence.length))}
          >
            {pending ? "保存中" : "保存问题"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

function EvidencePicker({
  documents,
  onAdd,
}: {
  documents: DocumentItem[];
  onAdd: (evidence: RetrievalEvaluationEvidence) => void;
}) {
  const [documentId, setDocumentId] = useState(documents[0]?.id ?? "");
  const [quote, setQuote] = useState("");
  const [sectionPath, setSectionPath] = useState("正文");
  const [pageStart, setPageStart] = useState("");
  const contentQuery = useQuery({
    queryKey: ["document-content", documentId, "tuning-evidence"],
    queryFn: () => api.getDocumentContent(documentId),
    enabled: !!documentId,
  });

  function captureSelection() {
    const selected = window.getSelection()?.toString().trim() ?? "";
    if (selected) setQuote(selected);
  }

  return (
    <div className="mt-3 space-y-3 border-t border-slate-200 pt-3">
      <div className="grid gap-3 sm:grid-cols-[minmax(0,1fr)_180px_110px]">
        <div>
          <label className="label">文档</label>
          <select className="input" value={documentId} onChange={(event) => { setDocumentId(event.target.value); setQuote(""); }}>
            {documents.map((item) => <option key={item.id} value={item.id}>{item.filename}</option>)}
          </select>
        </div>
        <div>
          <label className="label">章节路径</label>
          <input className="input" value={sectionPath} onChange={(event) => setSectionPath(event.target.value)} />
        </div>
        <div>
          <label className="label">页码（可选）</label>
          <input className="input" inputMode="numeric" value={pageStart} onChange={(event) => setPageStart(event.target.value)} />
        </div>
      </div>
      {contentQuery.isLoading && <LoadingState label="加载原文..." />}
      {contentQuery.error && <ErrorState message={contentQuery.error.message} />}
      {contentQuery.data && (
        <pre
          className="max-h-72 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-slate-50 p-3 font-sans text-xs leading-5 text-slate-700"
          onMouseUp={captureSelection}
        >
          {contentQuery.data.text}
        </pre>
      )}
      <div>
        <label className="label">已选证据</label>
        <textarea className="input min-h-24" value={quote} onChange={(event) => setQuote(event.target.value)} placeholder="在上方原文中选择文字，或直接粘贴最小充分片段" />
      </div>
      <div className="flex justify-end">
        <button
          type="button"
          className="btn-primary px-3 py-2 text-xs"
          disabled={!documentId || !quote.trim()}
          onClick={() => onAdd({
            id: `draft-${Date.now()}`,
            case_id: "",
            document_id: documentId,
            quote: quote.trim(),
            normalized_quote: quote.trim().replace(/\s+/g, "").toLocaleLowerCase(),
            char_start: null,
            char_end: null,
            section_path: sectionPath.trim() || "正文",
            page_start: pageStart ? Number(pageStart) : null,
            page_end: pageStart ? Number(pageStart) : null,
            created_at: new Date().toISOString(),
          })}
        >
          添加这条证据
        </button>
      </div>
    </div>
  );
}
