import { FormEvent, useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowLeft, BarChart3, ExternalLink, FileText, Gauge, Search } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import { ErrorState, LoadingState, PageHeader } from "../components/UI";
import SourcePreviewModal from "../components/SourcePreviewModal";
import type { SearchResult } from "../types";

function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : value.toFixed(4);
}

function relevanceScore(item: SearchResult): number | null {
  return item.relevance_score ?? item.rerank_score ?? item.rrf_score;
}

function relevanceSourceLabel(item: SearchResult): string {
  const source = item.relevance_source ?? (item.rerank_score != null ? "rerank" : "rrf");
  return source === "rerank" ? "Rerank 精排" : "RRF 融合";
}

export default function SearchQualityPage() {
  const [searchParams] = useSearchParams();
  const [kbId, setKbId] = useState(searchParams.get("kb") ?? "");
  const [documentId, setDocumentId] = useState(searchParams.get("document") ?? "");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState("10");
  const [rerank, setRerank] = useState(true);
  const [selected, setSelected] = useState<SearchResult | null>(null);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const documentsQuery = useQuery({
    queryKey: ["documents", kbId],
    queryFn: () => api.listDocuments(kbId),
    enabled: !!kbId,
  });
  const knowledgeBases = kbQuery.data ?? [];
  const selectedKb = useMemo(() => knowledgeBases.find((kb) => kb.id === kbId), [knowledgeBases, kbId]);
  const documents = documentsQuery.data ?? [];

  const searchMutation = useMutation({
    mutationFn: () => api.search({
      knowledge_base_id: kbId,
      query: query.trim(),
      top_k: Number(topK),
      rerank,
      document_id: documentId || undefined,
    }),
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    if (query.trim() && kbId) searchMutation.mutate();
  }

  if (kbQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;
  const response = searchMutation.data;

  return (
    <div>
      <Link to={kbId ? `/knowledge-bases/${kbId}` : "/knowledge-bases"} className="mb-4 inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800">
        <ArrowLeft className="h-4 w-4" />返回知识库
      </Link>
      <PageHeader title="检索质量检查" description="直接查看混合召回、统一相关度结果、RRF 与 Rerank 的分数、排名和命中文本，用于调参与定位召回问题。" />

      <form onSubmit={submit} className="panel p-5">
        <div className="grid gap-4 lg:grid-cols-[1fr_220px_180px_160px]">
          <div>
            <label className="label">知识库</label>
            <select className="input" value={kbId} onChange={(event) => { setKbId(event.target.value); setDocumentId(""); }} required>
              <option value="">请选择</option>
              {knowledgeBases.map((kb) => <option key={kb.id} value={kb.id}>{kb.name}</option>)}
            </select>
          </div>
          <div>
            <label className="label">检索范围</label>
            <select className="input" value={documentId} onChange={(event) => setDocumentId(event.target.value)}>
              <option value="">整个知识库</option>
              {documents.filter((doc) => doc.status === "ready").map((doc) => <option key={doc.id} value={doc.id}>{doc.filename}</option>)}
            </select>
          </div>
          <div><label className="label">Top K</label><input className="input" type="number" min={1} max={50} value={topK} onChange={(event) => setTopK(event.target.value)} /></div>
          <label className="flex items-center gap-2 self-end rounded-xl border border-slate-200 px-3 py-3 text-sm text-slate-700">
            <input type="checkbox" checked={rerank} onChange={(event) => setRerank(event.target.checked)} />启用重排
          </label>
        </div>
        <div className="mt-4 flex flex-col gap-3 sm:flex-row">
          <input className="input flex-1" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入用于测试召回质量的问题" required />
          <button type="submit" className="btn-primary" disabled={!kbId || !query.trim() || searchMutation.isPending}>
            <Search className="h-4 w-4" />{searchMutation.isPending ? "检索中..." : "执行检索"}
          </button>
        </div>
        {searchMutation.error && <div className="mt-4"><ErrorState message={searchMutation.error.message} /></div>}
      </form>

      {response && (
        <div className="mt-5 space-y-5">
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
            <Metric label="命中数量" value={String(response.quality.result_count)} />
            <Metric label="平均 RRF" value={score(response.quality.avg_rrf_score)} />
            <Metric label="最佳分数" value={score(response.quality.best_score)} />
            <Metric label="重排命中" value={String(response.quality.reranked_count)} />
            <Metric label="向量命中" value={String(response.quality.vector_hits)} />
            <Metric label="BM25 命中" value={String(response.quality.bm25_hits)} />
          </div>

          {response.rewritten_query && (
            <div className="rounded-xl border border-brand-200 bg-brand-50 p-3 text-sm text-brand-800">查询改写：{response.rewritten_query}</div>
          )}

          <div className="space-y-3">
            {response.results.map((item, index) => (
              <article key={item.chunk_id} className="panel p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2 text-sm font-semibold text-slate-900">
                      <span className="grid h-6 w-6 place-items-center rounded-lg bg-brand-50 text-xs text-brand-700">{index + 1}</span>
                      <FileText className="h-4 w-4 text-slate-400" />
                      <span className="truncate">{item.source}</span>
                    </div>
                    <div className="mt-1 text-xs text-slate-500">{item.section}</div>
                  </div>
                  <button type="button" onClick={() => setSelected(item)} className="btn-secondary px-3 py-2 text-xs"><ExternalLink className="h-4 w-4" />查看原文与分块</button>
                </div>
                <p className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-50 p-3 text-sm leading-6 text-slate-700">{item.content}</p>
                <div
                  className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-brand-100 bg-brand-50/70 px-3 py-2.5"
                  title="优先使用 Rerank 返回的相关度分；未启用重排时回退到 RRF 融合分"
                >
                  <div className="flex items-center gap-2 text-xs font-semibold text-brand-800">
                    <Gauge className="h-4 w-4" />相关度结果
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-semibold text-brand-900">{score(relevanceScore(item))}</span>
                    <span className="chip border border-brand-100 bg-white text-brand-700">{relevanceSourceLabel(item)}</span>
                  </div>
                </div>
                <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3 lg:grid-cols-6">
                  <Score label="Rerank" value={item.rerank_score} />
                  <Score label="RRF" value={item.rrf_score} />
                  <Score label="Vector" value={item.vector_score} />
                  <Score label="BM25" value={item.bm25_score} />
                  <Score label="向量排名" value={item.vector_rank} integer />
                  <Score label="BM25 排名" value={item.bm25_rank} integer />
                </div>
              </article>
            ))}
            {!response.results.length && <div className="panel p-10 text-center text-sm text-slate-400"><BarChart3 className="mx-auto mb-3 h-7 w-7" />没有召回结果，请检查文档入库状态、阈值和 Embedding 模型。</div>}
          </div>
        </div>
      )}

      <SourcePreviewModal
        open={!!selected}
        onClose={() => setSelected(null)}
        documentId={selected?.document_id ?? null}
        chunkId={selected?.chunk_id}
      />
      {selectedKb && <div className="sr-only">{selectedKb.name}</div>}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="panel p-3"><div className="text-xs text-slate-400">{label}</div><div className="mt-1 font-semibold text-slate-800">{value}</div></div>;
}

function Score({ label, value, integer = false }: { label: string; value: number | null; integer?: boolean }) {
  return <div className="rounded-lg bg-slate-50 px-3 py-2"><span className="text-slate-400">{label}</span><span className="ml-2 font-mono text-slate-700">{value === null ? "-" : integer ? value : value.toFixed(4)}</span></div>;
}
