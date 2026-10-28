import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronUp, FileText, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { api } from "../api/client";
import type { Citation } from "../types";

export default function CitationList({ citations }: { citations: Citation[] }) {
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  if (!citations.length) return null;

  return (
    <div className="mt-4 space-y-2">
      <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">参考来源 · 点击展开当前原文</div>
      {citations.map((citation) => {
        const key = `${citation.index}-${citation.chunk_id}`;
        const expanded = expandedKey === key;
        return (
          <div key={key} className={`overflow-hidden rounded-xl border transition ${expanded ? "border-brand-200 bg-brand-50/40" : "border-slate-200 bg-slate-50/80"}`}>
            <button
              type="button"
              onClick={() => setExpandedKey(expanded ? null : key)}
              className="group flex w-full items-start gap-3 p-3 text-left transition hover:bg-brand-50/60"
              aria-expanded={expanded}
            >
              <div className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-white text-xs font-semibold text-brand-700 shadow-sm">
                {citation.index}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 text-sm font-medium text-slate-800">
                  <FileText className="h-3.5 w-3.5 text-slate-400" />
                  <span className="truncate">{citation.source}</span>
                </div>
                {citation.knowledge_base_name && (
                  <div className="mt-1">
                    <span className="chip bg-brand-50 text-[11px] text-brand-700">{citation.knowledge_base_name}</span>
                  </div>
                )}
                <div className="mt-0.5 truncate text-xs text-slate-500">{citation.section}</div>
                {!expanded && <div className="mt-1.5 line-clamp-2 text-xs leading-5 text-slate-500">{citation.snippet}</div>}
              </div>
              <span className="mt-0.5 flex shrink-0 items-center gap-1 text-xs text-slate-400 group-hover:text-brand-600">
                {expanded ? "收起原文" : "查看原文"}
                {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </span>
            </button>
            {expanded && <CitationOriginal citation={citation} />}
          </div>
        );
      })}
    </div>
  );
}

function CitationOriginal({ citation }: { citation: Citation }) {
  const chunkQuery = useQuery({
    queryKey: ["chunk", citation.document_id, citation.chunk_id],
    queryFn: () => api.getChunk(citation.document_id, citation.chunk_id),
  });

  return (
    <div className="border-t border-brand-100 bg-white px-4 py-4">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-500">
        <span className="font-medium text-brand-700">当前引用分块原文</span>
        <span>分块 #{chunkQuery.data ? chunkQuery.data.chunk_index + 1 : "-"} · {citation.section}</span>
      </div>
      {chunkQuery.isLoading ? (
        <div className="flex items-center gap-2 py-3 text-sm text-slate-400"><LoaderCircle className="h-4 w-4 animate-spin" />正在加载当前原文...</div>
      ) : chunkQuery.error ? (
        <div className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{chunkQuery.error.message}</div>
      ) : (
        <pre className="thin-scrollbar max-h-80 overflow-y-auto whitespace-pre-wrap break-words rounded-xl bg-slate-50 p-4 font-sans text-sm leading-7 text-slate-700">
          {chunkQuery.data?.content}
        </pre>
      )}
    </div>
  );
}
