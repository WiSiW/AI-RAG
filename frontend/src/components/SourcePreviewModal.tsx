import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ExternalLink, FileText, Hash, Layers3, MapPin } from "lucide-react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api/client";
import type { Chunk } from "../types";
import { locateChunkInSource, remarkSourceHighlight } from "../utils/sourceLocation";
import { ErrorState, LoadingState, Modal } from "./UI";

export default function SourcePreviewModal({
  open,
  onClose,
  documentId,
  chunkId,
}: {
  open: boolean;
  onClose: () => void;
  documentId: string | null;
  chunkId?: string | null;
}) {
  const [activeChunkId, setActiveChunkId] = useState<string | null>(null);
  const sourceScrollRef = useRef<HTMLDivElement | null>(null);
  const selectedChunkRef = useRef<HTMLButtonElement | null>(null);

  const documentQuery = useQuery({
    queryKey: ["document", documentId],
    queryFn: () => api.getDocument(documentId!),
    enabled: open && !!documentId,
  });
  const chunksQuery = useQuery({
    queryKey: ["chunks", documentId],
    queryFn: () => api.listChunks(documentId!),
    enabled: open && !!documentId,
  });
  const contentQuery = useQuery({
    queryKey: ["document-content", documentId],
    queryFn: () => api.getDocumentContent(documentId!),
    enabled: open && !!documentId,
  });

  const document = documentQuery.data;
  const chunks = chunksQuery.data ?? [];
  const content = contentQuery.data;
  const isMarkdown = !!document?.filename.toLowerCase().match(/\.(md|markdown)$/);
  const selected = chunks.find((item) => item.id === activeChunkId)
    ?? chunks.find((item) => item.id === chunkId)
    ?? chunks[0];
  const sourceText = content?.text ?? "";
  const location = useMemo(
    () => locateChunkInSource(sourceText, selected ?? null, chunks),
    [chunks, selected, sourceText],
  );
  const isLoading = documentQuery.isLoading || chunksQuery.isLoading || contentQuery.isLoading;

  useEffect(() => {
    if (!open) {
      setActiveChunkId(null);
      return;
    }
    const requested = chunksQuery.data?.find((item) => item.id === chunkId);
    setActiveChunkId(requested?.id ?? chunksQuery.data?.[0]?.id ?? null);
  }, [chunkId, chunksQuery.data, open]);

  useEffect(() => {
    if (!open || isLoading || !selected) return;
    const frame = window.requestAnimationFrame(() => {
      sourceScrollRef.current
        ?.querySelector<HTMLElement>('[data-current-chunk="true"]')
        ?.scrollIntoView({ behavior: "smooth", block: "center", inline: "nearest" });
      selectedChunkRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "nearest" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [isLoading, isMarkdown, location?.end, location?.start, open, selected]);

  return (
    <Modal open={open} onClose={onClose} title="原文与命中分块" wide>
      {isLoading ? (
        <LoadingState />
      ) : documentQuery.error || chunksQuery.error || contentQuery.error ? (
        <ErrorState message={(documentQuery.error ?? chunksQuery.error ?? contentQuery.error)!.message} />
      ) : (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-slate-50 p-3">
            <div className="min-w-0">
              <div className="flex items-center gap-2 font-medium text-slate-800">
                <FileText className="h-4 w-4 text-brand-600" />
                <span className="truncate">{document?.filename}</span>
              </div>
              <div className="mt-1 text-xs text-slate-500">
                {chunks.length} 个分块 · {String(document?.chunk_mode ?? "")} · {document?.chunk_size} tokens
              </div>
            </div>
            <Link to={`/documents/${documentId}${selected ? `?chunk=${selected.id}` : ""}`} className="btn-secondary px-3 text-xs">
              <ExternalLink className="h-4 w-4" />打开完整预览
            </Link>
          </div>

          <div className="grid max-h-[68vh] gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
            <section className="flex min-h-80 flex-col overflow-hidden rounded-xl border border-slate-200">
              <div className="flex items-center gap-2 border-b border-slate-100 bg-white px-4 py-3 text-sm font-medium text-slate-800">
                <FileText className="h-4 w-4 text-brand-600" />提取原文
                <span className={`chip ml-auto gap-1 text-[10px] ${location ? "bg-amber-50 text-amber-700" : "bg-slate-100 text-slate-500"}`}>
                  <MapPin className="h-3 w-3" />{location ? "已定位" : "未定位"}
                </span>
              </div>
              <div ref={sourceScrollRef} className="source-location-content thin-scrollbar relative flex-1 overflow-y-auto bg-slate-50/60">
                {location ? (
                  <div className="sticky top-0 z-[1] flex items-center gap-2 border-b border-amber-100 bg-amber-50/95 px-4 py-2 text-[11px] font-medium text-amber-800 backdrop-blur">
                    <MapPin className="h-3.5 w-3.5" />{location.label}，已在原文中高亮
                  </div>
                ) : (
                  <div className="sticky top-0 z-[1] border-b border-slate-100 bg-white/95 px-4 py-2 text-[11px] text-slate-500 backdrop-blur">
                    未能自动定位当前分块，右侧仍显示完整分块内容
                  </div>
                )}
                <div className="p-4">
                  {sourceText ? (
                    isMarkdown ? (
                      <article className="markdown-body text-sm">
                        <ReactMarkdown
                          remarkPlugins={location ? [remarkGfm, remarkSourceHighlight(location.start, location.end)] : [remarkGfm]}
                        >
                          {sourceText}
                        </ReactMarkdown>
                      </article>
                    ) : (
                      <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-6 text-slate-700">
                        {location ? (
                          <>
                            {sourceText.slice(0, location.start)}
                            <mark data-current-chunk="true">{sourceText.slice(location.start, location.end)}</mark>
                            {sourceText.slice(location.end)}
                          </>
                        ) : sourceText}
                      </pre>
                    )
                  ) : (
                    <div className="py-16 text-center text-sm text-slate-400">暂无提取原文</div>
                  )}
                </div>
              </div>
            </section>

            <section className="flex min-h-80 flex-col overflow-hidden rounded-xl border border-slate-200">
              <div className="flex items-center gap-2 border-b border-slate-100 bg-white px-4 py-3 text-sm font-medium text-slate-800">
                <Layers3 className="h-4 w-4 text-brand-600" />全部分块
                <span className="ml-auto chip bg-slate-100 text-slate-600">{chunks.length}</span>
              </div>
              <div className="thin-scrollbar flex-1 space-y-2 overflow-y-auto bg-slate-50/60 p-3">
                {chunks.map((chunk: Chunk) => {
                  const active = chunk.id === selected?.id;
                  return (
                    <button
                      key={chunk.id}
                      ref={active ? selectedChunkRef : undefined}
                      type="button"
                      aria-pressed={active}
                      onClick={() => setActiveChunkId(chunk.id)}
                      className={`block w-full rounded-xl border p-3 text-left text-xs transition ${active ? "border-brand-300 bg-brand-50 ring-2 ring-brand-100" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"}`}
                    >
                      <div className="flex items-center justify-between gap-2 font-medium text-slate-700">
                        <span className="flex min-w-0 items-center gap-1.5">
                          <span>#{chunk.chunk_index + 1}</span>
                          <span className="truncate">{chunk.section_path}</span>
                          {active && <span className="chip shrink-0 bg-white text-[10px] text-brand-700">当前</span>}
                        </span>
                        <span className="flex shrink-0 items-center gap-1 text-slate-400"><Hash className="h-3 w-3" />{chunk.token_count}</span>
                      </div>
                      <p className="mt-2 whitespace-pre-wrap leading-5 text-slate-600">{chunk.content}</p>
                    </button>
                  );
                })}
                {!chunks.length && <div className="p-6 text-center text-sm text-slate-400">暂无分块</div>}
              </div>
            </section>
          </div>
          {selected && (
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
              <MapPin className="h-3.5 w-3.5" />
              当前命中分块：#{selected.chunk_index + 1} · {selected.section_path}
              <span className={`chip text-[10px] ${location ? "bg-amber-50 text-amber-700" : "bg-slate-100 text-slate-500"}`}>
                {location?.label ?? "原文位置未定位"}
              </span>
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}
