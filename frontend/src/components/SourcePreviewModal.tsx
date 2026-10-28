import { useQuery } from "@tanstack/react-query";
import { ExternalLink, FileText, Hash, Layers3 } from "lucide-react";
import { Link } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { api } from "../api/client";
import type { Chunk } from "../types";
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
  const isMarkdown = document?.filename.toLowerCase().match(/\.(md|markdown)$/);
  const selected = chunks.find((item) => item.id === chunkId) ?? chunks[0];

  return (
    <Modal open={open} onClose={onClose} title="原文与命中分块" wide>
      {documentQuery.isLoading || chunksQuery.isLoading || contentQuery.isLoading ? (
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
            <Link to={`/documents/${documentId}${chunkId ? `?chunk=${chunkId}` : ""}`} className="btn-secondary px-3 text-xs">
              <ExternalLink className="h-4 w-4" />打开完整预览
            </Link>
          </div>

          <div className="grid max-h-[68vh] gap-4 lg:grid-cols-2">
            <section className="flex min-h-80 flex-col overflow-hidden rounded-xl border border-slate-200">
              <div className="flex items-center gap-2 border-b border-slate-100 bg-white px-4 py-3 text-sm font-medium text-slate-800">
                <FileText className="h-4 w-4 text-brand-600" />提取原文
              </div>
              <div className="thin-scrollbar flex-1 overflow-y-auto bg-slate-50/60 p-4">
                {isMarkdown ? (
                  <article className="markdown-body text-sm"><ReactMarkdown remarkPlugins={[remarkGfm]}>{content?.text ?? ""}</ReactMarkdown></article>
                ) : (
                  <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-6 text-slate-700">{content?.text}</pre>
                )}
              </div>
            </section>

            <section className="flex min-h-80 flex-col overflow-hidden rounded-xl border border-slate-200">
              <div className="flex items-center gap-2 border-b border-slate-100 bg-white px-4 py-3 text-sm font-medium text-slate-800">
                <Layers3 className="h-4 w-4 text-brand-600" />全部分块
                <span className="ml-auto chip bg-slate-100 text-slate-600">{chunks.length}</span>
              </div>
              <div className="thin-scrollbar flex-1 space-y-2 overflow-y-auto bg-slate-50/60 p-3">
                {chunks.map((chunk: Chunk) => (
                  <div
                    key={chunk.id}
                    className={`rounded-xl border p-3 text-xs ${chunk.id === chunkId ? "border-brand-300 bg-brand-50 ring-2 ring-brand-100" : "border-slate-200 bg-white"}`}
                  >
                    <div className="flex items-center justify-between gap-2 font-medium text-slate-700">
                      <span>#{chunk.chunk_index + 1} · {chunk.section_path}</span>
                      <span className="flex items-center gap-1 text-slate-400"><Hash className="h-3 w-3" />{chunk.token_count}</span>
                    </div>
                    <p className="mt-2 whitespace-pre-wrap leading-5 text-slate-600">{chunk.content}</p>
                  </div>
                ))}
                {!chunks.length && <div className="p-6 text-center text-sm text-slate-400">暂无分块</div>}
              </div>
            </section>
          </div>
          {selected && <div className="text-xs text-slate-400">当前命中分块：#{selected.chunk_index + 1} · {selected.section_path}</div>}
        </div>
      )}
    </Modal>
  );
}
