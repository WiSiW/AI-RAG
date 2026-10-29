import { Children, cloneElement, isValidElement, useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Bot, ClipboardCheck, FileText, Hash, Layers3, Search, Sparkles, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import { Link, useParams, useSearchParams } from "react-router-dom";
import remarkGfm from "remark-gfm";
import { api } from "../api/client";
import { ErrorState, LoadingState, PageHeader, StatusBadge, formatBytes } from "../components/UI";
import type { Chunk } from "../types";

export default function DocumentPreviewPage() {
  const { documentId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedChunk = searchParams.get("chunk");
  const urlQuery = searchParams.get("q") ?? "";
  const submittedQuery = urlQuery.trim();
  const [query, setQuery] = useState(urlQuery);
  const chunkRefs = useRef<Record<string, HTMLAnchorElement | null>>({});

  const documentQuery = useQuery({ queryKey: ["document", documentId], queryFn: () => api.getDocument(documentId), enabled: !!documentId });
  const chunksQuery = useQuery({ queryKey: ["chunks", documentId], queryFn: () => api.listChunks(documentId), enabled: !!documentId });
  const contentQuery = useQuery({ queryKey: ["document-content", documentId], queryFn: () => api.getDocumentContent(documentId), enabled: !!documentId });
  const chunks = chunksQuery.data ?? [];

  const searchTerms = useMemo(() => toSearchTerms(submittedQuery), [submittedQuery]);
  const filteredChunks = useMemo(
    () => chunks.filter((chunk) => chunkMatches(chunk, searchTerms)),
    [chunks, searchTerms],
  );
  const matchCount = useMemo(
    () => filteredChunks.reduce(
      (total, chunk) => total + countOccurrences(chunkSearchText(chunk), searchTerms),
      0,
    ),
    [filteredChunks, searchTerms],
  );
  const markdownComponents = useMemo(() => highlightedMarkdownComponents(searchTerms), [searchTerms]);
  const hasSearch = searchTerms.length > 0;

  useEffect(() => {
    setQuery(urlQuery);
  }, [urlQuery]);

  useEffect(() => {
    if (selectedChunk && chunkRefs.current[selectedChunk]) {
      chunkRefs.current[selectedChunk]?.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedChunk, chunksQuery.data, submittedQuery]);

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const next = new URLSearchParams(searchParams.toString());
    const value = query.trim();
    if (value) next.set("q", value);
    else next.delete("q");
    setSearchParams(next, { replace: true });
  }

  function clearSearch() {
    setQuery("");
    const next = new URLSearchParams(searchParams.toString());
    next.delete("q");
    setSearchParams(next, { replace: true });
  }

  function chunkHref(chunkId: string): string {
    const next = new URLSearchParams(searchParams.toString());
    next.set("chunk", chunkId);
    return `/documents/${documentId}?${next.toString()}`;
  }

  if (documentQuery.isLoading || chunksQuery.isLoading || contentQuery.isLoading) return <LoadingState />;
  if (documentQuery.error) return <ErrorState message={documentQuery.error.message} />;
  if (chunksQuery.error) return <ErrorState message={chunksQuery.error.message} />;
  if (contentQuery.error) return <ErrorState message={contentQuery.error.message} />;

  const document = documentQuery.data!;
  const content = contentQuery.data!;
  const isMarkdown = document.filename.toLowerCase().endsWith(".md") || document.filename.toLowerCase().endsWith(".markdown");

  return (
    <div>
      <Link to={`/knowledge-bases/${document.kb_id}`} className="mb-4 inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-800">
        <ArrowLeft className="h-4 w-4" />返回知识库
      </Link>
      <PageHeader
        title={document.filename}
        description={`${formatBytes(document.size_bytes)} · ${chunks.length} 个分块 · ${document.chunk_size} tokens / 重叠 ${document.chunk_overlap}`}
        action={
          <div className="flex flex-wrap items-center gap-2">
            <StatusBadge status={document.status} />
            <Link to={`/chunk-evaluations?kb=${document.kb_id}&document=${document.id}`} className="btn-secondary"><ClipboardCheck className="h-4 w-4" />评估完整性</Link>
            <Link to={`/chat?kb=${document.kb_id}&document=${document.id}`} className="btn-secondary"><Bot className="h-4 w-4" />针对本文档问答</Link>
          </div>
        }
      />

      {document.error && <div className="mb-4"><ErrorState message={document.error} /></div>}

      <div className="mb-4 panel p-4">
        <div className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-4">
          <Metadata label="分块模式" value={String(document.chunk_mode ?? "structure")} />
          <Metadata label="文档标题" value={String(document.metadata_json.title ?? document.filename)} />
          <Metadata label="语言 / 格式" value={`${String(document.metadata_json.language ?? "-")} / ${String(document.metadata_json.format ?? "-")}`} />
          <Metadata label="SHA-256" value={String(document.metadata_json.sha256 ?? "-")} mono />
          <Metadata label="作者" value={String(document.metadata_json.author ?? document.metadata_json.authors ?? "-")} />
          <Metadata label="页数 / 章节" value={String(document.metadata_json.pages ?? document.metadata_json.chapter_count ?? document.metadata_json.section_count ?? "-")} />
          <Metadata label="字符数" value={String(document.metadata_json.characters ?? content.text.length)} />
          <Metadata label="上传来源" value={String(document.metadata_json.upload_source ?? "browser")} />
        </div>
      </div>

      <div className="grid min-h-[68vh] gap-5 xl:grid-cols-[430px_1fr]">
        <section className="panel flex max-h-[76vh] min-h-[520px] flex-col overflow-hidden">
          <div className="border-b border-slate-100 px-4 py-3.5">
            <div className="flex items-center gap-2">
              <Layers3 className="h-4 w-4 text-brand-600" />
              <div className="font-semibold text-slate-900">分块列表</div>
              <span className="ml-auto chip bg-slate-100 text-slate-600">
                {hasSearch ? `${filteredChunks.length} / ${chunks.length}` : chunks.length}
              </span>
            </div>
            <form onSubmit={submitSearch} className="mt-3 flex items-center gap-2">
              <div className="relative min-w-0 flex-1">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                <input
                  className="input py-2 pl-9 pr-9 text-xs"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="查询分块，多个关键词用空格分隔"
                  maxLength={200}
                  aria-label="查询文档分块"
                />
                {query && (
                  <button
                    type="button"
                    onClick={clearSearch}
                    className="absolute right-2 top-1/2 grid h-6 w-6 -translate-y-1/2 place-items-center rounded-full text-slate-400 hover:bg-slate-100 hover:text-slate-700"
                    aria-label="清除查询"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                )}
              </div>
              <button type="submit" className="btn-primary px-3 py-2 text-xs" disabled={!query.trim() && !hasSearch}>
                <Search className="h-3.5 w-3.5" />查询
              </button>
            </form>
            {hasSearch && (
              <div className="mt-2 flex items-center justify-between gap-3 text-xs" aria-live="polite">
                <span className={filteredChunks.length ? "text-slate-500" : "text-amber-700"}>
                  {filteredChunks.length
                    ? `“${submittedQuery}”命中 ${filteredChunks.length} 个分块、${matchCount} 处内容`
                    : `“${submittedQuery}”没有匹配的分块`}
                </span>
                <button type="button" onClick={clearSearch} className="shrink-0 text-brand-700 hover:text-brand-800">清除查询</button>
              </div>
            )}
          </div>
          <div className="thin-scrollbar flex-1 overflow-y-auto p-3">
            {!chunks.length ? (
              <div className="grid h-full place-items-center px-6 text-center text-sm text-slate-400">
                文档尚未生成分块，请等待入库完成或检查错误信息。
              </div>
            ) : !filteredChunks.length ? (
              <div className="grid h-full place-items-center px-6 text-center">
                <div>
                  <Search className="mx-auto mb-3 h-7 w-7 text-slate-300" />
                  <div className="text-sm font-medium text-slate-700">未找到匹配分块</div>
                  <p className="mt-1 text-xs leading-5 text-slate-400">尝试减少关键词、使用更短的词，或清除查询查看全部分块。</p>
                  <button type="button" onClick={clearSearch} className="btn-secondary mt-4 px-3 py-2 text-xs">查看全部分块</button>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                {filteredChunks.map((chunk) => {
                  const active = selectedChunk === chunk.id;
                  return (
                    <Link
                      key={chunk.id}
                      to={chunkHref(chunk.id)}
                      ref={(element) => { chunkRefs.current[chunk.id] = element; }}
                      className={`block scroll-mt-4 rounded-xl border p-3 transition ${
                        active ? "border-brand-300 bg-brand-50 ring-2 ring-brand-100" : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="flex items-center gap-2 font-semibold text-slate-700">
                          #{chunk.chunk_index + 1}
                          <span className={`chip text-[10px] ${chunk.granularity_level === "L1" ? "bg-violet-50 text-violet-700" : chunk.granularity_level === "L3" ? "bg-amber-50 text-amber-700" : "bg-brand-50 text-brand-700"}`}>{chunk.granularity_level}</span>
                          <span className="text-[10px] font-normal text-slate-400">{chunk.granule_type}</span>
                        </span>
                        <span className="flex items-center gap-1 text-slate-400"><Hash className="h-3 w-3" />{chunk.token_count}</span>
                      </div>
                      <div className="mt-1 truncate text-xs text-brand-700"><HighlightText text={chunk.section_path} terms={searchTerms} /></div>
                      {chunk.context_prefix && <div className="mt-1 text-[11px] leading-5 text-slate-400">{chunk.context_prefix}</div>}
                      <p className="mt-2 whitespace-pre-wrap text-xs leading-5 text-slate-600"><HighlightText text={chunk.content} terms={searchTerms} /></p>
                      {chunk.summary && chunk.granularity_level !== "L1" && (
                        <div className="mt-2 rounded-lg bg-emerald-50 px-2.5 py-2 text-xs leading-5 text-emerald-800">
                          <span className="mr-1 font-semibold">摘要</span>{chunk.summary}
                        </div>
                      )}
                      {!!chunk.keywords?.length && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {chunk.keywords.map((keyword) => <span key={keyword} className="chip bg-slate-100 text-[11px] text-slate-600">{keyword}</span>)}
                        </div>
                      )}
                      {!!chunk.hypothetical_questions?.length && (
                        <div className="mt-2 space-y-1 text-xs text-slate-500">
                          {chunk.hypothetical_questions.map((question) => <div key={question}><Sparkles className="mr-1 inline h-3 w-3 text-brand-500" />{question}</div>)}
                        </div>
                      )}
                      {chunk.enrichment_status !== "ready" && (
                        <div className="mt-2 text-[11px] font-medium text-amber-700">增强状态：{chunk.enrichment_status}</div>
                      )}
                    </Link>
                  );
                })}
              </div>
            )}
          </div>
        </section>

        <section className="panel flex max-h-[76vh] min-h-[520px] flex-col overflow-hidden">
          <div className="flex items-center gap-2 border-b border-slate-100 px-5 py-3.5">
            <FileText className="h-4 w-4 text-brand-600" />
            <div className="font-semibold text-slate-900">原文预览</div>
            {hasSearch && <span className="chip bg-amber-50 text-amber-700">{matchCount} 处命中</span>}
            <span className="ml-auto text-xs text-slate-400">{content.content_type}</span>
          </div>
          <div className="thin-scrollbar flex-1 overflow-y-auto bg-white px-6 py-6 sm:px-9">
            {isMarkdown ? (
              <article className="markdown-body mx-auto max-w-4xl">
                <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{content.text}</ReactMarkdown>
              </article>
            ) : (
              <pre className="mx-auto max-w-5xl whitespace-pre-wrap break-words font-sans text-sm leading-7 text-slate-700">
                <HighlightText text={content.text} terms={searchTerms} />
              </pre>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function Metadata({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0 rounded-xl bg-slate-50 p-3">
      <div className="text-slate-400">{label}</div>
      <div className={`mt-1 truncate text-slate-700 ${mono ? "font-mono" : "font-medium"}`} title={value}>{value}</div>
    </div>
  );
}

function toSearchTerms(query: string): string[] {
  return Array.from(new Set(query.toLocaleLowerCase().split(/\s+/).filter(Boolean))).slice(0, 16);
}

function chunkMatches(chunk: Chunk, terms: string[]): boolean {
  if (!terms.length) return true;
  const haystack = chunkSearchText(chunk).toLocaleLowerCase();
  return terms.every((term) => haystack.includes(term));
}

function chunkSearchText(chunk: Chunk): string {
  return [
    chunk.section_path,
    chunk.content,
    chunk.granularity_level,
    chunk.granule_type,
    chunk.summary,
    ...(chunk.keywords ?? []),
    ...(chunk.hypothetical_questions ?? []),
  ].join("\n");
}

function countOccurrences(text: string, terms: string[]): number {
  const haystack = text.toLocaleLowerCase();
  return terms.reduce((total, term) => {
    let count = 0;
    let position = 0;
    while (term && (position = haystack.indexOf(term, position)) !== -1) {
      count += 1;
      position += term.length;
    }
    return total + count;
  }, 0);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function HighlightText({ text, terms }: { text: string; terms: string[] }) {
  if (!terms.length || !text) return <>{text}</>;
  const matcher = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  const normalizedTerms = new Set(terms);
  return (
    <>
      {text.split(matcher).map((part, index) => (
        normalizedTerms.has(part.toLocaleLowerCase())
          ? <mark key={`${part}-${index}`} className="rounded bg-amber-200/90 px-0.5 text-slate-900">{part}</mark>
          : part
      ))}
    </>
  );
}

function highlightReactNode(node: ReactNode, terms: string[]): ReactNode {
  if (!terms.length) return node;
  if (typeof node === "string") return <HighlightText text={node} terms={terms} />;
  if (Array.isArray(node)) return Children.map(node, (child) => highlightReactNode(child, terms));
  if (isValidElement<{ children?: ReactNode }>(node)) {
    const children = node.props.children;
    if (children === undefined) return node;
    return cloneElement(node, undefined, highlightReactNode(children, terms));
  }
  return node;
}

function highlightedMarkdownComponents(terms: string[]): Components {
  const highlight = (children: ReactNode) => highlightReactNode(children, terms);
  return {
    h1: ({ children }) => <h1>{highlight(children)}</h1>,
    h2: ({ children }) => <h2>{highlight(children)}</h2>,
    h3: ({ children }) => <h3>{highlight(children)}</h3>,
    h4: ({ children }) => <h4>{highlight(children)}</h4>,
    p: ({ children }) => <p>{highlight(children)}</p>,
    li: ({ children }) => <li>{highlight(children)}</li>,
    blockquote: ({ children }) => <blockquote>{highlight(children)}</blockquote>,
    td: ({ children }) => <td>{highlight(children)}</td>,
    th: ({ children }) => <th>{highlight(children)}</th>,
    code: ({ children }) => <code>{highlight(children)}</code>,
  };
}
