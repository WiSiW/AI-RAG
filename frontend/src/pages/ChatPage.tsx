import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bot, ChevronDown, CircleStop, FileText, LoaderCircle, MessageSquarePlus, Send, Sparkles, ThumbsDown, ThumbsUp, UserRound } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { useNavigate, useSearchParams } from "react-router-dom";
import remarkGfm from "remark-gfm";
import { api, streamChat } from "../api/client";
import CitationList from "../components/CitationList";
import KnowledgeBaseSelector from "../components/KnowledgeBaseSelector";
import { ErrorState, LoadingState, PageHeader } from "../components/UI";
import type { Citation } from "../types";

interface ChatMessageView {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  messageId?: string;
  streaming?: boolean;
}

const prompts = [
  "请概括这个知识库的核心内容。",
  "文档中有哪些关键流程和注意事项？",
  "请给出与当前业务最相关的三条结论，并附上出处。",
];

export default function ChatPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const conversationParam = searchParams.get("conversation") ?? "";
  const kbParam = searchParams.get("kb") ?? "";
  const kbsParam = searchParams.get("kbs") ?? "";
  const documentParam = searchParams.get("document") ?? "";
  const initialKbIds = (kbsParam || kbParam)
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);

  const kbQuery = useQuery({ queryKey: ["knowledge-bases"], queryFn: api.listKnowledgeBases });
  const modelQuery = useQuery({ queryKey: ["models", "llm"], queryFn: () => api.listModels("llm") });
  const conversationQuery = useQuery({
    queryKey: ["conversation", conversationParam],
    queryFn: () => api.getConversation(conversationParam),
    enabled: !!conversationParam,
  });

  const [kbIds, setKbIds] = useState<string[]>(initialKbIds);
  const [documentId, setDocumentId] = useState(documentParam);
  const [llmId, setLlmId] = useState("");
  const [conversationId, setConversationId] = useState(conversationParam);
  const [messages, setMessages] = useState<ChatMessageView[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [feedback, setFeedback] = useState<Record<string, 1 | -1>>({});
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scopeInitializedRef = useRef(false);

  const documentsQuery = useQuery({
    queryKey: ["documents", kbIds[0]],
    queryFn: () => api.listDocuments(kbIds[0]),
    enabled: kbIds.length === 1,
  });
  const knowledgeBases = kbQuery.data ?? [];
  const llmModels = modelQuery.data ?? [];
  const documents = (documentsQuery.data ?? []).filter((doc) => doc.status === "ready");
  const selectedKbs = useMemo(
    () => knowledgeBases.filter((kb) => kbIds.includes(kb.id)),
    [knowledgeBases, kbIds],
  );

  useEffect(() => {
    if (scopeInitializedRef.current || !knowledgeBases.length) return;
    const availableIds = new Set(knowledgeBases.map((knowledgeBase) => knowledgeBase.id));
    const preferredIds = initialKbIds.filter((kbId) => availableIds.has(kbId));
    setKbIds(preferredIds.length ? preferredIds : [knowledgeBases[0].id]);
    scopeInitializedRef.current = true;
  }, [initialKbIds, kbIds.length, knowledgeBases]);

  useEffect(() => {
    if (!selectedKbs.length) return;
    const preferredIds = new Set(
      selectedKbs.map((kb) => kb.default_llm_profile_id).filter(Boolean),
    );
    const preferred = preferredIds.size === 1 ? [...preferredIds][0] : "";
    if (!llmId || !llmModels.some((model) => model.id === llmId)) {
      setLlmId(preferred || llmModels[0]?.id || "");
    }
  }, [selectedKbs, llmId, llmModels]);

  useEffect(() => {
    if (!conversationQuery.data) return;
    const conversation = conversationQuery.data;
    setConversationId(conversation.id);
    setKbIds(conversation.knowledge_base_ids);
    setDocumentId(conversation.document_id || "");
    setLlmId(conversation.llm_profile_id || "");
    setMessages(
      conversation.messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        citations: message.citations,
        messageId: message.role === "assistant" ? message.id : undefined,
      })),
    );
  }, [conversationQuery.data]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const feedbackMutation = useMutation({
    mutationFn: ({ messageId, rating }: { messageId: string; rating: 1 | -1 }) =>
      api.sendFeedback(messageId, rating),
    onSuccess: (_, variables) => setFeedback((current) => ({ ...current, [variables.messageId]: variables.rating })),
    onError: (err) => setError(err.message),
  });

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const query = input.trim();
    if (!query || !kbIds.length || !llmId || streaming) return;
    setError("");
    setWarning("");
    setInput("");
    const userView: ChatMessageView = { id: `user-${Date.now()}`, role: "user", content: query, citations: [] };
    const assistantView: ChatMessageView = { id: `assistant-${Date.now()}`, role: "assistant", content: "", citations: [], streaming: true };
    setMessages((current) => [...current, userView, assistantView]);
    setStreaming(true);
    const controller = new AbortController();
    abortRef.current = controller;
    let answer = "";
    let activeConversationId = conversationId;

    try {
      await streamChat(
        {
          knowledge_base_ids: kbIds,
          query,
          conversation_id: conversationId || undefined,
          llm_profile_id: llmId,
          document_id: documentId || undefined,
          stream: true,
        },
        {
          onEvent: (eventName, data) => {
            if (eventName === "start") {
              const startedConversationId = String(data.conversation_id ?? "");
              if (startedConversationId) {
                activeConversationId = startedConversationId;
                setConversationId(startedConversationId);
              }
              const failedIds = (data.failed_knowledge_base_ids ?? []) as string[];
              if (failedIds.length) {
                const failedNames = failedIds.map(
                  (failedId) => knowledgeBases.find((kb) => kb.id === failedId)?.name ?? failedId,
                );
                setWarning(`部分知识库检索失败：${failedNames.join("、")}`);
              }
            } else if (eventName === "token") {
              answer += String(data.content ?? "");
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantView.id ? { ...message, content: answer } : message,
                ),
              );
            } else if (eventName === "citations") {
              const citations = (data.citations ?? []) as Citation[];
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantView.id ? { ...message, citations } : message,
                ),
              );
            } else if (eventName === "done") {
              const messageId = String(data.message_id ?? "");
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantView.id
                    ? { ...message, messageId, streaming: false }
                    : message,
                ),
              );
              void queryClient.invalidateQueries({ queryKey: ["conversations"] });
              if (activeConversationId) {
                const documentQuery = documentId ? `&document=${documentId}` : "";
                navigate(`/chat?kbs=${kbIds.join(",")}${documentQuery}&conversation=${activeConversationId}`, { replace: true });
                void queryClient.invalidateQueries({ queryKey: ["conversation", activeConversationId] });
              }
            } else if (eventName === "error") {
              setError(String(data.message ?? "生成回答失败"));
              setMessages((current) =>
                current.map((message) =>
                  message.id === assistantView.id ? { ...message, streaming: false } : message,
                ),
              );
            }
          },
          onError: (streamError) => setError(streamError.message),
        },
        controller.signal,
      );
    } catch (streamError) {
      if ((streamError as Error).name !== "AbortError") {
        setError((streamError as Error).message);
      }
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantView.id ? { ...message, streaming: false } : message,
        ),
      );
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  function newConversation() {
    abortRef.current?.abort();
    setConversationId("");
    setMessages([]);
    setInput("");
    setError("");
    setWarning("");
    const documentQuery = documentId ? `&document=${documentId}` : "";
    navigate(`/chat?kbs=${kbIds.join(",")}${documentQuery}`, { replace: true });
  }

  function changeKnowledgeBases(nextIds: string[]) {
    abortRef.current?.abort();
    setKbIds(nextIds);
    setDocumentId("");
    setConversationId("");
    setMessages([]);
    setError("");
    setWarning("");
    const query = nextIds.length ? `?kbs=${nextIds.join(",")}` : "";
    navigate(`/chat${query}`, { replace: true });
  }

  if (kbQuery.isLoading || modelQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;
  if (modelQuery.error) return <ErrorState message={modelQuery.error.message} />;
  if (conversationParam && conversationQuery.error) return <ErrorState message={conversationQuery.error.message} />;

  return (
    <div>
      <PageHeader title="智能问答" description="支持跨知识库联合检索；每条回答都会标注知识库、文档和章节来源。" />

      {!knowledgeBases.length || !llmModels.length ? (
        <div className="panel p-8 text-center">
          <ErrorState message="请先配置 Embedding、问答模型并创建知识库。" />
        </div>
      ) : (
        <div className="grid min-h-[72vh] gap-5 xl:grid-cols-[290px_1fr]">
          <aside className="panel h-fit p-4 xl:sticky xl:top-24">
            <div>
              <label className="label">知识库范围</label>
              <KnowledgeBaseSelector
                knowledgeBases={knowledgeBases}
                value={kbIds}
                onChange={changeKnowledgeBases}
                disabled={streaming}
              />
            </div>
            <div className="mt-4">
              <label className="label">问答范围</label>
              <select className="input" value={documentId} onChange={(event) => { const nextDocument = event.target.value; setDocumentId(nextDocument); setConversationId(""); setMessages([]); const query = nextDocument ? `&document=${nextDocument}` : ""; navigate(`/chat?kbs=${kbIds.join(",")}${query}`, { replace: true }); }} disabled={streaming || kbIds.length !== 1}>
                <option value="">{kbIds.length > 1 ? "全部已选知识库" : "整个知识库"}</option>
                {documents.map((document) => <option key={document.id} value={document.id}>{document.filename}</option>)}
              </select>
              {kbIds.length > 1 ? (
                <div className="mt-1 text-xs text-slate-400">多知识库联动时不可限定单个文档</div>
              ) : documentId ? (
                <div className="mt-1 flex items-center gap-1 text-xs text-brand-700"><FileText className="h-3 w-3" />仅检索当前文档</div>
              ) : null}
            </div>
            <div className="mt-4">
              <label className="label">问答模型</label>
              <select className="input" value={llmId} onChange={(event) => setLlmId(event.target.value)} disabled={streaming}>
                {llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
            <button type="button" onClick={newConversation} className="btn-secondary mt-4 w-full"><MessageSquarePlus className="h-4 w-4" />新建对话</button>
            <div className="mt-5 rounded-xl bg-slate-50 p-3 text-xs leading-5 text-slate-500">
              <div className="mb-1 flex items-center gap-1.5 font-medium text-slate-700"><Sparkles className="h-3.5 w-3.5 text-brand-600" />跨库检索</div>
              各知识库按自身 Embedding 模型召回，经全局 RRF 融合与统一重排后生成回答。
            </div>
          </aside>

          <section className="panel flex min-h-[72vh] flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
              <div className="flex items-center gap-2 font-semibold text-slate-900"><Bot className="h-5 w-5 text-brand-600" />知源助手</div>
              {conversationId && <span className="chip bg-emerald-50 text-emerald-700">会话已保存</span>}
            </div>

            <div className="thin-scrollbar flex-1 overflow-y-auto px-4 py-6 sm:px-6">
              {!messages.length ? (
                <div className="mx-auto flex min-h-[42vh] max-w-2xl flex-col items-center justify-center text-center">
                  <div className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-brand-100 to-cyan-50 text-brand-700"><Bot className="h-7 w-7" /></div>
                  <h3 className="mt-5 text-lg font-semibold text-slate-900">向知识库提问</h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{documentId ? "回答只基于当前文档，并标注可点击的原文与分块来源。" : `回答只基于已选的 ${selectedKbs.length} 个知识库，并标注可点击的原文与分块来源。`}</p>
                  <div className="mt-6 grid w-full gap-2 sm:grid-cols-3">
                    {prompts.map((prompt) => (
                      <button key={prompt} type="button" onClick={() => setInput(prompt)} className="rounded-xl border border-slate-200 bg-white p-3 text-left text-xs leading-5 text-slate-600 transition hover:border-brand-200 hover:bg-brand-50/50">
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-4xl space-y-7">
                  {messages.map((message) => (
                    <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                      {message.role === "assistant" && <div className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-brand-50 text-brand-700"><Bot className="h-4 w-4" /></div>}
                      <div className={`min-w-0 ${message.role === "user" ? "max-w-[85%]" : "max-w-[92%] flex-1"}`}>
                        <div className={message.role === "user" ? "rounded-2xl rounded-tr-md bg-ink-900 px-4 py-3 text-sm leading-6 text-white" : "rounded-2xl rounded-tl-md border border-slate-200 bg-white px-4 py-4"}>
                          {message.role === "assistant" ? (
                            message.content ? (
                              <article className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></article>
                            ) : (
                              <div className="flex items-center gap-2 text-sm text-slate-400"><LoaderCircle className="h-4 w-4 animate-spin" />正在检索与生成...</div>
                            )
                          ) : (
                            <div className="whitespace-pre-wrap">{message.content}</div>
                          )}
                          {message.streaming && message.content && <span className="ml-1 inline-block h-4 w-1.5 animate-pulse bg-brand-500 align-middle" />}
                          {message.role === "assistant" && <CitationList citations={message.citations} />}
                        </div>
                        {message.role === "assistant" && message.messageId && (
                          <div className="mt-2 flex items-center gap-1 pl-1 text-slate-400">
                            <button type="button" onClick={() => feedbackMutation.mutate({ messageId: message.messageId!, rating: 1 })} className={`rounded-lg p-1.5 hover:bg-slate-100 ${feedback[message.messageId] === 1 ? "text-emerald-600" : ""}`} aria-label="回答有帮助"><ThumbsUp className="h-3.5 w-3.5" /></button>
                            <button type="button" onClick={() => feedbackMutation.mutate({ messageId: message.messageId!, rating: -1 })} className={`rounded-lg p-1.5 hover:bg-slate-100 ${feedback[message.messageId] === -1 ? "text-red-600" : ""}`} aria-label="回答待改进"><ThumbsDown className="h-3.5 w-3.5" /></button>
                          </div>
                        )}
                      </div>
                      {message.role === "user" && <div className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-slate-200 text-slate-600"><UserRound className="h-4 w-4" /></div>}
                    </div>
                  ))}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>

            <div className="border-t border-slate-100 bg-white/95 p-3 sm:p-4">
              {error && <div className="mb-3"><ErrorState message={error} /></div>}
              {warning && <div className="mb-3 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">{warning}</div>}
              <form onSubmit={send} className="mx-auto flex max-w-4xl items-end gap-2 rounded-2xl border border-slate-200 bg-slate-50 p-2 focus-within:border-brand-300 focus-within:ring-4 focus-within:ring-brand-50">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      void send();
                    }
                  }}
                  className="thin-scrollbar max-h-36 min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 outline-none placeholder:text-slate-400"
                  placeholder="输入问题，Enter 发送，Shift + Enter 换行"
                  disabled={streaming}
                />
                {streaming ? (
                  <button type="button" onClick={() => abortRef.current?.abort()} className="grid h-11 w-11 place-items-center rounded-xl bg-slate-200 text-slate-600" aria-label="停止生成"><CircleStop className="h-5 w-5" /></button>
                ) : (
                  <button type="submit" disabled={!input.trim() || !kbIds.length || !llmId} className="grid h-11 w-11 place-items-center rounded-xl bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:opacity-40" aria-label="发送"><Send className="h-5 w-5" /></button>
                )}
              </form>
              <div className="mt-2 flex items-center justify-center gap-1 text-[11px] text-slate-400"><ChevronDown className="h-3 w-3" />回答由模型生成，请结合引用来源核验关键事实。</div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}


