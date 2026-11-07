import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bot,
  ChevronDown,
  CircleStop,
  FileText,
  LoaderCircle,
  Menu,
  MessageSquare,
  MessageSquarePlus,
  Send,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  UserRound,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import { useNavigate, useSearchParams } from "react-router-dom";
import remarkGfm from "remark-gfm";
import { api } from "../api/client";
import {
  ChatMessageView,
  chatRuntimeStore,
  mapServerMessages,
  useChatConversation,
} from "../chat/runtimeStore";
import CitationList from "../components/CitationList";
import KnowledgeBaseSelector from "../components/KnowledgeBaseSelector";
import RetrievalTracePanel from "../components/RetrievalTracePanel";
import { ErrorState, LoadingState, PageHeader } from "../components/UI";
import type { Conversation } from "../types";

function summaryStatusMeta(status: string | undefined) {
  if (status === "running") return { label: "摘要更新中", className: "bg-amber-50 text-amber-700" };
  if (status === "failed") return { label: "摘要失败", className: "bg-red-50 text-red-700" };
  if (status === "degraded") return { label: "降级摘要", className: "bg-amber-50 text-amber-700" };
  if (status === "succeeded") return { label: "摘要已更新", className: "bg-emerald-50 text-emerald-700" };
  return null;
}

function conversationStatus(conversation: Conversation): { label: string; className: string } | null {
  const task = conversation.active_task;
  if (!task) return null;
  if (task.status === "queued") return { label: "排队中", className: "bg-slate-100 text-slate-600" };
  if (task.status === "running") {
    return task.phase === "generating"
      ? { label: "生成中", className: "bg-emerald-50 text-emerald-700" }
      : { label: "检索中", className: "bg-brand-50 text-brand-700" };
  }
  return null;
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
  const conversationsQuery = useQuery({
    queryKey: ["conversations"],
    queryFn: api.listConversations,
    refetchInterval: (query) =>
      (query.state.data ?? []).some(
        (conversation) =>
          conversation.active_task?.status === "queued" ||
          conversation.active_task?.status === "running" ||
          conversation.summary_status === "running",
      )
        ? 2000
        : false,
  });
  const conversationQuery = useQuery({
    queryKey: ["conversation", conversationParam],
    queryFn: () => api.getConversation(conversationParam),
    enabled: !!conversationParam,
    refetchInterval: (query) =>
      query.state.data?.summary_status === "running" ||
      query.state.data?.active_task?.status === "queued" ||
      query.state.data?.active_task?.status === "running"
        ? 3000
        : false,
  });

  const [kbIds, setKbIds] = useState<string[]>(initialKbIds);
  const [documentId, setDocumentId] = useState(documentParam);
  const [llmId, setLlmId] = useState("");
  const [retrievalMode, setRetrievalMode] = useState<"auto" | "single" | "multi">("auto");
  const [input, setInput] = useState("");
  const [localError, setLocalError] = useState("");
  const [sending, setSending] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [feedback, setFeedback] = useState<Record<string, 1 | -1>>({});
  const [sessionOpen, setSessionOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scopeInitializedRef = useRef(false);

  const runtime = useChatConversation(conversationParam);
  const documentsQuery = useQuery({
    queryKey: ["documents", kbIds[0]],
    queryFn: () => api.listDocuments(kbIds[0]),
    enabled: kbIds.length === 1,
  });
  const knowledgeBases = kbQuery.data ?? [];
  const llmModels = modelQuery.data ?? [];
  const conversations = conversationsQuery.data ?? [];
  const documents = (documentsQuery.data ?? []).filter((doc) => doc.status === "ready");
  const selectedKbs = useMemo(
    () => knowledgeBases.filter((kb) => kbIds.includes(kb.id)),
    [knowledgeBases, kbIds],
  );
  const serverMessages = useMemo(
    () => mapServerMessages(conversationQuery.data?.messages ?? []),
    [conversationQuery.data?.messages],
  );
  const messages: ChatMessageView[] = runtime.messages.length ? runtime.messages : serverMessages;
  const streaming = runtime.task?.status === "queued" || runtime.task?.status === "running";
  const busy = sending || streaming;
  const displayError = localError || runtime.error;
  const warning = runtime.warning;

  useEffect(() => {
    if (scopeInitializedRef.current || !knowledgeBases.length) return;
    const availableIds = new Set(knowledgeBases.map((knowledgeBase) => knowledgeBase.id));
    const preferredIds = initialKbIds.filter((kbId) => availableIds.has(kbId));
    setKbIds(preferredIds.length ? preferredIds : [knowledgeBases[0].id]);
    scopeInitializedRef.current = true;
  }, [initialKbIds, knowledgeBases]);

  useEffect(() => {
    if (!selectedKbs.length) return;
    const preferredIds = new Set(
      selectedKbs.map((knowledgeBase) => knowledgeBase.default_llm_profile_id).filter(Boolean),
    );
    const preferred = preferredIds.size === 1 ? [...preferredIds][0] : "";
    if (!llmId || !llmModels.some((model) => model.id === llmId)) {
      setLlmId(preferred || llmModels[0]?.id || "");
    }
  }, [selectedKbs, llmId, llmModels]);

  useEffect(() => {
    if (!conversationQuery.data) return;
    const conversation = conversationQuery.data;
    setKbIds(conversation.knowledge_base_ids);
    setDocumentId(conversation.document_id || "");
    setLlmId(conversation.llm_profile_id || "");
    const mapped = mapServerMessages(conversation.messages);
    const activeTask = conversation.active_task;
    if (!activeTask) {
      chatRuntimeStore.hydrate(conversation.id, mapped);
      return;
    }
    let cancelled = false;
    void api.getChatTask(activeTask.id)
      .then((task) => {
        if (!cancelled) chatRuntimeStore.recover(conversation.id, task, mapped);
      })
      .catch((error) => {
        if (!cancelled) setLocalError((error as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [conversationQuery.data]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  const feedbackMutation = useMutation({
    mutationFn: ({ messageId, rating }: { messageId: string; rating: 1 | -1 }) =>
      api.sendFeedback(messageId, rating),
    onSuccess: (_, variables) => setFeedback((current) => ({ ...current, [variables.messageId]: variables.rating })),
    onError: (err) => setLocalError(err.message),
  });

  async function send(event?: FormEvent) {
    event?.preventDefault();
    const query = input.trim();
    if (!query || !kbIds.length || !llmId || busy) return;
    setLocalError("");
    setInput("");
    setSending(true);
    try {
      const created = await chatRuntimeStore.send(
        conversationParam,
        {
          knowledge_base_ids: kbIds,
          query,
          llm_profile_id: llmId,
          document_id: documentId || undefined,
          retrieval_mode: retrievalMode,
        },
        serverMessages,
      );
      const documentQuery = documentId ? `&document=${documentId}` : "";
      navigate(`/chat?kbs=${kbIds.join(",")}${documentQuery}&conversation=${created.conversation_id}`, { replace: true });
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
    } catch (error) {
      setLocalError((error as Error).message);
    } finally {
      setSending(false);
    }
  }

  async function stopCurrentChat() {
    if (!streaming || stopping) return;
    setStopping(true);
    setLocalError("");
    try {
      await chatRuntimeStore.cancel(conversationParam);
    } catch (error) {
      setLocalError((error as Error).message);
    } finally {
      setStopping(false);
    }
  }

  function openConversation(conversation: Conversation) {
    setSessionOpen(false);
    const documentQuery = conversation.document_id ? `&document=${conversation.document_id}` : "";
    navigate(`/chat?kbs=${conversation.knowledge_base_ids.join(",")}${documentQuery}&conversation=${conversation.id}`);
  }

  function newConversation() {
    setSessionOpen(false);
    setInput("");
    setLocalError("");
    chatRuntimeStore.clear("");
    const documentQuery = documentId ? `&document=${documentId}` : "";
    navigate(`/chat?kbs=${kbIds.join(",")}${documentQuery}`, { replace: true });
  }

  function changeKnowledgeBases(nextIds: string[]) {
    setKbIds(nextIds);
    setDocumentId("");
    setInput("");
    setLocalError("");
    chatRuntimeStore.clear("");
    const query = nextIds.length ? `?kbs=${nextIds.join(",")}` : "";
    navigate(`/chat${query}`, { replace: true });
  }

  if (kbQuery.isLoading || modelQuery.isLoading) return <LoadingState />;
  if (kbQuery.error) return <ErrorState message={kbQuery.error.message} />;
  if (modelQuery.error) return <ErrorState message={modelQuery.error.message} />;
  if (conversationParam && conversationQuery.error) return <ErrorState message={conversationQuery.error.message} />;
  const summaryStatus = summaryStatusMeta(conversationQuery.data?.summary_status);
  const activeTitle = conversationQuery.data?.title || "新对话";

  const conversationList = (
    <div className="flex h-full flex-col">
      <button type="button" onClick={newConversation} className="btn-secondary w-full">
        <MessageSquarePlus className="h-4 w-4" />新建对话
      </button>
      <div className="mt-4 flex-1 space-y-1 overflow-y-auto pr-1">
        {conversations.length ? conversations.map((conversation) => {
          const status = conversationStatus(conversation);
          const active = conversation.id === conversationParam;
          return (
            <button
              key={conversation.id}
              type="button"
              onClick={() => openConversation(conversation)}
              className={`w-full rounded-xl px-3 py-3 text-left transition ${active ? "bg-brand-50 text-brand-800" : "text-slate-600 hover:bg-slate-50"}`}
            >
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4 shrink-0" />
                <span className="min-w-0 flex-1 truncate text-sm font-medium">{conversation.title}</span>
                {status && <span className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] ${status.className}`}>{status.label}</span>}
              </div>
              <div className="mt-1 truncate pl-6 text-[11px] text-slate-400">
                {new Date(conversation.updated_at).toLocaleString("zh-CN", { hour12: false })}
              </div>
            </button>
          );
        }) : (
          <p className="px-3 py-8 text-center text-xs leading-5 text-slate-400">还没有会话。发送第一个问题后会自动保存。</p>
        )}
      </div>
    </div>
  );

  return (
    <div>
      <PageHeader title="智能问答" description="支持多会话并行生成、切换与断点恢复；每条回答都会标注知识库、文档和章节来源。" />
      {!knowledgeBases.length || !llmModels.length ? (
        <div className="panel p-8 text-center"><ErrorState message="请先配置 Embedding、问答模型并创建知识库。" /></div>
      ) : (
        <div className="grid min-h-[72vh] gap-5 xl:grid-cols-[240px_280px_minmax(0,1fr)]">
          <aside className="panel hidden h-[72vh] p-4 xl:sticky xl:top-24 xl:block">{conversationList}</aside>

          <aside className="panel h-fit p-4 xl:sticky xl:top-24">
            <div>
              <label className="label">知识库范围</label>
              <KnowledgeBaseSelector knowledgeBases={knowledgeBases} value={kbIds} onChange={changeKnowledgeBases} disabled={busy} />
            </div>
            <div className="mt-4">
              <label className="label">问答范围</label>
              <select
                className="input"
                value={documentId}
                onChange={(event) => {
                  const nextDocument = event.target.value;
                  setDocumentId(nextDocument);
                  chatRuntimeStore.clear("");
                  const query = nextDocument ? `&document=${nextDocument}` : "";
                  navigate(`/chat?kbs=${kbIds.join(",")}${query}`, { replace: true });
                }}
                disabled={busy || kbIds.length !== 1}
              >
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
              <select className="input" value={llmId} onChange={(event) => setLlmId(event.target.value)} disabled={busy}>
                {llmModels.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
              </select>
            </div>
            <div className="mt-4">
              <label className="label">检索模式</label>
              <select className="input" value={retrievalMode} onChange={(event) => setRetrievalMode(event.target.value as "auto" | "single" | "multi")} disabled={busy}>
                <option value="auto">自动判断单跳/多跳</option>
                <option value="single">强制单跳</option>
                <option value="multi">强制多跳</option>
              </select>
            </div>
            <div className="mt-5 rounded-xl bg-slate-50 p-3 text-xs leading-5 text-slate-500">
              <div className="mb-1 flex items-center gap-1.5 font-medium text-slate-700"><Sparkles className="h-3.5 w-3.5 text-brand-600" />多会话并行</div>
              不同会话的任务独立执行；切换会话或离开页面不会停止后台回答。
            </div>
          </aside>

          <section className="panel flex min-h-[72vh] flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-100 px-5 py-3.5">
              <div className="flex min-w-0 items-center gap-2">
                <button type="button" onClick={() => setSessionOpen(true)} className="grid h-9 w-9 place-items-center rounded-xl bg-slate-100 text-slate-600 xl:hidden" aria-label="打开会话列表"><Menu className="h-4 w-4" /></button>
                <Bot className="h-5 w-5 shrink-0 text-brand-600" />
                <span className="truncate font-semibold text-slate-900">{activeTitle}</span>
              </div>
              <div className="flex items-center gap-2">
                {streaming && <span className="chip bg-emerald-50 text-emerald-700"><LoaderCircle className="h-3 w-3 animate-spin" />{runtime.task?.phase === "generating" ? "生成中" : "检索中"}</span>}
                {summaryStatus && <span className={`chip ${summaryStatus.className}`}>{summaryStatus.label}</span>}
              </div>
            </div>

            <div className="thin-scrollbar flex-1 overflow-y-auto px-4 py-6 sm:px-6">
              {!messages.length ? (
                <div className="mx-auto flex min-h-[42vh] max-w-2xl flex-col items-center justify-center text-center">
                  <div className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-brand-100 to-cyan-50 text-brand-700"><Bot className="h-7 w-7" /></div>
                  <h3 className="mt-5 text-lg font-semibold text-slate-900">向知识库提问</h3>
                  <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">{documentId ? "回答只基于当前文档，并标注可点击的原文与分块来源。" : `回答只基于已选的 ${selectedKbs.length} 个知识库，并标注可点击的原文与分块来源。`}</p>
                  <div className="mt-6 grid w-full gap-2 sm:grid-cols-3">
                    {prompts.map((prompt) => <button key={prompt} type="button" onClick={() => setInput(prompt)} className="rounded-xl border border-slate-200 bg-white p-3 text-left text-xs leading-5 text-slate-600 transition hover:border-brand-200 hover:bg-brand-50/50">{prompt}</button>)}
                  </div>
                </div>
              ) : (
                <div className="mx-auto max-w-4xl space-y-6">
                  {messages.map((message) => (
                    <div key={message.id} className={`flex gap-3 ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                      {message.role === "assistant" && <div className="grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-brand-600 text-white"><Bot className="h-4 w-4" /></div>}
                      <div className={`min-w-0 max-w-[85%] ${message.role === "user" ? "text-right" : "text-left"}`}>
                        <div className={`inline-block rounded-2xl px-4 py-3 text-sm leading-7 ${message.role === "user" ? "bg-brand-600 text-white" : "bg-slate-50 text-slate-800"}`}>
                          {message.role === "assistant" ? (
                            <>
                              {message.streaming && !message.content && <div className="flex items-center gap-2 text-slate-500"><LoaderCircle className="h-4 w-4 animate-spin" />{message.phase === "generating" ? "正在生成回答" : "正在检索知识库"}</div>}
                              {message.content && <div className="prose prose-sm max-w-none prose-slate"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>}
                              {!message.streaming && message.stopped && (
                                <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-500"><CircleStop className="h-3.5 w-3.5" />{message.stoppedStage === "retrieving" ? "已停止（检索阶段）" : message.stoppedStage === "generating" ? "已停止（生成阶段）" : "已停止"}</div>
                              )}
                            </>
                          ) : <div className="whitespace-pre-wrap">{message.content}</div>}
                          {message.streaming && message.content && <span className="ml-1 inline-block h-4 w-1.5 animate-pulse bg-brand-500 align-middle" />}
                          {message.role === "assistant" && message.retrievalTrace && <RetrievalTracePanel trace={message.retrievalTrace} />}
                          {message.role === "assistant" && message.uncertaintyNotice && !message.content.includes(message.uncertaintyNotice) && <div className="mt-2 rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">{message.uncertaintyNotice}</div>}
                          {message.role === "assistant" && <CitationList citations={message.citations} />}
                        </div>
                        {message.role === "assistant" && message.messageId && !message.stopped && !message.streaming && (
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
              {displayError && <div className="mb-3"><ErrorState message={displayError} /></div>}
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
                  disabled={busy}
                />
                {streaming ? (
                  <button type="button" onClick={stopCurrentChat} disabled={stopping} className="flex h-11 items-center gap-1.5 rounded-xl bg-slate-200 px-3 text-xs font-medium text-slate-700 transition hover:bg-slate-300 disabled:cursor-wait disabled:opacity-70" aria-label="停止当前问答"><CircleStop className="h-4 w-4" />{stopping ? "正在停止" : "停止"}</button>
                ) : (
                  <button type="submit" disabled={!input.trim() || !kbIds.length || !llmId || sending} className="grid h-11 w-11 place-items-center rounded-xl bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:opacity-40" aria-label="发送"><Send className="h-5 w-5" /></button>
                )}
              </form>
              <div className="mt-2 flex items-center justify-center gap-1 text-[11px] text-slate-400"><ChevronDown className="h-3 w-3" />回答由模型生成，请结合引用来源核验关键事实。</div>
            </div>
          </section>
        </div>
      )}

      {sessionOpen && (
        <div className="fixed inset-0 z-50 flex xl:hidden">
          <button type="button" className="absolute inset-0 bg-slate-950/30" onClick={() => setSessionOpen(false)} aria-label="关闭会话列表" />
          <aside className="relative z-10 h-full w-[300px] max-w-[85vw] bg-white p-4 shadow-2xl">
            <div className="mb-3 flex items-center justify-between"><div className="font-semibold text-slate-900">会话</div><button type="button" onClick={() => setSessionOpen(false)} className="grid h-8 w-8 place-items-center rounded-lg text-slate-500 hover:bg-slate-100"><X className="h-4 w-4" /></button></div>
            {conversationList}
          </aside>
        </div>
      )}
    </div>
  );
}
