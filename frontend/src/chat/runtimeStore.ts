import { useSyncExternalStore } from "react";
import { api, streamChatTaskEvents } from "../api/client";
import type {
  ChatStage,
  ChatTaskCreated,
  ChatTaskCreatePayload,
  Citation,
  ComplexityDecision,
  GenerationTask,
  GenerationTaskStatus,
  MultiHopStep,
  RetrievalTrace,
} from "../types";

export interface ChatMessageView {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  messageId?: string;
  streaming?: boolean;
  phase?: ChatStage;
  stopped?: boolean;
  stoppedStage?: ChatStage;
  retrievalTrace?: RetrievalTrace;
  uncertaintyNotice?: string;
}

interface RuntimeTask {
  taskId: string;
  conversationId: string;
  assistantMessageId: string;
  lastEventId: string | null;
  status: GenerationTaskStatus;
  phase: ChatStage;
  controller: AbortController | null;
  connecting: boolean;
}

export interface ConversationRuntime {
  messages: ChatMessageView[];
  task: RuntimeTask | null;
  error: string;
  warning: string;
}

const EMPTY_STATE: ConversationRuntime = { messages: [], task: null, error: "", warning: "" };
const DRAFT_KEY = "__draft__";

function parseStage(value: unknown): ChatStage | undefined {
  return value === "retrieving" || value === "generating" ? value : undefined;
}

function newId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

class ChatRuntimeStore {
  private states = new Map<string, ConversationRuntime>();
  private listeners = new Map<string, Set<() => void>>();
  private terminalHandler: ((conversationId: string) => void) | null = null;

  setTerminalHandler(handler: ((conversationId: string) => void) | null): void {
    this.terminalHandler = handler;
  }

  get(conversationId: string): ConversationRuntime {
    return this.states.get(conversationId || DRAFT_KEY) ?? EMPTY_STATE;
  }

  subscribe(conversationId: string, listener: () => void): () => void {
    const key = conversationId || DRAFT_KEY;
    const listeners = this.listeners.get(key) ?? new Set();
    listeners.add(listener);
    this.listeners.set(key, listeners);
    return () => {
      listeners.delete(listener);
      if (!listeners.size) this.listeners.delete(key);
    };
  }

  hydrate(conversationId: string, messages: ChatMessageView[]): void {
    const current = this.get(conversationId);
    if (current.task && (current.task.status === "queued" || current.task.status === "running")) return;
    this.set(conversationId, { messages, task: null, error: "", warning: "" });
  }

  clear(conversationId: string): void {
    this.states.delete(conversationId || DRAFT_KEY);
    this.notify(conversationId || DRAFT_KEY);
  }

  async send(
    conversationId: string,
    payload: Omit<ChatTaskCreatePayload, "conversation_id" | "client_request_id">,
    baseMessages: ChatMessageView[],
  ): Promise<ChatTaskCreated> {
    const key = conversationId || DRAFT_KEY;
    const current = this.get(key);
    const messages = current.messages.length ? current.messages : baseMessages;
    const localUser: ChatMessageView = {
      id: `pending-user-${newId()}`,
      role: "user",
      content: payload.query,
      citations: [],
    };
    const localAssistant: ChatMessageView = {
      id: `pending-assistant-${newId()}`,
      role: "assistant",
      content: "",
      citations: [],
      streaming: true,
      phase: "retrieving",
    };
    this.set(key, {
      messages: [...messages, localUser, localAssistant],
      task: null,
      error: "",
      warning: "",
    });

    try {
      const created = await api.createChatTask({
        ...payload,
        conversation_id: conversationId || undefined,
        client_request_id: newId(),
      });
      const targetKey = created.conversation_id;
      const latest = this.get(key);
      const nextMessages = latest.messages.map((message) => {
        if (message.id !== localUser.id && message.id !== localAssistant.id) return message;
        const isAssistant = message.id === localAssistant.id;
        return {
          ...message,
          id: isAssistant ? created.assistant_message_id : created.user_message_id,
          messageId: isAssistant ? created.assistant_message_id : undefined,
        };
      });
      const task: RuntimeTask = {
        taskId: created.task_id,
        conversationId: created.conversation_id,
        assistantMessageId: created.assistant_message_id,
        lastEventId: created.last_event_id,
        status: created.status,
        phase: "retrieving",
        controller: null,
        connecting: false,
      };
      if (key !== targetKey) {
        this.states.delete(key);
        this.listeners.delete(key);
      }
      this.set(targetKey, { messages: nextMessages, task, error: "", warning: "" });
      this.ensureTaskStream(targetKey);
      return created;
    } catch (error) {
      this.updateMessage(key, localAssistant.id, { streaming: false });
      this.patch(key, { error: (error as Error).message });
      throw error;
    }
  }

  recover(conversationId: string, task: GenerationTask, serverMessages: ChatMessageView[]): void {
    const current = this.get(conversationId);
    if (current.task?.taskId === task.id) return;
    const messages = serverMessages.length ? [...serverMessages] : [...current.messages];
    const assistantIndex = messages.findIndex((message) => message.id === task.assistant_message_id);
    const recovered: ChatMessageView = {
      id: task.assistant_message_id,
      messageId: task.assistant_message_id,
      role: "assistant",
      content: task.content,
      citations: task.citations,
      streaming: task.status === "queued" || task.status === "running",
      phase: task.phase,
      stopped: task.status === "cancelled",
      stoppedStage: task.phase,
      error: task.error ?? undefined,
    } as ChatMessageView & { error?: string };
    if (assistantIndex >= 0) messages[assistantIndex] = { ...messages[assistantIndex], ...recovered };
    else messages.push(recovered);
    const runtimeTask: RuntimeTask = {
      taskId: task.id,
      conversationId,
      assistantMessageId: task.assistant_message_id,
      lastEventId: task.last_event_id,
      status: task.status,
      phase: task.phase,
      controller: null,
      connecting: false,
    };
    this.set(conversationId, { messages, task: runtimeTask, error: task.error ?? "", warning: "" });
    if (task.status === "queued" || task.status === "running") this.ensureTaskStream(conversationId);
  }

  async cancel(conversationId: string): Promise<void> {
    const state = this.get(conversationId);
    const task = state.task;
    if (!task) return;
    try {
      await api.cancelChatTask(task.taskId);
    } finally {
      task.controller?.abort();
      this.patchTask(conversationId, { status: "cancelled", controller: null, connecting: false });
      this.updateAssistant(conversationId, task.assistantMessageId, {
        streaming: false,
        stopped: true,
        stoppedStage: task.phase,
      });
      this.terminalHandler?.(conversationId);
    }
  }

  private set(conversationId: string, state: ConversationRuntime): void {
    this.states.set(conversationId || DRAFT_KEY, state);
    this.notify(conversationId || DRAFT_KEY);
  }

  private patch(conversationId: string, patch: Partial<ConversationRuntime>): void {
    const key = conversationId || DRAFT_KEY;
    const current = this.get(key);
    this.states.set(key, { ...current, ...patch });
    this.notify(key);
  }

  private patchTask(conversationId: string, patch: Partial<RuntimeTask>): void {
    const current = this.get(conversationId);
    if (!current.task) return;
    this.patch(conversationId, { task: { ...current.task, ...patch } });
  }

  private updateMessage(
    conversationId: string,
    messageId: string,
    patch: Partial<ChatMessageView>,
  ): void {
    const current = this.get(conversationId);
    this.patch(conversationId, {
      messages: current.messages.map((message) =>
        message.id === messageId ? { ...message, ...patch } : message
      ),
    });
  }

  private updateAssistant(conversationId: string, messageId: string, patch: Partial<ChatMessageView>): void {
    this.updateMessage(conversationId, messageId, patch);
  }

  private notify(conversationId: string): void {
    for (const listener of this.listeners.get(conversationId) ?? []) listener();
  }

  private ensureTaskStream(conversationId: string): void {
    const task = this.get(conversationId).task;
    if (!task || task.connecting || task.status === "completed" || task.status === "cancelled" || task.status === "failed") return;
    void this.connectTask(conversationId, task.taskId);
  }

  private async connectTask(conversationId: string, taskId: string): Promise<void> {
    this.patchTask(conversationId, { connecting: true });
    let retries = 0;
    while (true) {
      const current = this.get(conversationId);
      const task = current.task;
      if (!task || task.taskId !== taskId) return;
      if (task.status === "completed" || task.status === "cancelled" || task.status === "failed") {
        this.patchTask(conversationId, { connecting: false, controller: null });
        return;
      }
      const controller = new AbortController();
      this.patchTask(conversationId, { controller });
      let terminal = false;
      let tokenBuffer = "";
      let flushTimer: number | undefined;
      const flush = () => {
        if (!tokenBuffer) return;
        const content = tokenBuffer;
        tokenBuffer = "";
        const message = this.get(conversationId).messages.find((item) => item.id === task.assistantMessageId);
        this.updateAssistant(conversationId, task.assistantMessageId, {
          content: `${message?.content ?? ""}${content}`,
        });
      };
      try {
        await streamChatTaskEvents(
          taskId,
          {
            onEvent: (event, rawData, eventId) => {
              const data = asRecord(rawData);
              if (eventId) this.patchTask(conversationId, { lastEventId: eventId });
              if (event === "token") {
                tokenBuffer += String(data.content ?? "");
                if (flushTimer === undefined) {
                  flushTimer = window.setTimeout(() => {
                    flushTimer = undefined;
                    flush();
                  }, 50);
                }
              } else if (event === "stage" || event === "queued") {
                const phase = parseStage(data.stage) ?? "retrieving";
                this.patchTask(conversationId, { phase, status: "running" });
                this.updateAssistant(conversationId, task.assistantMessageId, { phase });
              } else if (event === "start") {
                this.patchTask(conversationId, { phase: "generating", status: "running" });
                this.updateAssistant(conversationId, task.assistantMessageId, { phase: "generating" });
                const failedIds = Array.isArray(data.failed_knowledge_base_ids)
                  ? data.failed_knowledge_base_ids.map(String)
                  : [];
                if (failedIds.length) {
                  this.patch(conversationId, { warning: `部分知识库检索失败：${failedIds.join("、")}` });
                }
              } else if (event === "retrieval_plan") {
                const previous = this.get(conversationId).messages.find((item) => item.id === task.assistantMessageId)?.retrievalTrace;
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  retrievalTrace: {
                    requested_mode: previous?.requested_mode ?? "auto",
                    complexity: data as unknown as ComplexityDecision,
                    steps: previous?.steps ?? [],
                    stop_reason: "",
                    uncertain: false,
                    uncertainty_notice: "",
                    total_rounds: 0,
                    total_chunks: 0,
                  },
                });
              } else if (event === "retrieval_step") {
                const previous = this.get(conversationId).messages.find((item) => item.id === task.assistantMessageId)?.retrievalTrace;
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  retrievalTrace: {
                    requested_mode: previous?.requested_mode ?? "auto",
                    complexity: previous?.complexity ?? null,
                    steps: [...(previous?.steps ?? []), data as unknown as MultiHopStep],
                    stop_reason: previous?.stop_reason ?? "",
                    uncertain: previous?.uncertain ?? false,
                    uncertainty_notice: previous?.uncertainty_notice ?? "",
                    total_rounds: previous?.total_rounds ?? 0,
                    total_chunks: previous?.total_chunks ?? 0,
                  },
                });
              } else if (event === "retrieval_done") {
                const previous = this.get(conversationId).messages.find((item) => item.id === task.assistantMessageId)?.retrievalTrace;
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  retrievalTrace: {
                    requested_mode: previous?.requested_mode ?? "auto",
                    complexity: previous?.complexity ?? null,
                    steps: previous?.steps ?? [],
                    stop_reason: String(data.stop_reason ?? ""),
                    uncertain: Boolean(data.uncertain),
                    uncertainty_notice: String(data.uncertainty_notice ?? ""),
                    total_rounds: Number(data.total_rounds ?? 0),
                    total_chunks: Number(data.total_chunks ?? 0),
                  },
                });
              } else if (event === "citations") {
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  citations: (data.citations ?? []) as Citation[],
                });
              } else if (event === "notice") {
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  uncertaintyNotice: String(data.message ?? ""),
                });
              } else if (event === "done") {
                terminal = true;
                flush();
                this.patchTask(conversationId, { status: "completed", connecting: false, controller: null });
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  messageId: String(data.message_id ?? task.assistantMessageId),
                  streaming: false,
                  phase: undefined,
                });
                this.terminalHandler?.(conversationId);
              } else if (event === "cancelled") {
                terminal = true;
                flush();
                const stoppedStage = parseStage(data.stopped_stage) ?? this.get(conversationId).task?.phase ?? "retrieving";
                this.patchTask(conversationId, { status: "cancelled", phase: stoppedStage, connecting: false, controller: null });
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  streaming: false,
                  stopped: true,
                  stoppedStage,
                });
                this.terminalHandler?.(conversationId);
              } else if (event === "error") {
                terminal = true;
                flush();
                this.patchTask(conversationId, { status: "failed", connecting: false, controller: null });
                this.patch(conversationId, { error: String(data.message ?? "生成回答失败") });
                this.updateAssistant(conversationId, task.assistantMessageId, { streaming: false });
                this.terminalHandler?.(conversationId);
              } else if (event === "snapshot") {
                const snapshot = asRecord(data);
                const snapshotStatus = String(snapshot.status ?? "running") as GenerationTaskStatus;
                const snapshotPhase = parseStage(snapshot.phase) ?? "retrieving";
                this.updateAssistant(conversationId, task.assistantMessageId, {
                  content: String(snapshot.content ?? ""),
                  citations: (snapshot.citations ?? []) as Citation[],
                  phase: snapshotPhase,
                  streaming: snapshotStatus === "queued" || snapshotStatus === "running",
                  stopped: snapshotStatus === "cancelled",
                  stoppedStage: snapshotPhase,
                });
                this.patchTask(conversationId, { status: snapshotStatus, phase: snapshotPhase });
                if (snapshotStatus === "completed" || snapshotStatus === "cancelled" || snapshotStatus === "failed") {
                  terminal = true;
                  this.patchTask(conversationId, { connecting: false, controller: null });
                  if (snapshotStatus === "failed") {
                    this.patch(conversationId, { error: String(snapshot.error ?? "生成回答失败") });
                  }
                  this.terminalHandler?.(conversationId);
                }
              }
            },
            onError: () => undefined,
          },
          controller.signal,
          this.get(conversationId).task?.lastEventId,
        );
        if (flushTimer !== undefined) window.clearTimeout(flushTimer);
        flush();
        if (terminal) return;
        retries += 1;
        if (retries > 3) throw new Error("流式连接已中断");
      } catch (error) {
        if (flushTimer !== undefined) window.clearTimeout(flushTimer);
        flush();
        if ((error as Error).name === "AbortError") return;
        if (terminal) return;
        retries += 1;
        if (retries > 3) {
          this.patchTask(conversationId, { status: "failed", connecting: false, controller: null });
          this.patch(conversationId, { error: (error as Error).message });
          this.updateAssistant(conversationId, task.assistantMessageId, { streaming: false });
          this.terminalHandler?.(conversationId);
          return;
        }
      }
      await new Promise((resolve) => window.setTimeout(resolve, 500 * retries));
    }
  }
}

export const chatRuntimeStore = new ChatRuntimeStore();

export function useChatConversation(conversationId: string): ConversationRuntime {
  const key = conversationId || DRAFT_KEY;
  return useSyncExternalStore(
    (listener) => chatRuntimeStore.subscribe(key, listener),
    () => chatRuntimeStore.get(key),
    () => chatRuntimeStore.get(key),
  );
}

export function mapServerMessages(messages: Array<{
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  usage: Record<string, unknown>;
}>): ChatMessageView[] {
  return messages.map((message) => {
    const usage = message.usage ?? {};
    const trace = usage.retrieval_trace as RetrievalTrace | undefined;
    return {
      id: message.id,
      role: message.role,
      content: message.content,
      citations: message.citations,
      messageId: message.role === "assistant" ? message.id : undefined,
      streaming: (message as { status?: string }).status === "streaming",
      stopped: message.role === "assistant" && usage.stopped === true,
      stoppedStage: parseStage(usage.stopped_stage),
      retrievalTrace: trace,
      uncertaintyNotice: trace?.uncertainty_notice ?? "",
    };
  });
}
