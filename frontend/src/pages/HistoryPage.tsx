import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowUpRight, MessageSquare, Trash2 } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingState, PageHeader, formatDate } from "../components/UI";

export default function HistoryPage() {
  const queryClient = useQueryClient();
  const conversationsQuery = useQuery({ queryKey: ["conversations"], queryFn: api.listConversations });
  const deleteMutation = useMutation({
    mutationFn: api.deleteConversation,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["conversations"] }),
  });

  if (conversationsQuery.isLoading) return <LoadingState />;
  if (conversationsQuery.error) return <ErrorState message={conversationsQuery.error.message} />;
  const conversations = conversationsQuery.data ?? [];

  return (
    <div>
      <PageHeader title="历史问答" description="查看已保存的会话、回答引用与对话摘要。打开会话后可继续追问。" />
      {(deleteMutation.error) && <div className="mb-4"><ErrorState message={deleteMutation.error.message} /></div>}
      {!conversations.length ? (
        <div className="panel">
          <EmptyState
            title="还没有历史会话"
            description="前往智能问答开始第一次对话，消息会自动保存。"
            action={<Link to="/chat" className="btn-primary">开始问答</Link>}
          />
        </div>
      ) : (
        <div className="panel overflow-hidden">
          <div className="divide-y divide-slate-100">
            {conversations.map((conversation) => (
              <div key={conversation.id} className="group flex items-center gap-4 px-5 py-4 transition hover:bg-slate-50/70">
                <div className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-brand-50 text-brand-700"><MessageSquare className="h-5 w-5" /></div>
                <div className="min-w-0 flex-1">
                  <Link to={`/chat?kbs=${conversation.knowledge_base_ids.join(",")}&conversation=${conversation.id}`} className="block truncate font-medium text-slate-900 hover:text-brand-700">{conversation.title}</Link>
                  <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
                    <span>{conversation.message_count} 条消息</span>
                    <span>{conversation.knowledge_base_names.join("、") || "知识库已删除"}</span>
                    <span>更新于 {formatDate(conversation.updated_at)}</span>
                    {conversation.summary && <span className="text-emerald-600">已生成摘要</span>}
                  </div>
                  {conversation.summary && <p className="mt-2 line-clamp-1 text-xs text-slate-400">{conversation.summary}</p>}
                </div>
                <Link to={`/chat?kbs=${conversation.knowledge_base_ids.join(",")}&conversation=${conversation.id}`} className="grid h-9 w-9 place-items-center rounded-full text-slate-400 hover:bg-brand-50 hover:text-brand-700" aria-label="打开会话"><ArrowUpRight className="h-4 w-4" /></Link>
                <button
                  type="button"
                  onClick={() => {
                    if (window.confirm("确认删除此会话？")) deleteMutation.mutate(conversation.id);
                  }}
                  className="grid h-9 w-9 place-items-center rounded-full text-slate-400 hover:bg-red-50 hover:text-red-600"
                  aria-label="删除会话"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
