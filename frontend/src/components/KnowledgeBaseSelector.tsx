import { Check, Database, X } from "lucide-react";
import type { KnowledgeBase } from "../types";

const MAX_KNOWLEDGE_BASES = 20;

interface KnowledgeBaseSelectorProps {
  knowledgeBases: KnowledgeBase[];
  value: string[];
  onChange: (ids: string[]) => void;
  disabled?: boolean;
}

export default function KnowledgeBaseSelector({
  knowledgeBases,
  value,
  onChange,
  disabled = false,
}: KnowledgeBaseSelectorProps) {
  const selected = new Set(value);
  const selectedNames = knowledgeBases
    .filter((knowledgeBase) => selected.has(knowledgeBase.id))
    .map((knowledgeBase) => knowledgeBase.name);

  function toggle(id: string) {
    if (disabled) return;
    if (!selected.has(id) && value.length >= MAX_KNOWLEDGE_BASES) return;
    onChange(
      selected.has(id)
        ? value.filter((knowledgeBaseId) => knowledgeBaseId !== id)
        : [...value, id],
    );
  }

  return (
    <div className="rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-1.5 text-xs font-medium text-slate-700">
          <Database className="h-3.5 w-3.5 shrink-0 text-brand-600" />
          <span>已选 {value.length} 个知识库</span>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => onChange(knowledgeBases.slice(0, MAX_KNOWLEDGE_BASES).map((knowledgeBase) => knowledgeBase.id))}
            disabled={disabled || value.length === knowledgeBases.length}
            className="rounded-lg px-2 py-1 text-[11px] text-brand-700 hover:bg-brand-50 disabled:opacity-40"
          >
            <Check className="mr-1 inline h-3 w-3" />全选
          </button>
          <button
            type="button"
            onClick={() => onChange([])}
            disabled={disabled || !value.length}
            className="rounded-lg px-2 py-1 text-[11px] text-slate-500 hover:bg-slate-50 disabled:opacity-40"
          >
            <X className="mr-1 inline h-3 w-3" />清空
          </button>
        </div>
      </div>
      <div className="thin-scrollbar max-h-48 space-y-0.5 overflow-y-auto p-1.5">
        {knowledgeBases.map((knowledgeBase) => (
          <label
            key={knowledgeBase.id}
            className={`flex items-start gap-2 rounded-lg px-2 py-2 text-sm transition ${
              disabled ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-slate-50"
            }`}
          >
            <input
              type="checkbox"
              className="mt-0.5"
              checked={selected.has(knowledgeBase.id)}
              onChange={() => toggle(knowledgeBase.id)}
              disabled={disabled || (!selected.has(knowledgeBase.id) && value.length >= MAX_KNOWLEDGE_BASES)}
            />
            <span className="min-w-0">
              <span className="block truncate font-medium text-slate-700">{knowledgeBase.name}</span>
              {knowledgeBase.description && (
                <span className="mt-0.5 block line-clamp-1 text-[11px] text-slate-400">
                  {knowledgeBase.description}
                </span>
              )}
            </span>
          </label>
        ))}
      </div>
      {selectedNames.length > 0 && (
        <div className="flex flex-wrap gap-1 border-t border-slate-100 px-2.5 py-2">
          {selectedNames.map((name) => (
            <span key={name} className="chip bg-brand-50 text-brand-700">{name}</span>
          ))}
        </div>
      )}
    </div>
  );
}
