import re

from app.chunking.semantic_chunker import token_spans


def _single_line(value: str, max_chars: int = 240) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    return compact[:max_chars]


def build_context_prefix(
    document_title: str,
    section_path: str,
    *,
    summary: str = "",
    summary_label: str = "上级摘要",
    summary_chars: int = 120,
) -> str:
    parts = [
        f"文档: {_single_line(document_title, 160) or '未命名文档'}",
        f"章节: {_single_line(section_path, 240) or '正文'}",
    ]
    if summary.strip():
        parts.append(f"{summary_label}: {_single_line(summary, summary_chars)}")
    return "[" + " / ".join(parts) + "]"


def with_context_prefix(prefix: str, content: str) -> str:
    prefix = prefix.strip()
    return f"{prefix}\n{content}" if prefix else content


def clip_head_tokens(text: str, max_tokens: int) -> str:
    spans = token_spans(text)
    if not spans or max_tokens <= 0:
        return ""
    return text[: spans[min(len(spans), max_tokens) - 1][1]].strip()


def clip_tail_tokens(text: str, max_tokens: int) -> str:
    spans = token_spans(text)
    if not spans or max_tokens <= 0:
        return ""
    start = max(0, len(spans) - max_tokens)
    return text[spans[start][0] :].strip()


def clip_around_anchor(
    parent_text: str,
    child_text: str,
    *,
    before_tokens: int,
    after_tokens: int,
) -> str:
    if not parent_text.strip() or not child_text.strip():
        return ""
    anchor_start = parent_text.find(child_text)
    if anchor_start < 0:
        return ""
    anchor_end = anchor_start + len(child_text)
    spans = token_spans(parent_text)
    if not spans:
        return ""
    first = next(
        (index for index, (_, end) in enumerate(spans) if end > anchor_start),
        0,
    )
    last = next(
        (index for index in range(len(spans) - 1, -1, -1) if spans[index][0] < anchor_end),
        len(spans) - 1,
    )
    start = max(0, first - max(0, before_tokens))
    end = min(len(spans), last + max(0, after_tokens) + 1)
    return parent_text[spans[start][0] : spans[end - 1][1]].strip()
