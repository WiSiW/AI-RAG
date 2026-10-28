import math
import re
from collections.abc import Awaitable, Callable

from app.chunking.semantic_chunker import ChunkData, SemanticChunker, count_tokens, token_spans
from app.parsing.base import ParsedDocument, ParsedElement

CHUNK_MODES: list[dict[str, str]] = [
    {
        "value": "structure",
        "label": "结构感知（推荐）",
        "description": "按标题层级划分章节，再按 token 窗口切分，保留完整章节路径。",
    },
    {
        "value": "recursive",
        "label": "递归字符",
        "description": "依次按段落、句子和字符边界切分，适合通用长文本。",
    },
    {
        "value": "fixed",
        "label": "固定 token 窗口",
        "description": "严格按 token 数滑动切分，行为稳定，适合基线评测。",
    },
    {
        "value": "paragraph",
        "label": "段落聚合",
        "description": "优先保留完整段落，超过上限时再按 token 拆分。",
    },
    {
        "value": "sentence",
        "label": "句子聚合",
        "description": "按中英文句末标点切分并聚合，减少句子被截断。",
    },
    {
        "value": "semantic",
        "label": "语义边界",
        "description": "计算相邻语句向量相似度，在主题变化处切分，需要 Embedding 模型。",
    },
    {
        "value": "parent_child",
        "label": "父子分块",
        "description": "子块用于精确召回，命中后自动扩展父块作为问答上下文。",
    },
]
CHUNK_MODE_VALUES = {mode["value"] for mode in CHUNK_MODES}
EmbeddingFunction = Callable[[list[str]], Awaitable[list[list[float]]]]

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n+")
TAIL_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "；", ";", " ", "")


def validate_chunk_mode(mode: str) -> str:
    if mode not in CHUNK_MODE_VALUES:
        raise ValueError(f"不支持的分块模式：{mode}")
    return mode


def _group_sections(elements: list[ParsedElement]) -> list[tuple[str, list[ParsedElement]]]:
    sections: list[tuple[str, list[ParsedElement]]] = []
    stack: list[str] = []
    buffer: list[ParsedElement] = []

    def flush() -> None:
        if not buffer:
            return
        path = " > ".join(stack) if stack else "正文"
        sections.append((path, list(buffer)))
        buffer.clear()

    for element in elements:
        if element.type == "title":
            flush()
            while stack and len(stack) >= element.level:
                stack.pop()
            stack.append(element.text)
            buffer.append(element)
        else:
            buffer.append(element)
    flush()
    return sections


def _page_metadata(elements: list[ParsedElement]) -> dict:
    pages = sorted(
        {
            int(element.metadata["page"])
            for element in elements
            if element.metadata.get("page") is not None
        }
    )
    return {"pages": pages} if pages else {}


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    spans = token_spans(text)
    if not spans:
        return []
    step = max(1, chunk_size - chunk_overlap)
    pieces: list[str] = []
    start = 0
    while start < len(spans):
        end = min(start + chunk_size, len(spans))
        pieces.append(text[spans[start][0] : spans[end - 1][1]].strip())
        if end >= len(spans):
            break
        start += step
    return [piece for piece in pieces if piece]


def _tail_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    spans = token_spans(text)
    if not spans:
        return ""
    return text[spans[max(0, len(spans) - limit)][0] :].strip()


def _merge_units(
    units: list[str],
    *,
    section_path: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
    split_strategy: str,
) -> list[ChunkData]:
    chunks: list[ChunkData] = []
    current: list[str] = []
    current_tokens = 0

    def emit(content: str, extra: dict | None = None) -> None:
        content = content.strip()
        if not content:
            return
        chunks.append(
            ChunkData(
                content=content,
                section_path=section_path,
                token_count=count_tokens(content),
                chunk_index=0,
                metadata={
                    **metadata,
                    "chunk_mode": split_strategy,
                    "split": split_strategy,
                    **(extra or {}),
                },
            )
        )

    def flush() -> None:
        nonlocal current, current_tokens
        if not current:
            return
        content = "\n".join(current).strip()
        emit(content)
        overlap = _tail_tokens(content, chunk_overlap)
        current = [overlap] if overlap else []
        current_tokens = count_tokens(overlap)

    for raw_unit in units:
        unit = raw_unit.strip()
        if not unit:
            continue
        unit_tokens = count_tokens(unit)
        if unit_tokens > chunk_size:
            if current:
                content = "\n".join(current).strip()
                emit(content)
                current = []
                current_tokens = 0
            for piece in _split_long_text(unit, chunk_size, chunk_overlap):
                emit(piece, {"split": "recursive_window"})
            continue
        if current and current_tokens + unit_tokens > chunk_size:
            flush()
            if overlap_content := _tail_tokens("\n".join(current), 0):
                current = [overlap_content]
            if current and current_tokens + unit_tokens > chunk_size:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        emit("\n".join(current).strip())
    return chunks


def _recursive_units(text: str) -> list[str]:
    pieces = PARAGRAPH_SPLIT_PATTERN.split(text)
    units: list[str] = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        sentences = [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(piece) if item.strip()]
        units.extend(sentences or [piece])
    return units


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


async def _semantic_section_chunks(
    *,
    section_path: str,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    metadata: dict,
    embed_fn: EmbeddingFunction | None,
    threshold: float,
) -> list[ChunkData]:
    sentences = [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(text) if item.strip()]
    if not sentences:
        return []
    if embed_fn is None or len(sentences) == 1:
        return _merge_units(
            sentences,
            section_path=section_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata=metadata,
            split_strategy="semantic",
        )

    try:
        vectors = await embed_fn(sentences)
    except Exception:
        return _merge_units(
            sentences,
            section_path=section_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={**metadata, "semantic_fallback": True},
            split_strategy="semantic",
        )

    chunks: list[ChunkData] = []
    current: list[str] = []
    current_tokens = 0
    min_group_tokens = max(32, chunk_size // 4)

    def emit() -> None:
        nonlocal current, current_tokens
        content = "\n".join(current).strip()
        if content:
            chunks.append(
                ChunkData(
                    content=content,
                    section_path=section_path,
                    token_count=count_tokens(content),
                    chunk_index=0,
                    metadata={
                        **metadata,
                        "chunk_mode": "semantic",
                        "semantic_threshold": threshold,
                    },
                )
            )
        overlap = _tail_tokens(content, chunk_overlap)
        current = [overlap] if overlap else []
        current_tokens = count_tokens(overlap)

    for index, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence)
        similarity = (
            _cosine(vectors[index - 1], vectors[index])
            if index > 0 and index - 1 < len(vectors) and index < len(vectors)
            else 1.0
        )
        should_split = (
            current
            and similarity < threshold
            and current_tokens >= min_group_tokens
        )
        if sentence_tokens > chunk_size:
            if current:
                emit()
            for piece in _split_long_text(sentence, chunk_size, chunk_overlap):
                chunks.append(
                    ChunkData(
                        content=piece,
                        section_path=section_path,
                        token_count=count_tokens(piece),
                        chunk_index=0,
                        metadata={**metadata, "chunk_mode": "semantic", "split": "token_window"},
                    )
                )
            continue
        if current and (should_split or current_tokens + sentence_tokens > chunk_size):
            emit()
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        emit()
    return chunks


async def split_document(
    parsed_document: ParsedDocument,
    *,
    mode: str = "structure",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    embed_fn: EmbeddingFunction | None = None,
    semantic_threshold: float = 0.72,
) -> list[ChunkData]:
    validate_chunk_mode(mode)
    if chunk_size < 32:
        raise ValueError("chunk_size 不能小于 32")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0 且小于 chunk_size")

    if mode == "structure":
        chunks = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split(
            parsed_document
        )
        for chunk in chunks:
            chunk.metadata["chunk_mode"] = mode
        return chunks

    chunks: list[ChunkData] = []
    for section_path, elements in _group_sections(parsed_document.elements):
        rendered = [SemanticChunker._render_element(element) for element in elements]
        text = "\n".join(item for item in rendered if item).strip()
        if not text:
            continue
        metadata = _page_metadata(elements)
        section_chunks: list[ChunkData] = []
        if mode == "fixed":
            section_chunks = [
                ChunkData(
                    content=piece,
                    section_path=section_path,
                    token_count=count_tokens(piece),
                    chunk_index=0,
                    metadata={**metadata, "chunk_mode": mode},
                )
                for piece in _split_long_text(text, chunk_size, chunk_overlap)
            ]
        elif mode == "paragraph":
            section_chunks = _merge_units(
                [item.strip() for item in PARAGRAPH_SPLIT_PATTERN.split(text) if item.strip()],
                section_path=section_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                split_strategy=mode,
            )
        elif mode == "sentence":
            section_chunks = _merge_units(
                [item.strip() for item in SENTENCE_SPLIT_PATTERN.split(text) if item.strip()],
                section_path=section_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                split_strategy=mode,
            )
        elif mode == "recursive":
            section_chunks = _merge_units(
                _recursive_units(text),
                section_path=section_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                split_strategy=mode,
            )
        elif mode == "semantic":
            section_chunks = await _semantic_section_chunks(
                section_path=section_path,
                text=text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                metadata=metadata,
                embed_fn=embed_fn,
                threshold=semantic_threshold,
            )
        elif mode == "parent_child":
            parent_chunks = _merge_units(
                _recursive_units(text),
                section_path=section_path,
                chunk_size=max(chunk_size * 3, chunk_size + chunk_overlap),
                chunk_overlap=0,
                metadata=metadata,
                split_strategy="parent",
            )
            for parent_index, parent in enumerate(parent_chunks, 1):
                child_units = [
                    item.strip()
                    for item in SENTENCE_SPLIT_PATTERN.split(parent.content)
                    if item.strip()
                ]
                children = _merge_units(
                    child_units,
                    section_path=section_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    metadata={**metadata, "chunk_mode": mode},
                    split_strategy="parent_child",
                )
                for child in children:
                    child.metadata.update(
                        {
                            "parent_index": parent_index,
                            "parent_section": section_path,
                            "parent_content": parent.content,
                            "child_content": child.content,
                        }
                    )
                    section_chunks.append(child)
        chunks.extend(section_chunks)

    for index, chunk in enumerate(chunks):
        chunk.chunk_index = index
    return chunks
