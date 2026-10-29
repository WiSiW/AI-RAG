import re
from dataclasses import dataclass, field
from typing import Any

from app.chunking.semantic_chunker import count_tokens

MAX_L3_PER_L2 = 8
MAX_L3_CHARS = 320

_SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+|\n+")
_ERROR_RE = re.compile(
    r"(?:ERROR|WARN(?:ING)?|Exception|Traceback|错误|失败|异常|超时|拒绝|"
    r"HTTP\s*[45]\d{2}|\b[45]\d{2}\b|\bE[A-Z0-9_-]{2,}\b|\b[A-Z]{2,}_[A-Z0-9_]+\b)",
    re.IGNORECASE,
)
_CODE_RE = re.compile(
    r"^(?:def|class|import|from|SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|"
    r"curl|docker|npm|pip|git|kubectl|helm)\b|^[{}[\]]|^\s{2,}\S",
    re.IGNORECASE,
)
_PARAM_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{1,80}\s*(?:=|:)\s*\S+")


@dataclass
class SectionUnit:
    section_path: str
    content: str
    pages: list[int] = field(default_factory=list)
    parent_chunk_ids: list[str] = field(default_factory=list)
    parent_summaries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class SectionSummary:
    summary: str
    status: str = "ready"
    meta: dict = field(default_factory=dict)


@dataclass
class GranuleSpec:
    content: str
    section_path: str
    token_count: int
    granularity_level: str
    granule_type: str
    parent_chunk_id: str | None
    metadata: dict = field(default_factory=dict)
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    enrichment_status: str = "ready"
    enrichment_meta: dict = field(default_factory=dict)


def _pages(row: Any) -> list[int]:
    metadata = dict(getattr(row, "metadata_json", None) or {})
    return sorted({int(page) for page in metadata.get("pages", [])})


def build_section_units(rows: list[Any]) -> list[SectionUnit]:
    grouped: dict[str, SectionUnit] = {}
    for row in rows:
        section_path = row.section_path or "正文"
        unit = grouped.get(section_path)
        if unit is None:
            unit = SectionUnit(section_path=section_path, content="")
            grouped[section_path] = unit
        unit.content = f"{unit.content}\n\n{row.content}".strip()
        unit.parent_chunk_ids.append(row.id)
        if row.summary:
            unit.parent_summaries.append(row.summary)
        for page in _pages(row):
            if page not in unit.pages:
                unit.pages.append(page)
        for keyword in row.keywords or []:
            if keyword not in unit.keywords:
                unit.keywords.append(str(keyword))
    for unit in grouped.values():
        unit.pages.sort()
        unit.keywords = unit.keywords[:20]
    return list(grouped.values())


def fallback_section_summary(unit: SectionUnit, max_chars: int = 600) -> str:
    if unit.parent_summaries:
        joined = " ".join(unit.parent_summaries).strip()
        if joined:
            return joined[:max_chars]
    compact = re.sub(r"\s+", " ", unit.content).strip()
    return compact[:max_chars]


def build_l1_specs(
    units: list[SectionUnit], summaries: dict[str, SectionSummary]
) -> list[GranuleSpec]:
    specs: list[GranuleSpec] = []
    for unit in units:
        result = summaries.get(
            unit.section_path,
            SectionSummary(
                summary=fallback_section_summary(unit),
                status="degraded",
                meta={"generated_by": "rules", "fallback_reason": "section_summary_unavailable"},
            ),
        )
        summary = result.summary.strip() or fallback_section_summary(unit)
        content = f"{unit.section_path}\n{summary}".strip()
        specs.append(
            GranuleSpec(
                content=content,
                section_path=unit.section_path,
                token_count=count_tokens(content),
                granularity_level="L1",
                granule_type="section_summary",
                parent_chunk_id=None,
                metadata={
                    "granularity_level": "L1",
                    "parent_chunk_ids": unit.parent_chunk_ids,
                    "pages": unit.pages,
                    "summary_chars": len(summary),
                    "source_tokens": count_tokens(unit.content),
                },
                summary=summary,
                keywords=unit.keywords,
                enrichment_status=result.status,
                enrichment_meta={**result.meta, "granularity_level": "L1"},
            )
        )
    return specs


def _is_table_row(line: str) -> bool:
    return line.count("|") >= 2 and not re.fullmatch(r"[\s|:+-]+", line)


def _line_reason(line: str) -> tuple[str, str] | None:
    if _is_table_row(line):
        return "table_row", "table_row"
    if _ERROR_RE.search(line):
        return "error_line", "error_or_exception"
    if _PARAM_RE.match(line):
        return "parameter_line", "parameter_assignment"
    if _CODE_RE.search(line):
        return "code_line", "code_statement"
    return None


def _sentence_score(sentence: str, keywords: list[str]) -> float:
    lowered = sentence.casefold()
    keyword_score = sum(1.5 for keyword in keywords if keyword.casefold() in lowered)
    score = keyword_score
    if re.search(r"\d", sentence):
        score += 1.0
    if _ERROR_RE.search(sentence):
        score += 2.0
    if score > 0 and 12 <= len(sentence) <= 180:
        score += 0.5
    return score


def _append_atomic(
    chosen: list[tuple[str, str, str]],
    seen: set[str],
    content: str,
    granule_type: str,
    reason: str,
) -> None:
    normalized = re.sub(r"\s+", " ", content).strip()
    if not normalized or normalized in seen or len(normalized) > MAX_L3_CHARS:
        return
    seen.add(normalized)
    chosen.append((normalized, granule_type, reason))


def build_l3_specs(rows: list[Any]) -> list[GranuleSpec]:
    specs: list[GranuleSpec] = []
    for row in rows:
        chosen: list[tuple[str, str, str]] = []
        seen: set[str] = set()

        for raw_line in row.content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            reason = _line_reason(line)
            if reason:
                _append_atomic(chosen, seen, line, *reason)
            if len(chosen) >= MAX_L3_PER_L2:
                break

        candidates: list[tuple[float, str]] = []
        for sentence in _SENTENCE_RE.split(row.content):
            sentence = sentence.strip()
            if not sentence or sentence in seen or len(sentence) > MAX_L3_CHARS:
                continue
            score = _sentence_score(sentence, list(row.keywords or []))
            if score > 0:
                candidates.append((score, sentence))
        for _, sentence in sorted(candidates, key=lambda item: item[0], reverse=True):
            _append_atomic(
                chosen, seen, sentence, "sentence", "keyword_or_fact_sentence"
            )
            if len(chosen) >= MAX_L3_PER_L2:
                break

        pages = _pages(row)
        for content, granule_type, reason in chosen:
            specs.append(
                GranuleSpec(
                    content=content,
                    section_path=row.section_path,
                    token_count=count_tokens(content),
                    granularity_level="L3",
                    granule_type=granule_type,
                    parent_chunk_id=row.id,
                    metadata={
                        "granularity_level": "L3",
                        "parent_chunk_id": row.id,
                        "parent_chunk_index": row.chunk_index,
                        "pages": pages,
                        "element_type": granule_type,
                        "selection_reason": reason,
                    },
                    enrichment_status="ready",
                    enrichment_meta={"generated_by": "rules", "granularity_level": "L3"},
                )
            )
    return specs
