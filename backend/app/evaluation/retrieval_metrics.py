import math
import random
import re
from dataclasses import dataclass
from statistics import mean
from typing import Any

from app.chunking.semantic_chunker import count_tokens

QUESTION_TYPES = ("detail", "comparison", "summary", "reasoning", "table", "code", "no_answer")
QUERY_TYPES = ("bm25", "l1", "original", "summary", "questions", "l3")


@dataclass(frozen=True)
class TrialParameters:
    chunk_size: int
    chunk_overlap: int
    overlap_ratio: float
    label: str
    is_baseline: bool = False


def normalize_evidence_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def evidence_span(text: str, quote: str) -> tuple[int, int] | None:
    """Return source character offsets using whitespace-insensitive matching."""
    if not text or not quote:
        return None
    exact = text.find(quote)
    if exact >= 0:
        return exact, exact + len(quote)

    normalized_chars: list[str] = []
    source_positions: list[int] = []
    for index, char in enumerate(text):
        if char.isspace():
            continue
        normalized_chars.append(char.lower())
        source_positions.append(index)
    normalized = "".join(normalized_chars)
    needle = normalize_evidence_text(quote)
    start = normalized.find(needle)
    if start < 0 or not needle:
        return None
    end_index = start + len(needle) - 1
    return source_positions[start], source_positions[end_index] + 1


def build_trial_parameters(
    chunk_sizes: list[int],
    overlap_ratios: list[float],
    *,
    baseline_size: int,
    baseline_overlap: int,
) -> list[TrialParameters]:
    sizes = list(dict.fromkeys(chunk_sizes))
    ratios = list(dict.fromkeys(float(value) for value in overlap_ratios))
    if not sizes or not ratios:
        raise ValueError("分块大小和重叠比例不能为空")
    if len(sizes) * len(ratios) > 25:
        raise ValueError("参数组合最多 25 组")
    if not 32 <= baseline_size <= 8192:
        raise ValueError("当前参数 chunk_size 必须在 32 到 8192 之间")
    if not 0 <= baseline_overlap < baseline_size:
        raise ValueError("当前参数 chunk_overlap 必须大于等于 0 且小于 chunk_size")
    for size in sizes:
        if not 32 <= size <= 8192:
            raise ValueError("chunk_size 必须在 32 到 8192 之间")
    for ratio in ratios:
        if not 0 <= ratio < 100:
            raise ValueError("重叠比例必须在 0 到 100 之间")

    trials: list[TrialParameters] = []
    seen_pairs: set[tuple[int, int]] = set()
    for size in sizes:
        for ratio in ratios:
            overlap = min(size - 1, max(0, round(size * ratio / 100)))
            if (size, overlap) in seen_pairs:
                continue
            seen_pairs.add((size, overlap))
            trials.append(
                TrialParameters(
                    chunk_size=size,
                    chunk_overlap=overlap,
                    overlap_ratio=ratio,
                    label=f"{size} tokens / {ratio:g}% ({overlap})",
                )
            )

    baseline_matches = [
        item
        for item in trials
        if item.chunk_size == baseline_size and item.chunk_overlap == baseline_overlap
    ]
    if baseline_matches:
        matched = baseline_matches[0]
        trials[trials.index(matched)] = TrialParameters(
            chunk_size=matched.chunk_size,
            chunk_overlap=matched.chunk_overlap,
            overlap_ratio=matched.overlap_ratio,
            label=f"当前参数 · {matched.label}",
            is_baseline=True,
        )
    else:
        trials.insert(
            0,
            TrialParameters(
                chunk_size=baseline_size,
                chunk_overlap=baseline_overlap,
                overlap_ratio=round(baseline_overlap / max(1, baseline_size) * 100, 2),
                label="当前参数",
                is_baseline=True,
            ),
        )
    return trials


def _evidence_hits(
    evidence: list[dict[str, Any]], hits: list[dict[str, Any]], limit: int
) -> tuple[set[int], int | None]:
    matched: set[int] = set()
    first_rank: int | None = None
    for rank, hit in enumerate(hits[:limit], 1):
        chunk_text = normalize_evidence_text(str(hit.get("content", "")))
        hit_any = False
        for index, item in enumerate(evidence):
            needle = str(item.get("normalized_quote") or normalize_evidence_text(str(item.get("quote", ""))))
            if needle and needle in chunk_text:
                matched.add(index)
                hit_any = True
        if hit_any and first_rank is None:
            first_rank = rank
    return matched, first_rank


def score_retrieval_case(
    *,
    evidence: list[dict[str, Any]],
    hits: list[dict[str, Any]],
    latency_ms: float,
) -> dict[str, Any]:
    matched_5, first_5 = _evidence_hits(evidence, hits, 5)
    matched_10, first_10 = _evidence_hits(evidence, hits, 10)
    total = len(evidence)
    matched_tokens = sum(
        count_tokens(str(evidence[index].get("quote", ""))) for index in matched_10
    )
    retrieved_tokens = sum(count_tokens(str(hit.get("content", ""))) for hit in hits[:10])
    return {
        "recall_at_5": round(len(matched_5) / total, 6) if total else 0.0,
        "recall_at_10": round(len(matched_10) / total, 6) if total else 0.0,
        "all_evidence_hit_at_5": bool(total and len(matched_5) == total),
        "all_evidence_hit_at_10": bool(total and len(matched_10) == total),
        "reciprocal_rank": round(1 / first_10, 6) if first_10 else 0.0,
        "evidence_density": round(matched_tokens / retrieved_tokens, 6)
        if retrieved_tokens
        else 0.0,
        "latency_ms": round(max(0.0, latency_ms), 3),
        "evidence_results": [
            {
                "evidence_id": str(item.get("id", "")),
                "quote": str(item.get("quote", "")),
                "document_id": str(item.get("document_id", "")),
                "hit_at_5": index in matched_5,
                "hit_at_10": index in matched_10,
            }
            for index, item in enumerate(evidence)
        ],
    }


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * ratio
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 3)
    value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return round(value, 3)


def bootstrap_mean_ci(values: list[float], *, seed: int = 17, rounds: int = 500) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = [
        mean(rng.choice(values) for _ in values)
        for _ in range(rounds)
    ]
    return [percentile(samples, 0.025), percentile(samples, 0.975)]


def aggregate_case_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item.get("answerable", True)]
    recalls_5 = [float(item.get("recall_at_5", 0)) for item in answerable]
    recalls_10 = [float(item.get("recall_at_10", 0)) for item in answerable]
    latencies = [float(item.get("latency_ms", 0)) for item in results]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in answerable:
        by_type.setdefault(str(item.get("question_type", "detail")), []).append(item)
    type_metrics = {
        question_type: {
            "case_count": len(items),
            "recall_at_5": round(mean(float(item["recall_at_5"]) for item in items), 6),
            "recall_at_10": round(mean(float(item["recall_at_10"]) for item in items), 6),
        }
        for question_type, items in sorted(by_type.items())
    }
    quality_items = [
        item for item in answerable if isinstance(item.get("quality_scores"), dict)
        and item["quality_scores"]
    ]
    return {
        "answerable_cases": len(answerable),
        "no_answer_cases": len(results) - len(answerable),
        "recall_at_5": round(mean(recalls_5), 6) if recalls_5 else 0.0,
        "recall_at_10": round(mean(recalls_10), 6) if recalls_10 else 0.0,
        "all_evidence_hit_at_5": round(
            mean(1.0 if item.get("all_evidence_hit_at_5") else 0.0 for item in answerable),
            6,
        )
        if answerable
        else 0.0,
        "all_evidence_hit_at_10": round(
            mean(1.0 if item.get("all_evidence_hit_at_10") else 0.0 for item in answerable),
            6,
        )
        if answerable
        else 0.0,
        "mrr_at_10": round(mean(float(item.get("reciprocal_rank", 0)) for item in answerable), 6)
        if answerable
        else 0.0,
        "evidence_density": round(
            mean(float(item.get("evidence_density", 0)) for item in answerable), 6
        )
        if answerable
        else 0.0,
        "latency_p50_ms": percentile(latencies, 0.5),
        "latency_p95_ms": percentile(latencies, 0.95),
        "recall_at_5_ci": bootstrap_mean_ci(recalls_5),
        "recall_at_10_ci": bootstrap_mean_ci(recalls_10),
        "question_types": type_metrics,
        "quality": {
            "case_count": len(quality_items),
            "faithfulness": round(
                mean(float(item["quality_scores"].get("faithfulness", 0)) for item in quality_items),
                4,
            )
            if quality_items
            else 0.0,
            "completeness": round(
                mean(float(item["quality_scores"].get("completeness", 0)) for item in quality_items),
                4,
            )
            if quality_items
            else 0.0,
            "cited_gold_chunk_ratio": round(
                mean(
                    float(item["quality_scores"].get("cited_gold_chunk_ratio", 0))
                    for item in quality_items
                ),
                6,
            )
            if quality_items
            else 0.0,
        },
    }


def mark_pareto(trials: list[dict[str, Any]]) -> list[str]:
    completed = [
        item
        for item in trials
        if item.get("status") in {"completed", "partial"} and item.get("metrics")
    ]

    def dominated(candidate: dict[str, Any], other: dict[str, Any]) -> bool:
        left = candidate["metrics"]
        right = other["metrics"]
        left_cost = candidate.get("cost") or {}
        right_cost = other.get("cost") or {}
        goals = [
            (float(left.get("recall_at_5", 0)), float(right.get("recall_at_5", 0)), True),
            (float(left.get("recall_at_10", 0)), float(right.get("recall_at_10", 0)), True),
            (
                float(left.get("evidence_density", 0)),
                float(right.get("evidence_density", 0)),
                True,
            ),
            (
                float(left.get("latency_p95_ms", 0)),
                float(right.get("latency_p95_ms", 0)),
                False,
            ),
            (
                float(left_cost.get("index_tokens", 0)),
                float(right_cost.get("index_tokens", 0)),
                False,
            ),
        ]
        no_worse = all(
            left_value >= right_value if maximize else left_value <= right_value
            for left_value, right_value, maximize in goals
        )
        strictly_better = any(
            left_value > right_value if maximize else left_value < right_value
            for left_value, right_value, maximize in goals
        )
        return no_worse and strictly_better

    return [
        str(candidate["id"])
        for candidate in completed
        if not any(
            dominated(candidate, other)
            for other in completed
            if other["id"] != candidate["id"]
        )
    ]
