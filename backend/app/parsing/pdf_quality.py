import math
import re
from collections import Counter
from collections.abc import Iterable

from app.core.config import settings
from app.parsing.pdf_types import OCRQuality

_CORRUPTION_RE = re.compile(r"[\ufffd\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_MOJIBAKE_RE = re.compile(r"(?:Ã.|Â.|â€|锟斤拷|烫烫烫|屯屯屯|�)")
_REPEATED_RE = re.compile(r"([^\s])\1{5,}")
_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+")


def _effective_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _repetition_ratio(text: str, n: int = 5) -> float:
    compact = _effective_text(text)
    if len(compact) < n * 2:
        return 0.0
    grams = [compact[index : index + n] for index in range(len(compact) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def _language_ratio(text: str) -> float:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    total = cjk + latin
    if total == 0:
        return 0.0
    return max(cjk, latin) / total


def evaluate_text_quality(
    text: str,
    *,
    mean_confidence: float | None = None,
    min_chars: int | None = None,
    min_confidence: float | None = None,
    max_corruption_ratio: float | None = None,
    max_repetition_ratio: float | None = None,
) -> OCRQuality:
    compact = _effective_text(text)
    char_count = len(compact)
    corruption_count = len(_CORRUPTION_RE.findall(text or "")) + len(_MOJIBAKE_RE.findall(text or ""))
    corruption_ratio = corruption_count / max(char_count, 1)
    repetition_ratio = _repetition_ratio(text or "")
    repeated_chars = len(_REPEATED_RE.findall(text or ""))
    if repeated_chars:
        repetition_ratio = max(repetition_ratio, min(1.0, repeated_chars / max(char_count, 1)))
    language_ratio = _language_ratio(text or "")
    effective_chars = len(_TOKEN_RE.findall(text or ""))

    reasons: list[str] = []
    if effective_chars < (settings.pdf_ocr_min_chars if min_chars is None else min_chars):
        reasons.append("insufficient_text")
    if (
        mean_confidence is not None
        and mean_confidence < (settings.pdf_ocr_min_confidence if min_confidence is None else min_confidence)
    ):
        reasons.append("low_confidence")
    if corruption_ratio > (
        settings.pdf_ocr_max_corruption_ratio
        if max_corruption_ratio is None
        else max_corruption_ratio
    ):
        reasons.append("corrupted_characters")
    if repetition_ratio > (
        settings.pdf_ocr_max_repetition_ratio
        if max_repetition_ratio is None
        else max_repetition_ratio
    ):
        reasons.append("repetitive_text")
    if effective_chars >= 40 and language_ratio < 0.55:
        reasons.append("mixed_garbage")

    confidence_score = 0.7 if mean_confidence is None else max(0.0, min(1.0, mean_confidence))
    length_score = min(1.0, effective_chars / max(settings.pdf_ocr_min_chars * 4, 1))
    corruption_score = max(0.0, 1.0 - corruption_ratio * 20)
    repetition_score = max(0.0, 1.0 - repetition_ratio * 2)
    score = (
        confidence_score * 0.35
        + length_score * 0.25
        + corruption_score * 0.25
        + repetition_score * 0.15
    )
    if not math.isfinite(score):
        score = 0
    return OCRQuality(
        passed=not reasons,
        score=score,
        effective_chars=effective_chars,
        mean_confidence=mean_confidence,
        corruption_ratio=corruption_ratio,
        repetition_ratio=repetition_ratio,
        language_ratio=language_ratio,
        reasons=reasons,
    )


def choose_best_text(candidates: Iterable[tuple[str, str, OCRQuality]]) -> tuple[str, str, OCRQuality] | None:
    ranked = sorted(
        candidates,
        key=lambda item: (item[2].passed, item[2].score, item[2].effective_chars),
        reverse=True,
    )
    return ranked[0] if ranked else None