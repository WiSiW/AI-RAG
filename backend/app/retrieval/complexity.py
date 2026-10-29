import logging
import re
from dataclasses import dataclass

import httpx
import jieba.posseg as pseg

from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.schemas import ComplexityDecision

logger = logging.getLogger(__name__)

COMPARISON_PATTERN = re.compile(
    r"相比|更|区别|差异|对比|区别是什么|versus|vs\.?|compared with|difference",
    re.IGNORECASE,
)
LOGIC_PATTERN = re.compile(
    r"首先|然后|接着|最后|之前|之后|如果|则|因为|所以|由于|导致|因此|"
    r"先.+再|在.+之后|first|then|before|after|because|therefore",
    re.IGNORECASE,
)
BRIDGE_PATTERN = re.compile(
    r"通过|根据|对应|涉及|其中|基于|所在|属于|related to|which",
    re.IGNORECASE,
)
YES_PATTERN = re.compile(r"^(是|需要|yes|true|1)[。.!！\s]*$", re.IGNORECASE)
NO_PATTERN = re.compile(r"^(否|不需要|不是|no|false|0)[。.!！\s]*$", re.IGNORECASE)


@dataclass
class ComplexityFeatures:
    comparison_flag: bool
    logic_connector_flag: bool
    bridge_entity_count: int
    entity_count: int
    token_length: int

    def to_dict(self) -> dict:
        return {
            "comparison_flag": self.comparison_flag,
            "logic_connector_flag": self.logic_connector_flag,
            "bridge_entity_count": self.bridge_entity_count,
            "entity_count": self.entity_count,
            "token_length": self.token_length,
        }


def extract_complexity_features(question: str) -> ComplexityFeatures:
    entities: list[str] = []
    for word, flag in pseg.cut(question):
        term = word.strip()
        if not term:
            continue
        if flag.startswith(("nr", "ns", "nt", "nz", "eng", "vn")) or (
            len(term) >= 2 and flag.startswith("n")
        ):
            if term not in entities:
                entities.append(term)
    comparison = bool(COMPARISON_PATTERN.search(question))
    logic = bool(LOGIC_PATTERN.search(question))
    bridge_count = max(0, len(entities) - 1) if (comparison or logic or BRIDGE_PATTERN.search(question)) else 0
    return ComplexityFeatures(
        comparison_flag=comparison,
        logic_connector_flag=logic,
        bridge_entity_count=bridge_count,
        entity_count=len(entities),
        token_length=count_tokens(question),
    )


def heuristic_score(features: ComplexityFeatures) -> float:
    score = 0.0
    if features.comparison_flag:
        score += 0.3
    if features.logic_connector_flag:
        score += 0.3
    if features.bridge_entity_count > 0:
        score += 0.2
    if features.token_length >= 24:
        score += 0.1
    if features.entity_count >= 3:
        score += 0.1
    return round(min(1.0, score), 4)


class BERTComplexityClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url if base_url is not None else settings.multihop_classifier_url).rstrip("/")

    async def predict(self, question: str, features: ComplexityFeatures) -> tuple[float, str] | None:
        if not self.base_url:
            return None
        try:
            async with httpx.AsyncClient(timeout=settings.multihop_classifier_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/predict",
                    json={"question": question, "features": features.to_dict()},
                )
                response.raise_for_status()
                payload = response.json()
            probability = float(payload["multi_hop_probability"])
            return min(1.0, max(0.0, probability)), str(payload.get("model_version", ""))
        except Exception:
            logger.warning("BERT complexity classifier unavailable", exc_info=True)
            return None


async def classify_complexity(
    question: str,
    *,
    retrieval_mode: str,
    llm_profile,
    gateway,
    classifier_client: BERTComplexityClient | None = None,
) -> ComplexityDecision:
    features = extract_complexity_features(question)
    rule_score = heuristic_score(features)
    if retrieval_mode in {"single", "multi"}:
        return ComplexityDecision(
            is_multi_hop=retrieval_mode == "multi",
            decision_source="user_override",
            rule_score=rule_score,
            reason=f"用户指定 {retrieval_mode} 检索模式",
        )
    if rule_score >= 0.75:
        return ComplexityDecision(
            is_multi_hop=True,
            decision_source="rules",
            rule_score=rule_score,
            reason="比较词、逻辑连接或多个实体形成高置信多跳信号",
        )
    if rule_score <= 0.15 and features.entity_count <= 1:
        return ComplexityDecision(
            is_multi_hop=False,
            decision_source="rules",
            rule_score=rule_score,
            reason="问题较短且未发现比较、逻辑连接或桥梁实体",
        )

    bert_result = await (classifier_client or BERTComplexityClient()).predict(question, features)
    bert_probability = bert_result[0] if bert_result else None
    model_version = bert_result[1] if bert_result else ""
    high = settings.multihop_classifier_high_threshold
    low = settings.multihop_classifier_low_threshold
    if bert_probability is not None and bert_probability >= high:
        return ComplexityDecision(
            is_multi_hop=True,
            decision_source="bert",
            rule_score=rule_score,
            bert_probability=bert_probability,
            reason="BERT 分类器高置信判定为多跳",
            classifier_model_version=model_version,
        )
    if bert_probability is not None and bert_probability <= low:
        return ComplexityDecision(
            is_multi_hop=False,
            decision_source="bert",
            rule_score=rule_score,
            bert_probability=bert_probability,
            reason="BERT 分类器高置信判定为单跳",
            classifier_model_version=model_version,
        )

    if llm_profile is not None:
        try:
            answer = await gateway.complete(
                llm_profile,
                [
                    {
                        "role": "user",
                        "content": (
                            "请判断此问题是否需要查阅多个文档、进行分步推理？"
                            "只回答是或否，不要解释。\n问题："
                            f"{question}"
                        ),
                    }
                ],
                temperature=0,
                max_tokens=8,
            )
            cleaned = answer.strip()
            if YES_PATTERN.search(cleaned):
                return ComplexityDecision(
                    is_multi_hop=True,
                    decision_source="llm",
                    rule_score=rule_score,
                    bert_probability=bert_probability,
                    llm_decision=True,
                    reason="规则不确定，轻量 LLM 判定为多跳",
                    classifier_model_version=model_version,
                )
            if NO_PATTERN.search(cleaned):
                return ComplexityDecision(
                    is_multi_hop=False,
                    decision_source="llm",
                    rule_score=rule_score,
                    bert_probability=bert_probability,
                    llm_decision=False,
                    reason="规则不确定，轻量 LLM 判定为单跳",
                    classifier_model_version=model_version,
                )
        except Exception:
            logger.warning("LLM complexity classification failed", exc_info=True)

    is_multi = rule_score >= 0.4 or (bert_probability or 0) >= 0.5
    return ComplexityDecision(
        is_multi_hop=is_multi,
        decision_source="fallback",
        rule_score=rule_score,
        bert_probability=bert_probability,
        reason="分类器不可用，按规则和 BERT 概率回退",
        classifier_model_version=model_version,
    )
