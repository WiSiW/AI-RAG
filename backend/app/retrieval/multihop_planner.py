import json
import logging
import re
from dataclasses import dataclass

from app.chunking.semantic_chunker import count_tokens
from app.core.config import settings
from app.indexing.types import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class MultiHopPlan:
    need_more: bool
    next_question: str
    key_entities: list[str]
    confidence: float
    reason: str


def _extract_json(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


class MultiHopPlanner:
    def __init__(self, gateway):
        self.gateway = gateway

    async def plan(
        self,
        *,
        profile,
        original_question: str,
        current_question: str,
        round_number: int,
        evidence: list[RetrievedChunk],
        prior_questions: list[str],
        prior_entities: list[str],
    ) -> MultiHopPlan | None:
        if profile is None:
            return None
        evidence_parts: list[str] = []
        used_tokens = 0
        for index, chunk in enumerate(evidence, 1):
            text = chunk.context_text or chunk.content
            text = str(text)[:900]
            block = f"[{index}] {chunk.source} | {chunk.section}\n{text}"
            tokens = count_tokens(block)
            if evidence_parts and used_tokens + tokens > settings.multihop_planner_context_tokens:
                break
            evidence_parts.append(block)
            used_tokens += tokens
        prompt = f"""你是 RAG 多跳检索规划器。只做检索决策，不回答用户问题。
判断当前累计证据是否足以回答原问题；如果不足，提出一个依赖已发现线索的下一步检索问题。
不得输出隐式思维链，只给简短理由。next_question 必须自然、完整、可用于检索，不得重复历史问题。

只返回 JSON 对象：
{{"need_more":true,"next_question":"...","key_entities":["..."],"confidence":0.0,"reason":"..."}}

原问题：{original_question}
当前子问题：{current_question}
当前轮次：{round_number}
历史子问题：{json.dumps(prior_questions, ensure_ascii=False)}
已识别实体：{json.dumps(prior_entities, ensure_ascii=False)}
累计证据：
{chr(10).join(evidence_parts) if evidence_parts else "（无）"}"""
        try:
            response = await self.gateway.complete(
                profile,
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=settings.multihop_planner_max_tokens,
            )
            payload = _extract_json(response)
            if not isinstance(payload, dict):
                return None
            next_question = str(payload.get("next_question") or "").strip()
            if len(next_question) > 500:
                next_question = next_question[:500]
            entities = [
                str(item).strip()
                for item in (payload.get("key_entities") or [])
                if str(item).strip()
            ][:20]
            confidence = min(1.0, max(0.0, float(payload.get("confidence", 0))))
            return MultiHopPlan(
                need_more=bool(payload.get("need_more")),
                next_question=next_question,
                key_entities=entities,
                confidence=confidence,
                reason=str(payload.get("reason") or "")[:500],
            )
        except Exception:
            logger.warning("multi-hop planner failed", exc_info=True)
            return None
