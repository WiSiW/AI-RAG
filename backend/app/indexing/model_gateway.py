import json
import math
import time
from typing import Any

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.core.config import settings
from app.core.errors import UpstreamError
from app.core.security import decrypt_secret
from app.db.models import ModelProfile


class ChatMessage(BaseModel):
    role: str
    content: str


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class ModelGateway:
    @staticmethod
    def _api_key(profile: ModelProfile) -> str:
        return decrypt_secret(profile.api_key_encrypted) or "EMPTY"

    @staticmethod
    def _openai_client(profile: ModelProfile) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=ModelGateway._api_key(profile),
            base_url=profile.base_url.rstrip("/"),
            timeout=settings.model_request_timeout,
        )

    async def embed_texts(self, profile: ModelProfile, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._openai_client(profile)
        try:
            response = await client.embeddings.create(model=profile.model_name, input=texts)
        except Exception as exc:
            raise UpstreamError(f"Embedding 模型调用失败：{exc}") from exc
        finally:
            await client.close()
        vectors = [_normalize_vector(list(item.embedding)) for item in response.data]
        if len(vectors) != len(texts):
            raise UpstreamError("Embedding 模型返回数量与输入不一致")
        if profile.dimensions and any(len(vector) != profile.dimensions for vector in vectors):
            actual = len(vectors[0])
            raise UpstreamError(
                f"Embedding 维度不匹配：配置为 {profile.dimensions}，实际为 {actual}"
            )
        return vectors

    async def complete(
        self,
        profile: ModelProfile,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        payload = [message.model_dump() if isinstance(message, ChatMessage) else message for message in messages]
        client = self._openai_client(profile)
        try:
            response = await client.chat.completions.create(
                model=profile.model_name,
                messages=payload,
                temperature=(
                    float(profile.config.get("temperature", 0.1))
                    if temperature is None
                    else temperature
                ),
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise UpstreamError(f"问答模型调用失败：{exc}") from exc
        finally:
            await client.close()
        content = response.choices[0].message.content if response.choices else None
        return content or ""

    async def stream(
        self,
        profile: ModelProfile,
        messages: list[ChatMessage] | list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        payload = [message.model_dump() if isinstance(message, ChatMessage) else message for message in messages]
        client = self._openai_client(profile)
        try:
            response = await client.chat.completions.create(
                model=profile.model_name,
                messages=payload,
                temperature=(
                    float(profile.config.get("temperature", 0.1))
                    if temperature is None
                    else temperature
                ),
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content
                if content:
                    yield content
        except Exception as exc:
            raise UpstreamError(f"问答模型流式调用失败：{exc}") from exc
        finally:
            await client.close()

    async def rerank(self, profile: ModelProfile, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        path = str(profile.config.get("rerank_path", "rerank")).lstrip("/")
        url = f"{profile.base_url.rstrip('/')}/{path}"
        headers = {"Content-Type": "application/json"}
        api_key = self._api_key(profile)
        if api_key and api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {api_key}"
        body: dict[str, Any] = {
            "model": profile.model_name,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
        }
        if profile.config.get("return_documents"):
            body["return_documents"] = False
        try:
            async with httpx.AsyncClient(timeout=settings.model_request_timeout) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            raise UpstreamError(f"重排序模型调用失败：{exc}") from exc
        return self._parse_rerank_scores(payload, len(documents))

    @staticmethod
    def _parse_rerank_scores(payload: Any, count: int) -> list[float]:
        scores = [0.0] * count
        if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
            raw_scores = payload["scores"]
            for index, score in enumerate(raw_scores[:count]):
                scores[index] = float(score)
            return scores

        items = None
        if isinstance(payload, dict):
            for key in ("results", "data"):
                if isinstance(payload.get(key), list):
                    items = payload[key]
                    break
        elif isinstance(payload, list):
            items = payload

        if items is not None:
            for position, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                index = int(item.get("index", position))
                raw_score = item.get("relevance_score", item.get("score", item.get("relevance")))
                if raw_score is not None and 0 <= index < count:
                    scores[index] = float(raw_score)
            return scores
        raise UpstreamError("无法识别重排序模型响应格式")

    async def test_profile(self, profile: ModelProfile) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        details: dict[str, Any] = {}
        if profile.kind == "embedding":
            vectors = await self.embed_texts(profile, ["这是一条连通性测试。"])
            details["dimensions"] = len(vectors[0])
        elif profile.kind == "llm":
            answer = await self.complete(
                profile,
                [{"role": "user", "content": "只回复 OK"}],
                temperature=0,
                max_tokens=8,
            )
            details["answer"] = answer
        elif profile.kind == "rerank":
            scores = await self.rerank(profile, "测试查询", ["这是相关文档", "这是无关文档"])
            details["scores"] = scores
        else:
            raise UpstreamError(f"不支持的模型类型：{profile.kind}")
        details["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return "连接成功", details

    @staticmethod
    def profile_summary(profile: ModelProfile) -> str:
        return json.dumps(
            {
                "id": profile.id,
                "name": profile.name,
                "kind": profile.kind,
                "model": profile.model_name,
            },
            ensure_ascii=False,
        )
