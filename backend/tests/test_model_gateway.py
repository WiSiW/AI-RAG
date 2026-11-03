import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.errors import UpstreamError
from app.indexing.model_gateway import ModelGateway, _estimate_rerank_tokens


def _rerank_profile(**config):
    return SimpleNamespace(
        base_url="http://rerank.test",
        model_name="bge-reranker-v2-m3",
        api_key_encrypted=None,
        config=config,
    )


def test_parse_rerank_results_by_index():
    payload = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.2},
        ]
    }
    assert ModelGateway._parse_rerank_scores(payload, 2) == [0.2, 0.9]


def test_parse_rerank_scores_array():
    assert ModelGateway._parse_rerank_scores({"scores": [0.7, 0.1]}, 2) == [0.7, 0.1]


def test_parse_rerank_unknown_payload():
    with pytest.raises(UpstreamError):
        ModelGateway._parse_rerank_scores({"unexpected": True}, 2)


@pytest.mark.asyncio
async def test_rerank_splits_requests_and_preserves_document_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        results = [
            {
                "index": index,
                "relevance_score": float(document.rsplit("-", 1)[1]),
            }
            for index, document in enumerate(body["documents"])
        ]
        return httpx.Response(200, json={"results": results})

    gateway = ModelGateway(httpx.MockTransport(handler))
    profile = _rerank_profile(rerank_batch_documents=2, rerank_batch_tokens=1000)
    documents = [f"doc-{index}" for index in range(5)]

    scores = await gateway.rerank(profile, "测试查询", documents)

    assert [item["documents"] for item in requests] == [
        ["doc-0", "doc-1"],
        ["doc-2", "doc-3"],
        ["doc-4"],
    ]
    assert [request["top_n"] for request in requests] == [2, 2, 1]
    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_rerank_batches_respect_token_budget_and_split_oversized_documents():
    documents = ["甲" * 45, "乙"]

    batches = ModelGateway._rerank_batches(
        "q",
        documents,
        max_documents=8,
        max_tokens=21,
    )

    pieces = [piece for batch, _ in batches for piece in batch]
    assert len(batches) == 3
    assert [index for index, _ in pieces] == [0, 0, 0, 1]
    assert max(len(text) for _, text in pieces) <= 20
    assert all(tokens <= 21 for _, tokens in batches)


@pytest.mark.asyncio
async def test_rerank_uses_best_score_for_split_document():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        results = [
            {"index": index, "relevance_score": len(document) / 10}
            for index, document in enumerate(body["documents"])
        ]
        return httpx.Response(200, json={"results": results})

    gateway = ModelGateway(httpx.MockTransport(handler))
    profile = _rerank_profile(rerank_batch_documents=8, rerank_batch_tokens=21)

    scores = await gateway.rerank(profile, "q", ["甲" * 45, "乙"])

    assert scores == pytest.approx([2.0, 0.1])

def test_rerank_token_estimate_accounts_for_latin_subwords():
    assert _estimate_rerank_tokens("internationalization") > 1
    assert _estimate_rerank_tokens("短文本") == 3
