import pytest

from app.core.errors import UpstreamError
from app.indexing.model_gateway import ModelGateway


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
