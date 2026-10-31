from fastapi.testclient import TestClient

from app.db.session import Base, engine
from app.main import app


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "test-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_health_login_and_configuration_flow():
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as client:
        assert client.get("/api/v1/health/live").status_code == 200
        assert (
            client.post(
                "/api/v1/auth/login",
                json={"username": "admin", "password": "wrong"},
            ).status_code
            == 401
        )
        headers = auth_headers(client)
        engines = client.get("/api/v1/parsing-engines", headers=headers)
        assert engines.status_code == 200, engines.text
        assert {item["category"] for item in engines.json()} == {"ocr", "table", "layout"}
        adapters = client.get("/api/v1/parsing-engines/adapters", headers=headers)
        assert adapters.status_code == 200
        assert {item["adapter"] for item in adapters.json() if item["category"] == "ocr"} == {"http_ocr"}
        invalid_engine = client.post(
            "/api/v1/parsing-engines",
            headers=headers,
            json={
                "category": "ocr",
                "adapter": "http_ocr",
                "display_name": "不安全配置",
                "config": {"api_key": "should-not-store"},
            },
        )
        assert invalid_engine.status_code == 422

        embedding = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "测试 Embedding",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "model_name": "embedding-test",
                "dimensions": 3,
                "api_key": "sk-test",
            },
        )
        assert embedding.status_code == 201, embedding.text
        assert embedding.json()["has_api_key"] is True

        llm = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "测试问答",
                "kind": "llm",
                "base_url": "https://example.com/v1",
                "model_name": "chat-test",
            },
        )
        assert llm.status_code == 201, llm.text

        enrichment = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "测试增强",
                "kind": "enrichment",
                "base_url": "https://example.com/v1",
                "model_name": "enrichment-test",
            },
        )
        assert enrichment.status_code == 201, enrichment.text

        kb = client.post(
            "/api/v1/knowledge-bases",
            headers=headers,
            json={
                "name": "测试知识库",
                "description": "API smoke test",
                "embedding_profile_id": embedding.json()["id"],
                "enrichment_profile_id": enrichment.json()["id"],
                "default_llm_profile_id": llm.json()["id"],
            },
        )
        assert kb.status_code == 201, kb.text
        assert kb.json()["document_count"] == 0
        assert kb.json()["enrichment_profile_id"] == enrichment.json()["id"]
        assert set(kb.json()["settings"]["retrieval_weights"]) == {
            "bm25",
            "l1",
            "original",
            "summary",
            "questions",
            "l3",
        }

        listing = client.get("/api/v1/knowledge-bases", headers=headers)
        assert listing.status_code == 200
        assert listing.json()[0]["name"] == "测试知识库"
    Base.metadata.drop_all(bind=engine)
