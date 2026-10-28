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

        kb = client.post(
            "/api/v1/knowledge-bases",
            headers=headers,
            json={
                "name": "测试知识库",
                "description": "API smoke test",
                "embedding_profile_id": embedding.json()["id"],
                "default_llm_profile_id": llm.json()["id"],
            },
        )
        assert kb.status_code == 201, kb.text
        assert kb.json()["document_count"] == 0

        listing = client.get("/api/v1/knowledge-bases", headers=headers)
        assert listing.status_code == 200
        assert listing.json()[0]["name"] == "测试知识库"
    Base.metadata.drop_all(bind=engine)
