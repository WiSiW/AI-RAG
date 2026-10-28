import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.chunking.modes import CHUNK_MODES, split_document
from app.db.session import Base, engine
from app.main import app
from app.parsing.base import ParsedDocument, ParsedElement
from app.parsing.markdown_parser import MarkdownParser


def demo_document() -> ParsedDocument:
    text = "第一段说明。第二段继续说明。\n\n第三段讨论另一个主题。第四段补充细节。"
    return ParsedDocument(
        source="demo.md",
        title="演示文档",
        elements=[
            ParsedElement(type="title", text="第一章", level=1),
            ParsedElement(type="paragraph", text=text),
            ParsedElement(type="title", text="1.1 小节", level=2),
            ParsedElement(type="paragraph", text="小节内容。" * 30),
        ],
        full_text=text,
    )


async def fake_embed(texts: list[str]) -> list[list[float]]:
    return [[float(len(text) % 7), float(index % 3), 1.0] for index, text in enumerate(texts)]


@pytest.mark.asyncio
async def test_all_chunk_modes_are_available_and_produce_chunks():
    available = {mode["value"] for mode in CHUNK_MODES}
    assert available == {
        "structure",
        "recursive",
        "fixed",
        "paragraph",
        "sentence",
        "semantic",
        "parent_child",
    }
    for mode in available:
        chunks = await split_document(
            demo_document(),
            mode=mode,
            chunk_size=48,
            chunk_overlap=8,
            embed_fn=fake_embed,
        )
        assert chunks, mode
        assert all(chunk.token_count <= 48 for chunk in chunks), mode
    parent_chunks = await split_document(
        demo_document(),
        mode="parent_child",
        chunk_size=48,
        chunk_overlap=8,
    )
    assert any("parent_content" in chunk.metadata for chunk in parent_chunks)


def test_markdown_parser_extracts_front_matter_metadata(tmp_path: Path):
    path = tmp_path / "meta.md"
    path.write_text(
        "---\ntitle: 元数据测试\nauthor: 张三\ntags: rag, test\n---\n\n# 正文\n\n内容。\n",
        encoding="utf-8",
    )
    parsed = MarkdownParser().parse(str(path))
    assert parsed.title == "元数据测试"
    assert parsed.metadata["author"] == "张三"
    assert parsed.metadata["front_matter"]["tags"] == "rag, test"
    assert parsed.metadata["heading_count"] == 1


def test_archive_extractor_skips_traversal_and_unsupported(tmp_path: Path):
    from app.api.documents import _extract_archive

    archive_path = tmp_path / "docs.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("good/readme.md", "# hello")
        archive.writestr("../evil.md", "bad")
        archive.writestr("image.png", b"png")
    entries, errors = _extract_archive(archive_path)
    try:
        assert [Path(name).name for name, _, _ in entries] == ["readme.md"]
        assert any("非法路径" in error for error in errors)
        assert any("不支持的格式" in error for error in errors)
    finally:
        if entries:
            import shutil

            shutil.rmtree(entries[0][1].parent, ignore_errors=True)


def auth_headers(client: TestClient) -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "test-password"},
    )
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_model_edit_and_publication_auth_flow():
    Base.metadata.drop_all(bind=engine)
    with TestClient(app) as client:
        headers = auth_headers(client)
        embedding = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "embedding",
                "kind": "embedding",
                "base_url": "https://example.com/v1",
                "model_name": "embedding-test",
                "dimensions": 3,
            },
        ).json()
        llm = client.post(
            "/api/v1/model-profiles",
            headers=headers,
            json={
                "name": "llm",
                "kind": "llm",
                "base_url": "https://example.com/v1",
                "model_name": "chat-test",
            },
        ).json()
        edited = client.patch(
            f"/api/v1/model-profiles/{llm['id']}",
            headers=headers,
            json={"name": "llm-edited", "is_active": False},
        )
        assert edited.status_code == 200
        assert edited.json()["name"] == "llm-edited"
        assert edited.json()["is_active"] is False
        client.patch(
            f"/api/v1/model-profiles/{llm['id']}",
            headers=headers,
            json={"is_active": True},
        )
        kb = client.post(
            "/api/v1/knowledge-bases",
            headers=headers,
            json={
                "name": "kb",
                "embedding_profile_id": embedding["id"],
                "default_llm_profile_id": llm["id"],
            },
        ).json()
        publication = client.post(
            "/api/v1/publications",
            headers=headers,
            json={
                "name": "公开问答",
                "slug": "public-qa",
                "kind": "qa",
                "knowledge_base_ids": [kb["id"]],
                "llm_profile_id": llm["id"],
            },
        )
        assert publication.status_code == 201, publication.text
        api_key = publication.json()["api_key"]
        assert api_key.startswith("zy_")

        capabilities = client.get(
            "/api/v1/public/qa/public-qa/capabilities",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert capabilities.status_code == 200
        assert capabilities.json()["knowledge_bases"][0]["id"] == kb["id"]
        assert (
            client.get(
                "/api/v1/public/qa/public-qa/capabilities",
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        rotated = client.post(
            f"/api/v1/publications/{publication.json()['id']}/rotate-key",
            headers=headers,
        )
        assert rotated.status_code == 200
        assert rotated.json()["api_key"] != api_key
    Base.metadata.drop_all(bind=engine)

