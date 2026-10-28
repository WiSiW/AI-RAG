import hmac
from datetime import UTC, datetime
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.security import (
    create_download_token,
    decode_download_token,
    hash_api_key,
)
from app.core.storage import get_object_stream
from app.db.models import Document, KnowledgeBase, PublishedApi
from app.db.session import SessionLocal, get_db
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ChunkRead,
    PublicCapabilities,
    PublicChatRequest,
    PublicSearchRequest,
    SearchRequest,
    SearchResponse,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/public", tags=["public-api"])
chat_service = ChatService()


def _extract_api_key(authorization: str | None, x_api_key: str | None) -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value.strip()
    raise AppError("缺少 API Key", status_code=401, code="unauthorized")


def _authenticate(
    db: Session,
    *,
    slug: str,
    expected_kind: str,
    authorization: str | None,
    x_api_key: str | None,
) -> PublishedApi:
    api_key = _extract_api_key(authorization, x_api_key)
    publication = db.scalar(select(PublishedApi).where(PublishedApi.slug == slug))
    if publication is None:
        raise NotFoundError("发布接口不存在")
    if not publication.is_active or publication.kind != expected_kind:
        raise AppError("接口未启用或类型不匹配", status_code=403, code="publication_disabled")
    if not hmac.compare_digest(publication.api_key_hash, hash_api_key(api_key)):
        raise AppError("API Key 无效", status_code=401, code="invalid_api_key")
    publication.last_used_at = datetime.now(UTC)
    db.commit()
    return publication


def _select_kb(publication: PublishedApi, requested_id: str | None) -> str:
    allowed = publication.knowledge_base_ids or []
    if requested_id:
        if requested_id not in allowed:
            raise AppError("该知识库未授权", status_code=403, code="knowledge_base_forbidden")
        return requested_id
    if len(allowed) == 1:
        return allowed[0]
    raise ValidationError("该接口绑定多个知识库，请指定 knowledge_base_id")


def _public_base(request: Request) -> str:
    base = settings.public_base_url.rstrip("/") if settings.public_base_url else str(request.base_url).rstrip("/")
    return f"{base}{settings.api_prefix}"


def _urls_for_document(request: Request, publication: PublishedApi, document_id: str) -> tuple[str, str]:
    token = create_download_token(document_id, publication.id)
    base = _public_base(request)
    file_url = f"{base}/public/files/{quote(document_id)}?token={quote(token)}"
    preview_url = f"{base}/public/documents/{quote(document_id)}/preview?token={quote(token)}"
    return file_url, preview_url


def _validate_public_document(
    db: Session, publication: PublishedApi, document_id: str
) -> Document:
    document = db.get(Document, document_id)
    if document is None or document.kb_id not in (publication.knowledge_base_ids or []):
        raise NotFoundError("文档不存在")
    return document


@router.get("/qa/{slug}/capabilities", response_model=PublicCapabilities)
def qa_capabilities(
    slug: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> PublicCapabilities:
    publication = _authenticate(
        db,
        slug=slug,
        expected_kind="qa",
        authorization=authorization,
        x_api_key=x_api_key,
    )
    knowledge_bases = list(
        db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(publication.knowledge_base_ids or [])))
    )
    return PublicCapabilities(
        publication=publication.name,
        kind=publication.kind,
        knowledge_bases=[
            {"id": kb.id, "name": kb.name, "description": kb.description} for kb in knowledge_bases
        ],
        settings=publication.settings or {},
    )


@router.get("/search/{slug}/capabilities", response_model=PublicCapabilities)
def search_capabilities(
    slug: str,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> PublicCapabilities:
    publication = _authenticate(
        db,
        slug=slug,
        expected_kind="search",
        authorization=authorization,
        x_api_key=x_api_key,
    )
    knowledge_bases = list(
        db.scalars(select(KnowledgeBase).where(KnowledgeBase.id.in_(publication.knowledge_base_ids or [])))
    )
    return PublicCapabilities(
        publication=publication.name,
        kind=publication.kind,
        knowledge_bases=[
            {"id": kb.id, "name": kb.name, "description": kb.description} for kb in knowledge_bases
        ],
        settings=publication.settings or {},
    )


@router.post("/qa/{slug}", response_model=ChatResponse | None)
async def public_qa(
    slug: str,
    request: PublicChatRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    publication = _authenticate(
        db,
        slug=slug,
        expected_kind="qa",
        authorization=authorization,
        x_api_key=x_api_key,
    )
    settings_data = publication.settings or {}
    kb_id = _select_kb(publication, request.knowledge_base_id)
    chat_request = ChatRequest(
        knowledge_base_id=kb_id,
        query=request.query,
        conversation_id=request.conversation_id,
        llm_profile_id=publication.llm_profile_id,
        document_id=request.document_id,
        stream=request.stream,
    )
    def url_builder(document_id: str) -> tuple[str | None, str | None]:
        return _urls_for_document(raw_request, publication, document_id)

    if not request.stream:
        return await chat_service.chat(db, chat_request, citation_url_builder=url_builder)
    if not settings_data.get("stream_enabled", True):
        raise ValidationError("该接口未开放流式输出")

    async def event_stream():
        stream_db = SessionLocal()
        try:
            async for event in chat_service.stream_chat(
                stream_db, chat_request, citation_url_builder=url_builder
            ):
                yield event
        finally:
            stream_db.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/search/{slug}", response_model=SearchResponse)
async def public_search(
    slug: str,
    request: PublicSearchRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> SearchResponse:
    publication = _authenticate(
        db,
        slug=slug,
        expected_kind="search",
        authorization=authorization,
        x_api_key=x_api_key,
    )
    settings_data = publication.settings or {}
    kb_id = _select_kb(publication, request.knowledge_base_id)
    search_request = SearchRequest(
        knowledge_base_id=kb_id,
        query=request.query,
        top_k=request.top_k or int(settings_data.get("top_k", 5)),
        rerank=request.rerank if request.rerank is not None else bool(settings_data.get("rerank", True)),
        document_id=request.document_id,
    )
    def url_builder(document_id: str) -> tuple[str | None, str | None]:
        return _urls_for_document(raw_request, publication, document_id)

    return await chat_service.search(db, search_request, file_url_builder=url_builder)


@router.get("/documents/{document_id}/preview")
def public_document_preview(
    document_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    token_document_id, publication_id = decode_download_token(token)
    if token_document_id != document_id:
        raise AppError("文件地址与文档不匹配", status_code=401, code="invalid_file_token")
    publication = db.get(PublishedApi, publication_id)
    if publication is None or not publication.is_active:
        raise AppError("文件地址已失效", status_code=401, code="invalid_file_token")
    document = _validate_public_document(db, publication, document_id)
    from app.db.models import DocumentChunk

    chunks = list(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
    )
    return {
        "document_id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "text": document.extracted_text,
        "metadata": document.metadata_json or {},
        "chunks": [ChunkRead.model_validate(chunk).model_dump() for chunk in chunks],
    }


@router.get("/files/{document_id}")
def public_document_file(
    document_id: str,
    token: str = Query(...),
    db: Session = Depends(get_db),
):
    token_document_id, publication_id = decode_download_token(token)
    if token_document_id != document_id:
        raise AppError("文件地址与文档不匹配", status_code=401, code="invalid_file_token")
    publication = db.get(PublishedApi, publication_id)
    if publication is None or not publication.is_active:
        raise AppError("文件地址已失效", status_code=401, code="invalid_file_token")
    document = _validate_public_document(db, publication, document_id)
    response = get_object_stream(document.object_key)

    def iterator():
        try:
            while data := response.read(1024 * 1024):
                yield data
        finally:
            response.close()
            response.release_conn()

    safe_name = quote(document.filename)
    return StreamingResponse(
        iterator(),
        media_type=document.content_type or "application/octet-stream",
        headers={
            "Content-Disposition": f"inline; filename*=UTF-8''{safe_name}",
            "Cache-Control": "private, max-age=60",
        },
    )


