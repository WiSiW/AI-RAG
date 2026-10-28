import redis
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.storage import ensure_bucket, get_storage
from app.db.session import get_db
from app.indexing.bm25_store import BM25Store
from app.indexing.vector_store import MilvusVectorStore
from app.schemas import HealthComponent, HealthResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
def live() -> dict:
    return {"status": "ok", "service": settings.app_name, "version": settings.app_version}


@router.get("/ready", response_model=HealthResponse)
def ready(db: Session = Depends(get_db)) -> HealthResponse:
    components: dict[str, HealthComponent] = {}
    overall = "ok"

    try:
        db.execute(text("SELECT 1"))
        components["postgres"] = HealthComponent(status="ok")
    except Exception as exc:
        components["postgres"] = HealthComponent(status="error", detail=str(exc))
        overall = "degraded"

    try:
        redis.from_url(settings.redis_url, decode_responses=True).ping()
        components["redis"] = HealthComponent(status="ok")
    except Exception as exc:
        components["redis"] = HealthComponent(status="error", detail=str(exc))
        overall = "degraded"

    try:
        ensure_bucket()
        exists = get_storage().bucket_exists(settings.minio_bucket)
        components["minio"] = HealthComponent(
            status="ok" if exists else "error",
            detail="bucket exists" if exists else "bucket missing",
        )
        if not exists:
            overall = "degraded"
    except Exception as exc:
        components["minio"] = HealthComponent(status="error", detail=str(exc))
        overall = "degraded"

    milvus_ok, milvus_detail = MilvusVectorStore().health()
    components["milvus"] = HealthComponent(
        status="ok" if milvus_ok else "error", detail=milvus_detail
    )
    if not milvus_ok:
        overall = "degraded"

    bm25_ok, bm25_detail = BM25Store().health()
    components["bm25"] = HealthComponent(
        status="ok" if bm25_ok else "error", detail=bm25_detail
    )
    if not bm25_ok:
        overall = "degraded"

    return HealthResponse(
        status=overall,
        service=settings.app_name,
        version=settings.app_version,
        components=components,
    )
