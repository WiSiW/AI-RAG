import asyncio
import logging

from app.db.session import SessionLocal
from app.services.ingestion_service import IngestionService
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="zhiyuan.ingest_document", bind=True, max_retries=2)
def ingest_document_task(self, document_id: str, job_id: str) -> None:
    db = SessionLocal()
    try:
        asyncio.run(IngestionService().ingest(db, document_id, job_id))
    except Exception as exc:
        logger.exception("ingestion task failed document_id=%s", document_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=10 * (self.request.retries + 1)) from exc
        raise
    finally:
        db.close()
