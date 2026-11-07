import asyncio
import logging
from datetime import UTC, datetime

from app.core.config import settings
from app.db.models import KnowledgeBase
from app.db.session import SessionLocal
from app.evaluation.service import ChunkEvaluationService
from app.services.chat_task_service import ChatTaskService
from app.services.chunk_tuning_service import ChunkTuningService
from app.services.conversation_summary_service import ConversationSummaryService
from app.services.ingestion_service import IngestionService
from app.services.knowledge_base_transfer_service import KnowledgeBaseTransferService
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    name="zhiyuan.run_chat_generation",
    time_limit=30 * 60,
    soft_time_limit=29 * 60,
)
def run_chat_generation_task(task_id: str) -> None:
    asyncio.run(ChatTaskService().run(task_id))


@celery_app.task(
    name="zhiyuan.ingest_document",
    bind=True,
    max_retries=2,
    time_limit=settings.pdf_parse_task_time_limit,
    soft_time_limit=settings.pdf_parse_task_soft_time_limit,
)
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


@celery_app.task(
    name="zhiyuan.summarize_conversation",
    bind=True,
    max_retries=2,
    time_limit=5 * 60,
    soft_time_limit=4 * 60 + 30,
)
def summarize_conversation_task(
    self,
    conversation_id: str,
    summary_task_id: str,
) -> None:
    db = SessionLocal()
    service = ConversationSummaryService()
    try:
        asyncio.run(service.summarize(db, conversation_id, summary_task_id))
    except Exception as exc:
        logger.exception("conversation summary task failed conversation_id=%s", conversation_id)
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc, countdown=5 * (self.request.retries + 1)) from exc
        service.mark_failed(db, conversation_id, summary_task_id, str(exc))
        raise
    finally:
        db.close()


@celery_app.task(
    name="zhiyuan.rebuild_knowledge_base_index",
    time_limit=6 * 60 * 60,
    soft_time_limit=6 * 60 * 60 - 300,
)
def rebuild_knowledge_base_index_task(kb_id: str) -> dict:
    db = SessionLocal()
    try:
        return asyncio.run(IngestionService().rebuild_knowledge_base_index(db, kb_id))
    except Exception as exc:
        db.rollback()
        knowledge_base = db.get(KnowledgeBase, kb_id)
        if knowledge_base:
            settings_data = dict(knowledge_base.settings or {})
            migration = dict(settings_data.get("index_migration") or {})
            migration.update(
                {
                    "status": "failed",
                    "finished_at": datetime.now(UTC).isoformat(),
                    "errors": [
                        *(migration.get("errors") or []),
                        {"error": str(exc)[:1000]},
                    ],
                }
            )
            knowledge_base.settings = {**settings_data, "index_migration": migration}
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(
    name="zhiyuan.evaluate_chunks",
    time_limit=6 * 60 * 60,
    soft_time_limit=6 * 60 * 60 - 300,
)
def evaluate_chunks_task(run_id: str) -> None:
    db = SessionLocal()
    try:
        asyncio.run(ChunkEvaluationService().run(db, run_id))
    finally:
        db.close()


@celery_app.task(
    name="zhiyuan.run_chunk_tuning",
    time_limit=12 * 60 * 60,
    soft_time_limit=12 * 60 * 60 - 300,
)
def run_chunk_tuning_task(run_id: str) -> None:
    db = SessionLocal()
    try:
        asyncio.run(ChunkTuningService().run(db, run_id))
    finally:
        db.close()


@celery_app.task(
    name="zhiyuan.export_knowledge_base",
    time_limit=12 * 60 * 60,
    soft_time_limit=12 * 60 * 60 - 300,
)
def export_knowledge_base_task(job_id: str) -> None:
    db = SessionLocal()
    try:
        KnowledgeBaseTransferService().export_knowledge_base(db, job_id)
    finally:
        db.close()


@celery_app.task(
    name="zhiyuan.import_knowledge_base",
    time_limit=12 * 60 * 60,
    soft_time_limit=12 * 60 * 60 - 300,
)
def import_knowledge_base_task(job_id: str) -> None:
    db = SessionLocal()
    try:
        asyncio.run(KnowledgeBaseTransferService().import_knowledge_base(db, job_id))
    finally:
        db.close()
