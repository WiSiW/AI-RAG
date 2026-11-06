from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _apply_schema_migrations() -> None:
    """Keep existing installations usable without a separate migration service."""
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    additions: list[tuple[str, str, str]] = []
    if "documents" in tables:
        document_columns = {column["name"] for column in inspector.get_columns("documents")}
        if "metadata_json" not in document_columns:
            additions.append(("documents", "metadata_json", "JSON DEFAULT '{}'"))
        if "references_json" not in document_columns:
            additions.append(("documents", "references_json", "JSON DEFAULT '[]'"))
        if "chunk_mode" not in document_columns:
            additions.append(("documents", "chunk_mode", "VARCHAR(40) DEFAULT 'structure'"))
    if "model_profiles" in tables:
        model_columns = {
            column["name"] for column in inspector.get_columns("model_profiles")
        }
        if "request_timeout_seconds" not in model_columns:
            additions.append(
                ("model_profiles", "request_timeout_seconds", "INTEGER DEFAULT 1200")
            )
        if "request_max_retries" not in model_columns:
            additions.append(
                ("model_profiles", "request_max_retries", "INTEGER DEFAULT 2")
            )
    if "knowledge_bases" in tables:
        knowledge_base_columns = {
            column["name"] for column in inspector.get_columns("knowledge_bases")
        }
        if "enrichment_profile_id" not in knowledge_base_columns:
            additions.append(("knowledge_bases", "enrichment_profile_id", "VARCHAR(36)"))
    if "document_chunks" in tables:
        chunk_columns = {column["name"] for column in inspector.get_columns("document_chunks")}
        if "summary" not in chunk_columns:
            additions.append(("document_chunks", "summary", "TEXT DEFAULT ''"))
        if "keywords" not in chunk_columns:
            additions.append(("document_chunks", "keywords", "JSON DEFAULT '[]'"))
        if "hypothetical_questions" not in chunk_columns:
            additions.append(("document_chunks", "hypothetical_questions", "JSON DEFAULT '[]'"))
        if "enrichment_status" not in chunk_columns:
            additions.append(
                ("document_chunks", "enrichment_status", "VARCHAR(20) DEFAULT 'pending'")
            )
        if "enrichment_meta" not in chunk_columns:
            additions.append(("document_chunks", "enrichment_meta", "JSON DEFAULT '{}'"))
        if "granularity_level" not in chunk_columns:
            additions.append(
                ("document_chunks", "granularity_level", "VARCHAR(2) DEFAULT 'L2'")
            )
        if "granule_type" not in chunk_columns:
            additions.append(
                ("document_chunks", "granule_type", "VARCHAR(40) DEFAULT 'paragraph'")
            )
        if "parent_chunk_id" not in chunk_columns:
            additions.append(("document_chunks", "parent_chunk_id", "VARCHAR(36)"))
        if "context_prefix" not in chunk_columns:
            additions.append(("document_chunks", "context_prefix", "TEXT DEFAULT ''"))
        if "qa_pairs" not in chunk_columns:
            additions.append(("document_chunks", "qa_pairs", "JSON DEFAULT '[]'"))
    if "chunk_evaluation_runs" in tables:
        evaluation_columns = {
            column["name"] for column in inspector.get_columns("chunk_evaluation_runs")
        }
        if "stage" not in evaluation_columns:
            additions.append(
                ("chunk_evaluation_runs", "stage", "VARCHAR(30) DEFAULT 'queued'")
            )
        if "celery_task_id" not in evaluation_columns:
            additions.append(("chunk_evaluation_runs", "celery_task_id", "VARCHAR(100)"))
        if "pause_requested_at" not in evaluation_columns:
            additions.append(("chunk_evaluation_runs", "pause_requested_at", "TIMESTAMP"))
        if "paused_at" not in evaluation_columns:
            additions.append(("chunk_evaluation_runs", "paused_at", "TIMESTAMP"))
        if "heartbeat_at" not in evaluation_columns:
            additions.append(("chunk_evaluation_runs", "heartbeat_at", "TIMESTAMP"))
    if "conversations" in tables:
        conversation_columns = {
            column["name"] for column in inspector.get_columns("conversations")
        }
        if "document_id" not in conversation_columns:
            additions.append(("conversations", "document_id", "VARCHAR(36)"))
        if "summary_json" not in conversation_columns:
            additions.append(("conversations", "summary_json", "JSON DEFAULT '{}'"))
        if "summary_message_count" not in conversation_columns:
            additions.append(("conversations", "summary_message_count", "INTEGER DEFAULT 0"))
        if "summary_status" not in conversation_columns:
            additions.append(("conversations", "summary_status", "VARCHAR(20) DEFAULT 'idle'"))
        if "summary_version" not in conversation_columns:
            additions.append(("conversations", "summary_version", "INTEGER DEFAULT 0"))
        if "summary_started_at" not in conversation_columns:
            additions.append(("conversations", "summary_started_at", "TIMESTAMP"))
        if "summary_updated_at" not in conversation_columns:
            additions.append(("conversations", "summary_updated_at", "TIMESTAMP"))
        if "summary_task_id" not in conversation_columns:
            additions.append(("conversations", "summary_task_id", "VARCHAR(100)"))
        if "summary_error" not in conversation_columns:
            additions.append(("conversations", "summary_error", "TEXT"))
    with engine.begin() as connection:
        for table, column, ddl in additions:
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))
        if "chunk_evaluation_runs" in tables:
            connection.execute(
                text(
                    """
                    UPDATE chunk_evaluation_runs
                    SET stage = CASE
                        WHEN status = 'completed' THEN 'completed'
                        WHEN status = 'failed' THEN 'failed'
                        ELSE 'queued'
                    END
                    WHERE stage IS NULL OR stage = ''
                    """
                )
            )
        if "conversation_knowledge_bases" in tables and "conversations" in tables:
            connection.execute(
                text(
                    """
                    INSERT INTO conversation_knowledge_bases
                        (conversation_id, knowledge_base_id, position, created_at)
                    SELECT c.id, c.knowledge_base_id, 0, CURRENT_TIMESTAMP
                    FROM conversations c
                    WHERE c.knowledge_base_id IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1
                          FROM conversation_knowledge_bases ckb
                          WHERE ckb.conversation_id = c.id
                      )
                    """
                )
            )


def init_db() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _apply_schema_migrations()
