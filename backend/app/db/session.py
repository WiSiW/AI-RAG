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
        if "chunk_mode" not in document_columns:
            additions.append(("documents", "chunk_mode", "VARCHAR(40) DEFAULT 'structure'"))
    if "conversations" in tables:
        conversation_columns = {
            column["name"] for column in inspector.get_columns("conversations")
        }
        if "document_id" not in conversation_columns:
            additions.append(("conversations", "document_id", "VARCHAR(36)"))
    if not additions:
        return
    with engine.begin() as connection:
        for table, column, ddl in additions:
            connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}"))


def init_db() -> None:
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _apply_schema_migrations()
