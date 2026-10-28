from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ZHIYUAN_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "知源 - 企业知识中枢问答系统"
    app_version: str = "0.1.0"
    environment: str = "development"
    api_prefix: str = "/api/v1"

    secret_key: str = "change-me-to-a-long-random-string-at-least-32-chars"
    access_token_expire_minutes: int = 720
    admin_username: str = "admin"
    admin_password: str = "change-me"

    database_url: str = "postgresql+psycopg://zhiyuan:zhiyuan@localhost:5432/zhiyuan"
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "zhiyuan"
    minio_secure: bool = False
    minio_region: str = "us-east-1"

    milvus_uri: str = "http://localhost:19530"

    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    auto_create_tables: bool = True
    upload_max_mb: int = 2048
    archive_max_mb: int = 4096
    upload_temp_dir: str = "data/uploads"
    import_root: str = "data/imports"
    public_base_url: str = ""
    public_file_expire_minutes: int = 60
    model_request_timeout: float = 120.0
    embedding_batch_size: int = 32
    session_window_size: int = 10
    summary_trigger_messages: int = 16
    summary_keep_messages: int = 8
    summary_max_tokens: int = 512
    query_rewrite_max_tokens: int = 256
    rag_candidate_top_k: int = 20
    rag_final_top_k: int = 5

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def upload_max_bytes(self) -> int:
        return self.upload_max_mb * 1024 * 1024

    @property
    def archive_max_bytes(self) -> int:
        return self.archive_max_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()



