import io
import logging
import uuid
from pathlib import Path
from typing import BinaryIO

from minio import Minio
from minio.error import S3Error

from app.core.config import settings
from app.core.errors import AppError

logger = logging.getLogger(__name__)

_client: Minio | None = None


def get_storage() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
            region=settings.minio_region,
        )
    return _client


def ensure_bucket() -> None:
    client = get_storage()
    try:
        if not client.bucket_exists(settings.minio_bucket):
            client.make_bucket(settings.minio_bucket)
    except S3Error as exc:
        raise AppError("对象存储初始化失败", status_code=503, code="storage_unavailable") from exc


def build_object_key(filename: str, prefix: str = "documents") -> str:
    safe_name = filename.replace("\\", "_").replace("/", "_")
    return f"{prefix}/{uuid.uuid4().hex}/{safe_name}"


def upload_bytes(object_key: str, data: bytes, content_type: str) -> None:
    upload_stream(object_key, io.BytesIO(data), len(data), content_type)


def upload_stream(
    object_key: str,
    stream: BinaryIO,
    length: int,
    content_type: str,
    part_size: int = 16 * 1024 * 1024,
) -> None:
    ensure_bucket()
    stream.seek(0)
    get_storage().put_object(
        settings.minio_bucket,
        object_key,
        stream,
        length=length,
        content_type=content_type,
        part_size=part_size,
    )


def upload_file(object_key: str, file_path: str | Path, content_type: str) -> None:
    ensure_bucket()
    path = Path(file_path)
    get_storage().fput_object(
        settings.minio_bucket,
        object_key,
        str(path),
        content_type=content_type,
        part_size=16 * 1024 * 1024,
    )


def get_object_stream(object_key: str):
    ensure_bucket()
    return get_storage().get_object(settings.minio_bucket, object_key)


def download_bytes(object_key: str) -> bytes:
    response = get_object_stream(object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def delete_object(object_key: str) -> None:
    try:
        get_storage().remove_object(settings.minio_bucket, object_key)
    except S3Error:
        logger.warning("failed to delete object key=%s", object_key, exc_info=True)
