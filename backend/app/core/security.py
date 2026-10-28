import base64
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Annotated

from cryptography.fernet import Fernet
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.core.config import settings
from app.core.errors import AppError

bearer_scheme = HTTPBearer(auto_error=False)


def verify_admin_credentials(username: str, password: str) -> bool:
    return hmac.compare_digest(username, settings.admin_username) and hmac.compare_digest(
        password, settings.admin_password
    )


def create_access_token(subject: str) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=settings.access_token_expire_minutes)
    return jwt.encode(
        {"sub": subject, "exp": expires, "type": "admin"},
        settings.secret_key,
        algorithm="HS256",
    )


def decode_access_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except JWTError as exc:
        raise AppError("登录状态已失效，请重新登录", status_code=401, code="unauthorized") from exc
    if payload.get("type") != "admin" or not payload.get("sub"):
        raise AppError("无效的访问令牌", status_code=401, code="unauthorized")
    return str(payload["sub"])


def get_current_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise AppError("请先登录", status_code=401, code="unauthorized")
    return decode_access_token(credentials.credentials)


def _fernet() -> Fernet:
    digest = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(value: str | None) -> str | None:
    if not value:
        return None
    return _fernet().encrypt(value.encode("utf-8")).decode("ascii")


def decrypt_secret(value: str | None) -> str:
    if not value:
        return ""
    return _fernet().decrypt(value.encode("ascii")).decode("utf-8")


def generate_api_key() -> str:
    return f"zy_{secrets.token_urlsafe(32)}"


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def create_download_token(document_id: str, publication_id: str) -> str:
    expires = datetime.now(UTC) + timedelta(minutes=settings.public_file_expire_minutes)
    return jwt.encode(
        {
            "sub": document_id,
            "publication_id": publication_id,
            "exp": expires,
            "type": "document_download",
        },
        settings.secret_key,
        algorithm="HS256",
    )


def decode_download_token(token: str) -> tuple[str, str]:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    except JWTError as exc:
        raise AppError("文件地址已失效", status_code=401, code="invalid_file_token") from exc
    if payload.get("type") != "document_download" or not payload.get("sub"):
        raise AppError("无效的文件地址", status_code=401, code="invalid_file_token")
    return str(payload["sub"]), str(payload.get("publication_id", ""))

