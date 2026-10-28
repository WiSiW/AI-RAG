from fastapi import APIRouter, Depends

from app.core.config import settings
from app.core.errors import AppError
from app.core.security import create_access_token, get_current_admin, verify_admin_credentials
from app.schemas import LoginRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest) -> TokenResponse:
    if not verify_admin_credentials(request.username, request.password):
        raise AppError("用户名或密码错误", status_code=401, code="invalid_credentials")
    return TokenResponse(
        access_token=create_access_token(request.username),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me")
def me(admin: str = Depends(get_current_admin)) -> dict:
    return {"username": admin}
