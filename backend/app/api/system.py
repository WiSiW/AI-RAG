from fastapi import APIRouter, Depends

from app.chunking.modes import CHUNK_MODES
from app.core.config import settings
from app.core.security import get_current_admin
from app.parsing.registry import SUPPORTED_EXTENSIONS
from app.schemas import CapabilitiesResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(_: str = Depends(get_current_admin)) -> CapabilitiesResponse:
    return CapabilitiesResponse(
        upload_max_mb=settings.upload_max_mb,
        archive_max_mb=settings.archive_max_mb,
        chunk_modes=CHUNK_MODES,
        supported_extensions=list(SUPPORTED_EXTENSIONS),
    )
