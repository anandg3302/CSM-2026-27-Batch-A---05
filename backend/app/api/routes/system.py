from fastapi import APIRouter

from app.schemas.system import ApiRootResponse, HealthResponse
from app.services import system_service

router = APIRouter(tags=["System"])


@router.get("", response_model=ApiRootResponse)
def read_api_root() -> ApiRootResponse:
    """Simple root endpoint GET /api returning basic API information."""
    return system_service.get_api_root_info()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health-check endpoint GET /api/health."""
    return system_service.get_health_status()
