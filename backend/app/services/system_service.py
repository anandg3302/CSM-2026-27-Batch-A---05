from app.core.config import settings
from app.core.version import MEDIATRACE_APPLICATION_VERSION
from app.schemas.system import ApiRootResponse, HealthResponse


def get_health_status() -> HealthResponse:
    """Return service health metadata."""
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=MEDIATRACE_APPLICATION_VERSION,
    )


def get_api_root_info() -> ApiRootResponse:
    """Return baseline information about the MediaTrace API."""
    return ApiRootResponse(
        message="Welcome to MediaTrace API - Digital Media Forensics Platform",
        service=settings.app_name,
        version=MEDIATRACE_APPLICATION_VERSION,
        supported_modalities=["image", "video", "audio", "document"],
    )
