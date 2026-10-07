from typing import List
from pydantic import BaseModel

from app.core.version import MEDIATRACE_APPLICATION_VERSION


class HealthResponse(BaseModel):
    """Response model for GET /api/health."""

    status: str
    service: str
    version: str = MEDIATRACE_APPLICATION_VERSION


class ApiRootResponse(BaseModel):
    """Response model for GET /api."""

    message: str
    service: str
    version: str = MEDIATRACE_APPLICATION_VERSION
    supported_modalities: List[str]
