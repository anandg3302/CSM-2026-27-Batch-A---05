from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import settings
from app.core.version import MEDIATRACE_APPLICATION_VERSION

logger = logging.getLogger("mediatrace")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure required storage directories exist when the application starts."""
    settings.ensure_storage_directories()
    logger.info("MediaTrace API initialized (env=%s, version=%s)", settings.app_env, MEDIATRACE_APPLICATION_VERSION)
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    application = FastAPI(
        title=settings.app_name,
        description="Digital media forensics platform for analyzing images, videos, audio, and documents.",
        version=MEDIATRACE_APPLICATION_VERSION,
        lifespan=lifespan,
    )

    # Configure CORS for configured frontend origins
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        """Log unexpected server errors safely without leaking internal Python tracebacks."""
        logger.error("Unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred while processing the forensic request."},
        )

    application.include_router(api_router)
    return application


app = create_app()

