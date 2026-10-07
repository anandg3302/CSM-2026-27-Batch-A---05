from fastapi import APIRouter

from app.api.routes import (
    analysis,
    audio_analysis,
    comparison,
    document_analysis,
    evidence,
    forensics,
    history,
    media,
    reports,
    system,
    video_analysis,
)
from app.core.config import settings

api_router = APIRouter()

# Register system routes (/api and /api/health)
api_router.include_router(system.router, prefix=settings.api_prefix)

# Register media upload and forensic case routes (/api/media)
api_router.include_router(media.router, prefix=f"{settings.api_prefix}/media")

# Register forensic analysis routes (/api/analysis)
api_router.include_router(analysis.router, prefix=f"{settings.api_prefix}/analysis")

# Register video forensic analysis routes (/api/analysis/video)
api_router.include_router(
    video_analysis.router, prefix=f"{settings.api_prefix}/analysis/video"
)

# Register audio forensic analysis routes (/api/analysis/audio)
api_router.include_router(
    audio_analysis.router, prefix=f"{settings.api_prefix}/analysis/audio"
)

# Register document forensic analysis routes (/api/analysis/document)
api_router.include_router(
    document_analysis.router, prefix=f"{settings.api_prefix}/analysis/document"
)

# Register original vs edited comparison routes (/api/comparison)
api_router.include_router(comparison.router, prefix=f"{settings.api_prefix}/comparison")

# Register evidence correlation routes (/api/evidence)
api_router.include_router(evidence.router, prefix=f"{settings.api_prefix}/evidence")

# Register multi-signal forensic decision & interpretation routes (/api/forensics)
api_router.include_router(forensics.router, prefix=f"{settings.api_prefix}/forensics")

# Register possible edit-history reconstruction routes (/api/history)
api_router.include_router(history.router, prefix=f"{settings.api_prefix}/history")

# Register forensic report generation routes (/api/reports)
api_router.include_router(reports.router, prefix=f"{settings.api_prefix}/reports")



