from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.ai_detection import AiGenerationDetectionResult
from app.schemas.media import MediaCaseMetadata
from app.services.analysis import (
    image_analysis_service,
    image_detection_service,
    image_localization_service,
)

router = APIRouter(tags=["Analysis"])


@router.post("/image/{case_id}", response_model=MediaCaseMetadata)
def analyze_image_case(case_id: str) -> MediaCaseMetadata:
    """Run basic forensic analysis on an uploaded image case (Step 3)."""
    return image_analysis_service.analyze_image(case_id)


@router.post("/image/{case_id}/detect", response_model=MediaCaseMetadata)
def detect_image_case_manipulation(case_id: str) -> MediaCaseMetadata:
    """Run manipulation and AI-generation detection on an image case (Step 4 & Step 14.1)."""
    return image_detection_service.detect_image_manipulation(case_id)


@router.get("/image/{case_id}/ai-detection", response_model=AiGenerationDetectionResult)
def get_image_case_ai_detection(case_id: str) -> AiGenerationDetectionResult:
    """Return the stored AI-generation detection result for a case without rerunning inference (Step 14.1)."""
    return image_detection_service.get_stored_ai_detection(case_id)


@router.post("/image/{case_id}/localize", response_model=MediaCaseMetadata)
def localize_image_case_manipulation(case_id: str) -> MediaCaseMetadata:
    """Run spatial manipulation localization and generate heatmap/mask artifacts (Step 5)."""
    return image_localization_service.localize_image_manipulation(case_id)


@router.get("/image/{case_id}/localization/{artifact_name}")
def get_localization_visualization(case_id: str, artifact_name: str) -> FileResponse:
    """Serve generated localization heatmap.png or mask.png for display in the frontend."""
    path = image_localization_service.get_localization_artifact_path(case_id, artifact_name)
    return FileResponse(path, media_type="image/png")

