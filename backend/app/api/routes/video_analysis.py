from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.media import MediaCaseMetadata
from app.services.analysis import video_analysis_service

router = APIRouter(tags=["Video Analysis"])


@router.post("/{case_id}", response_model=MediaCaseMetadata)
def analyze_video_case(case_id: str) -> MediaCaseMetadata:
    """Execute non-destructive Step 10 video forensic analysis for an uploaded video case."""
    return video_analysis_service.analyze_video_for_case(case_id)


@router.get("/{case_id}/artifacts/{artifact_name}")
def get_video_frame_artifact(case_id: str, artifact_name: str) -> FileResponse:
    """Serve an extracted representative or suspicious frame artifact from storage/results/<case_id>/video/."""
    artifact_path = video_analysis_service.get_video_artifact_path(case_id, artifact_name)
    return FileResponse(path=str(artifact_path), media_type="image/jpeg")
