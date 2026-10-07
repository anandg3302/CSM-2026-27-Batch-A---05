from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.media import MediaCaseMetadata
from app.services.analysis import document_analysis_service

router = APIRouter()


@router.post(
    "/{case_id}",
    response_model=MediaCaseMetadata,
    summary="Run conservative forensic document analysis for an uploaded document case (Step 12)",
)
def run_document_analysis(case_id: str) -> MediaCaseMetadata:
    return document_analysis_service.analyze_document_for_case(case_id)


@router.get(
    "/{case_id}/artifacts/{artifact_name}",
    summary="Serve a derived document preview artifact (e.g. page_001.png)",
)
def get_document_artifact(case_id: str, artifact_name: str) -> FileResponse:
    artifact_path = document_analysis_service.get_document_artifact_path(case_id, artifact_name)
    return FileResponse(artifact_path, media_type="image/png")
