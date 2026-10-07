from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.schemas.media import MediaCaseMetadata
from app.services.analysis import audio_analysis_service

router = APIRouter()


@router.post(
    "/{case_id}",
    response_model=MediaCaseMetadata,
    summary="Run conservative forensic audio analysis for an uploaded audio case (Step 11)",
)
def run_audio_analysis(case_id: str) -> MediaCaseMetadata:
    return audio_analysis_service.analyze_audio_for_case(case_id)


@router.get(
    "/{case_id}/artifacts/{artifact_name}",
    summary="Serve a derived audio forensic artifact (waveform.png, spectrogram.png, discontinuity_<ms>.png)",
)
def get_audio_artifact(case_id: str, artifact_name: str) -> FileResponse:
    artifact_path = audio_analysis_service.get_audio_artifact_path(case_id, artifact_name)
    return FileResponse(artifact_path, media_type="image/png")
