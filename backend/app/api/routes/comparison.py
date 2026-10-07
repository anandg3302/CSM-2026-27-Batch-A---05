from typing import Optional
from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import FileResponse

from app.schemas.media import MediaCaseMetadata
from app.services.comparison import image_comparison_service

router = APIRouter(tags=["Comparison"])


@router.post(
    "/image/{case_id}/reference",
    response_model=MediaCaseMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_reference_image_for_case(
    case_id: str,
    file: Optional[UploadFile] = File(default=None),
) -> MediaCaseMetadata:
    """Upload a separate reference/original image for comparison against an existing case image."""
    return await image_comparison_service.upload_reference_image(case_id, file)


@router.post(
    "/image/{case_id}",
    response_model=MediaCaseMetadata,
)
def compare_case_image_with_reference(case_id: str) -> MediaCaseMetadata:
    """Compare the case image against its uploaded reference image and generate visual difference artifacts."""
    return image_comparison_service.compare_case_with_reference(case_id)


@router.get("/image/{case_id}/{artifact_name:path}")
def get_comparison_artifact(case_id: str, artifact_name: str) -> FileResponse:
    """Serve generated difference.png or difference_mask.png while preventing path traversal."""
    path = image_comparison_service.get_comparison_artifact_path(case_id, artifact_name)
    return FileResponse(path, media_type="image/png")
