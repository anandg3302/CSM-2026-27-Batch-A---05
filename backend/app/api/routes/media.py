from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, status

from app.schemas.media import MediaCaseMetadata
from app.services import media_service

router = APIRouter(tags=["Media"])


@router.post(
    "/upload",
    response_model=MediaCaseMetadata,
    status_code=status.HTTP_201_CREATED,
)
async def upload_media(file: Optional[UploadFile] = File(default=None)) -> MediaCaseMetadata:
    """Upload an image file and initialize a new forensic case."""
    return await media_service.create_media_case(file)


@router.get("", response_model=List[MediaCaseMetadata])
def list_cases() -> List[MediaCaseMetadata]:
    """List all previously uploaded forensic cases."""
    return media_service.list_media_cases()


@router.get("/{case_id}", response_model=MediaCaseMetadata)
def get_case(case_id: str) -> MediaCaseMetadata:
    """Return stored forensic case metadata for a specific case_id."""
    return media_service.get_media_case(case_id)
