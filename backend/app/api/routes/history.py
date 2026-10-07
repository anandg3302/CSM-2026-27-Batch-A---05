from fastapi import APIRouter

from app.schemas.history import EditHistoryReport
from app.schemas.media import MediaCaseMetadata
from app.services.history import history_service

router = APIRouter(tags=["History"])


@router.post("/{case_id}/reconstruct", response_model=MediaCaseMetadata)
def reconstruct_case_history(case_id: str) -> MediaCaseMetadata:
    """Reconstruct a conservative possible edit-history sequence for a case and persist it."""
    return history_service.reconstruct_history_for_case(case_id)


@router.get("/{case_id}", response_model=EditHistoryReport)
def get_case_history(case_id: str) -> EditHistoryReport:
    """Return the stored EditHistoryReport for a specific case_id."""
    return history_service.get_history_for_case(case_id)
