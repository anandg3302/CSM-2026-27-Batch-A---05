from fastapi import APIRouter

from app.schemas.evidence import EvidenceReport
from app.schemas.media import MediaCaseMetadata
from app.services.evidence import evidence_service

router = APIRouter(tags=["Evidence"])


@router.post("/{case_id}/generate", response_model=MediaCaseMetadata)
def generate_case_evidence(case_id: str) -> MediaCaseMetadata:
    """Correlate all completed forensic stages for a case and persist the EvidenceReport."""
    return evidence_service.generate_evidence_for_case(case_id)


@router.get("/{case_id}", response_model=EvidenceReport)
def get_case_evidence(case_id: str) -> EvidenceReport:
    """Return the stored EvidenceReport for a specific case_id."""
    return evidence_service.get_evidence_for_case(case_id)
