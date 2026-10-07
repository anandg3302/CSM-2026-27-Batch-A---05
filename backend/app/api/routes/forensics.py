from fastapi import APIRouter

from app.schemas.forensic_decision import ForensicDecisionResult
from app.schemas.reproducibility import (
    ForensicAnalysisSnapshot,
    ReproducibilityCheckResult,
)
from app.services.evidence import forensic_decision_service, reproducibility_service

router = APIRouter(tags=["Forensic Interpretation & Reproducibility"])


@router.post("/{case_id}/interpret", response_model=ForensicDecisionResult)
def interpret_case_forensics(case_id: str) -> ForensicDecisionResult:
    """
    Execute the conservative multi-signal forensic decision & interpretation layer (Step 14.2)
    over existing analysis and evidence results without re-running image models.
    """
    return forensic_decision_service.interpret_forensics_for_case(case_id)


@router.get("/{case_id}/interpret", response_model=ForensicDecisionResult)
def get_case_forensic_interpretation(case_id: str) -> ForensicDecisionResult:
    """Return the persisted ForensicDecisionResult for a specific case_id."""
    return forensic_decision_service.get_forensic_interpretation_for_case(case_id)


@router.post("/{case_id}/reproducibility/check", response_model=ReproducibilityCheckResult)
def check_case_forensic_reproducibility(case_id: str) -> ReproducibilityCheckResult:
    """
    Execute a non-destructive forensic reproducibility and integrity verification check (Step 14.3)
    without regenerating or overwriting any artifacts or case results.
    """
    return reproducibility_service.check_case_reproducibility(case_id)


@router.get("/{case_id}/reproducibility", response_model=ForensicAnalysisSnapshot)
def get_case_forensic_reproducibility(case_id: str) -> ForensicAnalysisSnapshot:
    """
    Return the immutable ForensicAnalysisSnapshot and audit trail for a specific case_id (Step 14.3).
    """
    return reproducibility_service.get_case_reproducibility_snapshot(case_id, refresh=True)

