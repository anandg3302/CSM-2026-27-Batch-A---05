from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.schemas.report import ForensicReport
from app.services.reporting import pdf_report_service, report_service

router = APIRouter(tags=["Reports"])


@router.post("/{case_id}/generate", response_model=ForensicReport)
def generate_case_forensic_report(case_id: str) -> ForensicReport:
    """Generate and persist a structured JSON forensic report for a case (Step 9)."""
    return report_service.generate_forensic_report(case_id)


@router.get("/{case_id}", response_model=ForensicReport)
def get_case_forensic_report(case_id: str) -> ForensicReport:
    """Return the persisted JSON ForensicReport for a case (Step 9)."""
    return report_service.get_forensic_report(case_id)


@router.post("/{case_id}/generate-pdf")
def generate_case_forensic_report_pdf(case_id: str) -> FileResponse:
    """Generate and return the human-readable PDF forensic report for a case (Step 9)."""
    pdf_path = pdf_report_service.generate_pdf_report_for_case(case_id)
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="MediaTrace_Forensic_Report.pdf",
    )


@router.get("/{case_id}/pdf")
def get_case_forensic_report_pdf(case_id: str) -> FileResponse:
    """Download the generated PDF forensic report for a case if it exists."""
    report_service._load_case_readonly(case_id)
    pdf_path = settings.reports_dir / case_id / "MediaTrace_Forensic_Report.pdf"
    if not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF forensic report for case '{case_id}' has not been generated yet.",
        )
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="MediaTrace_Forensic_Report.pdf",
    )
