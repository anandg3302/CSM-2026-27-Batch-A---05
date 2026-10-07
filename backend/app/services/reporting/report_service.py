from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings
from app.core.version import MEDIATRACE_APPLICATION_VERSION
from app.schemas.evidence import (
    AllowedEvidenceAssessment,
    EvidenceItem,
    EvidenceReport,
    EvidenceSummary,
)
from app.schemas.history import (
    EditHistoryReport,
    HistoryIntegrityInfo,
    KnownFactItem,
    ReconstructedHistoryEvent,
)
from app.schemas.media import MediaCaseMetadata
from app.schemas.report import (
    ForensicReport,
    ReportAssessmentSection,
    ReportCaseInfo,
    ReportComparisonSection,
    ReportCrossAnalysisItem,
    ReportEvidenceSection,
    ReportForensicsSection,
    ReportHistorySection,
    ReportIntegritySection,
    ReportLocalizationSection,
    ReportManipulationDetectionSection,
    ReportMediaSection,
)
from app.services.evidence.evidence_service import ASSESSMENT_LABELS
from app.services.evidence.forensic_decision_service import build_forensic_decision
from app.services.media_service import CASE_ID_PATTERN


def _load_case_readonly(case_id: str) -> MediaCaseMetadata:
    """Load a MediaCaseMetadata record from disk without modifying it."""
    if not CASE_ID_PATTERN.match(case_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format.",
        )

    result_path = settings.results_dir / f"{case_id}.json"
    if not result_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forensic case '{case_id}' was not found.",
        )

    try:
        with open(result_path, "r", encoding="utf-8") as file_in:
            raw_data = json.load(file_in)
        return MediaCaseMetadata(**raw_data)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metadata for case '{case_id}'.",
        ) from exc


def _build_deterministic_report_id(case_data: MediaCaseMetadata) -> str:
    """
    Generate a unique, deterministic report ID formatted as MTR-YYYYMMDD-XXXXXXXX
    that is distinct from case_id.
    """
    parts = case_data.case_id.split("-")
    date_part = parts[1] if len(parts) >= 3 and len(parts[1]) == 8 else "20260928"
    digest = hashlib.sha256(
        f"MTR:{case_data.case_id}:{case_data.sha256}".encode("utf-8")
    ).hexdigest()[:8].upper()
    return f"MTR-{date_part}-{digest}"


def _verify_hash_preserved(case_data: MediaCaseMetadata) -> bool:
    upload_path = settings.uploads_dir / case_data.stored_filename
    if not upload_path.is_file():
        return False
    current_sha256 = hashlib.sha256(upload_path.read_bytes()).hexdigest()
    return current_sha256.lower() == case_data.sha256.lower()


def _build_case_section(case_data: MediaCaseMetadata) -> ReportCaseInfo:
    return ReportCaseInfo(
        case_id=case_data.case_id,
        original_filename=case_data.original_filename,
        media_type=case_data.media_type,
        extension=case_data.extension,
        file_size=case_data.file_size,
        created_at=case_data.created_at,
        status=case_data.status,
    )


def _build_integrity_section(
    case_data: MediaCaseMetadata,
    hash_preserved: bool,
) -> ReportIntegritySection:
    return ReportIntegritySection(
        sha256=case_data.sha256,
        hash_preserved=hash_preserved,
    )


def _build_media_and_forensics_sections(
    case_data: MediaCaseMetadata,
) -> Tuple[ReportMediaSection, ReportForensicsSection]:
    ext_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    file_type = ext_map.get(case_data.extension.lower(), f"image/{case_data.extension.lstrip('.')}")

    if case_data.analysis is None:
        return (
            ReportMediaSection(
                available=False,
                file_type=file_type,
            ),
            ReportForensicsSection(
                available=False,
                exif_summary="Step 3 basic image forensic analysis has not been executed for this case.",
            ),
        )

    analysis = case_data.analysis
    img = analysis.image
    meta = analysis.metadata
    fields = meta.fields

    software_present = bool(fields.software and fields.software.strip())
    camera_present = bool((fields.make and fields.make.strip()) or (fields.model and fields.model.strip()))
    timestamp_present = bool(fields.datetime_original or fields.datetime)
    gps_present = bool(meta.gps_present)

    media_sec = ReportMediaSection(
        available=True,
        format=img.format,
        width=img.width,
        height=img.height,
        color_mode=img.mode,
        file_type=file_type,
        exif_present=meta.exif_present,
        software_metadata_present=software_present,
        camera_metadata_present=camera_present,
        timestamp_metadata_present=timestamp_present,
        gps_metadata_present=gps_present,
        gps_notice="GPS metadata present" if gps_present else None,
    )

    camera_str: Optional[str] = None
    if camera_present:
        cam_val = " ".join(p for p in (fields.make, fields.model) if p)
        camera_str = f"Camera metadata: {cam_val}"

    software_str: Optional[str] = None
    if software_present and fields.software:
        software_str = f"Software metadata: {fields.software.strip()}"

    timestamp_str: Optional[str] = None
    if timestamp_present:
        ts_val = fields.datetime_original or fields.datetime
        timestamp_str = f"Timestamp metadata: {ts_val}"

    jpeg_str: Optional[str] = None
    if (img.format or "").upper() == "JPEG":
        enc = img.jpeg_encoding or "standard"
        qt_cnt = img.quantization_tables_count if img.quantization_tables_count is not None else 0
        jpeg_str = f"JPEG encoding: {enc}, quantization tables: {qt_cnt}"

    exif_summary = (
        "EXIF metadata block present in image container."
        if meta.exif_present
        else "EXIF metadata not present in image container."
    )

    forensics_sec = ReportForensicsSection(
        available=True,
        analyzed_at=analysis.analyzed_at,
        dimensions=f"{img.width}x{img.height} px",
        color_mode=img.mode,
        exif_summary=exif_summary,
        camera_information=camera_str,
        software_metadata=software_str,
        timestamps=timestamp_str,
        jpeg_properties=jpeg_str,
        findings=analysis.findings,
    )

    return media_sec, forensics_sec


def _build_manipulation_detection_section(
    case_data: MediaCaseMetadata,
) -> ReportManipulationDetectionSection:
    det = case_data.manipulation_detection
    if det is None:
        return ReportManipulationDetectionSection(
            available=False,
            detector_status="not_evaluated",
            ai_generation_status="not_available",
            ai_generation_label="AI-generated content detection: Not available",
            ai_generation_explanation="No trained AI-generation detector is currently configured.",
        )

    comp_obs = [s for s in det.signals if s.type in {"compression", "recompression"}]
    meta_obs = [s for s in det.signals if s.type == "metadata"]

    ai_gen = getattr(det, "ai_generation_detection", None) or det.ai_generation
    ai_status = ai_gen.status if ai_gen else "not_available"
    if ai_status == "available" and ai_gen and ai_gen.prediction is not None:
        det_name = ai_gen.detector.name if ai_gen.detector else "AI detector"
        det_rev = ai_gen.detector.version if ai_gen.detector else "unknown"
        det_lic = ai_gen.detector.license if ai_gen.detector else "CC BY-ND 4.0"
        sel_cls = ai_gen.selected_class or ai_gen.prediction.raw_class or "unknown"
        art_score = (
            ai_gen.artificial_score
            if ai_gen.artificial_score is not None
            else ai_gen.prediction.score
        )
        hum_score = (
            ai_gen.human_score
            if ai_gen.human_score is not None
            else (round(max(0.0, 1.0 - float(art_score)), 4) if art_score is not None else None)
        )
        ai_label = (
            f"AI-generated content detection: {ai_gen.prediction.display_label} "
            f"(model: {det_name}@{det_rev}, license: {det_lic}, "
            f"selected_class: {sel_cls}, "
            f"artificial_score: {art_score:.4f}, "
            f"human_score: {hum_score:.4f}, "
            f"score_type: {ai_gen.prediction.score_type}, "
            f"decision_rule: {ai_gen.prediction.decision_rule}, threshold: null)"
        )
        interp = getattr(ai_gen, "ai_detector_interpretation", None)
        ai_explanation = (
            f"{interp.forensic_meaning} {interp.recommended_use}"
            if interp is not None
            else ai_gen.explanation
        )
    else:
        ai_label = "AI-generated content detection: Not available"
        ai_explanation = "No trained AI-generation detector is currently configured."


    return ReportManipulationDetectionSection(
        available=True,
        detector_status=det.status,
        evaluated_at=det.evaluated_at,
        overall_assessment=det.overall_assessment,
        assessment_label=det.assessment_label,
        recompression_status=det.recompression_status,
        forensic_signals=det.signals,
        compression_observations=comp_obs,
        metadata_observations=meta_obs,
        ai_generation_status=ai_status,
        ai_generation_label=ai_label,
        ai_generation_explanation=ai_explanation,
        ai_generation_detection=ai_gen,
    )



def _build_localization_section(case_data: MediaCaseMetadata) -> ReportLocalizationSection:
    loc = case_data.localization
    case_results_dir = settings.results_dir / case_data.case_id
    heatmap_exists = (case_results_dir / "localization" / "heatmap.png").is_file()
    mask_exists = (case_results_dir / "localization" / "mask.png").is_file()

    if loc is None:
        return ReportLocalizationSection(
            available=False,
            status="not_evaluated",
            localized_anomaly=False,
            finding_label="localized forensic anomaly (not evaluated)",
            heatmap_available=heatmap_exists,
            mask_available=mask_exists,
        )

    return ReportLocalizationSection(
        available=True,
        status=loc.status,
        evaluated_at=loc.localized_at,
        localized_anomaly=loc.localized_anomaly,
        finding_label="localized forensic anomaly",
        method=loc.method,
        summary=loc.message,
        regions=loc.regions,
        heatmap_available=heatmap_exists,
        mask_available=mask_exists,
    )


def _build_comparison_section(case_data: MediaCaseMetadata) -> ReportComparisonSection:
    comp = case_data.comparison
    case_results_dir = settings.results_dir / case_data.case_id
    diff_exists = (case_results_dir / "comparison" / "difference.png").is_file()
    mask_exists = (case_results_dir / "comparison" / "difference_mask.png").is_file()

    if comp is None:
        return ReportComparisonSection(
            available=False,
            status="not_evaluated",
            finding_label="visual differences",
            difference_image_available=diff_exists,
            difference_mask_available=mask_exists,
        )

    ref_filename = (
        case_data.reference_image.original_filename
        if case_data.reference_image is not None
        else None
    )
    ref_sha256 = (
        case_data.reference_image.sha256
        if case_data.reference_image is not None
        else comp.reference_image.sha256
    )

    case_dims = f"{comp.dimensions.case_dimensions.width}x{comp.dimensions.case_dimensions.height}"
    ref_dims = f"{comp.dimensions.reference_dimensions.width}x{comp.dimensions.reference_dimensions.height}"

    return ReportComparisonSection(
        available=True,
        status=comp.status,
        compared_at=comp.compared_at,
        reference_filename=ref_filename,
        reference_sha256=ref_sha256,
        same_dimensions=comp.dimensions.same,
        case_dimensions=case_dims,
        reference_dimensions=ref_dims,
        changed_pixel_percentage=comp.pixel_difference.changed_pixel_percentage,
        changed_region_count=len(comp.changed_regions),
        changed_regions=comp.changed_regions,
        metadata_differences=comp.metadata_comparison,
        difference_image_available=diff_exists,
        difference_mask_available=mask_exists,
        finding_label="visual differences",
        assessment=comp.assessment,
        assessment_label=comp.assessment_label,
    )


def _build_evidence_and_assessment_sections(
    case_data: MediaCaseMetadata,
    hash_preserved: bool,
) -> Tuple[ReportEvidenceSection, ReportAssessmentSection]:
    ev_report: Optional[EvidenceReport] = case_data.evidence

    if ev_report is None:
        fallback_items = [
            EvidenceItem(
                id="EV-001",
                category="integrity",
                type="original_hash",
                title="Original file integrity recorded",
                description="The original uploaded file is identified by a cryptographic SHA-256 digest.",
                value=case_data.sha256,
                strength="informational",
                source="step2_upload",
            ),
            EvidenceItem(
                id="EV-002",
                category="integrity",
                type="hash_preserved",
                title="Original file hash preserved" if hash_preserved else "Original file hash mismatch",
                description="The SHA-256 digest of the stored original remains unchanged after analysis.",
                value=hash_preserved,
                strength="informational" if hash_preserved else "moderate",
                source="integrity_check",
            ),
        ]
        fallback_assessment: AllowedEvidenceAssessment = "insufficient_evidence"
        fallback_label = ASSESSMENT_LABELS[fallback_assessment]
        summary = EvidenceSummary(
            total_evidence_items=len(fallback_items),
            informational=2 if hash_preserved else 1,
            weak=0,
            moderate=0 if hash_preserved else 1,
            analysis_stages_available=["integrity_only"],
        )
        return (
            ReportEvidenceSection(
                available=False,
                generated_at=None,
                assessment=fallback_assessment,
                assessment_label=fallback_label,
                summary=summary,
                items=fallback_items,
                cross_analysis_evidence=[],
                disclaimer=(
                    "Correlated forensic evidence connects observations across analysis stages "
                    "and does not by itself prove intentional manipulation."
                ),
            ),
            ReportAssessmentSection(
                assessment=fallback_assessment,
                assessment_label=fallback_label,
            ),
        )

    cross_items: List[ReportCrossAnalysisItem] = []
    for item in ev_report.items:
        if item.type == "cross_analysis_overlap" and isinstance(item.value, dict):
            cross_items.append(
                ReportCrossAnalysisItem(
                    id=item.id,
                    localization_region=str(item.value.get("localization_region", "")),
                    comparison_region=str(item.value.get("comparison_region", "")),
                    overlap_percentage=float(item.value.get("overlap_percentage", 0.0)),
                    intersection_area=item.value.get("intersection_area"),
                    description=item.description,
                    strength=item.strength,
                )
            )

    ev_sec = ReportEvidenceSection(
        available=True,
        generated_at=ev_report.generated_at,
        assessment=ev_report.assessment,
        assessment_label=ev_report.assessment_label,
        summary=ev_report.summary,
        items=ev_report.items,
        cross_analysis_evidence=cross_items,
        disclaimer=ev_report.disclaimer,
    )

    assessment_sec = ReportAssessmentSection(
        assessment=ev_report.assessment,
        assessment_label=ev_report.assessment_label,
    )

    return ev_sec, assessment_sec


def _build_history_section(
    case_data: MediaCaseMetadata,
    hash_preserved: bool,
) -> ReportHistorySection:
    hist: Optional[EditHistoryReport] = case_data.history

    if hist is None:
        fallback_event = ReconstructedHistoryEvent(
            id="EH-001",
            sequence=1,
            event_type="original_capture_or_creation",
            title="Image File Recorded in Case",
            description=(
                f"Uploaded file '{case_data.original_filename}' ({case_data.file_size} bytes) "
                f"stored and hashed with SHA-256."
            ),
            source="step2_upload",
            evidence_ids=["EV-001", "EV-002"],
            temporal_certainty="exact",
            confidence="high",
            status="observed",
            reasoning="The uploaded file bytes and SHA-256 digest were recorded when the case was created.",
            limitations=[
                "Does not prove when the original photograph or graphic was first captured or created."
            ],
        )
        fallback_fact = KnownFactItem(
            fact=(
                f"Uploaded image '{case_data.original_filename}' ({case_data.file_size} bytes) "
                f"was recorded in the case with SHA-256 digest {case_data.sha256}."
            ),
            source="step2_upload",
            evidence_ids=["EV-001", "EV-002"],
        )
        return ReportHistorySection(
            historical_accuracy="not_determined",
            events=[fallback_event],
            known_facts=[fallback_fact],
            inferred_events=[],
            unknowns=[
                "Exact chronological order of prior edits cannot be determined.",
                "Original camera capture time is not verifiable from file data alone.",
                "Specific editing operations cannot be confirmed from a single image.",
            ],
            limitations=[
                "Reconstructed edit history represents a possible sequence of processing events based on available forensic evidence.",
                "Evidence ordering reflects analytical sequence and plausible processing steps, not confirmed chronology.",
            ],
        )

    return ReportHistorySection(
        historical_accuracy=hist.historical_accuracy,
        events=hist.events,
        known_facts=hist.known_facts,
        inferred_events=hist.inferred_events,
        unknowns=hist.unknowns,
        limitations=hist.limitations,
    )


def _build_report_limitations() -> List[str]:
    return [
        "Forensic signals do not automatically prove intentional manipulation.",
        "Metadata is supporting evidence only.",
        "Compression characteristics do not establish an exact editing application.",
        "Pixel differences do not by themselves prove tampering.",
        "Localization identifies anomalies, not confirmed manipulation.",
        "Cross-analysis overlap indicates consistency between methods.",
        "AI-generation detection is unavailable until a trained detector is configured.",
        "Exact editing tools and operations cannot be established from the current evidence.",
        "User intent cannot be established.",
        "Unsaved intermediate edits cannot be reconstructed.",
        "The Step 8 timeline is an evidence-ordered hypothesis.",
        "Exact historical chronology cannot be guaranteed.",
        "Reference comparison does not independently establish which file came first.",
    ]


def generate_forensic_report(case_id: str) -> ForensicReport:
    """
    Generate and persist a structured Step 9 ForensicReport JSON at
    storage/reports/<case_id>/report.json without altering the original case JSON or media files.
    """
    case_data = _load_case_readonly(case_id)
    hash_preserved = _verify_hash_preserved(case_data)

    report_id = _build_deterministic_report_id(case_data)
    case_sec = _build_case_section(case_data)
    integrity_sec = _build_integrity_section(case_data, hash_preserved)
    media_sec, forensics_sec = _build_media_and_forensics_sections(case_data)
    detection_sec = _build_manipulation_detection_section(case_data)
    localization_sec = _build_localization_section(case_data)
    comparison_sec = _build_comparison_section(case_data)
    evidence_sec, assessment_sec = _build_evidence_and_assessment_sections(
        case_data, hash_preserved
    )
    multi_signal_sec = (
        case_data.forensic_decision
        if case_data.forensic_decision is not None
        else build_forensic_decision(case_data, include_integrity=True)
    )
    from app.services.evidence.reproducibility_service import (
        append_audit_event,
        create_forensic_snapshot,
    )

    case_copy = case_data.model_copy(deep=True)
    repro_snapshot = create_forensic_snapshot(case_copy, record_event=True)
    append_audit_event(
        case_data=case_copy,
        event_type="report_generated",
        summary=f"Generated forensic JSON report '{report_id}' linked to snapshot '{repro_snapshot.snapshot_id}'.",
        snapshot_id=repro_snapshot.snapshot_id,
        report_id=report_id,
        details={
            "report_id": report_id,
            "snapshot_id": repro_snapshot.snapshot_id,
            "export_format": "json",
            "relative_path": f"storage/reports/{case_id}/report.json",
        },
    )
    history_sec = _build_history_section(case_data, hash_preserved)
    limitations = _build_report_limitations()

    report = ForensicReport(
        report_id=report_id,
        case_id=case_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        report_version="1.0",
        application_version=MEDIATRACE_APPLICATION_VERSION,
        case=case_sec,
        integrity=integrity_sec,
        media=media_sec,
        forensics=forensics_sec,
        manipulation_detection=detection_sec,
        ai_generation_detection=detection_sec.ai_generation_detection,
        localization=localization_sec,
        comparison=comparison_sec,
        evidence=evidence_sec,
        multi_signal_interpretation=multi_signal_sec,
        reproducibility_audit_trail=repro_snapshot,
        audit_events=list(case_copy.audit_events or []),
        history=history_sec,
        assessment=assessment_sec,
        limitations=limitations,
    )

    case_report_dir = settings.reports_dir / case_id
    case_report_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_report_dir / "report.json"
    temp_path = report_path.with_suffix(".json.tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as file_out:
            json.dump(report.model_dump(), file_out, indent=2)
        os.replace(temp_path, report_path)
    except Exception as exc:
        if temp_path.is_file():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist forensic report JSON.",
        ) from exc

    return report


def get_forensic_report(case_id: str) -> ForensicReport:
    """Retrieve the persisted ForensicReport JSON from storage/reports/<case_id>/report.json."""
    _load_case_readonly(case_id)
    report_path = settings.reports_dir / case_id / "report.json"
    if not report_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Forensic report for case '{case_id}' has not been generated yet.",
        )

    try:
        with open(report_path, "r", encoding="utf-8") as file_in:
            raw_report = json.load(file_in)
        return ForensicReport(**raw_report)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read forensic report for case '{case_id}'.",
        ) from exc
