from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Tuple

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.analysis import SuspiciousRegion
from app.schemas.comparison import ChangedRegion
from app.schemas.evidence import (
    AllowedEvidenceAssessment,
    AllowedEvidenceCategory,
    AllowedEvidenceStrength,
    EvidenceItem,
    EvidenceReport,
    EvidenceSummary,
)
from app.schemas.media import MediaCaseMetadata
from app.services.media_service import CASE_ID_PATTERN

ASSESSMENT_LABELS = {
    "no_significant_evidence": "No significant manipulation evidence identified across completed stages",
    "multiple_forensic_signals": "Multiple independent forensic signals identified",
    "localized_visual_differences": "Localized visual differences observed between compared images",
    "insufficient_evidence": "Insufficient forensic evidence available",
}

# Minimum percentage of spatial intersection (relative to the smaller region area)
# required to report a cross-analysis spatial overlap.
MIN_SPATIAL_OVERLAP_PCT = 10.0


class _EvidenceCollector:
    """Helper to build sequentially numbered EvidenceItem objects (EV-001, EV-002, ...)."""

    def __init__(self) -> None:
        self._items: List[EvidenceItem] = []

    def add(
        self,
        category: AllowedEvidenceCategory,
        ev_type: str,
        title: str,
        description: str,
        value: Any,
        strength: AllowedEvidenceStrength,
        source: str,
    ) -> None:
        ev_id = f"EV-{len(self._items) + 1:03d}"
        self._items.append(
            EvidenceItem(
                id=ev_id,
                category=category,
                type=ev_type,
                title=title,
                description=description,
                value=value,
                strength=strength,
                source=source,
            )
        )

    @property
    def items(self) -> List[EvidenceItem]:
        return self._items


def _load_case(case_id: str) -> Tuple[MediaCaseMetadata, Path]:
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
        return MediaCaseMetadata(**raw_data), result_path
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metadata for case '{case_id}'.",
        ) from exc


def _collect_integrity_evidence(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """Extract Step 2 chain-of-custody SHA-256 and verify current disk integrity."""
    collector.add(
        category="integrity",
        ev_type="original_hash",
        title="Original file integrity recorded",
        description="The original uploaded file is identified by a cryptographic SHA-256 digest.",
        value=case_data.sha256,
        strength="informational",
        source="step2_upload",
    )

    upload_path = settings.uploads_dir / case_data.stored_filename
    hash_preserved = False
    if upload_path.is_file():
        current_sha256 = hashlib.sha256(upload_path.read_bytes()).hexdigest()
        hash_preserved = current_sha256 == case_data.sha256

    collector.add(
        category="integrity",
        ev_type="hash_preserved",
        title="Original file hash preserved" if hash_preserved else "Original file hash mismatch",
        description=(
            "The SHA-256 digest of the stored original remains unchanged after analysis."
            if hash_preserved
            else "The stored original file SHA-256 no longer matches the initial upload record."
        ),
        value=hash_preserved,
        strength="informational" if hash_preserved else "moderate",
        source="integrity_check",
    )


def _collect_step3_basic_forensics(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """Extract objective evidence from Step 3 Basic Image Forensics (never exposing GPS coordinates)."""
    analysis = case_data.analysis
    if analysis is None:
        return

    img_info = analysis.image
    meta = analysis.metadata

    collector.add(
        category="compression" if (img_info.format or "").upper() == "JPEG" else "metadata",
        ev_type="image_container_properties",
        title="Image container and geometry verified",
        description="Container format, pixel dimensions, and color channel structure were extracted.",
        value=f"{img_info.format or 'UNKNOWN'} {img_info.width}x{img_info.height} px, mode {img_info.mode}",
        strength="informational",
        source="step3_basic_forensics",
    )

    if meta.exif_present:
        collector.add(
            category="metadata",
            ev_type="exif_present",
            title="EXIF metadata present",
            description="The image container includes an EXIF metadata block.",
            value=True,
            strength="informational",
            source="step3_basic_forensics",
        )
        if meta.fields.software:
            collector.add(
                category="metadata",
                ev_type="software_metadata",
                title="Editing software metadata present",
                description="The image contains software metadata identifying an image-editing or processing application.",
                value=meta.fields.software,
                strength="weak",
                source="step3_basic_forensics",
            )
        if meta.fields.make or meta.fields.model:
            cam_str = " ".join(p for p in (meta.fields.make, meta.fields.model) if p)
            collector.add(
                category="metadata",
                ev_type="camera_metadata",
                title="Camera metadata present",
                description="The image contains capture hardware Make/Model EXIF metadata.",
                value=cam_str,
                strength="informational",
                source="step3_basic_forensics",
            )
        if meta.fields.datetime_original or meta.fields.datetime:
            ts_str = meta.fields.datetime_original or meta.fields.datetime
            collector.add(
                category="metadata",
                ev_type="timestamp_metadata",
                title="Timestamp metadata present",
                description="The image contains EXIF capture or file modification timestamp metadata.",
                value=ts_str,
                strength="informational",
                source="step3_basic_forensics",
            )
        if meta.gps_present:
            # Strictly report presence only; never expose raw coordinates
            collector.add(
                category="metadata",
                ev_type="gps_metadata",
                title="GPS metadata present",
                description="The image contains GPS metadata.",
                value="GPS metadata present",
                strength="informational",
                source="step3_basic_forensics",
            )
    else:
        collector.add(
            category="metadata",
            ev_type="exif_absent",
            title="EXIF metadata not present",
            description="No EXIF metadata block was found in the image container.",
            value="EXIF not available",
            strength="informational",
            source="step3_basic_forensics",
        )


def _collect_step4_manipulation_detection(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """
    Extract Step 4 manipulation detection signals.
    Does not create false AI-generation evidence when ai_generation.status == 'not_available'.
    """
    detection = case_data.manipulation_detection
    if detection is None:
        return

    meaningful_signals_added = 0
    for sig in detection.signals:
        if sig.strength in {"weak", "moderate"}:
            meaningful_signals_added += 1
            collector.add(
                category="manipulation",
                ev_type=sig.name,
                title=f"Detection signal: {sig.name.replace('_', ' ')}",
                description=sig.description,
                value=sig.value,
                strength=sig.strength,
                source="step4_manipulation_detection",
            )

    if meaningful_signals_added == 0:
        collector.add(
            category="manipulation",
            ev_type="manipulation_baseline_evaluation",
            title="No strong manipulation indicators detected in Step 4",
            description="Classical consistency, compression, and header heuristics did not flag weak or moderate anomalies.",
            value=detection.overall_assessment,
            strength="informational",
            source="step4_manipulation_detection",
        )

    ai_gen = getattr(detection, "ai_generation_detection", None) or detection.ai_generation
    if ai_gen and ai_gen.status == "available" and ai_gen.prediction is not None and ai_gen.detector is not None:
        threshold_info = (
            f", threshold: {ai_gen.prediction.threshold}"
            if ai_gen.prediction.threshold is not None
            else ", threshold: null"
        )
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
        interp = getattr(ai_gen, "ai_detector_interpretation", None)
        meaning_text = interp.forensic_meaning if interp else ai_gen.explanation
        collector.add(
            category="manipulation",
            ev_type="ai_generation_detector_signal",
            title="AI-generation detector signal (model-derived, uncalibrated)",
            description=meaning_text,
            value=(
                f"{ai_gen.prediction.display_label} "
                f"(model: {ai_gen.detector.name}@{ai_gen.detector.version}, "
                f"license: {ai_gen.detector.license}, "
                f"selected_class: {sel_cls}, "
                f"uncalibrated model score: {ai_gen.prediction.score:.4f}, "
                f"artificial_score: {art_score:.4f}, "
                f"human_score: {hum_score:.4f}, "
                f"score_type: {ai_gen.prediction.score_type}, "
                f"decision_rule: {ai_gen.prediction.decision_rule}{threshold_info}; "
                "not calibrated as a forensic probability and does not establish provenance or authenticity)"
            ),
            strength="informational",
            source="step14_1_ai_detection",
        )




def _collect_step5_localization(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """Extract Step 5 spatial localization evidence without claiming confirmed manipulation."""
    loc = case_data.localization
    if loc is None:
        return

    if loc.localized_anomaly and len(loc.regions) > 0:
        collector.add(
            category="localization",
            ev_type="localized_anomaly",
            title="Localized forensic anomaly detected",
            description="The baseline localization method identified spatially concentrated anomaly regions.",
            value={
                "region_count": len(loc.regions),
                "regions": [r.model_dump() for r in loc.regions],
            },
            strength="moderate",
            source="step5_localization",
        )
    else:
        collector.add(
            category="localization",
            ev_type="no_localized_anomaly",
            title="No localized anomaly detected",
            description="The baseline localization pass did not detect candidate anomaly regions above the threshold.",
            value={"region_count": 0},
            strength="informational",
            source="step5_localization",
        )


def _collect_step6_comparison(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """Extract Step 6 Original vs Edited comparison evidence."""
    comp = case_data.comparison
    if comp is None:
        return

    if not comp.dimensions.same:
        collector.add(
            category="comparison",
            ev_type="different_dimensions",
            title="Images have different dimensions",
            description="The case image and supplied reference image have different pixel dimensions.",
            value={
                "case_dimensions": comp.dimensions.case_dimensions.model_dump(),
                "reference_dimensions": comp.dimensions.reference_dimensions.model_dump(),
            },
            strength="weak",
            source="step6_comparison",
        )

    if len(comp.changed_regions) > 0:
        collector.add(
            category="comparison",
            ev_type="changed_regions",
            title="Localized visual differences detected",
            description="The supplied images contain spatially localized pixel differences.",
            value={
                "region_count": len(comp.changed_regions),
                "changed_pixel_percentage": comp.pixel_difference.changed_pixel_percentage,
            },
            strength="moderate",
            source="step6_comparison",
        )
    elif comp.pixel_difference.changed_pixel_percentage > 0.05:
        collector.add(
            category="comparison",
            ev_type="pixel_differences",
            title="Images contain changed pixels",
            description="Pixel-level differences were observed between the case image and reference image.",
            value={
                "changed_pixel_percentage": comp.pixel_difference.changed_pixel_percentage,
                "mean_absolute_difference": comp.pixel_difference.mean_absolute_difference,
            },
            strength="weak",
            source="step6_comparison",
        )
    else:
        collector.add(
            category="comparison",
            ev_type="no_significant_difference",
            title="No significant pixel differences detected",
            description="The case image and reference image have matching pixel values within the comparison threshold.",
            value={"changed_pixel_percentage": comp.pixel_difference.changed_pixel_percentage},
            strength="informational",
            source="step6_comparison",
        )

    meta_c = comp.metadata_comparison
    if any(
        [
            meta_c.format_changed,
            meta_c.exif_presence_changed,
            meta_c.software_changed,
            meta_c.timestamp_changed,
        ]
    ):
        collector.add(
            category="comparison",
            ev_type="metadata_difference",
            title="Metadata differs between versions",
            description="Container or EXIF metadata properties differ between the case image and reference image.",
            value={
                "format_changed": meta_c.format_changed,
                "exif_presence_changed": meta_c.exif_presence_changed,
                "software_changed": meta_c.software_changed,
                "timestamp_changed": meta_c.timestamp_changed,
            },
            strength="weak",
            source="step6_comparison",
        )


def _compute_bbox_overlap_percentage(
    loc_reg: SuspiciousRegion,
    comp_reg: ChangedRegion,
) -> Tuple[float, int]:
    """
    Calculate bounding-box intersection area and overlap percentage relative to the smaller box:
      overlap_pct = (intersection_area / min(loc_area, comp_area)) * 100.0
    """
    ix0 = max(loc_reg.x, comp_reg.x)
    iy0 = max(loc_reg.y, comp_reg.y)
    ix1 = min(loc_reg.x + loc_reg.width, comp_reg.x + comp_reg.width)
    iy1 = min(loc_reg.y + loc_reg.height, comp_reg.y + comp_reg.height)

    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0

    intersection_area = (ix1 - ix0) * (iy1 - iy0)
    min_area = max(1, min(loc_reg.area, comp_reg.area))
    overlap_pct = round((intersection_area / min_area) * 100.0, 1)
    return overlap_pct, intersection_area


def _collect_cross_analysis_correlations(
    collector: _EvidenceCollector,
    case_data: MediaCaseMetadata,
) -> None:
    """
    Section 8: Cross-Evidence Correlation.
    Identify when independent stages (e.g. Step 5 localization and Step 6 comparison)
    highlight spatially overlapping regions.
    """
    loc = case_data.localization
    comp = case_data.comparison

    if loc and loc.regions and comp and comp.changed_regions:
        for loc_reg in loc.regions:
            for comp_reg in comp.changed_regions:
                overlap_pct, inter_area = _compute_bbox_overlap_percentage(loc_reg, comp_reg)
                if overlap_pct >= MIN_SPATIAL_OVERLAP_PCT:
                    collector.add(
                        category="comparison",
                        ev_type="cross_analysis_overlap",
                        title="Independent analyses identify an overlapping region",
                        description="The localization and reference-image comparison results identify overlapping spatial regions.",
                        value={
                            "localization_region": loc_reg.id,
                            "comparison_region": comp_reg.id,
                            "overlap_percentage": overlap_pct,
                            "intersection_area": inter_area,
                        },
                        strength="moderate",
                        source="cross_analysis",
                    )

    det = case_data.manipulation_detection
    if det is not None:
        ai_gen = getattr(det, "ai_generation_detection", None) or det.ai_generation
        classical_assess = (
            det.classical_forensics.overall_assessment
            if det.classical_forensics is not None
            else det.overall_assessment
        )
        if ai_gen and ai_gen.status == "available" and ai_gen.prediction is not None:
            sel_cls = (ai_gen.selected_class or ai_gen.prediction.raw_class or "").lower()
            if sel_cls == "artificial" and classical_assess != "potential_manipulation":
                collector.add(
                    category="manipulation",
                    ev_type="ai_detector_cross_analysis_disagreement",
                    title="Cross-analysis observation: AI detector artificial vs. classical forensics",
                    description=(
                        "AI detector selected artificial while other available forensic signals "
                        "do not independently establish manipulation."
                    ),
                    value={
                        "ai_selected_class": "artificial",
                        "classical_assessment": classical_assess,
                        "interpretation": (
                            "AI detector selected artificial while other available forensic signals "
                            "do not independently establish manipulation."
                        ),
                    },
                    strength="informational",
                    source="cross_analysis",
                )
            elif sel_cls == "human" and (
                (loc is not None and loc.localized_anomaly)
                or classical_assess == "potential_manipulation"
            ):
                collector.add(
                    category="localization" if (loc is not None and loc.localized_anomaly) else "manipulation",
                    ev_type="ai_detector_cross_analysis_disagreement",
                    title="Cross-analysis observation: AI detector human vs. localized/classical anomaly",
                    description=(
                        "The AI detector selected human; separate localization analysis identified "
                        "candidate anomalous regions. These are distinct signals and neither result "
                        "establishes authenticity or manipulation on its own."
                    ),
                    value={
                        "ai_selected_class": "human",
                        "localized_anomaly": bool(loc and loc.localized_anomaly),
                        "classical_assessment": classical_assess,
                        "interpretation": (
                            "The AI detector selected human; separate localization analysis identified "
                            "candidate anomalous regions. These are distinct signals and neither result "
                            "establishes authenticity or manipulation on its own."
                        ),
                    },
                    strength="informational",
                    source="cross_analysis",
                )



def _determine_forensic_assessment(
    items: List[EvidenceItem],
    stages_available: List[str],
    case_data: MediaCaseMetadata,
) -> AllowedEvidenceAssessment:
    """
    Section 10: Conservative textual forensic assessment.
    Allowed values:
      - 'no_significant_evidence'
      - 'multiple_forensic_signals'
      - 'localized_visual_differences'
      - 'insufficient_evidence'
    """
    if not stages_available:
        return "insufficient_evidence"

    non_info_items = [item for item in items if item.strength in {"weak", "moderate"}]
    has_cross_overlap = any(item.type == "cross_analysis_overlap" for item in items)
    has_comparison_diff = bool(
        case_data.comparison
        and (
            len(case_data.comparison.changed_regions) > 0
            or case_data.comparison.pixel_difference.changed_pixel_percentage > 0.05
        )
    )

    # Distinct stages/sources that produced weak or moderate signals
    non_info_sources = {item.source for item in non_info_items}

    if has_cross_overlap or len(non_info_sources) >= 2 or len(non_info_items) >= 2:
        return "multiple_forensic_signals"

    if has_comparison_diff:
        return "localized_visual_differences"

    if len(non_info_items) == 1:
        return "multiple_forensic_signals"

    if "manipulation_detection" in stages_available or "comparison" in stages_available or "localization" in stages_available:
        return "no_significant_evidence"

    return "insufficient_evidence"


def generate_evidence_for_case(case_id: str) -> MediaCaseMetadata:
    """
    Correlate all existing analysis stages for a case into a structured EvidenceReport
    and persist it inside storage/results/<case_id>.json without modifying previous results.
    """
    case_data, result_path = _load_case(case_id)

    stages_available: List[str] = []
    if case_data.analysis is not None:
        stages_available.append("basic_forensics")
    if case_data.manipulation_detection is not None:
        stages_available.append("manipulation_detection")
    if case_data.localization is not None:
        stages_available.append("localization")
    if case_data.comparison is not None:
        stages_available.append("comparison")

    collector = _EvidenceCollector()
    _collect_integrity_evidence(collector, case_data)
    _collect_step3_basic_forensics(collector, case_data)
    _collect_step4_manipulation_detection(collector, case_data)
    _collect_step5_localization(collector, case_data)
    _collect_step6_comparison(collector, case_data)
    _collect_cross_analysis_correlations(collector, case_data)

    items = collector.items
    info_count = sum(1 for item in items if item.strength == "informational")
    weak_count = sum(1 for item in items if item.strength == "weak")
    mod_count = sum(1 for item in items if item.strength == "moderate")

    assessment = _determine_forensic_assessment(items, stages_available, case_data)

    summary = EvidenceSummary(
        total_evidence_items=len(items),
        informational=info_count,
        weak=weak_count,
        moderate=mod_count,
        analysis_stages_available=stages_available,
    )

    evidence_report = EvidenceReport(
        case_id=case_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        assessment=assessment,
        assessment_label=ASSESSMENT_LABELS[assessment],
        summary=summary,
        items=items,
    )

    updated_case = case_data.model_copy(
        update={
            "evidence": evidence_report,
        }
    )

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    record_analysis_execution(
        case_data=updated_case,
        service_name="evidence_service",
        status="completed",
        input_sha256=updated_case.sha256,
        output_summary=(
            f"Evidence assessment={assessment}; items={len(items)} "
            f"(informational={info_count}, weak={weak_count}, moderate={mod_count})."
        ),
        evidence_ids=[ev.id for ev in items],
        completed_at=evidence_report.generated_at,
    )
    create_forensic_snapshot(updated_case, record_event=True)

    temp_path = result_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as tmp_out:
            json.dump(updated_case.model_dump(), tmp_out, indent=2)
        os.replace(temp_path, result_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist evidence report.",
        ) from exc

    return updated_case


def get_evidence_for_case(case_id: str) -> EvidenceReport:
    """Return the stored EvidenceReport for a case, or HTTP 404 if not yet generated."""
    case_data, _ = _load_case(case_id)
    if case_data.evidence is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Evidence report has not been generated for case '{case_id}' yet.",
        )
    return case_data.evidence
