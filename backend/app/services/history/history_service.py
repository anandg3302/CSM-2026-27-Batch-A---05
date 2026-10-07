from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.evidence import EvidenceItem
from app.schemas.history import (
    AllowedHistoricalAccuracy,
    AllowedHistoryEventStatus,
    AllowedHistoryEventType,
    AllowedQualitativeConfidence,
    AllowedTemporalCertainty,
    EditHistoryReport,
    HistoryIntegrityInfo,
    InferredEventItem,
    KnownFactItem,
    ReconstructedHistoryEvent,
)
from app.schemas.media import MediaCaseMetadata
from app.services.evidence.evidence_service import generate_evidence_for_case
from app.services.media_service import CASE_ID_PATTERN


class _HistoryEventBuilder:
    """Helper to build deterministically numbered ReconstructedHistoryEvent items (EH-001, EH-002, ...)."""

    def __init__(self) -> None:
        self._events: List[ReconstructedHistoryEvent] = []

    def add(
        self,
        *,
        event_type: AllowedHistoryEventType,
        title: str,
        description: str,
        source: str,
        evidence_ids: List[str],
        temporal_certainty: AllowedTemporalCertainty,
        confidence: AllowedQualitativeConfidence,
        status: AllowedHistoryEventStatus,
        reasoning: str,
        limitations: List[str],
    ) -> None:
        seq = len(self._events) + 1
        event_id = f"EH-{seq:03d}"
        self._events.append(
            ReconstructedHistoryEvent(
                id=event_id,
                sequence=seq,
                event_type=event_type,
                title=title,
                description=description,
                source=source,
                evidence_ids=list(dict.fromkeys(evidence_ids)),
                temporal_certainty=temporal_certainty,
                confidence=confidence,
                status=status,
                reasoning=reasoning,
                limitations=limitations,
            )
        )

    @property
    def events(self) -> List[ReconstructedHistoryEvent]:
        return self._events


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


def _expected_evidence_stages(case_data: MediaCaseMetadata) -> List[str]:
    stages: List[str] = []
    if case_data.analysis is not None:
        stages.append("basic_forensics")
    if case_data.manipulation_detection is not None:
        stages.append("manipulation_detection")
    if case_data.localization is not None:
        stages.append("localization")
    if case_data.comparison is not None:
        stages.append("comparison")
    if not stages:
        stages.append("integrity_only")
    return stages


def _ensure_evidence_synchronized(case_id: str, case_data: MediaCaseMetadata) -> MediaCaseMetadata:
    """Ensure Step 7 EvidenceReport is present and covers all currently completed stages."""
    expected_stages = _expected_evidence_stages(case_data)
    if (
        case_data.evidence is None
        or case_data.evidence.summary.analysis_stages_available != expected_stages
    ):
        return generate_evidence_for_case(case_id)
    return case_data


def _find_evidence_ids(
    items: Iterable[EvidenceItem],
    *,
    ev_types: Optional[Iterable[str]] = None,
    categories: Optional[Iterable[str]] = None,
    sources: Optional[Iterable[str]] = None,
    fallback_ids: Optional[List[str]] = None,
) -> List[str]:
    type_set = set(ev_types) if ev_types is not None else None
    cat_set = set(categories) if categories is not None else None
    src_set = set(sources) if sources is not None else None

    matched: List[str] = []
    for item in items:
        if type_set is not None and item.type in type_set:
            matched.append(item.id)
            continue
        if type_set is None and cat_set is not None and item.category in cat_set:
            if src_set is None or item.source in src_set:
                matched.append(item.id)
                continue
        if type_set is None and cat_set is None and src_set is not None and item.source in src_set:
            matched.append(item.id)

    if matched:
        return list(dict.fromkeys(matched))
    return list(fallback_ids) if fallback_ids else []


def _verify_file_integrity(case_data: MediaCaseMetadata) -> HistoryIntegrityInfo:
    image_path = settings.uploads_dir / case_data.stored_filename
    hash_preserved = False
    if image_path.is_file():
        hasher = hashlib.sha256()
        with open(image_path, "rb") as file_in:
            while chunk := file_in.read(65536):
                hasher.update(chunk)
        hash_preserved = hasher.hexdigest().lower() == case_data.sha256.lower()

    return HistoryIntegrityInfo(
        original_sha256=case_data.sha256,
        hash_preserved=hash_preserved,
    )


def _has_extension_format_mismatch(case_data: MediaCaseMetadata) -> bool:
    if case_data.analysis is not None:
        for finding in case_data.analysis.findings:
            if "mismatch" in finding.finding.lower() and "extension" in finding.finding.lower():
                return True
    if case_data.manipulation_detection is not None:
        for sig in case_data.manipulation_detection.signals:
            if sig.name == "extension_format_mismatch" and sig.strength in {"weak", "moderate"}:
                return True
    return False


def _build_known_facts(
    case_data: MediaCaseMetadata,
    evidence_items: List[EvidenceItem],
) -> List[KnownFactItem]:
    facts: List[KnownFactItem] = []

    integrity_ids = _find_evidence_ids(
        evidence_items,
        ev_types=["original_hash", "hash_preserved"],
    )
    facts.append(
        KnownFactItem(
            fact=(
                f"Uploaded image '{case_data.original_filename}' ({case_data.file_size} bytes) "
                f"was recorded in the case with SHA-256 digest {case_data.sha256}."
            ),
            source="step2_upload",
            evidence_ids=integrity_ids,
        )
    )

    if case_data.analysis is not None:
        analysis = case_data.analysis
        img_info = analysis.image
        meta = analysis.metadata
        struct_ids = _find_evidence_ids(
            evidence_items,
            ev_types=["image_container_properties"],
            fallback_ids=integrity_ids,
        )
        facts.append(
            KnownFactItem(
                fact=(
                    f"Image dimensions are {img_info.width}x{img_info.height} in "
                    f"{img_info.format or 'UNKNOWN'} format ({img_info.mode} color mode)."
                ),
                source="step3_basic_analysis",
                evidence_ids=struct_ids,
            )
        )

        mismatch = _has_extension_format_mismatch(case_data)
        fmt_ids = _find_evidence_ids(
            evidence_items,
            ev_types=["extension_format_mismatch", "image_container_properties"],
            fallback_ids=struct_ids,
        )
        facts.append(
            KnownFactItem(
                fact=(
                    f"File extension '{case_data.extension}' "
                    f"{'does not match' if mismatch else 'matches'} "
                    f"detected container format '{img_info.format or 'UNKNOWN'}'."
                ),
                source="step3_basic_analysis",
                evidence_ids=fmt_ids,
            )
        )

        if meta.exif_present:
            fields = meta.fields
            if fields.software and fields.software.strip():
                sw_ids = _find_evidence_ids(
                    evidence_items,
                    ev_types=["software_metadata", "software_metadata_present"],
                    fallback_ids=struct_ids,
                )
                facts.append(
                    KnownFactItem(
                        fact=f"Embedded EXIF Software field contains '{fields.software.strip()}'.",
                        source="step3_basic_analysis",
                        evidence_ids=sw_ids,
                    )
                )
            if fields.make or fields.model:
                cam_ids = _find_evidence_ids(
                    evidence_items,
                    ev_types=["camera_metadata"],
                    fallback_ids=struct_ids,
                )
                facts.append(
                    KnownFactItem(
                        fact=(
                            f"Embedded EXIF camera header fields record make='{fields.make or 'N/A'}', "
                            f"model='{fields.model or 'N/A'}'."
                        ),
                        source="step3_basic_analysis",
                        evidence_ids=cam_ids,
                    )
                )
            if fields.datetime_original or fields.datetime:
                ts_ids = _find_evidence_ids(
                    evidence_items,
                    ev_types=["timestamp_metadata"],
                    fallback_ids=struct_ids,
                )
                ts_val = fields.datetime_original or fields.datetime
                facts.append(
                    KnownFactItem(
                        fact=f"Embedded EXIF timestamp field records '{ts_val}'.",
                        source="step3_basic_analysis",
                        evidence_ids=ts_ids,
                    )
                )
            if meta.gps_present:
                gps_ids = _find_evidence_ids(
                    evidence_items,
                    ev_types=["gps_metadata"],
                    fallback_ids=struct_ids,
                )
                facts.append(
                    KnownFactItem(
                        fact="Embedded EXIF GPS metadata presence was detected (coordinates withheld for privacy).",
                        source="step3_basic_analysis",
                        evidence_ids=gps_ids,
                    )
                )
        else:
            no_exif_ids = _find_evidence_ids(
                evidence_items,
                ev_types=["exif_absent"],
                fallback_ids=struct_ids,
            )
            facts.append(
                KnownFactItem(
                    fact="No EXIF metadata header is present in the uploaded image file.",
                    source="step3_basic_analysis",
                    evidence_ids=no_exif_ids,
                )
            )

    if case_data.comparison is not None:
        comp = case_data.comparison
        comp_ids = _find_evidence_ids(
            evidence_items,
            categories=["comparison"],
            sources=["step6_comparison"],
            fallback_ids=integrity_ids,
        )
        facts.append(
            KnownFactItem(
                fact=(
                    f"Comparison against reference image measured "
                    f"{comp.pixel_difference.changed_pixel_percentage:.2f}% changed pixels and "
                    f"{len(comp.changed_regions)} changed region(s)."
                ),
                source="step6_comparison",
                evidence_ids=comp_ids,
            )
        )

    return facts


def _build_events_and_inferences(
    case_data: MediaCaseMetadata,
    evidence_items: List[EvidenceItem],
) -> Tuple[List[ReconstructedHistoryEvent], List[InferredEventItem], AllowedHistoricalAccuracy]:
    builder = _HistoryEventBuilder()
    integrity_ids = _find_evidence_ids(
        evidence_items,
        ev_types=["original_hash", "hash_preserved"],
    )

    # 1. Known case ingestion (always Event 1)
    builder.add(
        event_type="original_capture_or_creation",
        title="Image File Recorded in Case",
        description=(
            f"Uploaded file '{case_data.original_filename}' ({case_data.file_size} bytes) "
            f"stored and hashed with SHA-256."
        ),
        source="step2_upload",
        evidence_ids=integrity_ids,
        temporal_certainty="exact",
        confidence="high",
        status="observed",
        reasoning="The uploaded file bytes and SHA-256 digest were recorded when the case was created.",
        limitations=[
            "Does not prove when the original photograph or graphic was first captured or created."
        ],
    )

    # 2. Metadata timestamp observations (if EXIF timestamps exist)
    if case_data.analysis is not None and case_data.analysis.metadata.exif_present:
        fields = case_data.analysis.metadata.fields
        if fields.datetime_original or fields.datetime:
            ts_parts = []
            if fields.datetime_original:
                ts_parts.append(f"DateTimeOriginal='{fields.datetime_original}'")
            if fields.datetime:
                ts_parts.append(f"DateTime='{fields.datetime}'")
            ts_ids = _find_evidence_ids(
                evidence_items,
                ev_types=["timestamp_metadata", "camera_metadata"],
                fallback_ids=integrity_ids,
            )
            builder.add(
                event_type="metadata_processing",
                title="EXIF Timestamp Metadata Recorded in File Header",
                description=f"Embedded EXIF header contains timestamp field(s): {', '.join(ts_parts)}.",
                source="step3_basic_analysis",
                evidence_ids=ts_ids,
                temporal_certainty="approximate",
                confidence="moderate",
                status="observed",
                reasoning="EXIF timestamp fields were extracted from the submitted image header during Step 3 inspection.",
                limitations=[
                    "EXIF timestamps reflect unverified header metadata and can be modified, stripped, or rewritten by software."
                ],
            )

    # 3. Compression observations / possible recompression (Rule 2: JPEG quantization tables, recompression indicators, or format/container mismatch)
    has_jpeg_qtables = bool(
        (
            case_data.analysis is not None
            and (case_data.analysis.image.format or "").upper() == "JPEG"
            and (case_data.analysis.image.quantization_tables_count or 0) > 0
        )
        or (
            case_data.manipulation_detection is not None
            and any(
                sig.name == "jpeg_quantization_tables"
                for sig in case_data.manipulation_detection.signals
            )
        )
    )
    has_container_mismatch = _has_extension_format_mismatch(case_data)
    recompression_ev_ids = _find_evidence_ids(
        evidence_items,
        ev_types=[
            "container_extension_mismatch",
            "extension_format_mismatch",
            "jpeg_recompression_indicator",
            "jpeg_high_compression_ratio",
        ],
    )
    if has_jpeg_qtables or has_container_mismatch:
        comp_cat_ids = _find_evidence_ids(
            evidence_items,
            categories=["compression"],
            ev_types=["image_container_properties", "container_extension_mismatch"],
        )
        for cid in comp_cat_ids:
            if cid not in recompression_ev_ids:
                recompression_ev_ids.append(cid)

    if has_jpeg_qtables or recompression_ev_ids or has_container_mismatch:
        recomp_source = (
            "step4_manipulation_detection"
            if case_data.manipulation_detection is not None
            else "step3_basic_analysis"
        )
        builder.add(
            event_type="possible_recompression",
            title="Possible Image Compression or Recompression",
            description=(
                "JPEG quantization tables, compression characteristics, or container-format "
                "properties indicate the image was lossily compressed or possibly re-saved."
            ),
            source=recomp_source,
            evidence_ids=recompression_ev_ids or integrity_ids,
            temporal_certainty="unknown",
            confidence="low",
            status="possible",
            reasoning=(
                "Image may have been saved or recompressed one or more times based on observed "
                "JPEG quantization tables, compression indicators, or container characteristics."
            ),
            limitations=[
                "Compression artifacts and quantization tables alone do not confirm intentional editing or the exact number of save generations."
            ],
        )

    # 3b. Step 14.1 / 14.1.6 AI-generation detector analysis & signal (never a historical creation claim)
    if case_data.manipulation_detection is not None:
        ai_gen = (
            getattr(case_data.manipulation_detection, "ai_generation_detection", None)
            or case_data.manipulation_detection.ai_generation
        )
        if ai_gen is not None and ai_gen.status == "available" and ai_gen.prediction is not None:
            ai_ev_ids = _find_evidence_ids(
                evidence_items,
                ev_types=["ai_generation_detector_signal"],
                fallback_ids=integrity_ids,
            )
            sel_cls = ai_gen.selected_class or ai_gen.prediction.raw_class or "unknown"
            builder.add(
                event_type="forensic_analysis_event",
                title="AI image detector analysis performed",
                description=(
                    f"AI image detector '{ai_gen.detector.name if ai_gen.detector else 'umm-maybe/AI-image-detector'}' "
                    f"(revision: {ai_gen.detector.version if ai_gen.detector else 'unknown'}) "
                    f"executed during forensic inspection (selected_class: {sel_cls}, "
                    f"uncalibrated model score: {ai_gen.prediction.score:.4f}, "
                    f"decision_rule: {ai_gen.prediction.decision_rule}, threshold: null)."
                ),
                source="step14_1_ai_detection",
                evidence_ids=ai_ev_ids,
                temporal_certainty="exact",
                confidence="high",
                status="observed",
                reasoning=(
                    "The AI image detector analysis run itself was observed during MediaTrace inspection. "
                    "Model output is an uncalibrated forensic signal, not a historical creation event, "
                    "and does not establish AI generation, camera capture, or provenance."
                ),
                limitations=list(ai_gen.limitations),
            )
            if ai_gen.prediction.label == "potentially_ai_generated":
                builder.add(
                    event_type="possible_synthetic_generation_signal",
                    title="AI-generation detector produced a model-derived signal",
                    description=ai_gen.explanation,
                    source="step14_1_ai_detection",
                    evidence_ids=ai_ev_ids,
                    temporal_certainty="unknown",
                    confidence="low",
                    status="possible",
                    reasoning=(
                        f"AI-generation detector produced a model-derived signal via "
                        f"'{ai_gen.detector.name if ai_gen.detector else 'AI detector'}' "
                        f"(revision: {ai_gen.detector.version if ai_gen.detector else 'unknown'}, "
                        f"uncalibrated model score: {ai_gen.prediction.score:.4f}, "
                        f"decision_rule: {ai_gen.prediction.decision_rule}). "
                        "Model output is an uncalibrated forensic signal, not a historical creation event. "
                        "This does not confirm AI generation or identify a specific generator."
                    ),
                    limitations=list(ai_gen.limitations),
                )



    # 4. Metadata processing inference (Software tags such as Adobe Photoshop, GIMP, Lightroom, etc.)
    software_name: Optional[str] = None
    if (
        case_data.analysis is not None
        and case_data.analysis.metadata.fields.software
        and case_data.analysis.metadata.fields.software.strip()
    ):
        software_name = case_data.analysis.metadata.fields.software.strip()
    elif case_data.manipulation_detection is not None:
        for sig in case_data.manipulation_detection.signals:
            if sig.name == "software_metadata_present" and sig.strength in {"weak", "moderate"}:
                software_name = str(sig.value) if sig.value else "Editing Software"
                break

    if software_name:
        sw_ev_ids = _find_evidence_ids(
            evidence_items,
            ev_types=[
                "software_metadata",
                "software_metadata_present",
            ],
            fallback_ids=integrity_ids,
        )
        builder.add(
            event_type="metadata_processing",
            title=f"Software Processing or Export Indicated ({software_name})",
            description=(
                f"File metadata references '{software_name}', supporting an inference that the image "
                f"was processed, opened, or exported using this software."
            ),
            source="step3_basic_analysis",
            evidence_ids=sw_ev_ids,
            temporal_certainty="unknown",
            confidence="moderate",
            status="supported_inference",
            reasoning=(
                f"Embedded EXIF Software tag ('{software_name}') provides direct metadata evidence "
                f"of software interaction prior to case upload."
            ),
            limitations=[
                "Software metadata does not reveal what specific visual operation, adjustment, or export was performed."
            ],
        )

    # 5. Possible visual-edit events from Step 5 Localization
    if (
        case_data.localization is not None
        and (
            case_data.localization.localized_anomaly
            or len(case_data.localization.regions) > 0
        )
    ):
        loc_count = len(case_data.localization.regions)
        loc_ev_ids = _find_evidence_ids(
            evidence_items,
            ev_types=["localized_anomaly"],
            categories=["localization"],
            sources=["step5_localization"],
            fallback_ids=integrity_ids,
        )
        builder.add(
            event_type="possible_visual_edit",
            title="Possible Localized Region Modification",
            description=(
                f"Spatial anomaly localization flagged {loc_count} region(s) with local noise, "
                f"edge, or error-level inconsistencies relative to the rest of the image."
            ),
            source="step5_localization",
            evidence_ids=loc_ev_ids,
            temporal_certainty="unknown",
            confidence="low",
            status="possible",
            reasoning="Suspicious region anomalies were detected during Step 5 spatial localization analysis.",
            limitations=[
                "Cannot determine the specific editing operation or tool used, or rule out complex scene textures."
            ],
        )

    # 6. Reference comparison observations from Step 6 Comparison
    has_reference_diff = False
    if case_data.comparison is not None:
        comp = case_data.comparison
        if (
            comp.pixel_difference.changed_pixel_percentage > 0.0
            or len(comp.changed_regions) > 0
            or not comp.dimensions.same
        ):
            has_reference_diff = True
            comp_ev_ids = _find_evidence_ids(
                evidence_items,
                categories=["comparison"],
                sources=["step6_comparison"],
                fallback_ids=integrity_ids,
            )
            ref_name = (
                case_data.reference_image.original_filename
                if case_data.reference_image is not None
                else "reference image"
            )
            builder.add(
                event_type="reference_difference_observed",
                title="Visual and Structural Differences Relative to Reference Image",
                description=(
                    f"Direct comparison against '{ref_name}' observed "
                    f"{comp.pixel_difference.changed_pixel_percentage:.2f}% changed pixels across "
                    f"{len(comp.changed_regions)} region(s)."
                ),
                source="step6_comparison",
                evidence_ids=comp_ev_ids,
                temporal_certainty="relative",
                confidence="high",
                status="observed",
                reasoning=(
                    "Direct comparison shows measurable pixel and regional differences between "
                    "the submitted reference image and the case image."
                ),
                limitations=[
                    "Comparison shows differences between two files, but does not independently prove which file was created first unless external provenance exists."
                ],
            )

    # 7. Cross-analysis correlation events from Step 7 Evidence Correlation
    overlap_ev_ids = _find_evidence_ids(
        evidence_items,
        ev_types=["cross_analysis_overlap"],
    )
    if overlap_ev_ids:
        builder.add(
            event_type="possible_visual_edit",
            title="Correlated Regional Modification Across Localization and Comparison",
            description=(
                "Single-image localization anomalies spatially overlap with changed regions "
                "measured during reference image comparison."
            ),
            source="step7_evidence_correlation",
            evidence_ids=overlap_ev_ids,
            temporal_certainty="unknown",
            confidence="moderate",
            status="supported_inference",
            reasoning=(
                "Both single-image localization and reference comparison highlight overlapping spatial regions."
            ),
            limitations=[
                "Confirms spatial convergence between analytical methods, but does not establish exact editing tool or intermediate steps."
            ],
        )

    # 8. Possible export event (when supported by software metadata, recompression/container mismatch, or reference comparison differences)
    if software_name or recompression_ev_ids or has_container_mismatch or has_reference_diff:
        export_ev_ids = _find_evidence_ids(
            evidence_items,
            ev_types=[
                "software_metadata",
                "software_metadata_present",
                "extension_format_mismatch",
                "jpeg_recompression_indicator",
                "jpeg_high_compression_ratio",
                "changed_regions",
                "pixel_difference_ratio",
            ],
            fallback_ids=integrity_ids,
        )
        builder.add(
            event_type="possible_export",
            title="Possible Derived File Save or Export",
            description=(
                "Observed metadata, encoding, or comparison characteristics are consistent with "
                "the file being saved or exported as a derivative prior to upload."
            ),
            source="step7_evidence_correlation",
            evidence_ids=export_ev_ids,
            temporal_certainty="unknown",
            confidence="low",
            status="possible",
            reasoning=(
                "Software tags, encoding characteristics, or differences from the reference image "
                "suggest a file save/export step occurred before case creation."
            ),
            limitations=[
                "Cannot determine the number of prior export generations or intermediate file formats."
            ],
        )

    # 9. MediaTrace forensic analysis workflow event
    completed_stages = _expected_evidence_stages(case_data)
    builder.add(
        event_type="forensic_analysis_event",
        title="MediaTrace Forensic Analysis and Correlation Recorded",
        description=(
            f"Executed non-destructive forensic pipeline stages ({', '.join(completed_stages)}) "
            f"and correlated {len(evidence_items)} evidence item(s)."
        ),
        source="step7_evidence_correlation",
        evidence_ids=integrity_ids,
        temporal_certainty="exact",
        confidence="high",
        status="observed",
        reasoning=(
            "MediaTrace recorded analytical outputs while preserving the original uploaded file "
            "and its SHA-256 digest unchanged."
        ),
        limitations=[
            "Forensic reconstruction organizes observed evidence into a possible sequence and does not recover deleted historical states."
        ],
    )

    # 10. Step 14.2 Multi-signal forensic interpretation analysis event (when present)
    if case_data.forensic_decision is not None:
        builder.add(
            event_type="forensic_analysis_event",
            title="Multi-signal forensic interpretation performed.",
            description="Multi-signal forensic interpretation performed.",
            source="step14_2_forensic_decision",
            evidence_ids=integrity_ids,
            temporal_certainty="exact",
            confidence="high",
            status="observed",
            reasoning=(
                "MediaTrace executed the conservative multi-signal forensic decision framework over existing case evidence."
            ),
            limitations=[
                "This is an analytical workflow event and does not represent a historical creation or editing event on the media file."
            ],
        )

    events = builder.events

    inferred_events: List[InferredEventItem] = [
        InferredEventItem(
            title=ev.title,
            event_type=ev.event_type,
            confidence=ev.confidence,
            evidence_ids=ev.evidence_ids,
            reasoning=ev.reasoning,
            limitations=ev.limitations,
        )
        for ev in events
        if ev.status in ("supported_inference", "possible")
    ]

    has_supported_history = any(
        ev.status == "supported_inference" or ev.event_type == "reference_difference_observed"
        for ev in events
    )
    historical_accuracy: AllowedHistoricalAccuracy = (
        "partially_supported" if has_supported_history else "not_determined"
    )

    return events, inferred_events, historical_accuracy


def _build_unknowns(case_data: MediaCaseMetadata) -> List[str]:
    unknowns: List[str] = [
        "Exact chronological order of prior edits cannot be determined.",
        "Original camera capture time is not verifiable from file data alone.",
        "Specific editing operations cannot be confirmed from a single image.",
    ]
    if case_data.analysis is None:
        unknowns.append(
            "Basic structural and EXIF metadata properties have not been evaluated (Step 3 not run)."
        )
    elif not case_data.analysis.metadata.exif_present:
        unknowns.append(
            "Original capture device and software metadata are unknown because no EXIF header is present."
        )

    if case_data.localization is None:
        unknowns.append(
            "Spatial anomaly localization has not been evaluated for this case (Step 5 not run)."
        )

    if case_data.comparison is None:
        unknowns.append(
            "No verified reference/original image has been compared to establish direct pixel changes (Step 6 not run)."
        )
    else:
        unknowns.append(
            "Reference comparison alone cannot prove which of the two compared images was created first without external provenance."
        )

    return unknowns


def _build_limitations() -> List[str]:
    return [
        "Reconstructed edit history represents a possible sequence of processing events based on available forensic evidence.",
        "Evidence ordering reflects analytical sequence and plausible processing steps, not confirmed chronology.",
        "Heuristic forensic signals and spatial anomalies may result from benign compression, resizing, or format conversion rather than intentional tampering.",
    ]


def reconstruct_history_for_case(case_id: str) -> MediaCaseMetadata:
    """
    Correlate available forensic evidence for a case and reconstruct a conservative,
    evidence-backed possible edit history (Step 8), persisting it to storage/results/<case_id>.json.
    """
    case_data, _ = _load_case(case_id)
    case_data = _ensure_evidence_synchronized(case_id, case_data)
    _, result_path = _load_case(case_id)

    evidence_items: List[EvidenceItem] = (
        case_data.evidence.items if case_data.evidence is not None else []
    )

    integrity = _verify_file_integrity(case_data)
    known_facts = _build_known_facts(case_data, evidence_items)
    events, inferred_events, historical_accuracy = _build_events_and_inferences(
        case_data, evidence_items
    )
    unknowns = _build_unknowns(case_data)
    limitations = _build_limitations()

    report = EditHistoryReport(
        case_id=case_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        reconstructed_sequence=True,
        sequence_type="evidence_ordered_hypothesis",
        historical_accuracy=historical_accuracy,
        events=events,
        known_facts=known_facts,
        inferred_events=inferred_events,
        unknowns=unknowns,
        limitations=limitations,
        integrity=integrity,
    )

    case_data.history = report

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    ev_ids = [ev.id for ev in evidence_items]
    record_analysis_execution(
        case_data=case_data,
        service_name="history_service",
        status="completed",
        input_sha256=case_data.sha256,
        output_summary=(
            f"Reconstructed possible edit history: {len(events)} event(s), "
            f"historical_accuracy={historical_accuracy}."
        ),
        evidence_ids=ev_ids,
        completed_at=report.generated_at,
    )
    create_forensic_snapshot(case_data, record_event=True)

    temp_path = result_path.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as file_out:
            json.dump(case_data.model_dump(), file_out, indent=2)
        os.replace(temp_path, result_path)
    except Exception as exc:
        if temp_path.is_file():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist reconstructed edit history.",
        ) from exc

    return case_data


def get_history_for_case(case_id: str) -> EditHistoryReport:
    """Return the stored EditHistoryReport for a forensic case, or raise 404 if not yet generated."""
    case_data, _ = _load_case(case_id)
    if case_data.history is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Edit history has not been reconstructed for case '{case_id}' yet.",
        )
    return case_data.history
