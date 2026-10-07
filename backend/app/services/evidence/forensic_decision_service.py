from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status

from app.core.config import settings
from app.schemas.evidence import EvidenceItem
from app.schemas.forensic_decision import (
    AllowedConsistencyValue,
    AllowedOverallForensicAssessment,
    AllowedSignalStrength,
    EvidenceClusterItem,
    ForensicDecisionResult,
    ForensicSignalItem,
    IndependenceGroupSummary,
    SignalRelationshipItem,
    TransformationContextSummary,
)
from app.schemas.history import ReconstructedHistoryEvent
from app.schemas.media import MediaCaseMetadata
from app.services.evidence.evidence_service import generate_evidence_for_case
from app.services.media_service import CASE_ID_PATTERN

OVERALL_ASSESSMENT_LABELS: Dict[AllowedOverallForensicAssessment, str] = {
    "insufficient_evidence": "Insufficient forensic evidence available",
    "no_significant_forensic_evidence": "No significant forensic anomalies observed across evaluated stages",
    "isolated_forensic_signal": "Isolated forensic observation identified in a single evidence group",
    "multiple_consistent_forensic_observations": "Multiple consistent forensic observations across independent groups",
    "mixed_or_conflicting_forensic_observations": "Mixed or conflicting forensic observations across evaluated signals",
}

STRENGTH_RANK: Dict[AllowedSignalStrength, int] = {
    "unavailable": 0,
    "informational": 1,
    "weak": 2,
    "moderate": 3,
}

INDEPENDENCE_GROUP_METADATA: Dict[str, Tuple[str, str]] = {
    "file_integrity": (
        "Group I: File & Cryptographic Integrity",
        "Upload hash and post-analysis hash verification reflect preservation of the stored file bytes and are treated as one integrity group.",
    ),
    "structural_container": (
        "Group S: Container & Structural Geometry",
        "Container format, dimensions, color channels, and extension checks derive from file header parsing and share one structural group.",
    ),
    "metadata_and_timestamps": (
        "Group A: Metadata & Timestamp Observations",
        "EXIF tags, software strings, device identifiers, and embedded timestamps originate from container metadata blocks and are not independent of each other.",
    ),
    "jpeg_and_compression": (
        "Group B: JPEG & Compression Observations",
        "Quantization tables, compression quality estimates, and recompression indicators analyze the same encoding stream and belong to one compression group.",
    ),
    "pixel_localization": (
        "Group C: Pixel & Spatial Localization Observations",
        "Candidate anomalous regions, ELA/noise heatmaps, and binary masks originate from the same spatial localization analysis and are counted as one signal group rather than multiple independent signals.",
    ),
    "reference_comparison": (
        "Group D: Reference Comparison Observations",
        "Changed pixel ratios, changed region bounding boxes, difference images, and metadata comparisons measure differences against the same supplied reference file.",
    ),
    "ai_detector": (
        "Group E: Auxiliary AI Detector Model",
        "Selected class, artificial score, and human score come from a single forward pass of the SwinForImageClassification model and represent a single auxiliary model signal.",
    ),
}


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


def _verify_file_hash(case_data: MediaCaseMetadata) -> bool:
    upload_path = settings.uploads_dir / case_data.stored_filename
    if not upload_path.is_file():
        # In in-memory unit tests where no disk file was written, fall back to evidence if present
        if case_data.evidence is not None:
            for item in case_data.evidence.items:
                if item.type == "hash_preserved" and isinstance(item.value, bool):
                    return item.value
        return True
    current_sha = hashlib.sha256(upload_path.read_bytes()).hexdigest()
    return current_sha.lower() == case_data.sha256.lower()


def _match_evidence_ids(
    evidence_items: List[EvidenceItem],
    *,
    categories: Optional[List[str]] = None,
    ev_types: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
) -> List[str]:
    matched: List[str] = []
    for item in evidence_items:
        if ev_types and item.type in ev_types:
            matched.append(item.id)
            continue
        if categories and item.category in categories:
            if not sources or item.source in sources:
                matched.append(item.id)
    return list(dict.fromkeys(matched))


def _build_signal_inventory(
    case_data: MediaCaseMetadata,
    hash_preserved: bool,
    include_integrity: bool = True,
) -> List[ForensicSignalItem]:
    signals: List[ForensicSignalItem] = []
    evidence_items: List[EvidenceItem] = (
        case_data.evidence.items if case_data.evidence is not None else []
    )

    def add_signal(
        *,
        signal_category: Any,
        availability: Any,
        observation: str,
        forensic_interpretation: str,
        strength: AllowedSignalStrength,
        evidentiary_role: Any,
        source_service: str,
        independence_group: Any,
        evidence_ids: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        supporting_artifacts: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        sig_id = f"SIG-{len(signals) + 1:03d}"
        signals.append(
            ForensicSignalItem(
                signal_id=sig_id,
                signal_category=signal_category,
                availability=availability,
                observation=observation,
                forensic_interpretation=forensic_interpretation,
                strength=strength,
                evidentiary_role=evidentiary_role,
                source_service=source_service,
                independence_group=independence_group,
                evidence_ids=evidence_ids or [],
                limitations=limitations or [],
                supporting_artifacts=supporting_artifacts or [],
                details=details,
            )
        )

    # 1. INTEGRITY SIGNAL (Rule J)
    if include_integrity and case_data.sha256:
        int_ev_ids = _match_evidence_ids(
            evidence_items, ev_types=["original_hash", "hash_preserved"]
        )
        if hash_preserved:
            add_signal(
                signal_category="integrity",
                availability="available",
                observation=(
                    f"SHA-256 digest ({case_data.sha256}) of stored file matches the original upload record."
                ),
                forensic_interpretation=(
                    "SHA-256 integrity confirms preservation of the analyzed file bytes relative to the stored hash, "
                    "but does not establish that the file itself is authentic or original."
                ),
                strength="informational",
                evidentiary_role="preservation_evidence",
                source_service="integrity_check",
                independence_group="file_integrity",
                evidence_ids=int_ev_ids,
                limitations=[
                    "Cryptographic hash preservation verifies chain-of-custody storage only and cannot determine pre-upload history."
                ],
                details={"sha256": case_data.sha256, "hash_preserved": True},
            )
        else:
            add_signal(
                signal_category="integrity",
                availability="available",
                observation=(
                    f"Stored file SHA-256 no longer matches the recorded upload digest ({case_data.sha256})."
                ),
                forensic_interpretation=(
                    "A byte-level discrepancy exists between the stored file and its initial upload hash record."
                ),
                strength="moderate",
                evidentiary_role="preservation_evidence",
                source_service="integrity_check",
                independence_group="file_integrity",
                evidence_ids=int_ev_ids,
                limitations=[
                    "Indicates post-upload storage change; does not explain pre-upload media provenance."
                ],
                details={"sha256": case_data.sha256, "hash_preserved": False},
            )

    # 2. STRUCTURAL & METADATA / TEMPORAL SIGNALS (Step 3 + Step 4 metadata)
    analysis = case_data.analysis
    det = case_data.manipulation_detection

    if analysis is not None:
        img_info = analysis.image
        meta = analysis.metadata
        struct_ev_ids = _match_evidence_ids(
            evidence_items, ev_types=["image_container_properties", "extension_format_mismatch"]
        )

        # Check extension/format mismatch
        has_mismatch = any(
            "mismatch" in f.finding.lower() and "extension" in f.finding.lower()
            for f in analysis.findings
        ) or (
            det is not None
            and any(
                s.name == "extension_format_mismatch" and s.strength in {"weak", "moderate"}
                for s in det.signals
            )
        )

        add_signal(
            signal_category="structural",
            availability="available",
            observation=(
                f"Container format is {img_info.format or 'UNKNOWN'} ({img_info.width}x{img_info.height} px, {img_info.mode} mode)"
                + (
                    f"; file extension '{case_data.extension}' differs from container format."
                    if has_mismatch
                    else "."
                )
            ),
            forensic_interpretation=(
                "Container and extension mismatch may reflect file renaming, export, or format conversion, and does not independently establish manipulation."
                if has_mismatch
                else "Container geometry and channel layout were observed without structural header anomalies."
            ),
            strength="weak" if has_mismatch else "informational",
            evidentiary_role="classical_forensic_observation" if has_mismatch else "contextual_metadata",
            source_service="step3_basic_forensics",
            independence_group="structural_container",
            evidence_ids=struct_ev_ids,
            limitations=[
                "Container dimensions and format headers do not establish original capture provenance."
            ],
            details={
                "format": img_info.format,
                "width": img_info.width,
                "height": img_info.height,
                "mode": img_info.mode,
                "extension_mismatch": has_mismatch,
            },
        )

        # Metadata & Temporal signals
        meta_ev_ids = _match_evidence_ids(
            evidence_items,
            ev_types=[
                "exif_present",
                "exif_absent",
                "software_metadata",
                "camera_metadata",
                "timestamp_metadata",
                "gps_metadata",
                "software_metadata_present",
                "exif_inconsistency",
            ],
        )
        meta_signals_from_det = [
            s for s in (det.signals if det else []) if s.type == "metadata" and s.strength in {"weak", "moderate"}
        ]
        sw_value = meta.fields.software.strip() if (meta.exif_present and meta.fields.software) else None
        cam_value = (
            " ".join(p for p in (meta.fields.make, meta.fields.model) if p).strip()
            if meta.exif_present
            else ""
        )
        meta_strength: AllowedSignalStrength = "informational"
        if any(s.strength == "moderate" for s in meta_signals_from_det):
            meta_strength = "moderate"
        elif sw_value or meta_signals_from_det:
            meta_strength = "weak"

        if meta.exif_present:
            parts = ["EXIF metadata block is present"]
            if cam_value:
                parts.append(f"camera Make/Model='{cam_value}'")
            if sw_value:
                parts.append(f"Software='{sw_value}'")
            if meta.gps_present:
                parts.append("GPS metadata present (coordinates redacted)")
            for ms in meta_signals_from_det:
                parts.append(f"{ms.name}={ms.value}")
            meta_obs = "; ".join(parts) + "."
            meta_interp = (
                "Embedded EXIF metadata was observed. Software tags or metadata inconsistencies may reflect export, processing, or file handling and do not independently establish manipulation."
                if meta_strength in {"weak", "moderate"}
                else "Embedded EXIF metadata was observed. Metadata fields can be stripped, preserved, or modified during benign processing and do not establish camera-original authenticity."
            )
        else:
            meta_obs = "No EXIF metadata block was found in the image container."
            meta_interp = (
                "Absence of EXIF metadata is common across web exports, messaging platforms, screenshots, and synthetic generators, and is not evidence of manipulation or AI generation."
            )

        add_signal(
            signal_category="metadata",
            availability="available",
            observation=meta_obs,
            forensic_interpretation=meta_interp,
            strength=meta_strength,
            evidentiary_role="classical_forensic_observation" if meta_strength in {"weak", "moderate"} else "contextual_metadata",
            source_service="step3_basic_forensics",
            independence_group="metadata_and_timestamps",
            evidence_ids=meta_ev_ids,
            limitations=[
                "EXIF metadata is mutable and neither proves camera-original capture nor confirms intentional tampering."
            ],
            details={
                "exif_present": meta.exif_present,
                "camera_make_model": cam_value or None,
                "software": sw_value,
                "gps_present": meta.gps_present,
            },
        )

        # Temporal observation if timestamps exist or timestamp signal reported
        dt_orig = meta.fields.datetime_original if meta.exif_present else None
        dt_mod = meta.fields.datetime if meta.exif_present else None
        if dt_orig or dt_mod:
            ts_differ = bool(dt_orig and dt_mod and dt_orig.strip() != dt_mod.strip())
            add_signal(
                signal_category="temporal",
                availability="available",
                observation=(
                    f"Modification timestamp ('{dt_mod}') differs from original timestamp ('{dt_orig}')."
                    if ts_differ
                    else f"Timestamp metadata recorded ('{dt_orig or dt_mod}')."
                ),
                forensic_interpretation=(
                    "A timestamp relationship was observed that may reflect processing, file handling, export, timezone adjustment, or other causes; it does not establish that the image was edited."
                ),
                strength="weak" if ts_differ else "informational",
                evidentiary_role="contextual_metadata",
                source_service="step3_basic_forensics",
                independence_group="metadata_and_timestamps",
                evidence_ids=_match_evidence_ids(evidence_items, ev_types=["timestamp_metadata"]),
                limitations=[
                    "Embedded clock timestamps are controlled by the host device or exporting software and are not cryptographically verified."
                ],
                details={
                    "datetime_original": dt_orig,
                    "datetime": dt_mod,
                    "timestamps_differ": ts_differ,
                },
            )
    else:
        add_signal(
            signal_category="metadata",
            availability="unavailable",
            observation="Basic container and EXIF metadata analysis signal unavailable (Step 3 not evaluated).",
            forensic_interpretation="Metadata and structural analysis has not been run for this case. Unavailability is not negative or positive forensic evidence.",
            strength="unavailable",
            evidentiary_role="unavailable_signal",
            source_service="step3_basic_forensics",
            independence_group="metadata_and_timestamps",
            limitations=["Step 3 basic image forensics has not been executed."],
        )

    # 3. COMPRESSION SIGNALS & AI DETECTOR SIGNALS (Step 4)
    if det is not None:
        comp_signals = [
            s for s in det.signals if s.type in {"compression", "recompression", "consistency"}
        ]
        active_comp = [s for s in comp_signals if s.strength in {"weak", "moderate"}]
        comp_strength: AllowedSignalStrength = "informational"
        if any(s.strength == "moderate" for s in active_comp):
            comp_strength = "moderate"
        elif any(s.strength == "weak" for s in active_comp):
            comp_strength = "weak"

        comp_ev_ids = _match_evidence_ids(
            evidence_items,
            categories=["compression", "manipulation"],
            sources=["step4_manipulation_detection"],
        )
        if active_comp:
            obs_parts = [f"{s.name}={s.value} ({s.description})" for s in active_comp]
            comp_obs = f"Compression/consistency observations recorded: {'; '.join(obs_parts)}"
            comp_interp = (
                "Compression or Error Level Analysis (ELA) variance was observed. Such characteristics may result from recompression, resizing, platform transcoding, or localized editing, and do not independently prove tampering."
            )
        else:
            comp_obs = (
                f"Compression and classical consistency heuristics evaluated (recompression_status={det.recompression_status or 'none_detected'}); no weak or moderate compression anomalies flagged."
            )
            comp_interp = (
                "Classical compression and consistency checks did not identify elevated anomalies. This does not establish authenticity or rule out uniform processing or synthetic generation."
            )

        add_signal(
            signal_category="compression",
            availability="available",
            observation=comp_obs,
            forensic_interpretation=comp_interp,
            strength=comp_strength,
            evidentiary_role="classical_forensic_observation",
            source_service="step4_manipulation_detection",
            independence_group="jpeg_and_compression",
            evidence_ids=comp_ev_ids,
            limitations=[
                "JPEG compression and ELA metrics depend on scene texture and prior save history and cannot establish exact editing operations."
            ],
            details={
                "recompression_status": det.recompression_status,
                "overall_assessment": det.overall_assessment,
                "active_signal_names": [s.name for s in active_comp],
            },
        )

        # AI DETECTOR SIGNAL (Section 10)
        ai_gen = getattr(det, "ai_generation_detection", None) or det.ai_generation
        ai_ev_ids = _match_evidence_ids(
            evidence_items, ev_types=["ai_generation_detection"]
        )
        if ai_gen is not None and ai_gen.status == "available" and ai_gen.prediction is not None:
            sel_class = ai_gen.selected_class or ai_gen.prediction.raw_class or "unknown"
            sel_class_id = (
                ai_gen.selected_class_id
                if ai_gen.selected_class_id is not None
                else (0 if sel_class == "artificial" else 1)
            )
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
            score_type = ai_gen.prediction.score_type or "uncalibrated_model_score"
            decision_rule = ai_gen.prediction.decision_rule or "model_class_selection"
            threshold_val = ai_gen.prediction.threshold

            if sel_class == "artificial":
                ai_obs = (
                    f"AI detector selected the artificial class (selected_class='artificial', selected_class_id={sel_class_id}, "
                    f"artificial_score={art_score:.4f}, human_score={hum_score:.4f}, score_type='{score_type}', "
                    f"decision_rule='{decision_rule}', threshold=null)."
                )
                ai_interp = (
                    "The configured model selected the artificial class. This is an uncalibrated model output and does not establish that the image was AI-generated, nor does it establish provenance or generator identity."
                )
                ai_strength: AllowedSignalStrength = "weak"
            else:
                ai_obs = (
                    f"AI detector selected the human class (selected_class='human', selected_class_id={sel_class_id}, "
                    f"artificial_score={art_score:.4f}, human_score={hum_score:.4f}, score_type='{score_type}', "
                    f"decision_rule='{decision_rule}', threshold=null)."
                )
                ai_interp = (
                    "The configured model selected the human class. This does not establish camera-original authenticity and does not rule out synthetic generation."
                )
                ai_strength = "informational"

            add_signal(
                signal_category="ai_detector",
                availability="available",
                observation=ai_obs,
                forensic_interpretation=ai_interp,
                strength=ai_strength,
                evidentiary_role="auxiliary_model_signal",
                source_service="step14_1_ai_image_detector",
                independence_group="ai_detector",
                evidence_ids=ai_ev_ids,
                limitations=list(ai_gen.limitations)
                if ai_gen.limitations
                else [
                    "Uncalibrated model score from umm-maybe/AI-image-detector; sensitive to resizing and JPEG recompression; never proof of provenance."
                ],
                details={
                    "repository": ai_gen.detector.name if ai_gen.detector else "umm-maybe/AI-image-detector",
                    "revision": ai_gen.detector.version if ai_gen.detector else "c7e223baf11bc40528af364ba7bdea030ef42f9e",
                    "architecture": "SwinForImageClassification",
                    "selected_class": sel_class,
                    "selected_class_id": sel_class_id,
                    "artificial_score": art_score,
                    "human_score": hum_score,
                    "score_type": score_type,
                    "decision_rule": decision_rule,
                    "threshold": threshold_val,
                },
            )
        else:
            add_signal(
                signal_category="ai_detector",
                availability="unavailable",
                observation="AI detector signal unavailable.",
                forensic_interpretation=(
                    "AI detector signal unavailable. This unavailability does not constitute evidence for or against AI generation."
                ),
                strength="unavailable",
                evidentiary_role="unavailable_signal",
                source_service="step14_1_ai_image_detector",
                independence_group="ai_detector",
                evidence_ids=ai_ev_ids,
                limitations=[
                    "AI detector runtime was unavailable or not evaluated for this case."
                ],
                details={"status": "not_available"},
            )
    else:
        add_signal(
            signal_category="compression",
            availability="unavailable",
            observation="Compression and classical manipulation detection signal unavailable (Step 4 not evaluated).",
            forensic_interpretation="Classical compression and manipulation heuristics have not been evaluated for this case.",
            strength="unavailable",
            evidentiary_role="unavailable_signal",
            source_service="step4_manipulation_detection",
            independence_group="jpeg_and_compression",
            limitations=["Step 4 manipulation detection has not been executed."],
        )
        add_signal(
            signal_category="ai_detector",
            availability="unavailable",
            observation="AI detector signal unavailable.",
            forensic_interpretation=(
                "AI detector signal unavailable. This unavailability does not constitute evidence for or against AI generation."
            ),
            strength="unavailable",
            evidentiary_role="unavailable_signal",
            source_service="step14_1_ai_image_detector",
            independence_group="ai_detector",
            limitations=["AI detector has not been evaluated for this case."],
            details={"status": "not_available"},
        )

    # 4. LOCALIZATION SIGNALS (Section 12)
    loc = case_data.localization
    case_results_dir = settings.results_dir / case_data.case_id
    heatmap_avail = bool(
        (loc and getattr(loc, "visualization", None) and loc.visualization.heatmap)
        or (case_results_dir / "localization" / "heatmap.png").is_file()
    )
    mask_avail = bool(
        (loc and getattr(loc, "visualization", None) and loc.visualization.mask)
        or (case_results_dir / "localization" / "mask.png").is_file()
    )

    if loc is not None and loc.status == "completed":
        loc_ev_ids = _match_evidence_ids(evidence_items, categories=["localization"])
        reg_count = len(loc.regions)
        artifacts: List[str] = []
        if heatmap_avail:
            artifacts.append("localization/heatmap.png")
        if mask_avail:
            artifacts.append("localization/mask.png")

        if loc.localized_anomaly and reg_count > 0:
            max_score = max((r.max_anomaly_score for r in loc.regions), default=0.0)
            loc_strength: AllowedSignalStrength = "moderate" if max_score >= 0.75 else "weak"
            locations_str = ", ".join(
                f"{r.id}(x={r.x},y={r.y},w={r.width},h={r.height})" for r in loc.regions[:4]
            )
            loc_obs = (
                f"Localization identified {reg_count} candidate anomalous region(s) via {loc.method or 'ela_noise_hybrid'} "
                f"[{locations_str}] with max anomaly score {max_score:.4f} (heatmap_available={heatmap_avail}, mask_available={mask_avail})."
            )
            loc_interp = (
                "Candidate anomalous regions were identified by the localization analysis. "
                "Localization identified regions with characteristics that differ from the analyzed baseline. "
                "This does not independently establish manipulation."
            )
        else:
            loc_strength = "informational"
            max_score = 0.0
            loc_obs = (
                f"Localization analysis completed ({loc.method or 'ela_noise_hybrid'}) with 0 candidate anomalous regions flagged "
                f"(heatmap_available={heatmap_avail}, mask_available={mask_avail})."
            )
            loc_interp = (
                "Spatial localization did not identify concentrated regional anomalies above baseline. "
                "This does not confirm that the image is unedited or authentic."
            )

        add_signal(
            signal_category="localization",
            availability="available",
            observation=loc_obs,
            forensic_interpretation=loc_interp,
            strength=loc_strength,
            evidentiary_role="spatial_anomaly_observation",
            source_service="step5_localization",
            independence_group="pixel_localization",
            evidence_ids=loc_ev_ids,
            limitations=[
                "High-contrast edges, textured surfaces, or localized recompression can produce candidate localization regions without intentional tampering."
            ],
            supporting_artifacts=artifacts,
            details={
                "region_count": reg_count,
                "localized_anomaly": bool(loc.localized_anomaly and reg_count > 0),
                "artifact_type": loc.method or "ela_noise_hybrid",
                "artifact_locations": [
                    {
                        "id": r.id,
                        "x": r.x,
                        "y": r.y,
                        "width": r.width,
                        "height": r.height,
                        "area": r.area,
                        "mean_anomaly_score": r.mean_anomaly_score,
                        "max_anomaly_score": r.max_anomaly_score,
                    }
                    for r in loc.regions
                ],
                "artifact_strength": loc_strength,
                "heatmap_available": heatmap_avail,
                "mask_available": mask_avail,
            },
        )
    else:
        add_signal(
            signal_category="localization",
            availability="unavailable",
            observation="Spatial anomaly localization signal unavailable (Step 5 not evaluated).",
            forensic_interpretation="Spatial anomaly localization has not been run for this case. Unavailability is not evidence of absence of localized anomalies.",
            strength="unavailable",
            evidentiary_role="unavailable_signal",
            source_service="step5_localization",
            independence_group="pixel_localization",
            limitations=["Step 5 spatial anomaly localization has not been executed."],
        )

    # 5. REFERENCE COMPARISON SIGNALS (Section 13)
    comp = case_data.comparison
    diff_img_avail = bool(
        (comp and getattr(comp, "visualization", None) and comp.visualization.difference)
        or (case_results_dir / "comparison" / "difference.png").is_file()
    )
    diff_mask_avail = bool(
        (comp and getattr(comp, "visualization", None) and comp.visualization.mask)
        or (case_results_dir / "comparison" / "difference_mask.png").is_file()
    )

    if comp is not None and comp.status == "completed":
        comp_ev_ids = _match_evidence_ids(evidence_items, categories=["comparison"])
        pct = float(comp.pixel_difference.changed_pixel_percentage)
        reg_cnt = len(comp.changed_regions)
        same_dims = bool(comp.dimensions.same)
        case_dims = f"{comp.dimensions.case_dimensions.width}x{comp.dimensions.case_dimensions.height}"
        ref_dims = f"{comp.dimensions.reference_dimensions.width}x{comp.dimensions.reference_dimensions.height}"
        comp_artifacts: List[str] = []
        if diff_img_avail:
            comp_artifacts.append("comparison/difference.png")
        if diff_mask_avail:
            comp_artifacts.append("comparison/difference_mask.png")

        has_diff = bool(not same_dims or pct > 0.05 or reg_cnt > 0)
        if not same_dims or pct >= 1.0 or reg_cnt > 0:
            ref_strength: AllowedSignalStrength = "moderate"
        elif pct > 0.05:
            ref_strength = "weak"
        else:
            ref_strength = "informational"

        if has_diff:
            ref_obs = (
                f"Comparison between case ({case_dims}) and reference ({ref_dims}) observed "
                f"{pct:.2f}% changed pixels across {reg_cnt} region(s) "
                f"(same_dimensions={same_dims}, difference_artifact_available={diff_img_avail or diff_mask_avail})."
            )
            ref_interp = (
                "Visual/content differences were observed between the analyzed file and the supplied reference. "
                "This comparison does not establish which file is original unless independent provenance establishes it, and does not infer malicious editing from differences alone."
            )
        else:
            ref_obs = (
                f"Comparison between case ({case_dims}) and reference ({ref_dims}) observed "
                f"{pct:.2f}% changed pixels and 0 changed regions."
            )
            ref_interp = (
                "No significant visual/content differences were observed between the analyzed file and the supplied reference."
            )

        add_signal(
            signal_category="reference_comparison",
            availability="available",
            observation=ref_obs,
            forensic_interpretation=ref_interp,
            strength=ref_strength,
            evidentiary_role="comparative_difference_observation",
            source_service="step6_reference_comparison",
            independence_group="reference_comparison",
            evidence_ids=comp_ev_ids,
            limitations=[
                "Reference comparison measures relative differences between two files and cannot determine chronological priority or user intent."
            ],
            supporting_artifacts=comp_artifacts,
            details={
                "case_dimensions": case_dims,
                "reference_dimensions": ref_dims,
                "same_dimensions": same_dims,
                "changed_pixel_percentage": pct,
                "changed_region_count": reg_cnt,
                "changed_regions": [
                    {
                        "id": r.id,
                        "x": r.x,
                        "y": r.y,
                        "width": r.width,
                        "height": r.height,
                        "area": r.area,
                    }
                    for r in comp.changed_regions
                ],
                "difference_artifact_available": bool(diff_img_avail or diff_mask_avail),
                "metadata_differences": comp.metadata_comparison.model_dump()
                if comp.metadata_comparison
                else None,
            },
        )
    else:
        add_signal(
            signal_category="reference_comparison",
            availability="unavailable",
            observation="Reference image comparison signal unavailable (no reference image compared).",
            forensic_interpretation="Reference image comparison has not been performed. Unavailability does not imply the image matches or differs from an external original.",
            strength="unavailable",
            evidentiary_role="unavailable_signal",
            source_service="step6_reference_comparison",
            independence_group="reference_comparison",
            limitations=["No reference image has been uploaded and compared for this case."],
        )

    return signals


def _build_independence_groups(
    signals: List[ForensicSignalItem],
) -> List[IndependenceGroupSummary]:
    groups: List[IndependenceGroupSummary] = []
    for group_id, (label, note) in INDEPENDENCE_GROUP_METADATA.items():
        group_sigs = [s for s in signals if s.independence_group == group_id]
        avail_sigs = [s for s in group_sigs if s.availability == "available"]
        is_avail = len(avail_sigs) > 0

        if not is_avail:
            groups.append(
                IndependenceGroupSummary(
                    group_id=group_id,  # type: ignore[arg-type]
                    group_label=label,
                    available=False,
                    signal_ids=[s.signal_id for s in group_sigs],
                    meaningful_observation_present=False,
                    highest_strength="unavailable",
                    summary_observation="No available signals in this independence group.",
                    independence_note=note,
                )
            )
            continue

        highest = max(avail_sigs, key=lambda s: STRENGTH_RANK[s.strength]).strength
        # Meaningful observation: weak or moderate strength in this group
        # Note: for file_integrity, hash_preserved=True is informational (not a forensic content anomaly),
        # whereas hash_preserved=False is moderate.
        meaningful = highest in {"weak", "moderate"}

        summary_obs = " | ".join(s.observation for s in avail_sigs)
        groups.append(
            IndependenceGroupSummary(
                group_id=group_id,  # type: ignore[arg-type]
                group_label=label,
                available=True,
                signal_ids=[s.signal_id for s in avail_sigs],
                meaningful_observation_present=meaningful,
                highest_strength=highest,
                summary_observation=summary_obs,
                independence_note=note,
            )
        )
    return groups


def _detect_transformation_context(
    case_data: MediaCaseMetadata,
) -> TransformationContextSummary:
    flags: List[str] = []

    # 1. Check AI detector transformation context if already populated
    det = case_data.manipulation_detection
    if det is not None:
        ai_gen = getattr(det, "ai_generation_detection", None) or det.ai_generation
        if ai_gen is not None and getattr(ai_gen, "transformation_context", None) is not None:
            for f in ai_gen.transformation_context.signals:
                if f in {
                    "dimension_change_observed",
                    "recompression_observed",
                    "processing_metadata_present",
                    "reference_difference_observed",
                } and f not in flags:
                    flags.append(f)

        if det.recompression_status in {"likely_recompressed", "possible_recompression"}:
            if "recompression_observed" not in flags:
                flags.append("recompression_observed")
        for sig in det.signals:
            if sig.name in {"jpeg_recompression_indicator", "double_jpeg_compression"} and sig.strength in {"weak", "moderate"}:
                if "recompression_observed" not in flags:
                    flags.append("recompression_observed")
            if sig.name in {"software_metadata", "software_metadata_present"} and sig.strength in {"weak", "moderate"}:
                if "processing_metadata_present" not in flags:
                    flags.append("processing_metadata_present")

    if case_data.analysis is not None and case_data.analysis.metadata.exif_present:
        if case_data.analysis.metadata.fields.software:
            if "processing_metadata_present" not in flags:
                flags.append("processing_metadata_present")

    if case_data.comparison is not None and case_data.comparison.status == "completed":
        if not case_data.comparison.dimensions.same:
            if "dimension_change_observed" not in flags:
                flags.append("dimension_change_observed")
        if (
            case_data.comparison.pixel_difference.changed_pixel_percentage > 0.05
            or len(case_data.comparison.changed_regions) > 0
        ):
            if "reference_difference_observed" not in flags:
                flags.append("reference_difference_observed")

    if flags:
        return TransformationContextSummary(
            documented=True,
            observed_flags=flags,
            statement=(
                f"Documented transformation indicators observed ({', '.join(flags)}). "
                "Resizing, JPEG recompression, format conversion, and software export can alter classical compression metrics "
                "and shift AI detector model scores without establishing malicious intent."
            ),
        )

    return TransformationContextSummary(
        documented=False,
        observed_flags=[],
        statement="No prior transformation is documented in the available evidence.",
    )


def _build_signal_relationships_and_consistency(
    signals: List[ForensicSignalItem],
    independence_groups: List[IndependenceGroupSummary],
    case_data: MediaCaseMetadata,
) -> Tuple[List[SignalRelationshipItem], AllowedConsistencyValue, bool]:
    """Build explicit pairwise signal relationships and overall consistency classification."""
    sig_by_cat: Dict[str, ForensicSignalItem] = {
        s.signal_category: s for s in signals
    }
    relationships: List[SignalRelationshipItem] = []

    ai_sig = sig_by_cat.get("ai_detector")
    loc_sig = sig_by_cat.get("localization")
    comp_sig = sig_by_cat.get("compression")
    meta_sig = sig_by_cat.get("metadata")
    ref_sig = sig_by_cat.get("reference_comparison")

    ai_avail = ai_sig is not None and ai_sig.availability == "available"
    ai_selected = (
        ai_sig.details.get("selected_class")
        if (ai_avail and ai_sig and ai_sig.details)
        else None
    )

    loc_avail = loc_sig is not None and loc_sig.availability == "available"
    loc_anomaly = bool(
        loc_avail and loc_sig and loc_sig.strength in {"weak", "moderate"}
    )

    comp_avail = comp_sig is not None and comp_sig.availability == "available"
    comp_anomaly = bool(
        comp_avail and comp_sig and comp_sig.strength in {"weak", "moderate"}
    )

    meta_avail = meta_sig is not None and meta_sig.availability == "available"
    meta_anomaly = bool(
        meta_avail and meta_sig and meta_sig.strength in {"weak", "moderate"}
    )
    has_camera_exif = bool(
        meta_avail
        and meta_sig
        and meta_sig.details
        and meta_sig.details.get("camera_make_model")
    )

    ref_avail = ref_sig is not None and ref_sig.availability == "available"
    ref_diff = bool(
        ref_avail and ref_sig and ref_sig.strength in {"weak", "moderate"}
    )

    material_conflict = False

    # Check if explicit model/category disagreement or synthetic generator metadata conflicts with AI=human
    sw_str = (
        str(meta_sig.details.get("software") or "").lower()
        if (meta_avail and meta_sig and meta_sig.details)
        else ""
    )
    fn_lower = (case_data.original_filename or "").lower()
    documented_ai_context = any(
        kw in sw_str or kw in fn_lower
        for kw in ("stable diffusion", "midjourney", "dall-e", "dalle", "comfyui", "ai_generated", "synthetic")
    )

    # 1. Relationship: ai_detector <-> localization
    if not ai_avail or not loc_avail:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="localization",
                relationship="unavailable",
                consistency="insufficient_information",
                reason="One or both of AI detector and spatial localization signals are unavailable for comparison.",
            )
        )
    elif ai_selected == "artificial" and loc_anomaly:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="localization",
                relationship="partial_consistency",
                consistency="partially_consistent",
                reason="The AI detector selected the artificial class while localization identified candidate anomalous regions; these signals measure distinct phenomena and neither establishes manipulation or AI generation alone.",
            )
        )
    elif ai_selected == "artificial" and not loc_anomaly:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="localization",
                relationship="independent",
                consistency="partially_consistent",
                reason="The AI detector selected the artificial class while spatial localization did not identify regional anomalies; full-image synthetic generation need not produce localized splicing anomalies.",
            )
        )
    elif ai_selected == "human" and loc_anomaly:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="localization",
                relationship="partial_consistency",
                consistency="partially_consistent",
                reason="The AI detector selected the human class independently of the candidate localization anomaly; neither signal establishes authenticity or manipulation alone.",
            )
        )
    else:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="localization",
                relationship="supports_consistency",
                consistency="consistent",
                reason="The AI detector selected the human class and localization identified no candidate regional anomalies; however, this consistency does not establish camera-original authenticity.",
            )
        )

    # 2. Relationship: ai_detector <-> metadata / compression
    if ai_avail and (meta_avail or comp_avail):
        if ai_selected == "human" and documented_ai_context:
            material_conflict = True
            relationships.append(
                SignalRelationshipItem(
                    signal_a="ai_detector",
                    signal_b="metadata",
                    relationship="conflict",
                    consistency="conflicting",
                    reason="Model/category disagreement observed: the AI detector selected the human class while metadata or documented case provenance indicates a synthetic/AI source.",
                )
            )
        elif ai_selected == "artificial" and has_camera_exif and not comp_anomaly and not loc_anomaly and not ref_diff:
            material_conflict = True
            relationships.append(
                SignalRelationshipItem(
                    signal_a="ai_detector",
                    signal_b="metadata",
                    relationship="conflict",
                    consistency="conflicting",
                    reason="The AI detector selected the artificial class, whereas capture hardware EXIF metadata is present and classical compression/localization analyses identified no anomalies.",
                )
            )
        elif ai_selected == "artificial" and not comp_anomaly and not meta_anomaly:
            relationships.append(
                SignalRelationshipItem(
                    signal_a="ai_detector",
                    signal_b="compression",
                    relationship="partial_consistency",
                    consistency="partially_consistent",
                    reason="The AI detector selected the artificial class while classical compression and metadata checks did not flag anomalies.",
                )
            )
        else:
            relationships.append(
                SignalRelationshipItem(
                    signal_a="ai_detector",
                    signal_b="compression",
                    relationship="supports_consistency" if (ai_selected == "human" and not comp_anomaly) else "partial_consistency",
                    consistency="consistent" if (ai_selected == "human" and not comp_anomaly) else "partially_consistent",
                    reason="Model class selection and classical compression observations were compared without direct contradiction.",
                )
            )
    else:
        relationships.append(
            SignalRelationshipItem(
                signal_a="ai_detector",
                signal_b="compression",
                relationship="unavailable",
                consistency="insufficient_information",
                reason="AI detector or compression analysis is unavailable.",
            )
        )

    # 3. Relationship: localization <-> reference_comparison
    if loc_avail and ref_avail:
        if loc_anomaly and ref_diff:
            relationships.append(
                SignalRelationshipItem(
                    signal_a="localization",
                    signal_b="reference_comparison",
                    relationship="supports_consistency",
                    consistency="consistent",
                    reason="Both single-image spatial localization and two-image reference comparison identified regional differences; this does not establish chronology or intent.",
                )
            )
        elif loc_anomaly and not ref_diff:
            material_conflict = True
            relationships.append(
                SignalRelationshipItem(
                    signal_a="localization",
                    signal_b="reference_comparison",
                    relationship="conflict",
                    consistency="conflicting",
                    reason="Spatial localization flagged candidate anomalous regions, whereas comparison with the supplied reference image found no significant pixel differences.",
                )
            )
        elif not loc_anomaly and ref_diff:
            relationships.append(
                SignalRelationshipItem(
                    signal_a="localization",
                    signal_b="reference_comparison",
                    relationship="partial_consistency",
                    consistency="partially_consistent",
                    reason="Reference comparison identified visual differences between files even though single-image localization did not isolate a concentrated anomaly.",
                )
            )
        else:
            relationships.append(
                SignalRelationshipItem(
                    signal_a="localization",
                    signal_b="reference_comparison",
                    relationship="supports_consistency",
                    consistency="consistent",
                    reason="Neither single-image localization nor reference image comparison identified regional differences.",
                )
            )
    else:
        relationships.append(
            SignalRelationshipItem(
                signal_a="localization",
                signal_b="reference_comparison",
                relationship="unavailable",
                consistency="not_applicable",
                reason="Reference comparison or spatial localization was not evaluated for this case.",
            )
        )

    # Determine overall consistency classification
    content_groups = [
        g for g in independence_groups
        if g.group_id not in {"file_integrity", "structural_container"} and g.available
    ]
    if not content_groups:
        return relationships, "insufficient_information", material_conflict

    if material_conflict:
        return relationships, "conflicting", True

    meaningful_groups = [
        g for g in independence_groups
        if g.group_id != "file_integrity" and g.meaningful_observation_present
    ]
    if len(content_groups) == 1 and len(meaningful_groups) == 1:
        return relationships, "not_applicable", False

    if (ai_selected == "artificial" and not (loc_anomaly or comp_anomaly or meta_anomaly or ref_diff)) or (
        ai_selected == "human" and (loc_anomaly or comp_anomaly or ref_diff)
    ):
        return relationships, "partially_consistent", False

    return relationships, "consistent", False


def _build_evidence_clusters(
    signals: List[ForensicSignalItem],
    consistency: AllowedConsistencyValue,
    material_conflict: bool,
) -> List[EvidenceClusterItem]:
    clusters: List[EvidenceClusterItem] = []

    category_to_cluster_meta = {
        "file_integrity": ("integrity_001", "integrity_observation"),
        "metadata_and_timestamps": ("metadata_001", "metadata_observation"),
        "jpeg_and_compression": ("compression_001", "compression_observation"),
        "ai_detector": ("ai_model_001", "ai_model_observation"),
        "pixel_localization": ("localization_001", "localization_observation"),
        "reference_comparison": ("comparison_001", "comparison_observation"),
    }

    for group_id, (cluster_id, cluster_cat) in category_to_cluster_meta.items():
        group_sigs = [
            s for s in signals if s.independence_group == group_id and s.availability == "available"
        ]
        if not group_sigs:
            continue
        _, note = INDEPENDENCE_GROUP_METADATA[group_id]
        ev_ids: List[str] = []
        for s in group_sigs:
            for eid in s.evidence_ids:
                if eid not in ev_ids:
                    ev_ids.append(eid)

        clusters.append(
            EvidenceClusterItem(
                cluster_id=cluster_id,
                cluster_category=cluster_cat,  # type: ignore[arg-type]
                independence_group=group_id,
                signals=[s.signal_id for s in group_sigs],
                evidence_ids=ev_ids,
                observation=" ".join(s.observation for s in group_sigs),
                interpretation=" ".join(dict.fromkeys(s.forensic_interpretation for s in group_sigs)),
                independence_note=note,
            )
        )

    # Cross-signal cluster if multiple groups are active or conflicting
    meaningful_sigs = [
        s for s in signals
        if s.availability == "available" and s.strength in {"weak", "moderate"} and s.independence_group != "file_integrity"
    ]
    distinct_meaningful_groups = list(dict.fromkeys(s.independence_group for s in meaningful_sigs))

    if material_conflict:
        clusters.append(
            EvidenceClusterItem(
                cluster_id="cross_conflict_001",
                cluster_category="cross_signal_conflict",
                independence_group="multi_group_correlation",
                signals=[s.signal_id for s in signals if s.availability == "available"],
                evidence_ids=[],
                observation="Material disagreement observed across distinct forensic signal sources.",
                interpretation="The available forensic signals point in different directions; neither set of signals overrides the other or establishes a definitive verdict.",
                independence_note="Aggregates cross-group disagreement without double-counting within-group outputs.",
            )
        )
    elif len(distinct_meaningful_groups) >= 2:
        clusters.append(
            EvidenceClusterItem(
                cluster_id="cross_consistency_001",
                cluster_category="cross_signal_consistency",
                independence_group="multi_group_correlation",
                signals=[s.signal_id for s in meaningful_sigs],
                evidence_ids=[],
                observation=f"Observations present across {len(distinct_meaningful_groups)} distinct independence groups ({', '.join(distinct_meaningful_groups)}).",
                interpretation="This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent.",
                independence_note="Each independence group is counted at most once regardless of how many artifacts or sub-metrics it produces.",
            )
        )

    return clusters


def _determine_overall_assessment_and_interpretation(
    signals: List[ForensicSignalItem],
    independence_groups: List[IndependenceGroupSummary],
    consistency: AllowedConsistencyValue,
    material_conflict: bool,
    hash_preserved: bool,
) -> Tuple[AllowedOverallForensicAssessment, str]:
    """
    Apply Conservative Assessment Rules A through J and Cross-Signal Interpretation Templates (Cases 1–5).
    """
    sig_by_cat = {s.signal_category: s for s in signals}
    ai_sig = sig_by_cat.get("ai_detector")
    loc_sig = sig_by_cat.get("localization")
    comp_sig = sig_by_cat.get("compression")
    meta_sig = sig_by_cat.get("metadata")
    ref_sig = sig_by_cat.get("reference_comparison")

    ai_avail = ai_sig is not None and ai_sig.availability == "available"
    ai_selected = (
        ai_sig.details.get("selected_class")
        if (ai_avail and ai_sig and ai_sig.details)
        else None
    )

    loc_anomaly = bool(
        loc_sig and loc_sig.availability == "available" and loc_sig.strength in {"weak", "moderate"}
    )
    comp_anomaly = bool(
        comp_sig and comp_sig.availability == "available" and comp_sig.strength in {"weak", "moderate"}
    )
    meta_anomaly = bool(
        meta_sig and meta_sig.availability == "available" and meta_sig.strength in {"weak", "moderate"}
    )
    ref_diff = bool(
        ref_sig and ref_sig.availability == "available" and ref_sig.strength in {"weak", "moderate"}
    )

    # Check how many forensic content stages were actually evaluated (excluding file_integrity)
    evaluated_content_groups = [
        g for g in independence_groups
        if g.group_id != "file_integrity" and g.available
    ]

    # Distinct independence groups with meaningful forensic observations (excluding file_integrity)
    meaningful_content_groups = [
        g for g in independence_groups
        if g.group_id != "file_integrity" and g.meaningful_observation_present
    ]
    meaningful_group_ids = [g.group_id for g in meaningful_content_groups]

    # Rule A: If no content forensic stages were evaluated at all
    if not evaluated_content_groups:
        if not hash_preserved:
            return (
                "isolated_forensic_signal",
                "A file hash discrepancy was observed relative to the upload record, while content forensic analyses have not yet been executed. SHA-256 integrity relates to storage preservation only and does not establish image provenance or manipulation.",
            )
        return (
            "insufficient_evidence",
            "No image content analysis stages have been evaluated yet. SHA-256 integrity confirms preservation of the analyzed file bytes relative to the stored hash, but does not establish that the file itself is authentic or original.",
        )

    # Rule D: If meaningful signals materially disagree (Case 5)
    if material_conflict:
        return (
            "mixed_or_conflicting_forensic_observations",
            "The available forensic signals are mixed or conflicting. The current evidence does not support a single definitive interpretation.",
        )

    # Case 2: AI human + localization anomaly (without other corroborating manipulation groups or with other groups)
    if ai_avail and ai_selected == "human" and loc_anomaly and not ref_diff and not comp_anomaly:
        return (
            "isolated_forensic_signal",
            "The AI detector selected the human class, while separate localization analysis identified candidate anomalous regions. These are distinct observations and neither establishes authenticity or manipulation independently.",
        )

    # Rule E & Rule F & Case 1: AI detector = artificial with NO non-model forensic anomalies
    if meaningful_group_ids == ["ai_detector"]:
        return (
            "isolated_forensic_signal",
            "The AI detector selected the artificial class, but the available non-model forensic signals do not independently establish manipulation or synthetic generation.",
        )

    # Rule B, G, H, I: Exactly one distinct independence group has a meaningful observation
    if len(meaningful_content_groups) == 1:
        sole_group = meaningful_group_ids[0]
        if sole_group == "pixel_localization":
            return (
                "isolated_forensic_signal",
                "Candidate anomalous regions were identified by the localization analysis as an isolated forensic signal. All localization outputs (regions, heatmap, mask) originate from the same analytical group and do not independently establish manipulation.",
            )
        if sole_group == "metadata_and_timestamps":
            return (
                "isolated_forensic_signal",
                "An isolated metadata or timestamp observation was recorded. Metadata characteristics may reflect software export, file handling, or benign processing and do not establish manipulation.",
            )
        if sole_group == "reference_comparison":
            return (
                "isolated_forensic_signal",
                "Visual/content differences were observed between the analyzed file and the supplied reference as an isolated comparative signal. Reference differences alone do not establish which file is original or prove malicious alteration.",
            )
        return (
            "isolated_forensic_signal",
            f"An isolated forensic observation was recorded in {sole_group}. A single forensic signal group does not establish manipulation, synthetic generation, or provenance.",
        )

    # Rule C & Case 3: Multiple (>= 2) genuinely distinct independence groups contain mutually consistent observations
    if len(meaningful_content_groups) >= 2:
        if ai_selected == "artificial" and (loc_anomaly or ref_diff):
            return (
                "multiple_consistent_forensic_observations",
                "Multiple forensic observations are present and are partially consistent with image processing or alteration. The available evidence does not by itself establish the exact cause, provenance, chronology, or intent. "
                "This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent.",
            )
        return (
            "multiple_consistent_forensic_observations",
            "Multiple distinct forensic independence groups contain mutually consistent observations. "
            "This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent.",
        )

    # Rule A & Case 4: Evaluated stages exist and NO meaningful forensic observations were flagged
    if ai_avail and ai_selected == "human":
        return (
            "no_significant_forensic_evidence",
            "The available analyses did not identify significant forensic anomalies, while the AI detector selected the human class. These observations do not establish camera-original authenticity.",
        )

    return (
        "no_significant_forensic_evidence",
        "The evaluated forensic analyses did not identify significant anomalies. Absence of detected anomalies does not establish camera-original authenticity or rule out prior processing.",
    )


def _build_supported_and_unsupported_conclusions(
    signals: List[ForensicSignalItem],
    transformation_context: TransformationContextSummary,
    hash_preserved: bool,
) -> Tuple[List[str], List[str]]:
    supported: List[str] = []
    unsupported: List[str] = []

    sig_by_cat = {s.signal_category: s for s in signals}

    # Integrity
    int_sig = sig_by_cat.get("integrity")
    if int_sig and int_sig.availability == "available":
        if hash_preserved:
            supported.append(
                "The stored file bytes match the cryptographic SHA-256 digest recorded at upload."
            )
        else:
            supported.append(
                "The stored file bytes differ from the cryptographic SHA-256 digest recorded at upload."
            )

    # AI detector
    ai_sig = sig_by_cat.get("ai_detector")
    if ai_sig and ai_sig.availability == "available" and ai_sig.details:
        sel = ai_sig.details.get("selected_class")
        if sel == "artificial":
            supported.append(
                "The configured AI detector selected the artificial class (uncalibrated model output)."
            )
        elif sel == "human":
            supported.append(
                "The configured AI detector selected the human class (uncalibrated model output)."
            )
    else:
        supported.append(
            "The AI detector signal is unavailable for this case and is not treated as evidence for or against synthetic generation."
        )

    # Localization
    loc_sig = sig_by_cat.get("localization")
    if loc_sig and loc_sig.availability == "available" and loc_sig.details:
        if loc_sig.details.get("localized_anomaly"):
            cnt = loc_sig.details.get("region_count", 0)
            supported.append(
                f"Candidate localized anomalies ({cnt} region(s)) were identified relative to the image baseline."
            )
        else:
            supported.append(
                "Spatial anomaly localization evaluated the image and did not flag candidate regional anomalies."
            )

    # Reference Comparison
    ref_sig = sig_by_cat.get("reference_comparison")
    if ref_sig and ref_sig.availability == "available" and ref_sig.details:
        pct = float(ref_sig.details.get("changed_pixel_percentage", 0.0))
        reg_cnt = int(ref_sig.details.get("changed_region_count", 0))
        if not ref_sig.details.get("same_dimensions", True) or pct > 0.05 or reg_cnt > 0:
            supported.append(
                f"Visual/content differences ({pct:.2f}% pixels across {reg_cnt} region(s)) were observed between the analyzed file and the supplied reference."
            )
        else:
            supported.append(
                "No significant pixel differences were observed between the analyzed file and the supplied reference."
            )

    # Metadata / Compression
    meta_sig = sig_by_cat.get("metadata")
    if meta_sig and meta_sig.availability == "available" and meta_sig.details:
        if meta_sig.details.get("exif_present"):
            supported.append("Embedded EXIF container metadata is present in the analyzed file.")
        else:
            supported.append("No EXIF metadata block is present in the analyzed file container.")

    if transformation_context.documented:
        supported.append(
            f"Indicators consistent with prior image transformation ({', '.join(transformation_context.observed_flags)}) were observed."
        )

    # Mandatory Unsupported Conclusions tailored to the case
    unsupported.extend(
        [
            "The image is definitively AI-generated.",
            "The image is definitively authentic or an unedited camera original.",
            "The exact editing operation or historical editing sequence is known.",
            "The image was intentionally or maliciously manipulated.",
        ]
    )
    if ai_sig and ai_sig.availability == "available":
        unsupported.append(
            "The AI detector score represents a calibrated probability or confidence percentage."
        )
        unsupported.append(
            "The specific generator model or capture provenance is established by the AI detector output."
        )
    if loc_sig and loc_sig.availability == "available" and loc_sig.details and loc_sig.details.get("localized_anomaly"):
        unsupported.append(
            "Candidate localization regions confirm intentional splicing or tampering without independent corroboration."
        )
    if ref_sig and ref_sig.availability == "available":
        unsupported.append(
            "Differences relative to the reference image establish which file was created first."
        )

    return supported, unsupported


def _build_alternative_explanations(
    signals: List[ForensicSignalItem],
    transformation_context: TransformationContextSummary,
) -> List[str]:
    explanations: List[str] = []
    for s in signals:
        if s.availability != "available":
            continue
        if s.signal_category == "ai_detector" and s.details and s.details.get("selected_class") == "artificial":
            explanations.extend(
                [
                    "JPEG recompression, resizing, post-processing filters, or social/messaging platform transcoding can shift uncalibrated AI detector scores toward the artificial class on human photographs.",
                    "Graphic design overlays, digital rendering, or heavy noise reduction may trigger artificial class selection.",
                ]
            )
        elif s.signal_category == "localization" and s.strength in {"weak", "moderate"}:
            explanations.extend(
                [
                    "Normal camera processing, localized contrast/luminance transitions, or non-uniform surface textures can produce elevated ELA/noise localization responses.",
                    "JPEG block boundary artifacts or benign cropping/resaving can create localized compression variance.",
                ]
            )
        elif s.signal_category == "compression" and s.strength in {"weak", "moderate"}:
            explanations.extend(
                [
                    "Benign JPEG recompression during file transfer, cloud storage sync, or messaging-app forwarding.",
                    "Routine export or format conversion from photo management software.",
                ]
            )
        elif s.signal_category == "metadata":
            if s.details and not s.details.get("exif_present"):
                explanations.append(
                    "Privacy-preserving metadata stripping by web platforms, messaging apps, or operating system screenshot utilities."
                )
            elif s.strength in {"weak", "moderate"}:
                explanations.append(
                    "Benign metadata rewriting during color-space conversion, resizing, or catalog export."
                )
        elif s.signal_category == "reference_comparison" and s.strength in {"weak", "moderate"}:
            explanations.append(
                "Legitimate workflow derivatives such as resizing, cropping, color grading, or re-encoding from a shared master image."
            )

    if transformation_context.documented and not explanations:
        explanations.append(
            "Observed characteristics may reflect routine resizing, JPEG recompression, or software export."
        )
    if not explanations:
        explanations.append(
            "Standard camera pipeline processing, routine file storage, or benign format handling."
        )
    return list(dict.fromkeys(explanations))


def _build_additional_evidence_recommendations(
    signals: List[ForensicSignalItem],
) -> List[str]:
    recs: List[str] = []
    sig_by_cat = {s.signal_category: s for s in signals}

    ref_sig = sig_by_cat.get("reference_comparison")
    if not ref_sig or ref_sig.availability != "available":
        recs.append(
            "Obtain an earlier file version or verified source reference image for direct pixel and container comparison."
        )

    meta_sig = sig_by_cat.get("metadata")
    if not meta_sig or meta_sig.availability != "available" or (meta_sig.details and not meta_sig.details.get("exif_present")):
        recs.append(
            "Request the original camera file or uncompressed master export with intact acquisition metadata."
        )
        recs.append(
            "Gather independent device provenance or transfer logs (e.g. original storage media or capture device logs)."
        )

    ai_sig = sig_by_cat.get("ai_detector")
    if ai_sig and ai_sig.availability == "available" and ai_sig.details and ai_sig.details.get("selected_class") == "artificial":
        recs.append(
            "Obtain a higher-resolution or uncompressed copy prior to social/messaging platform recompression, along with any known generation or capture source documentation."
        )

    loc_sig = sig_by_cat.get("localization")
    if not loc_sig or loc_sig.availability != "available":
        recs.append(
            "Execute spatial anomaly localization (Step 5) to evaluate regional compression and noise consistency."
        )

    if not recs:
        recs.extend(
            [
                "Original camera file or earliest available uncompressed master copy.",
                "Independent provenance documentation or chain-of-custody transfer records.",
            ]
        )

    return list(dict.fromkeys(recs))


def _build_framework_limitations(
    transformation_context: TransformationContextSummary,
) -> List[str]:
    lims = [
        "Multi-signal forensic interpretation correlates observations across analysis stages while preserving uncertainty; it never computes a probability of manipulation, AI generation, or authenticity.",
        "The AI detector (umm-maybe/AI-image-detector, SwinForImageClassification) outputs uncalibrated model scores (threshold=null) and is treated strictly as an auxiliary signal.",
        "Multiple outputs from the same analytical source (such as localization regions, heatmaps, and masks) share a single independence group and are not counted as independent evidence.",
        "Absence of detected forensic anomalies or selection of the human class does not establish camera-original authenticity.",
        "SHA-256 hash integrity verifies storage preservation of the uploaded bytes only and does not validate pre-upload originality.",
    ]
    if transformation_context.documented:
        lims.append(
            f"Documented transformation factors ({', '.join(transformation_context.observed_flags)}) can alter compression metrics and AI detector scores."
        )
    else:
        lims.append("No prior transformation is documented in the available evidence.")
    return lims


def build_forensic_decision(
    case_data: MediaCaseMetadata,
    *,
    include_integrity: bool = True,
) -> ForensicDecisionResult:
    """
    Pure, deterministic multi-signal forensic interpretation builder.
    Does not run any new image-analysis algorithms or re-invoke deep learning models.
    """
    hash_preserved = _verify_file_hash(case_data)
    signals = _build_signal_inventory(
        case_data, hash_preserved=hash_preserved, include_integrity=include_integrity
    )
    independence_groups = _build_independence_groups(signals)
    transformation_context = _detect_transformation_context(case_data)
    relationships, consistency, material_conflict = _build_signal_relationships_and_consistency(
        signals, independence_groups, case_data
    )
    clusters = _build_evidence_clusters(signals, consistency, material_conflict)
    overall_assessment, interpretation = _determine_overall_assessment_and_interpretation(
        signals, independence_groups, consistency, material_conflict, hash_preserved
    )
    supported, unsupported = _build_supported_and_unsupported_conclusions(
        signals, transformation_context, hash_preserved
    )
    alt_explanations = _build_alternative_explanations(signals, transformation_context)
    recommendations = _build_additional_evidence_recommendations(signals)
    limitations = _build_framework_limitations(transformation_context)

    # Determine status: 'available' if at least one content forensic stage was evaluated,
    # 'unavailable' if required content analysis is missing (e.g. no signals or integrity-only).
    has_content_analysis = any(
        case_data_stage is not None
        for case_data_stage in (
            case_data.analysis,
            case_data.manipulation_detection,
            case_data.localization,
            case_data.comparison,
        )
    )
    status_val = "available" if has_content_analysis else "unavailable"

    return ForensicDecisionResult(
        case_id=case_data.case_id,
        status=status_val,  # type: ignore[arg-type]
        signal_inventory=signals,
        independence_groups=independence_groups,
        consistency_classification=consistency,
        signal_relationships=relationships,
        evidence_clusters=clusters,
        transformation_context=transformation_context,
        overall_assessment=overall_assessment,
        overall_assessment_label=OVERALL_ASSESSMENT_LABELS[overall_assessment],
        interpretation=interpretation,
        supported_conclusions=supported,
        unsupported_conclusions=unsupported,
        alternative_explanations=alt_explanations,
        additional_evidence_recommendations=recommendations,
        limitations=limitations,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _append_history_analysis_event_if_present(case_data: MediaCaseMetadata) -> None:
    """
    Section 22 History Integration:
    Record ONLY 'Multi-signal forensic interpretation performed.' (status='observed', temporal_certainty='exact')
    as an analysis event without inventing historical creation/editing events.
    """
    if case_data.history is None:
        return

    existing_titles = {ev.title for ev in case_data.history.events}
    if "Multi-signal forensic interpretation performed." in existing_titles:
        return

    next_seq = len(case_data.history.events) + 1
    int_ev_ids = [
        item.id
        for item in (case_data.evidence.items if case_data.evidence else [])
        if item.type in {"original_hash", "hash_preserved"}
    ] or ["EV-001"]

    case_data.history.events.append(
        ReconstructedHistoryEvent(
            id=f"EH-{next_seq:03d}",
            sequence=next_seq,
            event_type="forensic_analysis_event",
            title="Multi-signal forensic interpretation performed.",
            description="Multi-signal forensic interpretation performed.",
            source="step14_2_forensic_decision",
            evidence_ids=int_ev_ids,
            temporal_certainty="exact",
            confidence="high",
            status="observed",
            reasoning="MediaTrace executed the conservative multi-signal forensic decision framework over existing case evidence.",
            limitations=[
                "This is an analytical workflow event and does not represent a historical creation or editing event on the media file."
            ],
        )
    )


def interpret_forensics_for_case(case_id: str) -> ForensicDecisionResult:
    """
    Load existing analysis and evidence for case_id, build the conservative multi-signal
    ForensicDecisionResult without re-running expensive image models, integrate with history
    if present, and persist atomically to storage/results/<case_id>.json.
    """
    case_data, result_path = _load_case(case_id)

    # Synchronize Step 7 evidence from existing stored stages if not yet generated
    if case_data.evidence is None and any(
        s is not None
        for s in (
            case_data.analysis,
            case_data.manipulation_detection,
            case_data.localization,
            case_data.comparison,
        )
    ):
        case_data = generate_evidence_for_case(case_id)
        _, result_path = _load_case(case_id)

    decision = build_forensic_decision(case_data, include_integrity=True)
    case_data.forensic_decision = decision
    _append_history_analysis_event_if_present(case_data)

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        record_analysis_execution,
    )

    ev_ids = [ev.id for ev in case_data.evidence.items] if (case_data.evidence and case_data.evidence.items) else []
    record_analysis_execution(
        case_data=case_data,
        service_name="forensic_decision_service",
        status="completed",
        input_sha256=case_data.sha256,
        output_summary=(
            f"Multi-signal interpretation overall_assessment={decision.overall_assessment}; "
            f"supported={len(decision.supported_conclusions)}, "
            f"unsupported={len(decision.unsupported_conclusions)}."
        ),
        evidence_ids=ev_ids,
        completed_at=decision.generated_at,
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
            detail="Failed to persist multi-signal forensic interpretation.",
        ) from exc

    return decision


def get_forensic_interpretation_for_case(case_id: str) -> ForensicDecisionResult:
    """Return the persisted ForensicDecisionResult for a case, or HTTP 404 if not yet generated."""
    case_data, _ = _load_case(case_id)
    if case_data.forensic_decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Multi-signal forensic interpretation has not been generated for case '{case_id}' yet.",
        )
    return case_data.forensic_decision
