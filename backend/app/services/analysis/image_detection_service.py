from datetime import datetime, timezone
import hashlib
import io
import json
import os
from typing import List, Tuple

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.ai_detection import (
    AiGenerationDetectionResult,
    AiTransformationContext,
    build_ai_detector_interpretation,
)
from app.schemas.analysis import (
    AllowedOverallAssessment,
    ClassicalForensicsSection,
    CombinedForensicAssessmentSection,
    DetectionSignal,
    ImageAnalysisResult,
    ManipulationDetectionResult,
)
from app.schemas.media import MediaCaseMetadata
from app.services.analysis import image_ai_detection_service, image_analysis_service
from app.services.media_service import CASE_ID_PATTERN


EXTENSION_TO_EXPECTED_FORMAT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
}

ASSESSMENT_LABELS = {
    "no_strong_indicators": "No strong manipulation indicators detected",
    "potential_manipulation": "Potential manipulation indicators detected",
    "insufficient_evidence": "Insufficient evidence",
}


def _check_image_consistency(
    img: Image.Image,
    case_data: MediaCaseMetadata,
) -> List[DetectionSignal]:
    """
    Component A: Image consistency checks.
    Verify container decode, extension alignment, resolution, and color mode.
    """
    signals: List[DetectionSignal] = []
    actual_format = (img.format or "UNKNOWN").upper()
    expected_format = EXTENSION_TO_EXPECTED_FORMAT.get(case_data.extension.lower())

    signals.append(
        DetectionSignal(
            type="consistency",
            name="container_decode_verification",
            description="Image container decoded cleanly and matches recorded SHA-256 hash.",
            value=f"{actual_format} ({img.width}x{img.height} px, mode {img.mode})",
            strength="informational",
        )
    )

    if expected_format and actual_format != expected_format:
        signals.append(
            DetectionSignal(
                type="consistency",
                name="container_extension_mismatch",
                description="The file extension does not match the actual decoded image container format, indicating re-saving or renaming.",
                value=f"Extension '{case_data.extension}' vs container '{actual_format}'",
                strength="moderate",
            )
        )

    if img.width < 64 or img.height < 64:
        signals.append(
            DetectionSignal(
                type="consistency",
                name="low_spatial_resolution",
                description="Image dimensions are very small, which limits spatial forensic reliability.",
                value=f"{img.width}x{img.height} px",
                strength="weak",
            )
        )
    else:
        aspect_ratio = max(img.width, img.height) / max(1, min(img.width, img.height))
        if aspect_ratio > 5.0:
            signals.append(
                DetectionSignal(
                    type="consistency",
                    name="unusual_aspect_ratio",
                    description="Image exhibits an extreme aspect ratio that may result from cropping or banner composition.",
                    value=f"{img.width}x{img.height} (ratio {aspect_ratio:.1f}:1)",
                    strength="weak",
                )
            )

    return signals


def _check_jpeg_forensics(img: Image.Image) -> List[DetectionSignal]:
    """
    Component B: JPEG forensic checks.
    Inspect quantization tables and progressive/sequential encoding without inventing quality numbers.
    """
    signals: List[DetectionSignal] = []
    if (img.format or "").upper() != "JPEG":
        return signals

    is_progressive = bool(img.info.get("progressive") or img.info.get("progression"))
    encoding_mode = "Progressive" if is_progressive else "Baseline sequential"

    signals.append(
        DetectionSignal(
            type="compression",
            name="jpeg_encoding_mode",
            description="Observed JPEG entropy scan structure.",
            value=encoding_mode,
            strength="informational",
        )
    )

    qtables = getattr(img, "quantization", None)
    if isinstance(qtables, dict) and len(qtables) > 0:
        table_lengths = [len(tbl) for tbl in qtables.values() if hasattr(tbl, "__len__")]
        signals.append(
            DetectionSignal(
                type="compression",
                name="jpeg_quantization_tables",
                description="JPEG quantization tables are present in the stream header.",
                value=f"{len(qtables)} table(s) present (entries per table: {table_lengths})",
                strength="informational",
            )
        )
    else:
        signals.append(
            DetectionSignal(
                type="compression",
                name="jpeg_quantization_tables",
                description="JPEG quantization tables could not be extracted from the stream header.",
                value="unknown",
                strength="informational",
            )
        )

    return signals


def _check_metadata_signals(step3_analysis: ImageAnalysisResult) -> List[DetectionSignal]:
    """
    Component C: Metadata signals reusing Step 3 analysis information.
    Metadata is treated strictly as a supporting indicator, never as proof of manipulation.
    """
    signals: List[DetectionSignal] = []
    meta = step3_analysis.metadata
    fields = meta.fields

    if not meta.exif_present:
        signals.append(
            DetectionSignal(
                type="metadata",
                name="exif_metadata_absent",
                description="EXIF metadata is absent. While common in social media transcodes, screenshots, or privacy-stripped exports, original capture provenance cannot be verified from headers.",
                value="No EXIF container present",
                strength="informational",
            )
        )
        return signals

    if fields.software:
        signals.append(
            DetectionSignal(
                type="metadata",
                name="software_metadata",
                description="Software metadata is present in the image EXIF data. This indicates the file passed through editing or processing software, though not necessarily malicious tampering.",
                value=fields.software,
                strength="weak",
            )
        )

    if fields.datetime and fields.datetime_original and fields.datetime != fields.datetime_original:
        signals.append(
            DetectionSignal(
                type="metadata",
                name="timestamp_modification_delta",
                description="EXIF file modification timestamp differs from the original capture timestamp.",
                value=f"Original: {fields.datetime_original} | Modified: {fields.datetime}",
                strength="weak",
            )
        )

    if fields.make or fields.model:
        camera_str = " ".join(part for part in (fields.make, fields.model) if part)
        signals.append(
            DetectionSignal(
                type="metadata",
                name="camera_provenance_metadata",
                description="Camera hardware Make/Model tags are recorded in EXIF metadata.",
                value=camera_str,
                strength="informational",
            )
        )

    return signals


def _check_recompression_indicator() -> Tuple[str, DetectionSignal]:
    """
    Component D: Recompression / double-compression indicator.
    Conservative implementation returns 'Not evaluated' rather than fabricating an unreliable estimate.
    """
    status_text = "Not evaluated"
    signal = DetectionSignal(
        type="recompression",
        name="double_compression_evaluation",
        description="Double JPEG compression detection requires DCT histogram periodicity modeling and is not evaluated in the baseline heuristic pass.",
        value=status_text,
        strength="informational",
    )
    return status_text, signal


def _evaluate_ai_generation(
    image_path,
    expected_sha256: str,
) -> AiGenerationDetectionResult:
    """
    Component E: Step 14.1 Real AI-generated image detection.
    Loads the configured pretrained detector (or returns status='not_available' gracefully).
    """
    return image_ai_detection_service.detect_ai_generated_image(
        image_path=image_path,
        expected_sha256=expected_sha256,
    )


def _determine_overall_assessment(
    signals: List[DetectionSignal],
    step3_analysis: ImageAnalysisResult,
) -> Tuple[AllowedOverallAssessment, str]:
    """
    Conservative classical assessment logic:
    - If weak/moderate manipulation indicators (e.g., software tag, timestamp delta, container mismatch) exist -> 'potential_manipulation'
    - Else if EXIF provenance with camera hardware metadata is present and no anomalies exist -> 'no_strong_indicators'
    - Else (e.g., stripped EXIF without provenance and no AI detector configured) -> 'insufficient_evidence'
    """
    has_indicator = any(sig.strength in {"weak", "moderate"} for sig in signals)
    if has_indicator:
        assessment: AllowedOverallAssessment = "potential_manipulation"
        return assessment, ASSESSMENT_LABELS[assessment]

    meta = step3_analysis.metadata
    has_camera_provenance = bool(
        meta.exif_present and (meta.fields.make or meta.fields.model or meta.fields.datetime_original)
    )
    if has_camera_provenance:
        assessment = "no_strong_indicators"
        return assessment, ASSESSMENT_LABELS[assessment]

    assessment = "insufficient_evidence"
    return assessment, ASSESSMENT_LABELS[assessment]


def detect_image_manipulation(case_id: str) -> MediaCaseMetadata:
    """
    Run Step 4 classical manipulation & Step 14.1 AI-generation detection for an image case.
    Preserves existing Step 3 analysis and original image bytes, and saves
    `manipulation_detection` into storage/results/<case_id>.json.
    """
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
            raw_case_dict = json.load(file_in)
        case_data = MediaCaseMetadata(**raw_case_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read metadata for case '{case_id}'.",
        ) from exc

    if case_data.media_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Case '{case_id}' has media_type '{case_data.media_type}', expected 'image'.",
        )

    upload_path = settings.uploads_dir / case_data.stored_filename
    if not upload_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Original uploaded file for case '{case_id}' was not found.",
        )

    # Ensure Step 3 analysis is available without overwriting it if already computed
    if case_data.analysis is None:
        case_data = image_analysis_service.analyze_image(case_id)

    step3_analysis = case_data.analysis
    assert step3_analysis is not None

    # Read original bytes in read-only mode and verify SHA-256 integrity BEFORE analysis
    file_bytes = upload_path.read_bytes()
    size_before = len(file_bytes)
    current_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if current_sha256 != case_data.sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="File integrity check failed: stored image SHA-256 does not match case record.",
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.load()
            signals: List[DetectionSignal] = []
            signals.extend(_check_image_consistency(img, case_data))
            signals.extend(_check_jpeg_forensics(img))
            signals.extend(_check_metadata_signals(step3_analysis))
            recompression_status, recompression_signal = _check_recompression_indicator()
            signals.append(recompression_signal)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid decodable image container.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run manipulation detection on image.",
        ) from exc

    classical_assessment, classical_label = _determine_overall_assessment(signals, step3_analysis)
    ai_gen_result = _evaluate_ai_generation(upload_path, case_data.sha256)

    observed_tx_signals: List[str] = []
    if case_data.comparison is not None:
        if not case_data.comparison.dimensions.same:
            observed_tx_signals.append("dimension_change_observed")
        if (
            case_data.comparison.pixel_difference.changed_pixel_percentage > 0.0
            or len(case_data.comparison.changed_regions) > 0
        ):
            observed_tx_signals.append("reference_difference_observed")

    if any(
        sig.name in {"container_extension_mismatch", "jpeg_high_compression_ratio"}
        or (sig.type in {"compression", "recompression"} and sig.strength in {"weak", "moderate"})
        for sig in signals
    ):
        observed_tx_signals.append("recompression_observed")

    if bool(
        (step3_analysis.metadata.fields.software and step3_analysis.metadata.fields.software.strip())
        or any(
            sig.name in {"software_metadata_present", "timestamp_modification_delta"}
            and sig.strength in {"weak", "moderate"}
            for sig in signals
        )
    ):
        observed_tx_signals.append("processing_metadata_present")

    tx_documented = len(observed_tx_signals) > 0
    tx_ctx = AiTransformationContext(
        documented=tx_documented,
        signals=observed_tx_signals,
        note=(
            f"Documented transformation/processing context signals ({', '.join(observed_tx_signals)}) were "
            "observed in case evidence. Controlled transformation testing showed that resizing and "
            "JPEG recompression can materially shift model scores and flip the selected class; "
            "however, these observations do not prove why the detector produced its output."
            if tx_documented
            else "No prior resizing, recompression, or processing transformation is documented unless "
            "supported by explicit metadata, container, or reference-comparison evidence."
        ),
    )

    ai_gen_result = ai_gen_result.model_copy(
        update={
            "transformation_context": tx_ctx,
            "ai_detector_interpretation": build_ai_detector_interpretation(
                status=ai_gen_result.status,
                selected_class=ai_gen_result.selected_class,
                selected_class_id=ai_gen_result.selected_class_id,
                artificial_score=ai_gen_result.artificial_score,
                human_score=ai_gen_result.human_score,
                decision_rule=(
                    ai_gen_result.prediction.decision_rule
                    if ai_gen_result.prediction is not None
                    else "model_class_selection"
                ),
                threshold=(
                    ai_gen_result.prediction.threshold
                    if ai_gen_result.prediction is not None
                    else None
                ),
                limitations=list(ai_gen_result.limitations),
                transformation_context=tx_ctx,
            ),
        }
    )

    # Verify SHA-256 and file size AFTER AI detector execution
    bytes_after = upload_path.read_bytes()
    sha256_after = hashlib.sha256(bytes_after).hexdigest()
    if sha256_after != current_sha256 or len(bytes_after) != size_before:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Integrity violation: original uploaded image changed during detection.",
        )

    classical_section = ClassicalForensicsSection(
        overall_assessment=classical_assessment,
        assessment_label=classical_label,
        recompression_status=recompression_status,
        metadata_signals=[s for s in signals if s.type == "metadata"],
        compression_signals=[s for s in signals if s.type in {"compression", "recompression"}],
        consistency_signals=[s for s in signals if s.type == "consistency"],
        signals=signals,
    )

    # Step 14.1.6 Requirement 9: Do NOT allow the AI detector alone to produce potential_manipulation
    overall_assessment: AllowedOverallAssessment = classical_assessment
    assessment_label = ASSESSMENT_LABELS[overall_assessment]

    cross_analysis_interpretation = None
    if ai_gen_result.status == "available" and ai_gen_result.prediction is not None:
        threshold_clause = (
            f", threshold: {ai_gen_result.prediction.threshold}"
            if ai_gen_result.prediction.threshold is not None
            else ""
        )
        ai_summary = (
            f"Model '{ai_gen_result.detector.name}' (revision: {ai_gen_result.detector.version}, "
            f"license: {ai_gen_result.detector.license}) "
            f"classification: {ai_gen_result.prediction.display_label} "
            f"(selected_class: {ai_gen_result.selected_class}, "
            f"uncalibrated model score: {ai_gen_result.prediction.score:.4f}, "
            f"decision_rule: {ai_gen_result.prediction.decision_rule}{threshold_clause})."
        )
        if (
            ai_gen_result.selected_class == "artificial"
            and classical_assessment != "potential_manipulation"
        ):
            cross_analysis_interpretation = (
                "AI detector selected artificial while other available forensic signals "
                "do not independently establish manipulation."
            )
        elif ai_gen_result.selected_class == "human" and (
            (case_data.localization is not None and case_data.localization.localized_anomaly)
            or classical_assessment == "potential_manipulation"
        ):
            cross_analysis_interpretation = (
                "The AI detector selected human; separate localization analysis identified "
                "candidate anomalous regions. These are distinct signals and neither result "
                "establishes authenticity or manipulation on its own."
            )
        elif ai_gen_result.selected_class == "human":
            cross_analysis_interpretation = (
                "The AI detector selected human; this uncalibrated model selection does not "
                "establish camera-original authenticity and does not rule out synthetic generation."
            )
    else:
        ai_summary = ai_gen_result.explanation

    combined_interpretation = (
        f"Classical forensic observations: {classical_label}. "
        f"AI-generation detector output: {ai_summary} "
        + (f"{cross_analysis_interpretation} " if cross_analysis_interpretation else "")
        + "Classical metadata/compression findings and model-based AI-generation signals are evaluated "
        "independently and are not combined into a synthetic numerical probability."
    )

    assessment_section = CombinedForensicAssessmentSection(
        overall_assessment=overall_assessment,
        assessment_label=assessment_label,
        classical_forensic_observations=classical_label,
        ai_detector_output=ai_summary,
        combined_interpretation=combined_interpretation,
        cross_analysis_interpretation=cross_analysis_interpretation,
        transformation_context=tx_ctx,
    )


    detection_result = ManipulationDetectionResult(
        status="completed",
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        overall_assessment=overall_assessment,
        assessment_label=assessment_label,
        recompression_status=recompression_status,
        signals=signals,
        ai_generation=ai_gen_result,
        classical_forensics=classical_section,
        ai_generation_detection=ai_gen_result,
        assessment=assessment_section,
    )

    updated_case = case_data.model_copy(
        update={
            "status": "detected",
            "manipulation_detection": detection_result,
        }
    )

    from app.services.evidence.reproducibility_service import (
        create_forensic_snapshot,
        extract_ai_model_provenance,
        record_analysis_execution,
    )

    ai_prov = extract_ai_model_provenance(updated_case)
    record_analysis_execution(
        case_data=updated_case,
        service_name="image_detection_service",
        status="completed",
        input_sha256=updated_case.sha256,
        output_summary=(
            f"Classical assessment={detection_result.overall_assessment}; "
            f"AI detector status={detection_result.ai_generation_detection.status}"
            + (
                f" (class={detection_result.ai_generation_detection.prediction.label})"
                if detection_result.ai_generation_detection.prediction
                else ""
            )
            + "."
        ),
        model_provenance=ai_prov,
        completed_at=detection_result.evaluated_at,
    )
    create_forensic_snapshot(updated_case, record_event=True)

    # Atomically write updated case JSON so previous JSON is preserved on failure
    temp_path = settings.results_dir / f"{case_id}.json.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as tmp_out:
            json.dump(updated_case.model_dump(), tmp_out, indent=2)
        os.replace(temp_path, result_path)
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist manipulation detection results.",
        ) from exc

    return updated_case


def get_stored_ai_detection(case_id: str) -> AiGenerationDetectionResult:
    """Return the stored AI-generation detection result for a case without rerunning inference."""
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
            raw_case_dict = json.load(file_in)
        case_data = MediaCaseMetadata(**raw_case_dict)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read metadata for case '{case_id}'.",
        ) from exc

    if case_data.manipulation_detection is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"AI-generation detection has not been executed yet for case '{case_id}'.",
        )

    return case_data.manipulation_detection.ai_generation_detection

