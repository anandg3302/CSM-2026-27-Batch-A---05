r"""Step 14.1.3 & Step 14.1.7 — Controlled Multi-Image AI Detector Validation Harness.

Characterizes the behavior of the pinned AI-generated image detector
(`umm-maybe/AI-image-detector` @ `c7e223baf11bc40528af364ba7bdea030ef42f9e`)
across known-source validation images defined in a provenance manifest.

IMPORTANT:
- This is a controlled engineering validation harness, NOT a scientific or forensic
  model validation benchmark.
- Known categories (`human`, `ai_generated`, `transformed`) and provenance metadata
  are read strictly from `manifest.json` and are NEVER inferred from model predictions,
  EXIF, filenames, or classical forensic heuristics.
- Uses the exact production detector service (`image_ai_detection_service`)
  and preprocessing path (`validation inference == production inference`).
- Produces descriptive validation results only ("controlled dataset descriptive counts",
  no accuracy/precision/recall/F1/AUC, no model rankings, and no combined synthetic probabilities).
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.ai_detection import (
    DEFAULT_AI_DETECTOR_LIMITATIONS,
    OFFICIAL_MODEL_ARCHITECTURE,
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_LICENSE,
    OFFICIAL_PINNED_REVISION,
)
from app.services.analysis import (
    image_ai_detection_service,
    image_analysis_service,
    image_detection_service,
    image_localization_service,
)


ALLOWED_KNOWN_CATEGORIES = {"human", "ai_generated", "transformed"}
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
DEFAULT_TARGET_SAMPLE_COUNTS = {
    "human": 10,
    "ai_generated": 10,
    "transformed": 1,
}

VALIDATION_LIMITATIONS = [
    "This is a controlled descriptive engineering validation harness, not a scientific or forensic benchmark.",
    "Known categories come exclusively from manifest.json and are never inferred from model predictions, EXIF metadata, or visual heuristics.",
    "Model outputs are uncalibrated model-derived scores (uncalibrated_model_score) with decision_rule='model_class_selection' and threshold=null.",
    "This small convenience sample is not a representative statistical benchmark; results must not be generalized to arbitrary internet imagery or newer generators.",
    "Transformed images are reported separately and excluded from the source-level human-vs-AI descriptive confusion counts.",
    "No authenticity, provenance, benchmark accuracy, or manipulation probability conclusion is generated from detector scores or classical forensic observations.",
]


class ValidationHarnessError(ValueError):
    """Raised when the validation manifest, dataset images, or configuration is invalid."""


class ModelIntegrityError(ValidationHarnessError):
    """Raised when the local AI image detector model fails integrity or signature verification."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_integrity_for_validation() -> Dict[str, Any]:
    """Verify model repository, pinned revision, SHA-256 hashes, architecture, and ONNX signatures.

    Aborts validation with ModelIntegrityError if any check fails.
    """
    if not settings.ai_image_detector_enabled:
        raise ModelIntegrityError("AI image detector is disabled in settings.")

    model_id = (settings.ai_image_detector_model_id or "").strip()
    pinned_rev = (settings.ai_image_detector_model_version or "").strip()
    if not image_ai_detection_service._is_valid_pinned_revision(pinned_rev):
        raise ModelIntegrityError(
            f"Unpinned or invalid model revision '{pinned_rev}'. Must be a pinned 40-char commit SHA."
        )

    model_dir = image_ai_detection_service._resolve_safe_model_directory(model_id)
    config_path = model_dir / "config.json"
    onnx_path = model_dir / "model.onnx"
    pt_path = model_dir / "pytorch_model.bin"

    if not config_path.is_file():
        raise ModelIntegrityError(f"Missing model config.json in '{model_dir.name}'.")
    if not onnx_path.is_file():
        raise ModelIntegrityError(f"Missing model.onnx artifact in '{model_dir.name}'.")

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelIntegrityError(f"Failed to parse model config.json: {exc}") from exc

    cfg_rev = str(cfg.get("pinned_revision") or cfg.get("_commit_hash") or pinned_rev).strip()
    if cfg_rev != pinned_rev:
        raise ModelIntegrityError(
            f"Model config.json revision '{cfg_rev}' does not match configured pinned revision '{pinned_rev}'."
        )

    archs = cfg.get("architectures") or []
    arch = str(archs[0]) if isinstance(archs, list) and archs else OFFICIAL_MODEL_ARCHITECTURE
    if model_id == OFFICIAL_MODEL_ID and arch != OFFICIAL_MODEL_ARCHITECTURE:
        raise ModelIntegrityError(
            f"Unexpected model architecture '{arch}'; expected '{OFFICIAL_MODEL_ARCHITECTURE}'."
        )

    actual_onnx_sha256 = _sha256_file(onnx_path)
    expected_onnx_sha256 = cfg.get("onnx_sha256")
    if expected_onnx_sha256 and actual_onnx_sha256.lower() != str(expected_onnx_sha256).lower():
        raise ModelIntegrityError(
            f"model.onnx SHA-256 mismatch: expected {expected_onnx_sha256}, got {actual_onnx_sha256}."
        )

    actual_pt_sha256: Optional[str] = None
    expected_pt_sha256 = cfg.get("pytorch_model_sha256")
    if pt_path.is_file():
        actual_pt_sha256 = _sha256_file(pt_path)
        if expected_pt_sha256 and actual_pt_sha256.lower() != str(expected_pt_sha256).lower():
            raise ModelIntegrityError(
                f"pytorch_model.bin SHA-256 mismatch: expected {expected_pt_sha256}, got {actual_pt_sha256}."
            )
    elif expected_pt_sha256:
        actual_pt_sha256 = str(expected_pt_sha256)

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise ModelIntegrityError(f"ONNX Runtime failed to load model.onnx: {exc}") from exc

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ModelIntegrityError("model.onnx must have exactly 1 input and 1 output node.")

    inp = inputs[0]
    out = outputs[0]
    inp_shape = list(inp.shape or [])
    out_shape = list(out.shape or [])
    if (
        inp.type != "tensor(float)"
        or len(inp_shape) != 4
        or inp_shape[1] != 3
        or inp_shape[2] != 224
        or inp_shape[3] != 224
        or len(out_shape) != 2
        or out_shape[1] != 2
    ):
        raise ModelIntegrityError(
            f"Invalid ONNX signature: input={inp.name}:{inp_shape}:{inp.type}, output={out.name}:{out_shape}:{out.type}."
        )

    image_ai_detection_service.clear_ai_detector_cache()
    runtime = image_ai_detection_service._load_detector_runtime()
    if runtime is None:
        raise ModelIntegrityError("Production AI image detector runtime returned unavailable.")

    return {
        "model_repository": str(cfg.get("model_repository") or model_id),
        "pinned_revision": pinned_rev,
        "architecture": arch,
        "license": OFFICIAL_MODEL_LICENSE,
        "onnx_is_upstream": bool(cfg.get("onnx_is_upstream", False)),
        "pytorch_model_sha256": actual_pt_sha256,
        "model_onnx_sha256": actual_onnx_sha256,
        "onnx_input_signature": {
            "name": inp.name,
            "shape": [str(d) if not isinstance(d, int) else d for d in (inp.shape or [])],
            "dtype": inp.type,
        },
        "onnx_output_signature": {
            "name": out.name,
            "shape": [str(d) if not isinstance(d, int) else d for d in (out.shape or [])],
            "dtype": out.type,
        },
        "class_mapping": {"0": "artificial", "1": "human"},
        "preprocessing": {
            "image_mode": "RGB",
            "resize": "224x224",
            "resample": "BICUBIC (resample=3)",
            "value_scaling": "[0, 255] -> [0.0, 1.0]",
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "tensor_layout": "NCHW float32",
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def load_and_validate_manifest(
    manifest_path: Path,
    strict_provenance: bool = False,
) -> Dict[str, Any]:
    """Load and validate the controlled validation manifest with provenance and SHA-256 checks."""
    if not manifest_path.is_file():
        raise ValidationHarnessError(f"Validation manifest file not found: {manifest_path}")

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationHarnessError(f"Invalid JSON in validation manifest '{manifest_path}': {exc}") from exc

    if isinstance(raw_manifest, list):
        raw_samples = raw_manifest
        target_counts = dict(DEFAULT_TARGET_SAMPLE_COUNTS)
        schema_version = "1.0"
    elif isinstance(raw_manifest, dict):
        if "samples" not in raw_manifest or not isinstance(raw_manifest["samples"], list):
            raise ValidationHarnessError("Manifest object must contain a 'samples' list.")
        raw_samples = raw_manifest["samples"]
        target_counts = dict(DEFAULT_TARGET_SAMPLE_COUNTS)
        if isinstance(raw_manifest.get("target_sample_counts"), dict):
            for k, v in raw_manifest["target_sample_counts"].items():
                if k in ALLOWED_KNOWN_CATEGORIES and isinstance(v, int) and v >= 0:
                    target_counts[k] = v
        schema_version = str(raw_manifest.get("schema_version", "1.0"))
    else:
        raise ValidationHarnessError("Manifest root must be a JSON object with 'samples' or a JSON array.")

    is_v2_or_strict = strict_provenance or schema_version == "2.0"

    dataset_root = manifest_path.resolve().parent
    seen_ids: set[str] = set()
    seen_sha256: Dict[str, str] = {}
    validated_samples: List[Dict[str, Any]] = []

    for idx, entry in enumerate(raw_samples):
        if not isinstance(entry, dict):
            raise ValidationHarnessError(f"Manifest entry at index {idx} must be a JSON object.")

        val_id = str(entry.get("sample_id") or entry.get("id") or "").strip()
        if not val_id:
            raise ValidationHarnessError(f"Manifest entry at index {idx} is missing a non-empty 'id' or 'sample_id'.")
        if val_id in seen_ids:
            raise ValidationHarnessError(f"Duplicate validation ID '{val_id}' in manifest.")
        seen_ids.add(val_id)

        known_category = entry.get("category") if entry.get("category") is not None else entry.get("known_category")
        if not isinstance(known_category, str) or not known_category.strip():
            raise ValidationHarnessError(f"Missing category / known_category for ID '{val_id}'.")
        if known_category not in ALLOWED_KNOWN_CATEGORIES:
            raise ValidationHarnessError(
                f"Invalid known_category '{known_category}' for ID '{val_id}'. "
                f"Allowed values: {sorted(ALLOWED_KNOWN_CATEGORIES)}."
            )

        rel_path_str = str(entry.get("path") or "").strip()
        if not rel_path_str:
            raise ValidationHarnessError(f"Manifest entry '{val_id}' is missing a valid 'path'.")

        transformations = entry.get("transformations")
        if transformations is None:
            transformations = []
        elif not isinstance(transformations, list):
            raise ValidationHarnessError(f"'transformations' for ID '{val_id}' must be a list.")

        # Step 14.1.7 Requirement 8 & 11: Transformed derivatives must never be counted as independent source samples
        norm_rel_path = rel_path_str.replace("\\", "/")
        if known_category in {"human", "ai_generated"} and (
            len(transformations) > 0
            or entry.get("parent_sample_id")
            or entry.get("parent_sha256")
            or norm_rel_path.startswith("transformed/")
            or norm_rel_path.startswith("transformation_robustness/")
        ):
            raise ValidationHarnessError(
                f"Sample '{val_id}' in source category '{known_category}' contains transformation metadata or path "
                f"('{rel_path_str}'). Transformed files must not be counted as independent source samples."
            )

        if is_v2_or_strict:
            required_v2_keys = (
                "sample_id",
                "category",
                "path",
                "sha256",
                "byte_size",
                "width",
                "height",
                "format",
                "source_description",
                "generator_or_camera_if_known",
                "provenance_status",
                "notes",
            )
            missing_keys = [
                k for k in required_v2_keys
                if k not in entry or entry[k] is None or (isinstance(entry[k], str) and not entry[k].strip() and k != "notes")
            ]
            if missing_keys:
                raise ValidationHarnessError(
                    f"Manifest entry '{val_id}' is missing required provenance fields: {missing_keys}"
                )
            if known_category == "transformed":
                if not entry.get("parent_sample_id") or not entry.get("parent_sha256"):
                    raise ValidationHarnessError(
                        f"Transformed entry '{val_id}' is missing required 'parent_sample_id' or 'parent_sha256'."
                    )

        resolved_image_path = (dataset_root / rel_path_str).resolve()
        try:
            resolved_image_path.relative_to(dataset_root)
        except ValueError as exc:
            raise ValidationHarnessError(
                f"Path '{rel_path_str}' for ID '{val_id}' escapes dataset directory '{dataset_root}'."
            ) from exc

        if not resolved_image_path.is_file():
            raise ValidationHarnessError(
                f"Validation image file does not exist for ID '{val_id}': {resolved_image_path}"
            )

        ext = resolved_image_path.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValidationHarnessError(
                f"Unsupported image extension '{ext}' for ID '{val_id}'. "
                f"Supported extensions: {sorted(SUPPORTED_IMAGE_EXTENSIONS)}."
            )

        img_bytes = resolved_image_path.read_bytes()
        actual_sha256 = hashlib.sha256(img_bytes).hexdigest()
        actual_byte_size = len(img_bytes)

        # Verify documented SHA-256 if provided in manifest
        expected_sha256 = entry.get("sha256")
        if expected_sha256 is not None:
            if str(expected_sha256).strip().lower() != actual_sha256.lower():
                raise ValidationHarnessError(
                    f"Changed SHA-256 detected for ID '{val_id}': manifest expected '{expected_sha256}', "
                    f"but file has '{actual_sha256}'."
                )

        # Verify documented byte_size if provided in manifest
        expected_byte_size = entry.get("byte_size")
        if expected_byte_size is not None:
            if int(expected_byte_size) != actual_byte_size:
                raise ValidationHarnessError(
                    f"Changed byte_size detected for ID '{val_id}': manifest expected {expected_byte_size}, "
                    f"but file has {actual_byte_size}."
                )

        # Reject duplicate SHA-256 values across samples
        if actual_sha256 in seen_sha256:
            prev_id = seen_sha256[actual_sha256]
            raise ValidationHarnessError(
                f"Duplicate SHA-256 value '{actual_sha256}' detected across samples '{prev_id}' and '{val_id}'. "
                "Duplicate files must not be counted as independent samples."
            )
        seen_sha256[actual_sha256] = val_id

        try:
            with Image.open(io.BytesIO(img_bytes)) as im:
                img_width = int(im.width)
                img_height = int(im.height)
                img_format = str(im.format or ext.lstrip(".").upper())
                img_dimensions = f"{img_width}x{img_height}"
                im.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValidationHarnessError(
                f"Unsupported or unreadable image file for ID '{val_id}': {resolved_image_path}"
            ) from exc

        if entry.get("width") is not None and int(entry["width"]) != img_width:
            raise ValidationHarnessError(
                f"Width mismatch for ID '{val_id}': manifest expected {entry['width']}, got {img_width}."
            )
        if entry.get("height") is not None and int(entry["height"]) != img_height:
            raise ValidationHarnessError(
                f"Height mismatch for ID '{val_id}': manifest expected {entry['height']}, got {img_height}."
            )

        expected_category = entry.get("expected_category")
        if expected_category is not None and expected_category not in {"human", "ai_generated"}:
            raise ValidationHarnessError(
                f"Invalid expected_category '{expected_category}' for ID '{val_id}'."
            )

        gen_or_cam = str(
            entry.get("generator_or_camera_if_known")
            or entry.get("generator")
            or "unknown"
        ).strip()
        prov_status = str(
            entry.get("provenance_status")
            or ("documented" if entry.get("source_description") else "unknown")
        ).strip()

        validated_samples.append(
            {
                "id": val_id,
                "sample_id": val_id,
                "path": rel_path_str,
                "resolved_path": resolved_image_path,
                "category": known_category,
                "known_category": known_category,
                "original_category": entry.get("original_category"),
                "expected_category": expected_category,
                "parent_sample_id": entry.get("parent_sample_id"),
                "parent_sha256": entry.get("parent_sha256"),
                "transformation_description": entry.get("transformation_description"),
                "source_description": str(entry.get("source_description") or ""),
                "generator_or_camera_if_known": gen_or_cam,
                "provenance_status": prov_status,
                "generator": entry.get("generator") or (gen_or_cam if known_category == "ai_generated" else None),
                "transformations": [str(t) for t in transformations],
                "notes": str(entry.get("notes") or ""),
                "image_format": img_format,
                "width": img_width,
                "height": img_height,
                "image_dimensions": img_dimensions,
                "initial_byte_size": actual_byte_size,
                "initial_sha256": actual_sha256,
            }
        )

    return {
        "manifest_path": str(manifest_path.resolve()),
        "dataset_root": str(dataset_root),
        "schema_version": schema_version,
        "target_sample_counts": target_counts,
        "samples": validated_samples,
    }


def _run_optional_classical_forensics(image_path: Path, image_bytes: bytes) -> Dict[str, Any]:
    """Run MediaTrace classical forensics in an isolated temporary workspace without touching production storage."""
    orig_storage_root = settings.storage_root
    orig_uploads_dir = settings.uploads_dir
    orig_results_dir = settings.results_dir
    orig_reports_dir = settings.reports_dir

    with tempfile.TemporaryDirectory(prefix="mediatrace_val_") as tmp_dir_str:
        tmp_root = Path(tmp_dir_str)
        tmp_uploads = tmp_root / "uploads"
        tmp_results = tmp_root / "results"
        tmp_reports = tmp_root / "reports"
        tmp_uploads.mkdir(parents=True, exist_ok=True)
        tmp_results.mkdir(parents=True, exist_ok=True)
        tmp_reports.mkdir(parents=True, exist_ok=True)

        try:
            settings.storage_root = tmp_root
            settings.uploads_dir = tmp_uploads
            settings.results_dir = tmp_results
            settings.reports_dir = tmp_reports

            case_id = "MT-20260929-ABCDEF01"
            stored_filename = f"{case_id}_{image_path.name}"
            (tmp_uploads / stored_filename).write_bytes(image_bytes)
            case_payload = {
                "case_id": case_id,
                "original_filename": image_path.name,
                "stored_filename": stored_filename,
                "media_type": "image",
                "extension": image_path.suffix.lower(),
                "file_size": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "status": "uploaded",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            (tmp_results / f"{case_id}.json").write_text(json.dumps(case_payload), encoding="utf-8")

            analysis_resp = image_analysis_service.analyze_image(case_id)
            detect_resp = image_detection_service.detect_image_manipulation(case_id)
            localize_resp = image_localization_service.localize_image_manipulation(case_id)

            basic_dict = analysis_resp.analysis.model_dump() if analysis_resp.analysis else None
            classical_dict = (
                detect_resp.manipulation_detection.classical_forensics.model_dump()
                if detect_resp.manipulation_detection and detect_resp.manipulation_detection.classical_forensics
                else None
            )
            loc_dict = localize_resp.localization.model_dump() if localize_resp.localization else None

            return {
                "basic_image_analysis": {
                    "format": basic_dict["image"]["format"] if basic_dict else None,
                    "dimensions": (
                        f"{basic_dict['image']['width']}x{basic_dict['image']['height']}"
                        if basic_dict
                        else None
                    ),
                    "exif_present": basic_dict["metadata"]["exif_present"] if basic_dict else None,
                    "gps_present": basic_dict["metadata"]["gps_present"] if basic_dict else None,
                    "gps_info": "GPS coordinates redacted" if (basic_dict and basic_dict["metadata"]["gps_present"]) else None,
                    "camera_make": basic_dict["metadata"]["fields"]["make"] if basic_dict else None,
                    "camera_model": basic_dict["metadata"]["fields"]["model"] if basic_dict else None,
                    "software": basic_dict["metadata"]["fields"]["software"] if basic_dict else None,
                },
                "classical_manipulation_detection": {
                    "overall_assessment": classical_dict.get("overall_assessment") if classical_dict else None,
                    "assessment_label": classical_dict.get("assessment_label") if classical_dict else None,
                    "recompression_status": classical_dict.get("recompression_status") if classical_dict else None,
                    "signal_names": [s["name"] for s in classical_dict.get("signals", [])] if classical_dict else [],
                },
                "localization": {
                    "localized_anomaly": loc_dict.get("localized_anomaly") if loc_dict else None,
                    "region_count": len(loc_dict.get("regions", [])) if loc_dict else 0,
                    "threshold": loc_dict.get("threshold") if loc_dict else None,
                },
                "reference_comparison_available": False,
                "cross_analysis_note": (
                    "Additional forensic signals were observed; this experiment does not establish whether those signals represent manipulation."
                ),
                "separation_notice": (
                    "Classical MediaTrace forensic observations are recorded separately from the AI detector "
                    "and are never combined into a synthetic probability."
                ),
            }
        finally:
            settings.storage_root = orig_storage_root
            settings.uploads_dir = orig_uploads_dir
            settings.results_dir = orig_results_dir
            settings.reports_dir = orig_reports_dir


def _evaluate_category_disagreement(
    known_category: str,
    selected_class: str,
    expected_category: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Compare model output with manifest known_category using neutral wording."""
    if known_category == "human" and selected_class == "artificial":
        return True, "model/category disagreement observed"
    if known_category == "ai_generated" and selected_class == "human":
        return True, "model/category disagreement observed"
    if known_category == "transformed" and expected_category in {"human", "ai_generated"}:
        if (expected_category == "human" and selected_class == "artificial") or (
            expected_category == "ai_generated" and selected_class == "human"
        ):
            return True, "model/category disagreement observed"
    return False, None


def _compute_category_stats(
    category: str,
    items: List[Dict[str, Any]],
    target_count: int,
) -> Dict[str, Any]:
    art_scores = [float(r["artificial_score"]) for r in items if r.get("artificial_score") is not None]
    hum_scores = [float(r["human_score"]) for r in items if r.get("human_score") is not None]
    selected_art = sum(1 for r in items if r.get("selected_class") == "artificial")
    selected_hum = sum(1 for r in items if r.get("selected_class") == "human")

    return {
        "known_category": category,
        "sample_count": len(items),
        "target_sample_count": target_count,
        "category_complete": len(items) >= target_count,
        "selected_as_artificial_count": selected_art,
        "selected_as_human_count": selected_hum,
        "artificial_score_min": round(min(art_scores), 4) if art_scores else None,
        "artificial_score_max": round(max(art_scores), 4) if art_scores else None,
        "artificial_score_mean": round(statistics.mean(art_scores), 4) if art_scores else None,
        "artificial_score_median": round(statistics.median(art_scores), 4) if art_scores else None,
        "human_score_min": round(min(hum_scores), 4) if hum_scores else None,
        "human_score_max": round(max(hum_scores), 4) if hum_scores else None,
        "human_score_mean": round(statistics.mean(hum_scores), 4) if hum_scores else None,
        "human_score_median": round(statistics.median(hum_scores), 4) if hum_scores else None,
    }


def compute_stratified_dataset_summary(
    per_image_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute Step 14.1.7 stratified dataset and provenance summary without exposing GPS coordinates."""
    human_rows = [r for r in per_image_results if r["known_category"] == "human"]
    ai_rows = [r for r in per_image_results if r["known_category"] == "ai_generated"]
    transformed_rows = [r for r in per_image_results if r["known_category"] == "transformed"]
    source_rows = human_rows + ai_rows

    human_known_cam = sum(
        1 for r in human_rows
        if str(r.get("generator_or_camera_if_known") or "unknown").strip().lower() not in {"unknown", "", "none"}
    )
    human_unknown_cam = len(human_rows) - human_known_cam

    ai_known_gen = sum(
        1 for r in ai_rows
        if str(r.get("generator_or_camera_if_known") or "unknown").strip().lower() not in {"unknown", "", "none"}
    )
    ai_unknown_gen = len(ai_rows) - ai_known_gen

    source_sha256s = {r["sha256"] for r in source_rows}

    return {
        "gps_coordinates_exposed": False,
        "unique_source_sha256_count": len(source_sha256s),
        "total_source_sample_count": len(source_rows),
        "total_transformed_sample_count": len(transformed_rows),
        "human": {
            "count": len(human_rows),
            "known_camera_source": human_known_cam,
            "unknown_source": human_unknown_cam,
            "formats": dict(Counter(r["image_format"] for r in human_rows)),
            "dimensions": dict(Counter(r["image_dimensions"] for r in human_rows)),
            "known_devices": dict(
                Counter(
                    str(r.get("generator_or_camera_if_known") or "unknown")
                    for r in human_rows
                )
            ),
            "provenance_status_counts": dict(
                Counter(str(r.get("provenance_status") or "unknown") for r in human_rows)
            ),
        },
        "ai_generated": {
            "count": len(ai_rows),
            "known_generator": ai_known_gen,
            "unknown_generator": ai_unknown_gen,
            "formats": dict(Counter(r["image_format"] for r in ai_rows)),
            "dimensions": dict(Counter(r["image_dimensions"] for r in ai_rows)),
            "generators": dict(
                Counter(
                    str(r.get("generator_or_camera_if_known") or "unknown")
                    for r in ai_rows
                )
            ),
            "provenance_status_counts": dict(
                Counter(str(r.get("provenance_status") or "unknown") for r in ai_rows)
            ),
        },
        "transformed_separate_experiment": {
            "count": len(transformed_rows),
            "counted_as_independent_source": False,
            "formats": dict(Counter(r["image_format"] for r in transformed_rows)),
            "dimensions": dict(Counter(r["image_dimensions"] for r in transformed_rows)),
            "parent_sample_ids": [r.get("parent_sample_id") for r in transformed_rows],
        },
    }


def compute_dataset_quality_warnings(
    per_image_results: List[Dict[str, Any]],
    target_sample_counts: Dict[str, int],
) -> List[Dict[str, str]]:
    """Generate Step 14.1.7 informational dataset quality warnings (never turned into scores)."""
    warnings: List[Dict[str, str]] = []
    human_rows = [r for r in per_image_results if r["known_category"] == "human"]
    ai_rows = [r for r in per_image_results if r["known_category"] == "ai_generated"]
    source_rows = human_rows + ai_rows

    human_target = max(10, int(target_sample_counts.get("human", 10)))
    ai_target = max(10, int(target_sample_counts.get("ai_generated", 10)))

    if len(human_rows) < human_target:
        warnings.append({
            "code": "human_count_below_target",
            "severity": "informational",
            "message": (
                f"Known-human source sample count ({len(human_rows)}) is below the target of "
                f"{human_target}; dataset is marked incomplete and does not support statistical generalizations."
            ),
        })

    if len(ai_rows) < ai_target:
        warnings.append({
            "code": "ai_count_below_target",
            "severity": "informational",
            "message": (
                f"Known-AI source sample count ({len(ai_rows)}) is below the target of "
                f"{ai_target}; dataset is marked incomplete and does not support statistical generalizations."
            ),
        })

    known_generators = {
        str(r.get("generator_or_camera_if_known") or "unknown").strip()
        for r in ai_rows
        if str(r.get("generator_or_camera_if_known") or "unknown").strip().lower() not in {"unknown", "", "none"}
    }
    if len(known_generators) <= 1:
        warnings.append({
            "code": "limited_known_generator_diversity",
            "severity": "informational",
            "message": (
                f"Only {len(known_generators)} distinct known AI generator(s) explicitly identified "
                f"across {len(ai_rows)} AI sample(s); generator model diversity is limited."
            ),
        })

    known_devices = {
        str(r.get("generator_or_camera_if_known") or "unknown").strip()
        for r in human_rows
        if str(r.get("generator_or_camera_if_known") or "unknown").strip().lower() not in {"unknown", "", "none"}
    }
    if len(known_devices) <= 1:
        warnings.append({
            "code": "single_human_source_device",
            "severity": "informational",
            "message": (
                f"Only {len(known_devices)} human camera/device source(s) represented "
                f"({sorted(known_devices) if known_devices else ['none']})."
            ),
        })

    dim_counts = Counter(r["image_dimensions"] for r in source_rows)
    repeated_dims = {dim: cnt for dim, cnt in dim_counts.items() if cnt >= 2}
    if repeated_dims:
        warnings.append({
            "code": "shared_dimensions_observed",
            "severity": "informational",
            "message": f"Multiple source images share identical dimensions: {repeated_dims}.",
        })

    sha_counts = Counter(r["sha256"] for r in per_image_results)
    dup_shas = [sha for sha, cnt in sha_counts.items() if cnt > 1]
    if dup_shas:
        warnings.append({
            "code": "duplicate_sha256_present",
            "severity": "informational",
            "message": f"Duplicate SHA-256 values detected: {dup_shas}.",
        })

    unknown_prov_count = sum(
        1 for r in source_rows
        if str(r.get("generator_or_camera_if_known") or "unknown").strip().lower() in {"unknown", "", "none"}
    )
    if unknown_prov_count > 0:
        warnings.append({
            "code": "unknown_generator_or_camera_provenance",
            "severity": "informational",
            "message": (
                f"{unknown_prov_count} of {len(source_rows)} source sample(s) have 'unknown' specific "
                "generator or camera model provenance."
            ),
        })

    source_desc_counts = Counter(r.get("source_description", "") for r in source_rows if r.get("source_description"))
    if source_rows and source_desc_counts:
        top_desc, top_count = source_desc_counts.most_common(1)[0]
        if top_count > len(source_rows) / 2 and len(source_rows) >= 2:
            warnings.append({
                "code": "dataset_dominated_by_single_source",
                "severity": "informational",
                "message": (
                    f"Source dataset is dominated by a single source description "
                    f"({top_count}/{len(source_rows)} samples: '{top_desc}')."
                ),
            })

    accidental_transformed = [
        r["sample_id"]
        for r in source_rows
        if r.get("transformations") or r.get("parent_sample_id")
    ]
    if accidental_transformed:
        warnings.append({
            "code": "transformed_counted_as_source",
            "severity": "informational",
            "message": f"Transformed derivatives found in source category: {accidental_transformed}.",
        })

    return warnings


def compute_descriptive_validation_summary(
    per_image_results: List[Dict[str, Any]],
    model_integrity: Dict[str, Any],
    target_sample_counts: Dict[str, int],
) -> Dict[str, Any]:
    """Compute descriptive validation statistics without accuracy/precision/recall/F1/AUC claims."""
    by_category: Dict[str, List[Dict[str, Any]]] = {
        "human": [],
        "ai_generated": [],
        "transformed": [],
    }
    for row in per_image_results:
        cat = row["known_category"]
        if cat in by_category:
            by_category[cat].append(row)

    category_stats = {
        cat: _compute_category_stats(cat, by_category[cat], target_sample_counts.get(cat, 0))
        for cat in ("human", "ai_generated", "transformed")
    }

    incomplete_categories = [
        cat for cat, st in category_stats.items() if not st["category_complete"]
    ]

    stratified_summary = compute_stratified_dataset_summary(per_image_results)
    quality_warnings = compute_dataset_quality_warnings(per_image_results, target_sample_counts)

    confusion_counts = {
        "label": "controlled dataset descriptive counts",
        "table_type": "descriptive_confusion_counts",
        "excluded_categories": ["transformed"],
        "known_human_selected_human": category_stats["human"]["selected_as_human_count"],
        "known_human_selected_artificial": category_stats["human"]["selected_as_artificial_count"],
        "known_ai_selected_artificial": category_stats["ai_generated"]["selected_as_artificial_count"],
        "known_ai_selected_human": category_stats["ai_generated"]["selected_as_human_count"],
        "rows": {
            "known_human": {
                "predicted_artificial": category_stats["human"]["selected_as_artificial_count"],
                "predicted_human": category_stats["human"]["selected_as_human_count"],
            },
            "known_ai_generated": {
                "predicted_artificial": category_stats["ai_generated"]["selected_as_artificial_count"],
                "predicted_human": category_stats["ai_generated"]["selected_as_human_count"],
            },
        },
    }

    disagreements = [
        {
            "sample_id": r["sample_id"],
            "validation_id": r["validation_id"],
            "filename": r["filename"],
            "known_category": r["known_category"],
            "selected_class": r["selected_class"],
            "artificial_score": r["artificial_score"],
            "human_score": r["human_score"],
            "image_dimensions": r["image_dimensions"],
            "format": r["image_format"],
            "provenance_information": {
                "source_description": r.get("source_description"),
                "generator_or_camera_if_known": r.get("generator_or_camera_if_known"),
                "provenance_status": r.get("provenance_status"),
            },
            "transformation_information": {
                "transformations": r.get("transformations", []),
                "parent_sample_id": r.get("parent_sample_id"),
                "parent_sha256": r.get("parent_sha256"),
            },
            "observation": r["disagreement_observation"],
            "cross_analysis_note": r.get("cross_analysis_note"),
        }
        for r in per_image_results
        if r.get("category_disagreement") is True
    ]

    transformed_details = [
        {
            "sample_id": r["sample_id"],
            "validation_id": r["validation_id"],
            "filename": r["filename"],
            "original_category": r.get("original_category"),
            "parent_sample_id": r.get("parent_sample_id"),
            "parent_sha256": r.get("parent_sha256"),
            "transformations": r.get("transformations", []),
            "selected_class": r["selected_class"],
            "artificial_score": r["artificial_score"],
            "human_score": r["human_score"],
            "category_disagreement": r["category_disagreement"],
        }
        for r in by_category["transformed"]
    ]

    return {
        "report_type": "descriptive validation results",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_complete": len(incomplete_categories) == 0,
        "model_integrity": model_integrity,
        "stratified_dataset_summary": stratified_summary,
        "sample_availability": {
            "human_available": len(by_category["human"]),
            "human_target": target_sample_counts.get("human", 10),
            "ai_generated_available": len(by_category["ai_generated"]),
            "ai_generated_target": target_sample_counts.get("ai_generated", 10),
            "transformed_available": len(by_category["transformed"]),
            "transformed_target": target_sample_counts.get("transformed", 1),
            "incomplete_categories": incomplete_categories,
            "availability_notice": (
                "All target categories meet minimum sample counts."
                if not incomplete_categories
                else f"Categories below target sample count ({', '.join(incomplete_categories)}) are reported descriptively as incomplete without fabricating or downloading synthetic images."
            ),
        },
        "category_statistics": category_stats,
        "confusion_count_table": confusion_counts,
        "transformed_category_results": {
            "included_in_confusion_counts": False,
            "statistics": category_stats["transformed"],
            "samples": transformed_details,
        },
        "category_disagreements": {
            "count": len(disagreements),
            "known_human_selected_artificial": [
                d for d in disagreements if d["known_category"] == "human"
            ],
            "known_ai_selected_human": [
                d for d in disagreements if d["known_category"] == "ai_generated"
            ],
            "items": disagreements,
        },
        "dataset_quality_warnings": quality_warnings,
        "limitations": list(VALIDATION_LIMITATIONS),
    }


def _render_validation_markdown_summary(
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> str:
    """Render Step 14.1.7 validation_summary.md with all 10 required sections."""
    mi = summary["model_integrity"]
    strat = summary["stratified_dataset_summary"]
    avail = summary["sample_availability"]
    cat_stats = summary["category_statistics"]
    conf = summary["confusion_count_table"]
    disagreements = summary["category_disagreements"]["items"]
    warnings = summary["dataset_quality_warnings"]

    human_rows = [r for r in results if r["known_category"] == "human"]
    ai_rows = [r for r in results if r["known_category"] == "ai_generated"]
    tx_rows = [r for r in results if r["known_category"] == "transformed"]

    lines: List[str] = [
        "# MediaTrace AI Image Detector — Controlled Validation Summary (Step 14.1.7)",
        "",
        "> **Controlled Engineering Validation Statement**: This report documents a controlled descriptive engineering validation exercise for the MediaTrace AI image detector (`umm-maybe/AI-image-detector`). It is **not** a claim of benchmark accuracy, real-world reliability, or forensic model validation.",
        "",
        "## 1. Dataset Composition",
        f"- **Dataset Completeness Status**: `{'complete' if summary['dataset_complete'] else 'incomplete'}` ({avail['availability_notice']})",
        f"- **Known-Human Source Images**: `{avail['human_available']}` (Target: `{avail['human_target']}`)",
        f"- **Known-AI Source Images**: `{avail['ai_generated_available']}` (Target: `{avail['ai_generated_target']}`)",
        f"- **Unique Source SHA-256 Digests**: `{strat['unique_source_sha256_count']}`",
        f"- **Transformed Derivatives (Tracked Separately)**: `{avail['transformed_available']}` (Excluded from source human-vs-AI counts)",
        "",
        "## 2. Provenance Summary",
        f"- **GPS Coordinates Exposed**: `{strat['gps_coordinates_exposed']}`",
        f"- **Known-Human Provenance**: `count={strat['human']['count']}`, `known_camera_source={strat['human']['known_camera_source']}`, `unknown_source={strat['human']['unknown_source']}`, `devices={json.dumps(strat['human']['known_devices'])}`, `formats={json.dumps(strat['human']['formats'])}`, `dimensions={json.dumps(strat['human']['dimensions'])}`",
        f"- **Known-AI Provenance**: `count={strat['ai_generated']['count']}`, `known_generator={strat['ai_generated']['known_generator']}`, `unknown_generator={strat['ai_generated']['unknown_generator']}`, `generators={json.dumps(strat['ai_generated']['generators'])}`, `formats={json.dumps(strat['ai_generated']['formats'])}`, `dimensions={json.dumps(strat['ai_generated']['dimensions'])}`",
        "",
        "## 3. Model Configuration",
        f"- **Repository**: `{mi['model_repository']}`",
        f"- **Pinned Revision**: `{mi['pinned_revision']}`",
        f"- **Architecture**: `{mi['architecture']}`",
        f"- **License**: `{mi.get('license', OFFICIAL_MODEL_LICENSE)}`",
        f"- **Class Mapping**: `0 = artificial`, `1 = human`",
        f"- **Score Type**: `uncalibrated_model_score`",
        f"- **Decision Rule**: `model_class_selection`",
        f"- **Threshold**: `null`",
        f"- **Preprocessing**: RGB, resize `224x224` (`BICUBIC`), `[0,255] -> [0.0,1.0]`, ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`, `NCHW float32`",
        "",
        "## 4. Known-Human Results",
        "| Sample ID | Filename | Format | Dimensions | Camera / Source | Selected Class | Artificial Score | Human Score | Disagreement |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in human_rows:
        lines.append(
            f"| `{r['sample_id']}` | `{r['filename']}` | `{r['image_format']}` | `{r['image_dimensions']}` | "
            f"`{r.get('generator_or_camera_if_known', 'unknown')}` | `{r['selected_class']}` | "
            f"`{r['artificial_score']:.4f}` | `{r['human_score']:.4f}` | `{r['category_disagreement']}` |"
        )

    lines.extend([
        "",
        "## 5. Known-AI Results",
        "| Sample ID | Filename | Format | Dimensions | Generator | Selected Class | Artificial Score | Human Score | Disagreement |",
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for r in ai_rows:
        lines.append(
            f"| `{r['sample_id']}` | `{r['filename']}` | `{r['image_format']}` | `{r['image_dimensions']}` | "
            f"`{r.get('generator_or_camera_if_known', 'unknown')}` | `{r['selected_class']}` | "
            f"`{r['artificial_score']:.4f}` | `{r['human_score']:.4f}` | `{r['category_disagreement']}` |"
        )

    lines.extend([
        "",
        "## 6. Model/Category Disagreements",
        f"- **Total Disagreements Observed**: `{len(disagreements)}`",
    ])
    if disagreements:
        for d in disagreements:
            lines.append(
                f"  - **`{d['sample_id']}` (`{d['filename']}`)**: known_category=`{d['known_category']}`, "
                f"selected_class=`{d['selected_class']}`, artificial_score=`{d['artificial_score']:.4f}`, "
                f"human_score=`{d['human_score']:.4f}`, dimensions=`{d['image_dimensions']}`, format=`{d['format']}`, "
                f"observation=`{d['observation']}`."
            )
    else:
        lines.append("  - None observed in current sample set.")

    lines.extend([
        "",
        "## 7. Transformation Experiment Reference (Kept Separate)",
        f"- **Manifest Transformed Derivatives (`transformed/`)**: `{len(tx_rows)}` sample(s) evaluated separately and excluded from source human-vs-AI counts.",
    ])
    for r in tx_rows:
        lines.append(
            f"  - `{r['sample_id']}` (parent=`{r.get('parent_sample_id')}`, parent_sha256=`{r.get('parent_sha256')}`): "
            f"selected_class=`{r['selected_class']}`, artificial_score=`{r['artificial_score']:.4f}`, human_score=`{r['human_score']:.4f}`."
        )
    lines.append(
        "- **Step 14.1.5 Controlled Transformation Robustness Experiment**: Maintained separately in `datasets/ai_detector_validation/transformation_robustness_results.json` and `transformation_robustness_summary.md` (24 records across 4 source images × 6 conditions)."
    )

    h_st = cat_stats["human"]
    a_st = cat_stats["ai_generated"]
    lines.extend([
        "",
        "## 8. Descriptive Statistics (`controlled dataset descriptive counts`)",
        f"- **Label**: `{conf['label']}`",
        f"- **Known-Human (`n={h_st['sample_count']}`)**: selected_human=`{conf['known_human_selected_human']}`, selected_artificial=`{conf['known_human_selected_artificial']}`, artificial_score min/max/mean/median=`{h_st['artificial_score_min']}` / `{h_st['artificial_score_max']}` / `{h_st['artificial_score_mean']}` / `{h_st['artificial_score_median']}`, human_score min/max/mean/median=`{h_st['human_score_min']}` / `{h_st['human_score_max']}` / `{h_st['human_score_mean']}` / `{h_st['human_score_median']}`",
        f"- **Known-AI (`n={a_st['sample_count']}`)**: selected_artificial=`{conf['known_ai_selected_artificial']}`, selected_human=`{conf['known_ai_selected_human']}`, artificial_score min/max/mean/median=`{a_st['artificial_score_min']}` / `{a_st['artificial_score_max']}` / `{a_st['artificial_score_mean']}` / `{a_st['artificial_score_median']}`, human_score min/max/mean/median=`{a_st['human_score_min']}` / `{a_st['human_score_max']}` / `{a_st['human_score_mean']}` / `{a_st['human_score_median']}`",
        "",
        "## 9. Dataset Quality Warnings & Limitations",
        "### Dataset Quality Warnings (Informational)",
    ])
    for w in warnings:
        lines.append(f"- **[`{w['code']}`]**: {w['message']}")
    lines.append("")
    lines.append("### Forensic & Validation Limitations")
    for lim in summary["limitations"]:
        lines.append(f"- {lim}")

    lines.extend([
        "",
        "## 10. Explicit Controlled Engineering Validation Statement",
        "This validation run is a **controlled descriptive engineering experiment** using locally available, provenance-documented files. It does **not** establish production accuracy, benchmark accuracy, forensic reliability, or calibrated performance for `umm-maybe/AI-image-detector`.",
        "",
    ])
    return "\n".join(lines)


def _write_validation_csv(csv_path: Path, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "validation_id",
        "filename",
        "category",
        "known_category",
        "original_category",
        "parent_sample_id",
        "parent_sha256",
        "source_description",
        "generator_or_camera_if_known",
        "provenance_status",
        "generator",
        "transformations",
        "image_format",
        "width",
        "height",
        "image_dimensions",
        "byte_size",
        "sha256",
        "sha256_preserved",
        "detector_status",
        "selected_class",
        "selected_class_id",
        "raw_class",
        "artificial_score",
        "human_score",
        "score_type",
        "decision_rule",
        "threshold",
        "category_disagreement",
        "disagreement_observation",
        "model_id",
        "model_repository",
        "model_revision",
        "pinned_revision",
        "model_architecture",
        "model_onnx_sha256",
        "validated_at",
        "runtime_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames}
            if isinstance(row.get("transformations"), list):
                row["transformations"] = ";".join(str(x) for x in row["transformations"])
            writer.writerow(row)


def run_validation_harness(
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    skip_forensics: bool = False,
) -> Dict[str, Any]:
    """Run the complete Step 14.1.3 / 14.1.7 controlled multi-image AI detector validation harness."""
    repo_root = Path(__file__).resolve().parents[2]
    default_manifest = (repo_root / "datasets" / "ai_detector_validation" / "manifest.json").resolve()
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else default_manifest
    )
    dataset_results_dir = (resolved_manifest.parent / "results").resolve()
    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else dataset_results_dir
    )

    # 1. Verify model integrity before processing any sample
    model_integrity = verify_model_integrity_for_validation()

    # 2. Load and validate manifest and dataset files
    manifest_info = load_and_validate_manifest(resolved_manifest)

    per_image_results: List[Dict[str, Any]] = []

    for sample in manifest_info["samples"]:
        img_path: Path = sample["resolved_path"]
        pre_bytes = img_path.read_bytes()
        pre_size = len(pre_bytes)
        pre_sha256 = hashlib.sha256(pre_bytes).hexdigest()

        if pre_sha256 != sample["initial_sha256"] or pre_size != sample["initial_byte_size"]:
            raise ValidationHarnessError(
                f"Dataset integrity violation prior to inference for '{sample['id']}'."
            )

        # Invoke exact production detector service
        runtime_error: Optional[str] = None
        ai_res = image_ai_detection_service.detect_ai_generated_image(
            img_path,
            expected_sha256=pre_sha256,
        )

        # Verify source image on disk was not modified
        post_bytes = img_path.read_bytes()
        post_size = len(post_bytes)
        post_sha256 = hashlib.sha256(post_bytes).hexdigest()
        if post_sha256 != pre_sha256 or post_size != pre_size or post_bytes != pre_bytes:
            raise ValidationHarnessError(
                f"Dataset image '{img_path.name}' was modified during validation!"
            )

        if ai_res.status != "available" or ai_res.prediction is None or ai_res.detector is None:
            runtime_error = ai_res.reason or "Detector returned unavailable during validation."
            raise ModelIntegrityError(
                f"Detector failed to produce an available prediction for '{sample['id']}': {runtime_error}"
            )

        artificial_score = round(float(ai_res.prediction.score), 4)
        human_score = (
            round(float(ai_res.human_score), 4)
            if ai_res.human_score is not None
            else round(float(1.0 - artificial_score), 4)
        )
        selected_class = str(ai_res.selected_class or ai_res.prediction.raw_class)
        selected_class_id = (
            ai_res.selected_class_id
            if ai_res.selected_class_id is not None
            else (0 if selected_class == "artificial" else 1)
        )

        disagreement, disagreement_note = _evaluate_category_disagreement(
            known_category=sample["known_category"],
            selected_class=selected_class,
            expected_category=sample.get("expected_category"),
        )

        forensic_signals: Optional[Dict[str, Any]] = None
        cross_analysis_note: Optional[str] = None
        if not skip_forensics:
            forensic_signals = _run_optional_classical_forensics(img_path, pre_bytes)
            cross_analysis_note = forensic_signals.get("cross_analysis_note")
            after_forensics_bytes = img_path.read_bytes()
            if (
                hashlib.sha256(after_forensics_bytes).hexdigest() != pre_sha256
                or len(after_forensics_bytes) != pre_size
            ):
                raise ValidationHarnessError(
                    f"Dataset image '{img_path.name}' was modified after classical forensic check!"
                )

        per_image_results.append(
            {
                "sample_id": sample["sample_id"],
                "validation_id": sample["id"],
                "filename": img_path.name,
                "relative_path": sample["path"],
                "category": sample["category"],
                "known_category": sample["known_category"],
                "original_category": sample.get("original_category"),
                "expected_category": sample.get("expected_category"),
                "parent_sample_id": sample.get("parent_sample_id"),
                "parent_sha256": sample.get("parent_sha256"),
                "transformation_description": sample.get("transformation_description"),
                "source_description": sample["source_description"],
                "generator_or_camera_if_known": sample["generator_or_camera_if_known"],
                "provenance_status": sample["provenance_status"],
                "generator": sample["generator"],
                "transformations": sample["transformations"],
                "notes": sample["notes"],
                "image_format": sample["image_format"],
                "width": sample["width"],
                "height": sample["height"],
                "image_dimensions": sample["image_dimensions"],
                "byte_size": pre_size,
                "sha256": pre_sha256,
                "sha256_preserved": True,
                "detector_status": ai_res.status,
                "selected_class": selected_class,
                "selected_class_id": selected_class_id,
                "raw_class": ai_res.prediction.raw_class,
                "artificial_score": artificial_score,
                "human_score": human_score,
                "score_type": ai_res.prediction.score_type,
                "decision_rule": ai_res.prediction.decision_rule,
                "threshold": ai_res.prediction.threshold,
                "category_disagreement": disagreement,
                "disagreement_observation": disagreement_note,
                "cross_analysis_note": cross_analysis_note,
                "model_id": ai_res.detector.name,
                "model_repository": ai_res.detector.model_repository or ai_res.detector.name,
                "model_revision": ai_res.detector.pinned_revision,
                "pinned_revision": ai_res.detector.pinned_revision,
                "model_architecture": ai_res.detector.architecture,
                "preprocessing": ai_res.detector.preprocessing.model_dump(),
                "transformation_context": ai_res.transformation_context.model_dump(),
                "detector_limitations": list(ai_res.limitations or DEFAULT_AI_DETECTOR_LIMITATIONS),
                "pytorch_model_sha256": ai_res.detector.pytorch_model_sha256,
                "model_onnx_sha256": ai_res.detector.onnx_sha256,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "runtime_error": runtime_error,
                "mediatrace_forensic_signals": forensic_signals,
            }
        )

    summary = compute_descriptive_validation_summary(
        per_image_results=per_image_results,
        model_integrity=model_integrity,
        target_sample_counts=manifest_info["target_sample_counts"],
    )

    markdown_summary = _render_validation_markdown_summary(summary, per_image_results)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    results_json_path = resolved_output_dir / "validation_results.json"
    summary_json_path = resolved_output_dir / "validation_summary.json"
    summary_md_path = resolved_output_dir / "validation_summary.md"
    results_csv_path = resolved_output_dir / "validation_results.csv"

    results_payload = {
        "report_type": "descriptive validation results",
        "generated_at": summary["generated_at"],
        "manifest_schema_version": manifest_info["schema_version"],
        "dataset_complete": summary["dataset_complete"],
        "skip_forensics": skip_forensics,
        "model_integrity": model_integrity,
        "stratified_dataset_summary": summary["stratified_dataset_summary"],
        "sample_availability": summary["sample_availability"],
        "category_statistics": summary["category_statistics"],
        "confusion_count_table": summary["confusion_count_table"],
        "category_disagreements": summary["category_disagreements"],
        "dataset_quality_warnings": summary["dataset_quality_warnings"],
        "results": per_image_results,
        "limitations": list(VALIDATION_LIMITATIONS),
    }

    results_json_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.write_text(markdown_summary, encoding="utf-8")
    _write_validation_csv(results_csv_path, per_image_results)

    # If running on the default repository manifest and output_dir was custom (or storage/results),
    # also mirror into datasets/ai_detector_validation/results/ so Requirement 10 artifacts stay current.
    if resolved_manifest == default_manifest and resolved_output_dir != dataset_results_dir:
        dataset_results_dir.mkdir(parents=True, exist_ok=True)
        (dataset_results_dir / "validation_results.json").write_text(
            json.dumps(results_payload, indent=2), encoding="utf-8"
        )
        (dataset_results_dir / "validation_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (dataset_results_dir / "validation_summary.md").write_text(
            markdown_summary, encoding="utf-8"
        )
        _write_validation_csv(dataset_results_dir / "validation_results.csv", per_image_results)

    return {
        "results_payload": results_payload,
        "summary": summary,
        "markdown_summary": markdown_summary,
        "output_files": {
            "validation_results_json": str(results_json_path),
            "validation_summary_json": str(summary_json_path),
            "validation_summary_md": str(summary_md_path),
            "validation_results_csv": str(results_csv_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 14.1.3 / 14.1.7 — Controlled Multi-Image AI Detector Validation CLI"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to manifest.json (default: datasets/ai_detector_validation/manifest.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for validation output artifacts (default: datasets/ai_detector_validation/results)",
    )
    parser.add_argument(
        "--skip-forensics",
        action="store_true",
        help="Skip optional classical MediaTrace forensic stages (basic analysis, manipulation, localization)",
    )
    args = parser.parse_args()

    out = run_validation_harness(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        skip_forensics=args.skip_forensics,
    )
    summary = out["summary"]
    mi = summary["model_integrity"]
    avail = summary["sample_availability"]

    print("========================================================================")
    print("MediaTrace AI Image Detector — Controlled Validation Harness (14.1.7)")
    print("========================================================================")
    print(f"report_type      : {summary['report_type']}")
    print(f"dataset_complete : {summary['dataset_complete']}")
    print(f"model_repository : {mi['model_repository']}")
    print(f"pinned_revision  : {mi['pinned_revision']}")
    print(f"architecture     : {mi['architecture']}")
    print(f"pytorch_sha256   : {mi['pytorch_model_sha256']}")
    print(f"model_onnx_sha256: {mi['model_onnx_sha256']}")
    print("------------------------------------------------------------------------")
    print(
        f"Samples available: human={avail['human_available']}/{avail['human_target']}, "
        f"ai_generated={avail['ai_generated_available']}/{avail['ai_generated_target']}, "
        f"transformed={avail['transformed_available']}/{avail['transformed_target']}"
    )
    print(f"Availability note: {avail['availability_notice']}")
    print("------------------------------------------------------------------------")
    for item in out["results_payload"]["results"]:
        print(
            f"[{item['sample_id']}] ({item['known_category']}) {item['filename']} | "
            f"selected={item['selected_class']} | "
            f"artificial={item['artificial_score']:.4f} | human={item['human_score']:.4f} | "
            f"disagreement={item['category_disagreement']}"
        )
    print("------------------------------------------------------------------------")
    print("Output files written:")
    for k, v in out["output_files"].items():
        print(f"  {k}: {v}")
    print("========================================================================")


if __name__ == "__main__":
    main()
