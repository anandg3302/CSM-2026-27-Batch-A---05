from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict

from app.core.config import PROJECT_ROOT, settings

MEDIATRACE_APPLICATION_NAME: str = "MediaTrace"
MEDIATRACE_APPLICATION_VERSION: str = "1.0.0"

SCHEMA_VERSIONS: Dict[str, str] = {
    "case_schema_version": "1.3",
    "snapshot_schema_version": "1.0",
    "evidence_schema_version": "1.0",
    "interpretation_schema_version": "1.0",
    "report_schema_version": "1.3",
    "audit_event_schema_version": "1.0",
}

SERVICE_VERSIONS: Dict[str, str] = {
    "image_analysis_service": "1.0.0",
    "image_forensics_service": "1.0.0",
    "image_detection_service": "1.4.1",
    "image_manipulation_service": "1.4.1",
    "image_ai_detection_service": "1.4.1",
    "ai_image_detection_service": "1.4.1",
    "image_localization_service": "1.0.0",
    "image_comparison_service": "1.0.0",
    "evidence_service": "1.0.0",
    "evidence_correlation_service": "1.0.0",
    "forensic_decision_service": "1.4.2",
    "history_reconstruction_service": "1.0.0",
    "report_service": "1.4.3",
    "pdf_report_service": "1.4.3",
    "report_generation_service": "1.4.3",
    "video_analysis_service": "1.0.0",
    "audio_analysis_service": "1.0.0",
    "document_analysis_service": "1.0.0",
}

_FORBIDDEN_CONFIG_KEY_SUBSTRINGS = (
    "secret",
    "token",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "credential",
    "auth",
    "private_key",
    "timestamp",
    "created_at",
    "generated_at",
    "evaluated_at",
    "analyzed_at",
    "localized_at",
    "compared_at",
    "started_at",
    "completed_at",
)

_WINDOWS_ABS_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']*")
_UNIX_USER_PATH_RE = re.compile(r"/(?:Users|home)/[^\s\"'/]+(?:/[^\s\"']*)*")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api_key|apikey|access_token|auth_token|bearer)\s*[:=]\s*([^\s,;\"']+)"
)


def normalize_storage_path(raw_path: Path | str | None, base_dir: Any = None) -> str:
    """
    Convert any filesystem path into a privacy-safe relative path (e.g. 'storage/uploads/...')
    without exposing developer usernames or machine-specific root directories.
    """
    if raw_path is None:
        return ""
    s = str(raw_path).strip()
    if not s:
        return ""

    try:
        p = Path(s)
        if p.is_absolute():
            for base_prefix, rel_prefix in (
                (settings.storage_root.resolve(), "storage"),
                (PROJECT_ROOT.resolve(), ""),
            ):
                try:
                    rel = p.resolve().relative_to(base_prefix)
                    rel_posix = rel.as_posix()
                    return f"{rel_prefix}/{rel_posix}".strip("/") if rel_prefix else rel_posix
                except ValueError:
                    pass
            # If outside workspace root, strip drive/user hierarchy and keep only storage tail or basename
            posix_str = p.as_posix()
            if "/storage/" in posix_str:
                return "storage/" + posix_str.split("/storage/", 1)[1]
            return f"storage/{p.name}"
    except Exception:
        pass

    norm = s.replace("\\", "/")
    if "/storage/" in norm:
        return "storage/" + norm.split("/storage/", 1)[1]
    norm = _WINDOWS_ABS_RE.sub(lambda m: "storage/" + Path(m.group(0).replace("\\", "/")).name, norm)
    norm = _UNIX_USER_PATH_RE.sub(lambda m: "storage/" + Path(m.group(0)).name, norm)
    return norm.lstrip("/")


def sanitize_audit_text(text: str | None) -> str:
    """
    Sanitize descriptions, warnings, and exception messages before persisting in audit records.
    Removes secrets, tokens, passwords, and absolute developer paths.
    """
    if not text:
        return ""
    cleaned = _SECRET_ASSIGNMENT_RE.sub(r"\1=[REDACTED]", str(text))
    cleaned = _WINDOWS_ABS_RE.sub(
        lambda m: normalize_storage_path(m.group(0)), cleaned
    )
    cleaned = _UNIX_USER_PATH_RE.sub(
        lambda m: normalize_storage_path(m.group(0)), cleaned
    )
    return cleaned


def _canonicalize_config_value(val: Any) -> Any:
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return round(float(val), 8)
    if isinstance(val, (str, Path)):
        s = str(val).strip()
        if _WINDOWS_ABS_RE.search(s) or _UNIX_USER_PATH_RE.search(s) or "\\" in s:
            return normalize_storage_path(s)
        return sanitize_audit_text(s)
    if isinstance(val, (list, tuple)):
        return [_canonicalize_config_value(item) for item in val]
    if isinstance(val, dict):
        out: Dict[str, Any] = {}
        for k in sorted(val.keys(), key=lambda x: str(x)):
            k_str = str(k).strip()
            k_lower = k_str.lower()
            if any(sub in k_lower for sub in _FORBIDDEN_CONFIG_KEY_SUBSTRINGS):
                continue
            if k_lower.endswith("_dir") or k_lower.endswith("_path") or k_lower == "storage_root":
                continue
            out[k_str] = _canonicalize_config_value(val[k])
        return out
    return str(val)


def canonicalize_analysis_configuration(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized, secret-free, path-independent configuration dictionary."""
    res = _canonicalize_config_value(config_dict)
    return res if isinstance(res, dict) else {}


def sanitize_dict_for_audit(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize a dictionary for audit logging (strips absolute user paths and secrets)."""
    if not isinstance(data, dict):
        return {}
    return canonicalize_analysis_configuration(data)


def compute_configuration_hash(
    service_or_config: Any,
    config_dict: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a deterministic SHA-256 hex digest over normalized analysis configuration."""
    if isinstance(service_or_config, str):
        base_cfg = (
            dict(config_dict)
            if isinstance(config_dict, dict)
            else get_default_analysis_configuration(service_or_config)
        )
        base_cfg.setdefault("service_name", service_or_config)
        canonical_dict = canonicalize_analysis_configuration(base_cfg)
    elif isinstance(service_or_config, dict):
        canonical_dict = canonicalize_analysis_configuration(service_or_config)
    else:
        canonical_dict = {}
    serialized = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def get_default_analysis_configuration(analysis_type: str) -> Dict[str, Any]:
    """Return the canonical configuration parameters for a given MediaTrace analysis stage."""
    if analysis_type in {"image_forensics", "image_analysis_service", "image_forensics_service"}:
        return {
            "analysis_type": "image_forensics",
            "exif_extraction": True,
            "gps_redaction": "strict_presence_only",
            "jpeg_quantization_inspection": True,
        }
    if analysis_type in {
        "image_detection",
        "ai_image_detection",
        "image_detection_service",
        "image_manipulation_service",
        "image_ai_detection_service",
        "ai_image_detection_service",
    }:
        return {
            "analysis_type": analysis_type,
            "ai_detector_enabled": bool(settings.ai_image_detector_enabled),
            "model_repository": "umm-maybe/AI-image-detector",
            "model_revision": "c7e223baf11bc40528af364ba7bdea030ef42f9e",
            "architecture": "SwinForImageClassification",
            "class_mapping": {"0": "artificial", "1": "human"},
            "preprocessing": {
                "image_mode": "RGB",
                "resize": "224x224",
                "resample": "BICUBIC",
                "value_scaling": "[0, 255] -> [0.0, 1.0]",
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "layout": "NCHW float32",
            },
            "score_type": "uncalibrated_model_score",
            "decision_rule": "model_class_selection",
            "threshold": None,
            "ela_recompression_quality": 90,
        }
    if analysis_type in {"image_localization", "image_localization_service"}:
        return {
            "analysis_type": "image_localization",
            "method": "classical_localization_baseline",
            "ela_quality": 90,
            "anomaly_threshold": 0.65,
            "generates_heatmap": True,
            "generates_mask": True,
        }
    if analysis_type in {"image_comparison", "image_comparison_service"}:
        return {
            "analysis_type": "image_comparison",
            "difference_threshold": 10,
            "generates_difference_image": True,
            "generates_difference_mask": True,
        }
    if analysis_type in {"evidence_correlation", "evidence_service", "evidence_correlation_service"}:
        return {
            "analysis_type": "evidence_correlation",
            "min_spatial_overlap_pct": 10.0,
            "strength_levels": ["informational", "weak", "moderate"],
        }
    if analysis_type in {"forensic_interpretation", "forensic_decision_service"}:
        return {
            "analysis_type": "forensic_interpretation",
            "framework_version": "14.2",
            "ai_detector_role": "auxiliary_model_signal",
            "numeric_confidence_enabled": False,
            "threshold": None,
        }
    if analysis_type in {"history_reconstruction", "history_reconstruction_service"}:
        return {
            "analysis_type": "history_reconstruction",
            "sequence_type": "evidence_ordered_hypothesis",
        }
    if analysis_type in {"report_generation", "report_service", "pdf_report_service", "report_generation_service"}:
        return {
            "analysis_type": "report_generation",
            "report_schema_version": SCHEMA_VERSIONS["report_schema_version"],
            "includes_multi_signal_interpretation": True,
            "includes_reproducibility_audit_trail": True,
        }
    return {"analysis_type": analysis_type}

