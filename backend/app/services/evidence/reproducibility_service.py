"""Step 14.3 — Forensic Case Reproducibility & Audit Trail Service.

Provides:
- Input file provenance & cryptographic SHA-256 verification (without ever
  overwriting the original recorded SHA-256 on mismatch).
- Analysis execution registry with rerun/reuse & configuration-change tracking.
- Artifact registry with versioned immutable archive copies so reruns never
  silently overwrite historical artifact SHA-256 records.
- Evidence dependency graph linking input bytes -> analyses -> artifacts ->
  evidence items -> multi-signal interpretation -> forensic report.
- Append-only system audit event log (`AUD-YYYYMMDD-XXXXXXXX`), strictly separated
  from Step 8 media edit history reconstruction.
- Immutable ForensicAnalysisSnapshot (`MTS-YYYYMMDD-XXXXXXXX`) with canonical
  SHA-256 digest (`snapshot_sha256`).
- Non-destructive Reproducibility Check (`reproducible`, `partially_reproducible`,
  `not_reproducible`, `cannot_determine`) with full backward compatibility for
  pre-Step-14.3 legacy cases (`historical_analysis_registry = "unavailable"`).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.core.config import settings
from app.core.version import (
    MEDIATRACE_APPLICATION_NAME,
    MEDIATRACE_APPLICATION_VERSION,
    SCHEMA_VERSIONS,
    SERVICE_VERSIONS,
    compute_configuration_hash,
    get_default_analysis_configuration,
    normalize_storage_path,
    sanitize_audit_text,
    sanitize_dict_for_audit,
)
from app.schemas.media import MediaCaseMetadata
from app.schemas.reproducibility import (
    DEFAULT_REPRODUCIBILITY_LIMITATIONS,
    AiModelProvenanceRecord,
    AnalysisEnvironmentInfo,
    AnalysisRecord,
    ArtifactRecord,
    EvidenceDependencyNode,
    ForensicAnalysisSnapshot,
    InputFileProvenance,
    InputIntegrityVerification,
    ReportGenerationAuditRecord,
    ReproducibilityCheckResult,
    SystemAuditEvent,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _date_part() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _generate_prefixed_id(prefix: str) -> str:
    return f"{prefix}-{_date_part()}-{uuid.uuid4().hex[:8].upper()}"


def _compute_file_sha256_and_size(file_path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    return hasher.hexdigest(), size


_FILE_SHA256_CACHE: Dict[tuple[str, int, int], str] = {}


def _get_cached_file_sha256(file_path: Path) -> str:
    stat = file_path.stat()
    cache_key = (str(file_path.resolve()), stat.st_mtime_ns, stat.st_size)
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    sha, _ = _compute_file_sha256_and_size(file_path)
    _FILE_SHA256_CACHE[cache_key] = sha
    return sha


def get_analysis_environment_info() -> AnalysisEnvironmentInfo:
    """Build privacy-safe deterministic execution environment metadata."""
    onnx_ver: Optional[str] = None
    try:
        import onnxruntime as ort
        onnx_ver = getattr(ort, "__version__", None)
    except Exception:
        onnx_ver = None

    pil_ver: Optional[str] = None
    try:
        import PIL
        pil_ver = getattr(PIL, "__version__", None)
    except Exception:
        pil_ver = None

    device = "cpu"
    if (settings.ai_image_detector_device or "cpu").strip().lower() == "cuda":
        device = "cuda"

    return AnalysisEnvironmentInfo(
        application_name=MEDIATRACE_APPLICATION_NAME,
        application_version=MEDIATRACE_APPLICATION_VERSION,
        python_version=platform.python_version(),
        platform_category=platform.system().lower() or "unknown",
        execution_device=device,
        onnxruntime_version=onnx_ver,
        pillow_version=pil_ver,
        schema_versions=dict(SCHEMA_VERSIONS),
        service_versions=dict(SERVICE_VERSIONS),
    )


def ensure_case_audit_structures(
    case_data: MediaCaseMetadata,
    initialize_analysis_registry: bool = False,
) -> None:
    """Initialize Step 14.3 registries on a case without fabricating historical analysis registries for legacy cases."""
    if initialize_analysis_registry:
        if not case_data.case_schema_version:
            case_data.case_schema_version = SCHEMA_VERSIONS["case_schema_version"]
        if case_data.analysis_registry is None:
            case_data.analysis_registry = []
    if case_data.artifact_registry is None:
        case_data.artifact_registry = []
    if case_data.audit_events is None:
        case_data.audit_events = []
    if case_data.snapshots is None:
        case_data.snapshots = []
    if case_data.report_generations is None:
        case_data.report_generations = []


def append_audit_event(
    case_data: MediaCaseMetadata,
    event_type: str,
    summary: str,
    analysis_id: Optional[str] = None,
    artifact_id: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    report_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> SystemAuditEvent:
    """Append an immutable system audit event (`AUD-YYYYMMDD-XXXXXXXX`) to the case."""
    ensure_case_audit_structures(case_data, initialize_analysis_registry=False)
    safe_summary = sanitize_audit_text(summary) or event_type
    safe_details = sanitize_dict_for_audit(details or {})

    event = SystemAuditEvent(
        event_id=_generate_prefixed_id("AUD"),
        case_id=case_data.case_id,
        event_type=event_type,  # type: ignore[arg-type]
        event_category="system_audit_event",
        timestamp=_utc_now_iso(),
        summary=safe_summary,
        analysis_id=analysis_id,
        artifact_id=artifact_id,
        snapshot_id=snapshot_id,
        report_id=report_id,
        details=safe_details,
    )
    case_data.audit_events.append(event)  # type: ignore[union-attr]
    return event


def _get_case_relative_storage_path(case_data: MediaCaseMetadata) -> str:
    raw = getattr(case_data, "storage_path", None)
    if raw:
        return normalize_storage_path(raw, settings.storage_root)
    return f"storage/uploads/{case_data.stored_filename}"


def _resolve_storage_relative_path(relative_path: str) -> Path:
    """
    Resolve a normalized relative path ('storage/uploads/...', 'storage/results/...', 'storage/reports/...')
    against the active configured storage directories (`settings.uploads_dir`, `settings.results_dir`,
    `settings.reports_dir`), supporting both production and isolated test storage directories.
    """
    clean_rel = normalize_storage_path(relative_path, settings.storage_root).replace("\\", "/")
    rel_after = clean_rel[len("storage/"):] if clean_rel.startswith("storage/") else clean_rel
    if rel_after.startswith("uploads/"):
        return settings.uploads_dir / rel_after[len("uploads/"):]
    if rel_after.startswith("results/"):
        return settings.results_dir / rel_after[len("results/"):]
    if rel_after.startswith("reports/"):
        return settings.reports_dir / rel_after[len("reports/"):]
    if rel_after.startswith("models/"):
        return settings.models_dir / rel_after[len("models/"):]
    candidate = settings.storage_root / rel_after
    if candidate.exists():
        return candidate
    return settings.storage_root / clean_rel


def verify_input_file_integrity(
    case_data: MediaCaseMetadata,
    record_event_on_mismatch: bool = True,
) -> InputIntegrityVerification:
    """
    Recompute SHA-256 and byte size of the stored input file and compare against
    the immutable `case_data.sha256` and `case_data.file_size` recorded at upload.

    CRITICAL RULE: Never overwrite `case_data.sha256` or `case_data.file_size`.
    """
    rel_path = _get_case_relative_storage_path(case_data)
    abs_path = settings.uploads_dir / case_data.stored_filename
    checked_at = _utc_now_iso()

    if not abs_path.is_file():
        if record_event_on_mismatch:
            append_audit_event(
                case_data=case_data,
                event_type="integrity_mismatch_detected",
                summary=f"Stored original input file '{rel_path}' is unavailable on disk.",
                details={
                    "original_sha256": case_data.sha256,
                    "recomputed_sha256": None,
                    "status": "unavailable",
                },
            )
        return InputIntegrityVerification(
            status="unavailable",
            original_sha256=case_data.sha256,
            recomputed_sha256=None,
            original_byte_size=case_data.file_size,
            recomputed_byte_size=None,
            checked_at=checked_at,
            hash_match=False,
            byte_size_match=False,
            detail=(
                f"Input integrity unavailable: stored file at '{rel_path}' could not be found."
            ),
        )

    recomputed_sha256, recomputed_size = _compute_file_sha256_and_size(abs_path)
    hash_match = recomputed_sha256.lower() == case_data.sha256.lower()
    size_match = recomputed_size == case_data.file_size

    if hash_match and size_match:
        return InputIntegrityVerification(
            status="verified",
            original_sha256=case_data.sha256,
            recomputed_sha256=recomputed_sha256,
            original_byte_size=case_data.file_size,
            recomputed_byte_size=recomputed_size,
            checked_at=checked_at,
            hash_match=True,
            byte_size_match=True,
            detail=(
                "Stored original file bytes match the SHA-256 digest and byte size recorded at upload."
            ),
        )

    # Mismatch detected — NEVER overwrite case_data.sha256!
    if record_event_on_mismatch:
        already_logged = any(
            ev.event_type == "integrity_mismatch_detected"
            and ev.details.get("recomputed_sha256") == recomputed_sha256
            for ev in (case_data.audit_events or [])
        )
        if not already_logged:
            append_audit_event(
                case_data=case_data,
                event_type="integrity_mismatch_detected",
                summary=(
                    "Cryptographic SHA-256 mismatch detected between stored file bytes and original upload SHA-256."
                ),
                details={
                    "original_sha256": case_data.sha256,
                    "recomputed_sha256": recomputed_sha256,
                    "original_byte_size": case_data.file_size,
                    "recomputed_byte_size": recomputed_size,
                    "status": "mismatch",
                },
            )

    return InputIntegrityVerification(
        status="mismatch",
        original_sha256=case_data.sha256,
        recomputed_sha256=recomputed_sha256,
        original_byte_size=case_data.file_size,
        recomputed_byte_size=recomputed_size,
        checked_at=checked_at,
        hash_match=hash_match,
        byte_size_match=size_match,
        detail=(
            "INTEGRITY MISMATCH: Stored file SHA-256 or byte size differs from the original upload record. "
            "Original SHA-256 has been preserved unchanged."
        ),
    )


def register_artifact(
    case_data: MediaCaseMetadata,
    artifact_type: str,
    relative_path: str,
    media_type: str,
    source_analysis_id: str,
    source_service: str,
) -> Optional[ArtifactRecord]:
    """
    Register a generated forensic artifact (`MTA-YYYYMMDD-XXXXXXXX`) in `case_data.artifact_registry`.

    To guarantee artifact immutability across reruns (Sections 12 & 13):
    - Copies the artifact bytes to an immutable archive file under
      `storage/results/<case_id>/artifacts/<artifact_id>_<filename>`.
    - If a prior active ArtifactRecord exists for the same `relative_path`, preserves
      the prior record in `artifact_registry` and sets its `superseded_by_artifact_id`
      to the newly created `artifact_id`.
    """
    ensure_case_audit_structures(case_data, initialize_analysis_registry=False)
    clean_rel_path = normalize_storage_path(relative_path, settings.storage_root)
    abs_path = _resolve_storage_relative_path(clean_rel_path)
    if not abs_path.is_file():
        return None

    sha256_hex, byte_size = _compute_file_sha256_and_size(abs_path)

    # Check if an active record for this relative_path already exists
    prior_active: Optional[ArtifactRecord] = None
    for existing in case_data.artifact_registry or []:
        if existing.relative_path == clean_rel_path and existing.superseded_by_artifact_id is None:
            prior_active = existing
            break

    if (
        prior_active is not None
        and prior_active.sha256.lower() == sha256_hex.lower()
        and prior_active.source_analysis_id == source_analysis_id
    ):
        return prior_active

    artifact_id = _generate_prefixed_id("MTA")
    archive_dir = settings.results_dir / case_data.case_id / "artifacts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_filename = f"{artifact_id}_{abs_path.name}"
    archive_abs_path = archive_dir / archive_filename
    shutil.copy2(abs_path, archive_abs_path)
    archive_rel_path = f"storage/results/{case_data.case_id}/artifacts/{archive_filename}"

    record = ArtifactRecord(
        artifact_id=artifact_id,
        case_id=case_data.case_id,
        artifact_type=artifact_type,  # type: ignore[arg-type]
        relative_path=clean_rel_path,
        immutable_archive_path=archive_rel_path,
        media_type=media_type,
        sha256=sha256_hex,
        derived_artifact_sha256=sha256_hex,
        byte_size=byte_size,
        created_at=_utc_now_iso(),
        source_analysis_id=source_analysis_id,
        source_service=source_service,
        superseded_by_artifact_id=None,
    )

    if prior_active is not None:
        prior_active.superseded_by_artifact_id = artifact_id

    case_data.artifact_registry.append(record)  # type: ignore[union-attr]
    append_audit_event(
        case_data=case_data,
        event_type="artifact_created",
        summary=f"Registered derived forensic artifact '{clean_rel_path}' ({artifact_type}).",
        analysis_id=source_analysis_id,
        artifact_id=artifact_id,
        details={
            "artifact_type": artifact_type,
            "relative_path": clean_rel_path,
            "immutable_archive_path": archive_rel_path,
            "sha256": sha256_hex,
            "byte_size": byte_size,
            "supersedes_artifact_id": prior_active.artifact_id if prior_active else None,
        },
    )
    return record


def extract_ai_model_provenance(case_data: MediaCaseMetadata) -> AiModelProvenanceRecord:
    """
    Extract AI detector model provenance from the case's image (Step 14.1) or video (Stage A)
    AI detection result and verify pinned revision & model file hashes.
    """
    if case_data.media_type == "video" or case_data.video_analysis is not None:
        v_model_dir = settings.models_dir / "Vansh180--VideoMae-ffc23-deepfake-detector"
        v_safetensors = v_model_dir / "model.safetensors"
        v_config = v_model_dir / "config.json"
        v_preproc = v_model_dir / "preprocessor_config.json"

        v_artifact_hashes: Dict[str, str] = {}
        for p in (v_safetensors, v_config, v_preproc):
            if p.is_file():
                try:
                    v_artifact_hashes[p.name] = _get_cached_file_sha256(p)
                except Exception:
                    pass

        v_ai = (
            case_data.video_analysis.ai_generation
            if case_data.video_analysis is not None
            else None
        )
        if v_ai is not None and v_ai.artifact_hashes:
            for k, val in v_ai.artifact_hashes.items():
                v_artifact_hashes.setdefault(k, val)

        v_weights_sha = v_artifact_hashes.get("model.safetensors")
        v_preproc_params = (
            dict(v_ai.preprocessing)
            if (v_ai is not None and v_ai.preprocessing)
            else {
                "processor_class": "VideoMAEImageProcessor",
                "num_frames": 16,
                "size": {"shortest_edge": 224},
                "crop_size": {"height": 224, "width": 224},
                "color_mode": "RGB",
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            }
        )
        is_completed = bool(v_ai is not None and v_ai.status == "completed")
        return AiModelProvenanceRecord(
            available=is_completed,
            status=v_ai.status if v_ai is not None else "not_available",
            model_name=(v_ai.model_name if (v_ai and v_ai.model_name) else settings.ai_video_detector_model_id)
            or "Vansh180/VideoMae-ffc23-deepfake-detector",
            pinned_revision=(v_ai.model_version if (v_ai and v_ai.model_version) else settings.ai_video_detector_model_version)
            or "ea26518b527d938d66e568c52bcf1c19a86cf295",
            architecture=(v_ai.architecture if (v_ai and v_ai.architecture) else "VideoMAEForVideoClassification"),
            license=(v_ai.license if (v_ai and v_ai.license) else "MIT"),
            training_domain=(v_ai.training_domain if (v_ai and v_ai.training_domain) else "FaceForensics++ C23"),
            framework="safetensors_numpy_runtime" if is_completed else None,
            device=settings.ai_video_detector_device or "cpu",
            weights_sha256=v_weights_sha,
            onnx_sha256=None,
            artifact_hashes=v_artifact_hashes,
            class_mapping=dict(v_ai.id2label) if (v_ai and v_ai.id2label) else {"0": "real", "1": "fake"},
            preprocessing_parameters=v_preproc_params,
            score_type="uncalibrated_model_score",
            decision_rule="model_class_selection",
            threshold=None,
            selected_class=v_ai.selected_class if is_completed else None,
            real_score=v_ai.real_score if is_completed else None,
            fake_score=v_ai.fake_score if is_completed else None,
            frames_sampled=v_ai.frames_sampled if v_ai else None,
            usable_detector_frames=v_ai.usable_detector_frames if v_ai else None,
            detector_scope=v_ai.detector_scope if v_ai else "facial_video_deepfake_detection",
        )

    model_dir = settings.models_dir / "umm-maybe--AI-image-detector"
    config_path = model_dir / "config.json"
    pt_path = model_dir / "pytorch_model.bin"
    onnx_path = model_dir / "model.onnx"

    config_data: Dict[str, Any] = {}
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config_data = {}

    pt_sha256: Optional[str] = config_data.get("pytorch_model_sha256")
    if pt_path.is_file():
        try:
            pt_sha256 = _get_cached_file_sha256(pt_path)
        except Exception:
            pass

    onnx_sha256: Optional[str] = config_data.get("onnx_sha256")
    if onnx_path.is_file():
        try:
            onnx_sha256 = _get_cached_file_sha256(onnx_path)
        except Exception:
            pass

    ai_res = (
        case_data.manipulation_detection.ai_generation_detection
        if case_data.manipulation_detection is not None
        else None
    )

    if ai_res is None or ai_res.status != "completed":
        return AiModelProvenanceRecord(
            available=False,
            status=ai_res.status if ai_res is not None else "not_available",
            model_name=settings.ai_image_detector_model_id or "umm-maybe/AI-image-detector",
            pinned_revision=settings.ai_image_detector_model_version
            or "c7e223baf11bc40528af364ba7bdea030ef42f9e",
            architecture="SwinForImageClassification",
            framework=None,
            device=settings.ai_image_detector_device or "cpu",
            weights_sha256=pt_sha256,
            onnx_sha256=onnx_sha256,
            class_mapping={"0": "artificial", "1": "human"},
            preprocessing_parameters={
                "color_mode": "RGB",
                "target_size": [224, 224],
                "interpolation": "BICUBIC",
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
            },
            score_type="uncalibrated_model_score",
            decision_rule="model_class_selection",
            threshold=None,
        )

    det = ai_res.detector
    pred = ai_res.prediction
    preproc = ai_res.preprocessing

    return AiModelProvenanceRecord(
        available=True,
        status=ai_res.status,
        model_name=det.name if det else "umm-maybe/AI-image-detector",
        pinned_revision=det.version if det else "c7e223baf11bc40528af364ba7bdea030ef42f9e",
        architecture=det.architecture if det else "SwinForImageClassification",
        framework=det.framework if det else "onnxruntime",
        device=det.device if det else "cpu",
        weights_sha256=pt_sha256,
        onnx_sha256=onnx_sha256,
        class_mapping={"0": "artificial", "1": "human"},
        preprocessing_parameters={
            "color_mode": preproc.color_mode if preproc else "RGB",
            "target_size": list(preproc.target_size) if (preproc and preproc.target_size) else [224, 224],
            "interpolation": preproc.interpolation if preproc else "BICUBIC",
            "image_mean": list(preproc.image_mean) if (preproc and preproc.image_mean) else [0.485, 0.456, 0.406],
            "image_std": list(preproc.image_std) if (preproc and preproc.image_std) else [0.229, 0.224, 0.225],
        },
        score_type="uncalibrated_model_score",
        decision_rule="model_class_selection",
        threshold=None,
        selected_class=pred.label if pred else None,
        artificial_score=pred.synthetic_score if pred else None,
        human_score=pred.real_score if pred else None,
    )


def record_analysis_execution(
    case_data: MediaCaseMetadata,
    service_name: str,
    status: str,
    input_sha256: str,
    output_summary: str,
    configuration: Optional[Dict[str, Any]] = None,
    model_provenance: Optional[AiModelProvenanceRecord] = None,
    artifact_specs: Optional[List[Dict[str, str]]] = None,
    evidence_ids: Optional[List[str]] = None,
    was_reused: bool = False,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    duration_ms: Optional[float] = None,
) -> AnalysisRecord:
    """
    Record an analysis stage execution (`ANL-YYYYMMDD-XXXXXXXX`) in `case_data.analysis_registry`,
    register any derived artifacts, and append corresponding audit events.
    """
    ensure_case_audit_structures(case_data, initialize_analysis_registry=True)
    now_iso = _utc_now_iso()
    start_ts = started_at or now_iso
    end_ts = completed_at or now_iso

    config_dict = (
        sanitize_dict_for_audit(configuration)
        if configuration is not None
        else get_default_analysis_configuration(service_name)
    )
    config_hash = compute_configuration_hash(service_name, config_dict)
    service_version = SERVICE_VERSIONS.get(service_name, "1.0.0")

    prior_records = [
        r for r in (case_data.analysis_registry or []) if r.service_name == service_name
    ]
    latest_prior = prior_records[-1] if prior_records else None

    if was_reused and latest_prior is not None:
        analysis_id = _generate_prefixed_id("ANL")
        record = AnalysisRecord(
            analysis_id=analysis_id,
            service_name=service_name,
            service_version=service_version,
            status="reused",
            started_at=start_ts,
            completed_at=end_ts,
            duration_ms=duration_ms or 0.0,
            input_sha256=input_sha256,
            configuration=config_dict,
            configuration_hash=config_hash,
            model_provenance=model_provenance or latest_prior.model_provenance,
            output_summary=sanitize_audit_text(output_summary),
            artifact_ids=list(latest_prior.artifact_ids),
            evidence_ids=list(evidence_ids or latest_prior.evidence_ids),
            was_reused=True,
            reused_analysis_id=latest_prior.analysis_id,
            rerun_of_analysis_id=None,
            configuration_changed=False,
            model_changed=False,
        )
        case_data.analysis_registry.append(record)  # type: ignore[union-attr]
        append_audit_event(
            case_data=case_data,
            event_type="analysis_reused",
            summary=f"Reused cached result for '{service_name}' from {latest_prior.analysis_id}.",
            analysis_id=analysis_id,
            details={
                "service_name": service_name,
                "reused_analysis_id": latest_prior.analysis_id,
                "configuration_hash": config_hash,
            },
        )
        return record

    analysis_id = _generate_prefixed_id("ANL")
    rerun_of_id = latest_prior.analysis_id if latest_prior is not None else None
    config_changed = (
        latest_prior is not None and latest_prior.configuration_hash != config_hash
    )
    model_changed = False
    if latest_prior is not None and latest_prior.model_provenance and model_provenance:
        model_changed = (
            latest_prior.model_provenance.model_name != model_provenance.model_name
            or latest_prior.model_provenance.pinned_revision != model_provenance.pinned_revision
            or latest_prior.model_provenance.onnx_sha256 != model_provenance.onnx_sha256
        )

    append_audit_event(
        case_data=case_data,
        event_type="analysis_started",
        summary=f"Started '{service_name}' (v{service_version}).",
        analysis_id=analysis_id,
        details={
            "service_name": service_name,
            "service_version": service_version,
            "configuration_hash": config_hash,
            "input_sha256": input_sha256,
        },
    )

    registered_artifact_ids: List[str] = []
    for spec in artifact_specs or []:
        art_rec = register_artifact(
            case_data=case_data,
            artifact_type=spec["artifact_type"],
            relative_path=spec["relative_path"],
            media_type=spec.get("media_type", "image/png"),
            source_analysis_id=analysis_id,
            source_service=service_name,
        )
        if art_rec is not None:
            registered_artifact_ids.append(art_rec.artifact_id)

    record = AnalysisRecord(
        analysis_id=analysis_id,
        service_name=service_name,
        service_version=service_version,
        status=status,  # type: ignore[arg-type]
        started_at=start_ts,
        completed_at=end_ts,
        duration_ms=duration_ms,
        input_sha256=input_sha256,
        configuration=config_dict,
        configuration_hash=config_hash,
        model_provenance=model_provenance,
        output_summary=sanitize_audit_text(output_summary),
        artifact_ids=registered_artifact_ids,
        evidence_ids=list(evidence_ids or []),
        was_reused=False,
        reused_analysis_id=None,
        rerun_of_analysis_id=rerun_of_id,
        configuration_changed=config_changed,
        model_changed=model_changed,
    )
    case_data.analysis_registry.append(record)  # type: ignore[union-attr]

    if rerun_of_id is not None:
        append_audit_event(
            case_data=case_data,
            event_type="analysis_rerun",
            summary=(
                f"Reran '{service_name}' ({analysis_id} supersedes {rerun_of_id}; "
                f"configuration_changed={config_changed}, model_changed={model_changed})."
            ),
            analysis_id=analysis_id,
            details={
                "service_name": service_name,
                "rerun_of_analysis_id": rerun_of_id,
                "configuration_changed": config_changed,
                "model_changed": model_changed,
                "configuration_hash": config_hash,
            },
        )
    else:
        append_audit_event(
            case_data=case_data,
            event_type="analysis_completed" if status == "completed" else "analysis_failed",
            summary=f"Completed '{service_name}' with status '{status}'.",
            analysis_id=analysis_id,
            details={
                "service_name": service_name,
                "status": status,
                "artifact_ids": registered_artifact_ids,
                "configuration_hash": config_hash,
            },
        )

    return record


def build_evidence_dependency_graph(case_data: MediaCaseMetadata) -> List[EvidenceDependencyNode]:
    """
    Build the explicit Evidence Dependency Graph tracing:
    input_bytes -> analysis_execution -> derived_artifact -> evidence_item ->
    forensic_interpretation -> forensic_report.
    """
    nodes: List[EvidenceDependencyNode] = []
    input_node_id = f"INP-{case_data.case_id}"
    nodes.append(
        EvidenceDependencyNode(
            node_id=input_node_id,
            node_type="input_bytes",
            label=f"Original Input Bytes ({case_data.original_filename})",
            source_service="media_upload_service",
            sha256=case_data.sha256,
            depends_on=[],
        )
    )

    service_to_latest_analysis_id: Dict[str, str] = {}
    for anl in case_data.analysis_registry or []:
        service_to_latest_analysis_id[anl.service_name] = anl.analysis_id
        nodes.append(
            EvidenceDependencyNode(
                node_id=anl.analysis_id,
                node_type="analysis_execution",
                label=f"{anl.service_name} (v{anl.service_version})",
                source_service=anl.service_name,
                sha256=anl.configuration_hash,
                depends_on=[input_node_id],
            )
        )

    active_artifact_ids: List[str] = []
    for art in case_data.artifact_registry or []:
        if art.superseded_by_artifact_id is None:
            active_artifact_ids.append(art.artifact_id)
        nodes.append(
            EvidenceDependencyNode(
                node_id=art.artifact_id,
                node_type="derived_artifact",
                label=f"Artifact: {art.artifact_type} ({art.relative_path})",
                source_service=art.source_service,
                sha256=art.sha256,
                depends_on=[art.source_analysis_id] if art.source_analysis_id else [input_node_id],
            )
        )

    evidence_node_ids: List[str] = []
    if case_data.evidence and case_data.evidence.items:
        for ev in case_data.evidence.items:
            ev_id = ev.id
            evidence_node_ids.append(ev_id)
            deps: List[str] = [input_node_id]
            src_lower = (ev.source or "").lower()
            if "metadata" in src_lower or "image_analysis" in src_lower or "step3" in src_lower:
                if "image_analysis_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_analysis_service"])
            if "manipulation" in src_lower or "compression" in src_lower or "step4" in src_lower:
                if "image_detection_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_detection_service"])
            if "ai" in src_lower or "detector" in src_lower:
                if "image_ai_detection_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_ai_detection_service"])
                elif "image_detection_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_detection_service"])
            if "localization" in src_lower or "step5" in src_lower:
                if "image_localization_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_localization_service"])
            if "comparison" in src_lower or "step6" in src_lower:
                if "image_comparison_service" in service_to_latest_analysis_id:
                    deps.append(service_to_latest_analysis_id["image_comparison_service"])
            if "evidence_service" in service_to_latest_analysis_id:
                deps.append(service_to_latest_analysis_id["evidence_service"])
            nodes.append(
                EvidenceDependencyNode(
                    node_id=ev_id,
                    node_type="evidence_item",
                    label=f"{ev.id}: {ev.title}",
                    source_service=ev.source,
                    sha256=None,
                    depends_on=sorted(set(deps)),
                )
            )

    interp_node_id: Optional[str] = None
    if case_data.forensic_decision is not None:
        interp_node_id = f"INT-{case_data.case_id}"
        interp_deps = list(evidence_node_ids) + list(service_to_latest_analysis_id.values())
        if not interp_deps:
            interp_deps = [input_node_id]
        nodes.append(
            EvidenceDependencyNode(
                node_id=interp_node_id,
                node_type="forensic_interpretation",
                label=(
                    f"Multi-Signal Forensic Interpretation ({case_data.forensic_decision.overall_assessment})"
                ),
                source_service="forensic_decision_service",
                sha256=None,
                depends_on=sorted(set(interp_deps)),
            )
        )

    for rep in case_data.report_generations or []:
        rep_deps = [input_node_id] + list(evidence_node_ids) + list(active_artifact_ids)
        if interp_node_id:
            rep_deps.append(interp_node_id)
        nodes.append(
            EvidenceDependencyNode(
                node_id=rep.report_id,
                node_type="forensic_report",
                label=f"Forensic Report {rep.report_id} ({rep.export_format.upper()})",
                source_service="report_service",
                sha256=rep.report_sha256,
                depends_on=sorted(set(rep_deps)),
            )
        )

    return nodes


def evaluate_case_reproducibility(
    case_data: MediaCaseMetadata,
    record_audit_event: bool = False,
) -> ReproducibilityCheckResult:
    """
    Non-destructively verify all 7 forensic reproducibility conditions (Section 18).
    Never modifies or regenerates any artifacts or case outputs.
    """
    checked_at = _utc_now_iso()
    reasons: List[str] = []
    notes: List[str] = []

    # 1 & 2. Input file existence & SHA-256 verification
    integrity = verify_input_file_integrity(
        case_data, record_event_on_mismatch=record_audit_event
    )
    input_file_present = integrity.status != "unavailable"
    input_sha256_match = integrity.status == "verified" and integrity.hash_match

    if not input_file_present:
        reasons.append("Original uploaded input file is missing from storage.")
    elif not input_sha256_match:
        reasons.append(
            "Stored input file SHA-256 or byte size does not match the original upload record."
        )

    # 3. Historical analysis registry check (Backward compatibility for pre-Step-14.3 cases)
    has_legacy_analyses = any(
        [
            case_data.analysis is not None,
            case_data.manipulation_detection is not None,
            case_data.localization is not None,
            case_data.comparison is not None,
            case_data.evidence is not None,
            case_data.video_analysis is not None,
            case_data.audio_analysis is not None,
            case_data.document_analysis is not None,
            case_data.history is not None,
        ]
    )
    if case_data.analysis_registry is None:
        historical_registry_status = "unavailable"
        analysis_records_present = False
        if has_legacy_analyses:
            reasons.append(
                "Case was analyzed prior to Step 14.3; historical analysis registry is unavailable."
            )
            notes.append(
                "Legacy pre-Step-14.3 case detected: historical execution records were not fabricated."
            )
        else:
            reasons.append("No analysis records have been registered for this case yet.")
    else:
        historical_registry_status = "available"
        analysis_records_present = len(case_data.analysis_registry) > 0
        if not analysis_records_present:
            reasons.append("Analysis registry is initialized but contains no completed analysis records.")

    # 4. Configuration hashes match current deterministic defaults
    config_hash_match = True
    if case_data.analysis_registry:
        latest_by_service: Dict[str, AnalysisRecord] = {}
        for rec in case_data.analysis_registry:
            latest_by_service[rec.service_name] = rec
        for svc_name, rec in latest_by_service.items():
            expected_cfg = get_default_analysis_configuration(svc_name)
            expected_hash = compute_configuration_hash(svc_name, expected_cfg)
            if rec.configuration_hash != expected_hash:
                config_hash_match = False
                reasons.append(
                    f"Configuration hash for '{svc_name}' ({rec.configuration_hash[:12]}...) differs "
                    f"from current default configuration ({expected_hash[:12]}...)."
                )

    # 5. AI model reference and weights/ONNX SHA-256 verification
    model_reference_match = True
    ai_prov = extract_ai_model_provenance(case_data)
    if ai_prov.available:
        expected_model_id = (settings.ai_image_detector_model_id or "").strip()
        expected_rev = (settings.ai_image_detector_model_version or "").strip()
        if ai_prov.model_name != expected_model_id or ai_prov.pinned_revision != expected_rev:
            model_reference_match = False
            reasons.append(
                f"Recorded AI detector ({ai_prov.model_name}@{ai_prov.pinned_revision}) differs from "
                f"currently configured model ({expected_model_id}@{expected_rev})."
            )
        model_dir = settings.models_dir / "umm-maybe--AI-image-detector"
        config_path = model_dir / "config.json"
        if config_path.is_file():
            try:
                cfg = json.loads(config_path.read_text(encoding="utf-8"))
                expected_onnx_sha = cfg.get("onnx_sha256")
                if (
                    expected_onnx_sha
                    and ai_prov.onnx_sha256
                    and ai_prov.onnx_sha256.lower() != str(expected_onnx_sha).lower()
                ):
                    model_reference_match = False
                    reasons.append("AI detector ONNX artifact SHA-256 does not match pinned config.json hash.")
            except Exception:
                pass

    # 6. Artifacts present on disk & SHA-256 hashes match
    artifacts_present = True
    artifact_hashes_match = True

    active_artifacts = [
        art
        for art in (case_data.artifact_registry or [])
        if art.superseded_by_artifact_id is None
    ]
    for art in active_artifacts:
        abs_art_path = _resolve_storage_relative_path(art.relative_path)
        if not abs_art_path.is_file():
            artifacts_present = False
            artifact_hashes_match = False
            reasons.append(f"Registered artifact '{art.relative_path}' ({art.artifact_id}) is missing from disk.")
        else:
            actual_sha, actual_size = _compute_file_sha256_and_size(abs_art_path)
            if actual_sha.lower() != art.sha256.lower() or actual_size != art.byte_size:
                artifact_hashes_match = False
                reasons.append(
                    f"Artifact integrity mismatch for '{art.relative_path}' ({art.artifact_id}): "
                    f"on-disk SHA-256 ({actual_sha[:12]}...) differs from recorded SHA-256 ({art.sha256[:12]}...)."
                )

    # Also verify any legacy artifact paths referenced in localization/comparison if artifact_registry is empty
    if not active_artifacts:
        candidate_rel_paths: List[str] = []
        if case_data.localization and case_data.localization.visualization:
            vis_loc = case_data.localization.visualization
            if getattr(vis_loc, "heatmap", None):
                candidate_rel_paths.append(f"storage/results/{case_data.case_id}/{vis_loc.heatmap}")
            if getattr(vis_loc, "mask", None):
                candidate_rel_paths.append(f"storage/results/{case_data.case_id}/{vis_loc.mask}")
        if case_data.comparison and case_data.comparison.visualization:
            vis_comp = case_data.comparison.visualization
            if getattr(vis_comp, "difference", None):
                candidate_rel_paths.append(f"storage/results/{case_data.case_id}/{vis_comp.difference}")
            if getattr(vis_comp, "mask", None):
                candidate_rel_paths.append(f"storage/results/{case_data.case_id}/{vis_comp.mask}")
        for rel_p in candidate_rel_paths:
            clean_p = normalize_storage_path(rel_p, settings.storage_root)
            if not _resolve_storage_relative_path(clean_p).is_file():
                artifacts_present = False
                reasons.append(f"Referenced case artifact '{clean_p}' is missing from disk.")

    # 7. Schema and service versions match current codebase versions
    schema_versions_match = True
    if case_data.case_schema_version and case_data.case_schema_version != SCHEMA_VERSIONS["case_schema_version"]:
        schema_versions_match = False
        reasons.append(
            f"Case schema version '{case_data.case_schema_version}' differs from current "
            f"'{SCHEMA_VERSIONS['case_schema_version']}'."
        )

    service_versions_match = True
    for rec in case_data.analysis_registry or []:
        expected_ver = SERVICE_VERSIONS.get(rec.service_name)
        if expected_ver and rec.service_version != expected_ver:
            service_versions_match = False
            reasons.append(
                f"Service '{rec.service_name}' version '{rec.service_version}' differs from current '{expected_ver}'."
            )

    # Determine overall status conservatively
    if not input_file_present or not input_sha256_match or not artifacts_present or not artifact_hashes_match:
        status = "not_reproducible"
    elif historical_registry_status == "unavailable":
        status = "partially_reproducible" if has_legacy_analyses else "cannot_determine"
    elif not analysis_records_present:
        status = "cannot_determine"
    elif (
        not config_hash_match
        or not model_reference_match
        or not schema_versions_match
        or not service_versions_match
    ):
        status = "partially_reproducible"
    else:
        status = "reproducible"
        notes.append(
            "All input bytes, analysis execution records, configuration hashes, AI model references, "
            "and derived artifact SHA-256 digests match."
        )

    result = ReproducibilityCheckResult(
        case_id=case_data.case_id,
        checked_at=checked_at,
        status=status,  # type: ignore[arg-type]
        input_file_present=input_file_present,
        input_sha256_match=input_sha256_match,
        analysis_records_present=analysis_records_present,
        historical_analysis_registry=historical_registry_status,  # type: ignore[arg-type]
        configuration_hash_match=config_hash_match,
        model_reference_match=model_reference_match,
        artifacts_present=artifacts_present,
        artifact_hashes_match=artifact_hashes_match,
        schema_versions_match=schema_versions_match,
        service_versions_match=service_versions_match,
        reasons=reasons,
        verification_notes=notes,
    )

    if record_audit_event:
        ensure_case_audit_structures(case_data, initialize_analysis_registry=False)
        append_audit_event(
            case_data=case_data,
            event_type="reproducibility_checked",
            summary=f"Executed non-destructive reproducibility verification: status='{status}'.",
            details={
                "status": status,
                "input_file_present": input_file_present,
                "input_sha256_match": input_sha256_match,
                "analysis_records_present": analysis_records_present,
                "historical_analysis_registry": historical_registry_status,
                "configuration_hash_match": config_hash_match,
                "model_reference_match": model_reference_match,
                "artifacts_present": artifacts_present,
                "artifact_hashes_match": artifact_hashes_match,
                "reasons": reasons,
            },
        )

    return result


def compute_snapshot_sha256(snapshot_dict: Dict[str, Any]) -> str:
    """
    Compute canonical SHA-256 of a ForensicAnalysisSnapshot dictionary with
    `snapshot_sha256` excluded (set to empty string).
    """
    payload = dict(snapshot_dict)
    payload["snapshot_sha256"] = ""
    canonical_bytes = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


def create_forensic_snapshot(
    case_data: MediaCaseMetadata,
    record_event: bool = True,
) -> ForensicAnalysisSnapshot:
    """
    Build an immutable ForensicAnalysisSnapshot (`MTS-YYYYMMDD-XXXXXXXX`) for `case_data`,
    compute its canonical `snapshot_sha256`, and update `case_data.latest_snapshot`.
    """
    ensure_case_audit_structures(case_data, initialize_analysis_registry=False)

    snapshot_id = _generate_prefixed_id("MTS")
    created_at = _utc_now_iso()
    clean_storage_path = _get_case_relative_storage_path(case_data)

    # Discover any generated report.json / MediaTrace_Forensic_Report.pdf on disk if not yet recorded
    report_json_file = settings.reports_dir / case_data.case_id / "report.json"
    if report_json_file.is_file():
        has_json_rec = any(r.export_format == "json" for r in (case_data.report_generations or []))
        if not has_json_rec:
            try:
                rep_data = json.loads(report_json_file.read_text(encoding="utf-8"))
                rep_id = rep_data.get("report_id", f"MTR-{_date_part()}-00000000")
            except Exception:
                rep_id = f"MTR-{_date_part()}-00000000"
            rep_sha, rep_size = _compute_file_sha256_and_size(report_json_file)
            register_artifact(
                case_data=case_data,
                artifact_type="report_json",
                relative_path=f"storage/reports/{case_data.case_id}/report.json",
                media_type="application/json",
                source_analysis_id=rep_id,
                source_service="report_service",
            )
            case_data.report_generations.append(  # type: ignore[union-attr]
                ReportGenerationAuditRecord(
                    report_id=rep_id,
                    case_id=case_data.case_id,
                    snapshot_id=snapshot_id,
                    report_schema_version=SCHEMA_VERSIONS["report_schema_version"],
                    generated_at=created_at,
                    export_format="json",
                    relative_path=f"storage/reports/{case_data.case_id}/report.json",
                    report_sha256=rep_sha,
                    byte_size=rep_size,
                )
            )
            append_audit_event(
                case_data=case_data,
                event_type="report_generated",
                summary=f"Generated forensic JSON report '{rep_id}'.",
                snapshot_id=snapshot_id,
                report_id=rep_id,
                details={
                    "report_id": rep_id,
                    "export_format": "json",
                    "relative_path": f"storage/reports/{case_data.case_id}/report.json",
                    "report_sha256": rep_sha,
                },
            )

    report_pdf_file = settings.reports_dir / case_data.case_id / "MediaTrace_Forensic_Report.pdf"
    if report_pdf_file.is_file():
        has_pdf_rec = any(r.export_format == "pdf" for r in (case_data.report_generations or []))
        if not has_pdf_rec:
            rep_id = (
                case_data.report_generations[0].report_id
                if case_data.report_generations
                else f"MTR-{_date_part()}-00000000"
            )
            pdf_sha, pdf_size = _compute_file_sha256_and_size(report_pdf_file)
            register_artifact(
                case_data=case_data,
                artifact_type="report_pdf",
                relative_path=f"storage/reports/{case_data.case_id}/MediaTrace_Forensic_Report.pdf",
                media_type="application/pdf",
                source_analysis_id=rep_id,
                source_service="pdf_report_service",
            )
            case_data.report_generations.append(  # type: ignore[union-attr]
                ReportGenerationAuditRecord(
                    report_id=rep_id,
                    case_id=case_data.case_id,
                    snapshot_id=snapshot_id,
                    report_schema_version=SCHEMA_VERSIONS["report_schema_version"],
                    generated_at=created_at,
                    export_format="pdf",
                    relative_path=f"storage/reports/{case_data.case_id}/MediaTrace_Forensic_Report.pdf",
                    report_sha256=pdf_sha,
                    byte_size=pdf_size,
                )
            )
            append_audit_event(
                case_data=case_data,
                event_type="report_pdf_generated",
                summary=f"Generated forensic PDF report '{rep_id}'.",
                snapshot_id=snapshot_id,
                report_id=rep_id,
                details={
                    "report_id": rep_id,
                    "export_format": "pdf",
                    "relative_path": f"storage/reports/{case_data.case_id}/MediaTrace_Forensic_Report.pdf",
                    "report_sha256": pdf_sha,
                },
            )

    input_provenance = InputFileProvenance(
        case_id=case_data.case_id,
        original_filename=case_data.original_filename,
        safe_filename=case_data.stored_filename,
        relative_storage_path=clean_storage_path,
        media_type=case_data.media_type,
        extension=case_data.extension,
        mime_type=getattr(case_data, "mime_type", None),
        byte_size=case_data.file_size,
        sha256=case_data.sha256,
        uploaded_at=case_data.created_at,
    )

    integrity = verify_input_file_integrity(case_data, record_event_on_mismatch=False)
    env_info = get_analysis_environment_info()
    ai_prov = extract_ai_model_provenance(case_data)
    dep_graph = build_evidence_dependency_graph(case_data)
    repro_check = evaluate_case_reproducibility(case_data, record_audit_event=False)

    evidence_ids: List[str] = []
    if case_data.evidence and case_data.evidence.items:
        evidence_ids = [ev.id for ev in case_data.evidence.items]

    interp_summary: Optional[Dict[str, Any]] = None
    if case_data.forensic_decision is not None:
        fd = case_data.forensic_decision
        tensions_count = len(
            [
                rel
                for rel in getattr(fd, "signal_relationships", [])
                if getattr(rel, "consistency", "") == "contradiction_or_tension"
            ]
        )
        interp_summary = {
            "overall_assessment": fd.overall_assessment,
            "overall_assessment_label": getattr(fd, "overall_assessment_label", ""),
            "interpretation": fd.interpretation,
            "assessment_rationale": fd.interpretation,
            "generated_at": fd.generated_at,
            "evaluated_at": fd.generated_at,
            "consistency_classification": getattr(fd, "consistency_classification", ""),
            "supported_conclusions_count": len(fd.supported_conclusions),
            "unsupported_conclusions_count": len(fd.unsupported_conclusions),
            "contradictions_or_tensions_count": tensions_count,
        }

    draft_snapshot = ForensicAnalysisSnapshot(
        snapshot_id=snapshot_id,
        snapshot_schema_version=SCHEMA_VERSIONS["snapshot_schema_version"],
        case_id=case_data.case_id,
        created_at=created_at,
        snapshot_sha256="0" * 64,
        input_file_provenance=input_provenance,
        input_integrity=integrity,
        environment_info=env_info,
        analyses_executed=list(case_data.analysis_registry or []),
        ai_model_provenance=ai_prov,
        artifacts_produced=list(case_data.artifact_registry or []),
        evidence_items_consumed=evidence_ids,
        evidence_dependency_graph=dep_graph,
        interpretation_summary=interp_summary,
        reproducibility_check=repro_check,
        audit_events=list(case_data.audit_events or []),
        reproducibility_limitations=list(DEFAULT_REPRODUCIBILITY_LIMITATIONS),
    )

    snapshot_dict = draft_snapshot.model_dump()
    canonical_sha = compute_snapshot_sha256(snapshot_dict)
    final_snapshot = draft_snapshot.model_copy(update={"snapshot_sha256": canonical_sha})

    case_data.latest_snapshot = final_snapshot
    case_data.snapshots.append(final_snapshot)  # type: ignore[union-attr]

    if record_event:
        append_audit_event(
            case_data=case_data,
            event_type="snapshot_created",
            summary=f"Created immutable forensic analysis snapshot '{snapshot_id}' (SHA-256: {canonical_sha[:12]}...).",
            snapshot_id=snapshot_id,
            details={
                "snapshot_id": snapshot_id,
                "snapshot_sha256": canonical_sha,
                "reproducibility_status": repro_check.status,
            },
        )

    return final_snapshot


def record_report_generation(
    case_data: MediaCaseMetadata,
    report_id: str,
    export_format: str,
    relative_path: str,
) -> Optional[ReportGenerationAuditRecord]:
    """
    Register a JSON or PDF forensic report generation event (`MTR-YYYYMMDD-XXXXXXXX`),
    archive the report artifact, and append an audit event.
    """
    ensure_case_audit_structures(case_data, initialize_analysis_registry=False)
    clean_rel = normalize_storage_path(relative_path, settings.storage_root)
    abs_path = _resolve_storage_relative_path(clean_rel)
    if not abs_path.is_file():
        return None

    rep_sha, rep_size = _compute_file_sha256_and_size(abs_path)
    snap_id = (
        case_data.latest_snapshot.snapshot_id
        if case_data.latest_snapshot is not None
        else create_forensic_snapshot(case_data, record_event=True).snapshot_id
    )

    art_type = "report_pdf" if export_format.lower() == "pdf" else "report_json"
    mime = "application/pdf" if export_format.lower() == "pdf" else "application/json"

    register_artifact(
        case_data=case_data,
        artifact_type=art_type,
        relative_path=clean_rel,
        media_type=mime,
        source_analysis_id=report_id,
        source_service="pdf_report_service" if export_format.lower() == "pdf" else "report_service",
    )

    rec = ReportGenerationAuditRecord(
        report_id=report_id,
        case_id=case_data.case_id,
        snapshot_id=snap_id,
        report_schema_version=SCHEMA_VERSIONS["report_schema_version"],
        generated_at=_utc_now_iso(),
        export_format=export_format.lower(),  # type: ignore[arg-type]
        relative_path=clean_rel,
        report_sha256=rep_sha,
        byte_size=rep_size,
    )
    case_data.report_generations.append(rec)  # type: ignore[union-attr]

    append_audit_event(
        case_data=case_data,
        event_type="report_pdf_generated" if export_format.lower() == "pdf" else "report_generated",
        summary=f"Generated forensic {export_format.upper()} report '{report_id}' linked to snapshot '{snap_id}'.",
        snapshot_id=snap_id,
        report_id=report_id,
        details={
            "report_id": report_id,
            "snapshot_id": snap_id,
            "export_format": export_format.lower(),
            "relative_path": clean_rel,
            "report_sha256": rep_sha,
            "byte_size": rep_size,
        },
    )
    return rec


def check_case_reproducibility(case_id: str) -> ReproducibilityCheckResult:
    """
    Endpoint-level non-destructive reproducibility check (`POST /api/forensics/{case_id}/reproducibility/check`).
    """
    from app.services.media_service import get_media_case, save_media_case

    case_data = get_media_case(case_id)
    result = evaluate_case_reproducibility(case_data, record_audit_event=True)
    create_forensic_snapshot(case_data, record_event=True)
    save_media_case(case_data)
    return result


def get_case_reproducibility_snapshot(
    case_id: str,
    refresh: bool = True,
) -> ForensicAnalysisSnapshot:
    """
    Endpoint-level snapshot & audit trail retrieval (`GET /api/forensics/{case_id}/reproducibility`).
    Re-evaluates live integrity/reproducibility checks without modifying artifacts.
    """
    from app.services.media_service import get_media_case, save_media_case

    case_data = get_media_case(case_id)
    if refresh or case_data.latest_snapshot is None:
        snapshot = create_forensic_snapshot(case_data, record_event=False)
        save_media_case(case_data)
        return snapshot
    return case_data.latest_snapshot
