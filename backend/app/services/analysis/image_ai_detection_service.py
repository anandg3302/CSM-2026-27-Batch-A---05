"""Step 14.1 / 14.1.1 — Real AI-Generated Image Detection Service.

Loads the pretrained vision classifier `umm-maybe/AI-image-detector`
(`SwinForImageClassification` backbone, pinned revision
`c7e223baf11bc40528af364ba7bdea030ef42f9e`, license `CC BY-ND 4.0`) from
`storage/models/` using safe model formats (`ONNX` via `onnxruntime` exported locally
from the pinned upstream PyTorch checkpoint, or `safetensors` via `transformers`
with `trust_remote_code=False` and `local_files_only=True`).

Forensic Guarantees:
- Never modifies, overwrites, resizes-on-disk, or recompresses the original uploaded image.
- Preprocessing occurs strictly on an in-memory tensor copy `(1, 3, 224, 224)` using
  exact `preprocessor_config.json` parameters (`RGB`, `224x224` Bicubic resize,
  mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`).
- Verifies SHA-256 and file size before and after inference.
- Treats model outputs as uncalibrated classifier scores (`uncalibrated_model_score`)
  with `decision_rule="model_class_selection"` (`threshold=None`), never as calibrated
  statistical probabilities or definitive proof of provenance.
- Rejects unpinned branch names (such as `main`) and returns `status="not_available"`
  gracefully when the model is disabled or not installed in `storage/models/`.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.ai_detection import (
    DEFAULT_AI_DETECTOR_LIMITATIONS,
    OFFICIAL_MODEL_ARCHITECTURE,
    OFFICIAL_MODEL_ID,
    OFFICIAL_MODEL_LICENSE,
    OFFICIAL_PINNED_REVISION,
    AiDetectorMetadata,
    AiDetectorPrediction,
    AiDetectorPreprocessingInfo,
    AiGenerationDetectionResult,
    AiTransformationContext,
    AllowedAiDecisionRule,
    AllowedAiPredictionLabel,
    build_ai_detector_interpretation,
)


logger = logging.getLogger("mediatrace")

OFFICIAL_IMAGE_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
OFFICIAL_IMAGE_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)
UNPINNED_BRANCH_NAMES = {"main", "master", "latest", "head", "default", "trunk"}
COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{7,40}$")

POSITIVE_SIGNAL_EXPLANATION = (
    "Potential AI-generation signal detected: the configured detector selected the "
    "'artificial' class. Model-derived score; not calibrated as a forensic probability. "
    "This output is one supporting forensic signal and does not independently establish image provenance."
)

NEGATIVE_OR_UNCERTAIN_EXPLANATION = (
    "No AI-generation signal detected by the configured detector (model-derived score; "
    "not calibrated as a forensic probability). Absence of an AI-generation signal does "
    "not prove that the image is human-created or authentic."
)

UNAVAILABLE_REASON = (
    "AI image detection model is not configured or could not be loaded. "
    "No trained AI-generation detection model is currently configured."
)

UNPINNED_REVISION_REASON = (
    "AI image detection model revision is not pinned to an immutable commit SHA. "
    "Unpinned branch names (such as 'main') are rejected for forensic reproducibility."
)

UNAVAILABLE_EXPLANATION = (
    "AI-generated image detection is currently unavailable because no compatible "
    "trained detector is configured."
)

SYNTHETIC_LABEL_KEYWORDS = ("artificial", "ai", "fake", "synthetic", "generated", "sdxl", "diffusion")


@dataclass
class _LoadedDetectorRuntime:
    model_id: str
    version: str
    architecture: str
    source: str
    framework: str
    device: str
    upstream_format: str
    runtime_format: str
    onnx_is_upstream: bool
    model_repository: str
    pytorch_model_sha256: Optional[str]
    onnx_sha256: Optional[str]
    exported_at: Optional[str]
    conversion_info: str
    input_width: int
    input_height: int
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]
    id2label: Dict[int, str]
    synthetic_class_index: int
    license_name: str
    predict_fn: Callable[[np.ndarray], np.ndarray]


_MODEL_CACHE: Dict[Tuple[str, str, str, str], _LoadedDetectorRuntime] = {}
_CUSTOM_PREDICTOR: Optional[Callable[[np.ndarray, Path], Dict[str, Any]]] = None


def clear_ai_detector_cache() -> None:
    """Clear in-memory cached detector sessions (used when configuration changes or in tests)."""
    _MODEL_CACHE.clear()


def set_custom_predictor_for_testing(
    predictor: Optional[Callable[[np.ndarray, Path], Dict[str, Any]]]
) -> None:
    """Allow unit tests to inject a deterministic model predictor without downloading multi-GB weights."""
    global _CUSTOM_PREDICTOR
    _CUSTOM_PREDICTOR = predictor


def _compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as file_in:
        while chunk := file_in.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_valid_pinned_revision(revision: str) -> bool:
    """Verify that the model revision is pinned to an immutable commit SHA and not a mutable branch."""
    cleaned = (revision or "").strip()
    if not cleaned:
        return False
    if cleaned.lower() in UNPINNED_BRANCH_NAMES:
        return False
    return bool(COMMIT_SHA_PATTERN.match(cleaned))


def _normalize_license_name(model_id: str, raw_license: Optional[str]) -> str:
    """Ensure the official CC BY-ND 4.0 license is reported for umm-maybe/AI-image-detector."""
    cleaned = (raw_license or "").strip()
    if model_id.strip().lower() == OFFICIAL_MODEL_ID.lower():
        return OFFICIAL_MODEL_LICENSE
    if not cleaned or cleaned.lower() in {"apache-2.0", "apache 2.0", "unknown"}:
        return OFFICIAL_MODEL_LICENSE
    return cleaned


def _sanitize_model_id_to_dirname(model_id: str) -> str:
    """Convert a model identifier like 'umm-maybe/AI-image-detector' into a safe directory slug."""
    cleaned = model_id.strip().replace("\\", "/").strip("/")
    if not cleaned or ".." in cleaned or cleaned.startswith("/"):
        raise ValueError("Invalid AI detector model identifier.")
    slug = cleaned.replace("/", "--")
    if not re.match(r"^[A-Za-z0-9_.\-]+$", slug):
        raise ValueError("Model identifier contains unsafe path characters.")
    return slug


def _resolve_safe_model_directory(model_id: str) -> Path:
    """Resolve and validate the model checkpoint folder inside settings.models_dir."""
    default_storage_root = (Path(__file__).resolve().parents[4] / "storage").resolve()
    if (
        settings.models_dir.resolve() == (default_storage_root / "models").resolve()
        and settings.storage_root.resolve() != default_storage_root
    ):
        models_root = (settings.storage_root.resolve() / "models").resolve()
    else:
        models_root = settings.models_dir.resolve()
    slug = _sanitize_model_id_to_dirname(model_id)
    candidate = (models_root / slug).resolve()
    try:
        candidate.relative_to(models_root)
    except ValueError as exc:
        raise ValueError("Model path escaped configured models directory.") from exc
    return candidate


def _softmax(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.array([0.5, 0.5], dtype=np.float64)
    if arr.size == 1:
        val = float(arr[0])
        prob_ai = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, val))))
        return np.array([prob_ai, 1.0 - prob_ai], dtype=np.float64)
    shifted = arr - np.max(arr)
    exps = np.exp(np.clip(shifted, -50.0, 50.0))
    total = float(np.sum(exps))
    if total <= 0.0:
        return np.full_like(arr, 1.0 / arr.size, dtype=np.float64)
    return exps / total


def _resolve_synthetic_class_index(id2label: Dict[int, str]) -> int:
    """Identify which output logit index corresponds to the AI/synthetic class."""
    for idx, label in id2label.items():
        lower = str(label).lower()
        if any(keyword in lower for keyword in SYNTHETIC_LABEL_KEYWORDS):
            return int(idx)
    return 0


def _build_preprocessing_metadata(
    input_width: int,
    input_height: int,
    image_mean: Tuple[float, float, float],
    image_std: Tuple[float, float, float],
    id2label: Dict[int, str],
) -> AiDetectorPreprocessingInfo:
    return AiDetectorPreprocessingInfo(
        image_mode="RGB",
        resize_dimensions=f"{input_width}x{input_height}",
        crop_behavior="none (direct resize to 224x224)",
        resample_filter="Image.Resampling.BICUBIC (resample=3)",
        value_scaling="[0, 255] uint8 -> [0.0, 1.0] float32",
        normalization_mean=[round(float(x), 6) for x in image_mean],
        normalization_std=[round(float(x), 6) for x in image_std],
        tensor_layout=f"NCHW (1, 3, {input_height}, {input_width}) float32",
        class_ordering={str(k): str(v) for k, v in sorted(id2label.items())},
    )


def _load_detector_runtime() -> Optional[_LoadedDetectorRuntime]:
    """Load and cache the configured ONNX or Safetensors AI image detector from storage/models/."""
    if not settings.ai_image_detector_enabled:
        return None

    model_id = (settings.ai_image_detector_model_id or "").strip()
    version = (settings.ai_image_detector_model_version or "").strip()
    requested_device = (settings.ai_image_detector_device or "cpu").strip().lower()
    if not model_id or not _is_valid_pinned_revision(version):
        logger.warning(
            "Rejected AI detector load because model_id='%s' or revision='%s' is not pinned.",
            model_id,
            version,
        )
        return None

    try:
        model_dir = _resolve_safe_model_directory(model_id)
    except ValueError as exc:
        logger.warning("Rejected unsafe AI image detector model_id '%s': %s", model_id, exc)
        return None

    cache_key = (str(model_dir), model_id, version, requested_device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not model_dir.is_dir():
        return None

    config_path = model_dir / "config.json"
    config_data: Dict[str, Any] = {}
    if config_path.is_file():
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse detector config.json in %s: %s", model_dir, exc)
            return None

    preproc_path = model_dir / "preprocessor_config.json"
    preproc_data: Dict[str, Any] = {}
    if preproc_path.is_file():
        try:
            preproc_data = json.loads(preproc_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse preprocessor_config.json in %s: %s", model_dir, exc)

    raw_id2label = config_data.get("id2label", {"0": "artificial", "1": "human"})
    id2label: Dict[int, str] = {}
    if isinstance(raw_id2label, dict):
        for k, v in raw_id2label.items():
            try:
                id2label[int(k)] = str(v)
            except ValueError:
                continue
    if not id2label:
        id2label = {0: "artificial", 1: "human"}

    if len(id2label) != 2 or id2label.get(0, "").lower() != "artificial" or id2label.get(1, "").lower() != "human":
        logger.warning("Rejected detector in %s due to invalid class ordering: %s", model_dir, id2label)
        return None

    synthetic_idx = _resolve_synthetic_class_index(id2label)
    if model_id.lower() == OFFICIAL_MODEL_ID.lower():
        architecture = OFFICIAL_MODEL_ARCHITECTURE
    else:
        architecture = str(
            config_data.get("architectures", [OFFICIAL_MODEL_ARCHITECTURE])[0]
            if isinstance(config_data.get("architectures"), list) and config_data.get("architectures")
            else config_data.get("architecture", OFFICIAL_MODEL_ARCHITECTURE)
        )

    source_url = str(
        config_data.get("source", f"https://huggingface.co/{model_id}/tree/{version}")
    )
    license_name = _normalize_license_name(model_id, config_data.get("license"))
    input_size = int(preproc_data.get("size", config_data.get("image_size", 224)))

    if model_id.lower() == OFFICIAL_MODEL_ID.lower():
        image_mean = OFFICIAL_IMAGE_MEAN
        image_std = OFFICIAL_IMAGE_STD
    else:
        mean_vals = preproc_data.get("image_mean", config_data.get("image_mean", list(OFFICIAL_IMAGE_MEAN)))
        std_vals = preproc_data.get("image_std", config_data.get("image_std", list(OFFICIAL_IMAGE_STD)))
        image_mean = (float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2]))
        image_std = (float(std_vals[0]), float(std_vals[1]), float(std_vals[2]))

    recorded_pt_sha = config_data.get("pytorch_model_sha256")
    pt_bin_path = model_dir / "pytorch_model.bin"
    actual_pt_sha: Optional[str] = recorded_pt_sha
    if pt_bin_path.is_file() and recorded_pt_sha:
        actual_pt_sha = _compute_file_sha256(pt_bin_path)
        if actual_pt_sha.lower() != str(recorded_pt_sha).lower():
            logger.warning(
                "Rejected detector: pytorch_model.bin SHA-256 mismatch (%s != %s).",
                actual_pt_sha,
                recorded_pt_sha,
            )
            return None

    onnx_path = model_dir / "model.onnx"
    if onnx_path.is_file():
        try:
            actual_onnx_sha = _compute_file_sha256(onnx_path)
            recorded_onnx_sha = config_data.get("onnx_sha256")
            if recorded_onnx_sha and actual_onnx_sha.lower() != str(recorded_onnx_sha).lower():
                logger.warning(
                    "Rejected detector: model.onnx SHA-256 mismatch (%s != %s).",
                    actual_onnx_sha,
                    recorded_onnx_sha,
                )
                return None

            import onnxruntime as ort

            available_providers = ort.get_available_providers()
            if requested_device == "cuda" and "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                active_device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                active_device = "cpu"

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=providers,
            )
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                logger.warning("Rejected ONNX model with %d inputs and %d outputs.", len(inputs), len(outputs))
                return None

            inp = inputs[0]
            out = outputs[0]
            inp_shape = list(inp.shape)
            out_shape = list(out.shape)
            if inp.type != "tensor(float)":
                logger.warning("Rejected ONNX model with unsupported input dtype '%s'.", inp.type)
                return None
            if len(inp_shape) != 4 or inp_shape[1] != 3 or inp_shape[2] != input_size or inp_shape[3] != input_size:
                logger.warning("Rejected ONNX model with incompatible input shape %s.", inp_shape)
                return None
            if len(out_shape) != 2 or out_shape[1] != 2:
                logger.warning("Rejected ONNX model with incompatible output shape %s (expected 2 classes).", out_shape)
                return None

            input_name = inp.name

            def _run_onnx(tensor_nchw: np.ndarray) -> np.ndarray:
                outputs_arr = session.run(None, {input_name: tensor_nchw})
                return np.asarray(outputs_arr[0])

            runtime = _LoadedDetectorRuntime(
                model_id=model_id,
                version=version,
                architecture=architecture,
                source=source_url,
                framework="onnxruntime",
                device=active_device,
                upstream_format=f"PyTorch (pytorch_model.bin, {architecture})",
                runtime_format="ONNX (model.onnx via ONNX Runtime)",
                onnx_is_upstream=False,
                model_repository=str(config_data.get("model_repository", model_id)),
                pytorch_model_sha256=actual_pt_sha,
                onnx_sha256=actual_onnx_sha,
                exported_at=config_data.get("exported_at"),
                conversion_info=(
                    f"upstream PyTorch weights (pytorch_model.bin, {architecture}) -> "
                    "MediaTrace local ONNX export (opset 17) -> ONNX Runtime inference"
                ),
                input_width=input_size,
                input_height=input_size,
                image_mean=image_mean,
                image_std=image_std,
                id2label=id2label,
                synthetic_class_index=synthetic_idx,
                license_name=license_name,
                predict_fn=_run_onnx,
            )
            _MODEL_CACHE[cache_key] = runtime
            return runtime
        except Exception as exc:
            logger.warning("Failed to initialize ONNX Runtime session from %s: %s", onnx_path, exc)
            return None

    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.is_file():
        try:
            import torch
            from transformers import AutoModelForImageClassification

            active_device = (
                "cuda"
                if requested_device == "cuda" and torch.cuda.is_available()
                else "cpu"
            )
            hf_model = AutoModelForImageClassification.from_pretrained(
                str(model_dir),
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            hf_model.to(active_device)
            hf_model.eval()

            def _run_torch(tensor_nchw: np.ndarray) -> np.ndarray:
                with torch.inference_mode():
                    inp = torch.from_numpy(tensor_nchw).to(active_device)
                    out = hf_model(pixel_values=inp)
                    return out.logits.detach().cpu().numpy()

            runtime = _LoadedDetectorRuntime(
                model_id=model_id,
                version=version,
                architecture=architecture,
                source=source_url,
                framework="transformers+torch",
                device=active_device,
                upstream_format=f"PyTorch (pytorch_model.bin, {architecture})",
                runtime_format="Safetensors (model.safetensors via transformers+torch)",
                onnx_is_upstream=False,
                model_repository=str(config_data.get("model_repository", model_id)),
                pytorch_model_sha256=actual_pt_sha,
                onnx_sha256=None,
                exported_at=config_data.get("exported_at"),
                conversion_info=(
                    f"upstream PyTorch weights (pytorch_model.bin, {architecture}) -> "
                    "local Safetensors serialization -> transformers+torch inference"
                ),
                input_width=input_size,
                input_height=input_size,
                image_mean=image_mean,
                image_std=image_std,
                id2label=id2label,
                synthetic_class_index=synthetic_idx,
                license_name=license_name,
                predict_fn=_run_torch,
            )
            _MODEL_CACHE[cache_key] = runtime
            return runtime
        except Exception as exc:
            logger.warning("Failed to initialize safetensors model from %s: %s", safetensors_path, exc)
            return None

    return None


def _preprocess_image_in_memory(
    raw_bytes: bytes,
    target_width: int = 224,
    target_height: int = 224,
    image_mean: Tuple[float, float, float] = OFFICIAL_IMAGE_MEAN,
    image_std: Tuple[float, float, float] = OFFICIAL_IMAGE_STD,
) -> np.ndarray:
    """Decode image from memory and return normalized float32 NCHW tensor (1, 3, H, W)
    aligned with official `preprocessor_config.json` (RGB, 224x224 Bicubic, ImageNet mean/std).
    """
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize((target_width, target_height), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype=np.float32) / 255.0

    mean = np.array(image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(image_std, dtype=np.float32).reshape(1, 1, 3)
    normalized = (arr - mean) / np.where(std == 0.0, 1.0, std)
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def _interpret_model_score(
    score: float,
    threshold: Optional[float],
    raw_class: str,
) -> Tuple[AllowedAiPredictionLabel, str, str, AllowedAiDecisionRule, Optional[float]]:
    """Apply conservative classification rules without claiming an unvalidated forensic threshold."""
    if threshold is not None and (0.0 < threshold < 1.0):
        decision_rule: AllowedAiDecisionRule = "configured_score_threshold"
        if abs(score - threshold) < 0.05:
            return (
                "undetermined",
                "Undetermined",
                NEGATIVE_OR_UNCERTAIN_EXPLANATION,
                decision_rule,
                threshold,
            )
        if score >= threshold:
            return (
                "potentially_ai_generated",
                "Potential AI-generation signal",
                POSITIVE_SIGNAL_EXPLANATION,
                decision_rule,
                threshold,
            )
        return (
            "no_ai_signal_detected",
            "No AI-generation signal detected",
            NEGATIVE_OR_UNCERTAIN_EXPLANATION,
            decision_rule,
            threshold,
        )

    # Default when threshold is None: use model_class_selection (argmax over model logits/scores)
    decision_rule = "model_class_selection"
    raw_lower = (raw_class or "").strip().lower()
    is_synthetic = any(keyword in raw_lower for keyword in SYNTHETIC_LABEL_KEYWORDS)
    if is_synthetic:
        return (
            "potentially_ai_generated",
            "Potential AI-generation signal",
            POSITIVE_SIGNAL_EXPLANATION,
            decision_rule,
            None,
        )

    return (
        "no_ai_signal_detected",
        "No AI-generation signal detected",
        NEGATIVE_OR_UNCERTAIN_EXPLANATION,
        decision_rule,
        None,
    )


def detect_ai_generated_image(
    image_path: Path,
    expected_sha256: Optional[str] = None,
) -> AiGenerationDetectionResult:
    """Run real AI-generated image detection on the original uploaded file without modifying it."""
    if not image_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Stored image file for this case is missing on disk.",
        )

    pre_bytes = image_path.read_bytes()
    pre_size = len(pre_bytes)
    pre_sha256 = hashlib.sha256(pre_bytes).hexdigest()

    if expected_sha256 and pre_sha256.lower() != expected_sha256.lower():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File integrity check failed: stored image hash does not match case SHA-256.",
        )

    # Validate that the file is a decodable image before running model or fallback
    try:
        with Image.open(io.BytesIO(pre_bytes)) as verify_img:
            verify_img.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is corrupted or is not a readable image container.",
        ) from exc

    configured_version = (settings.ai_image_detector_model_version or "").strip()
    if settings.ai_image_detector_enabled and not _is_valid_pinned_revision(configured_version):
        tx_ctx = AiTransformationContext()
        return AiGenerationDetectionResult(
            status="not_available",
            modality="image",
            reason=UNPINNED_REVISION_REASON,
            detector=None,
            prediction=None,
            explanation=UNPINNED_REVISION_REASON,
            limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
            transformation_context=tx_ctx,
            ai_detector_interpretation=build_ai_detector_interpretation(
                status="not_available",
                limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
                transformation_context=tx_ctx,
            ),
        )

    threshold = settings.ai_image_detector_threshold

    # Path 1: Custom injected predictor (for unit tests without requiring multi-GB downloads)
    if _CUSTOM_PREDICTOR is not None and settings.ai_image_detector_enabled:
        tensor = _preprocess_image_in_memory(pre_bytes)
        pred_info = _CUSTOM_PREDICTOR(tensor, image_path)
        post_bytes = image_path.read_bytes()
        post_sha256 = hashlib.sha256(post_bytes).hexdigest()
        if post_sha256 != pre_sha256 or len(post_bytes) != pre_size:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Integrity violation: original image file changed during AI detection.",
            )

        score = round(float(pred_info["score"]), 4)
        human_score = round(float(pred_info.get("human_score", max(0.0, 1.0 - score))), 4)
        eff_threshold = pred_info.get("threshold", threshold)
        default_raw_class = (
            "artificial"
            if (eff_threshold is not None and score >= eff_threshold) or (eff_threshold is None and score >= 0.5)
            else "human"
        )
        raw_class = str(pred_info.get("raw_class", default_raw_class)).strip().lower()
        selected_class = "artificial" if raw_class == "artificial" else "human"
        selected_class_id = 0 if selected_class == "artificial" else 1
        label, display_label, explanation, decision_rule, reported_threshold = _interpret_model_score(
            score, eff_threshold, raw_class
        )
        model_name = str(pred_info.get("name", settings.ai_image_detector_model_id))
        pinned_rev = str(pred_info.get("version", configured_version or OFFICIAL_PINNED_REVISION))
        arch = str(pred_info.get("architecture", OFFICIAL_MODEL_ARCHITECTURE))
        lic = _normalize_license_name(model_name, pred_info.get("license"))
        tx_ctx = AiTransformationContext()
        interp = build_ai_detector_interpretation(
            status="available",
            selected_class=selected_class,
            selected_class_id=selected_class_id,
            artificial_score=score,
            human_score=human_score,
            decision_rule=decision_rule,
            threshold=reported_threshold,
            limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
            transformation_context=tx_ctx,
        )

        return AiGenerationDetectionResult(
            status="available",
            modality="image",
            reason=explanation,
            detector=AiDetectorMetadata(
                name=model_name,
                version=pinned_rev,
                pinned_revision=pinned_rev,
                architecture=arch,
                source=str(pred_info.get("source", f"https://huggingface.co/{model_name}/tree/{pinned_rev}")),
                framework=str(pred_info.get("framework", "onnxruntime")),
                device=str(pred_info.get("device", settings.ai_image_detector_device)),
                upstream_format=f"PyTorch (pytorch_model.bin, {arch})",
                runtime_format="ONNX (model.onnx via ONNX Runtime)",
                onnx_is_upstream=False,
                model_repository=model_name,
                pytorch_model_sha256=pred_info.get("pytorch_model_sha256"),
                onnx_sha256=pred_info.get("onnx_sha256"),
                exported_at=pred_info.get("exported_at"),
                conversion_info=(
                    f"upstream PyTorch weights (pytorch_model.bin, {arch}) -> "
                    "MediaTrace local ONNX export (opset 17) -> ONNX Runtime inference"
                ),
                input_resolution="224x224",
                preprocessing=_build_preprocessing_metadata(
                    224, 224, OFFICIAL_IMAGE_MEAN, OFFICIAL_IMAGE_STD, {0: "artificial", 1: "human"}
                ),
                license=lic,
            ),
            prediction=AiDetectorPrediction(
                label=label,
                display_label=display_label,
                raw_class=raw_class,
                selected_class=selected_class,
                selected_class_id=selected_class_id,
                score=score,
                artificial_score=score,
                human_score=human_score,
                score_type="uncalibrated_model_score",
                decision_rule=decision_rule,
                threshold=reported_threshold,
            ),
            selected_class=selected_class,
            selected_class_id=selected_class_id,
            artificial_score=score,
            human_score=human_score,
            explanation=explanation,
            limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
            transformation_context=tx_ctx,
            ai_detector_interpretation=interp,
        )

    # Path 2: Real ONNX Runtime / Safetensors pretrained model execution
    runtime = _load_detector_runtime()
    if runtime is None:
        tx_ctx = AiTransformationContext()
        return AiGenerationDetectionResult(
            status="not_available",
            modality="image",
            reason=UNAVAILABLE_REASON,
            detector=None,
            prediction=None,
            explanation=UNAVAILABLE_EXPLANATION,
            limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
            transformation_context=tx_ctx,
            ai_detector_interpretation=build_ai_detector_interpretation(
                status="not_available",
                limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
                transformation_context=tx_ctx,
            ),
        )

    try:
        tensor_nchw = _preprocess_image_in_memory(
            pre_bytes,
            target_width=runtime.input_width,
            target_height=runtime.input_height,
            image_mean=runtime.image_mean,
            image_std=runtime.image_std,
        )
        logits = runtime.predict_fn(tensor_nchw)
        probs = _softmax(logits)
        syn_idx = runtime.synthetic_class_index if runtime.synthetic_class_index < probs.size else 0
        hum_idx = 1 - syn_idx if probs.size >= 2 else 1
        score = round(float(probs[syn_idx]), 4)
        human_score = round(float(probs[hum_idx]) if hum_idx < probs.size else max(0.0, 1.0 - score), 4)
        pred_idx = int(np.argmax(probs))
        raw_class = str(runtime.id2label.get(pred_idx, "artificial" if pred_idx == syn_idx else "human")).strip().lower()
        selected_class = "artificial" if raw_class == "artificial" else "human"
        selected_class_id = 0 if selected_class == "artificial" else 1
    except Exception as exc:
        logger.error("AI image detector inference failed: %s", type(exc).__name__)
        tx_ctx = AiTransformationContext()
        return AiGenerationDetectionResult(
            status="not_available",
            modality="image",
            reason=UNAVAILABLE_REASON,
            detector=None,
            prediction=None,
            explanation=UNAVAILABLE_EXPLANATION,
            limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
            transformation_context=tx_ctx,
            ai_detector_interpretation=build_ai_detector_interpretation(
                status="not_available",
                limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
                transformation_context=tx_ctx,
            ),
        )

    post_bytes = image_path.read_bytes()
    post_sha256 = hashlib.sha256(post_bytes).hexdigest()
    if post_sha256 != pre_sha256 or len(post_bytes) != pre_size:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Integrity violation: original image file changed during AI detection.",
        )

    label, display_label, explanation, decision_rule, reported_threshold = _interpret_model_score(
        score, threshold, raw_class
    )
    tx_ctx = AiTransformationContext()
    interp = build_ai_detector_interpretation(
        status="available",
        selected_class=selected_class,
        selected_class_id=selected_class_id,
        artificial_score=score,
        human_score=human_score,
        decision_rule=decision_rule,
        threshold=reported_threshold,
        limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
        transformation_context=tx_ctx,
    )

    return AiGenerationDetectionResult(
        status="available",
        modality="image",
        reason=explanation,
        detector=AiDetectorMetadata(
            name=runtime.model_id,
            version=runtime.version,
            pinned_revision=runtime.version,
            architecture=runtime.architecture,
            source=runtime.source,
            framework=runtime.framework,
            device=runtime.device,
            upstream_format=runtime.upstream_format,
            runtime_format=runtime.runtime_format,
            onnx_is_upstream=runtime.onnx_is_upstream,
            model_repository=runtime.model_repository,
            pytorch_model_sha256=runtime.pytorch_model_sha256,
            onnx_sha256=runtime.onnx_sha256,
            exported_at=runtime.exported_at,
            conversion_info=runtime.conversion_info,
            input_resolution=f"{runtime.input_width}x{runtime.input_height}",
            preprocessing=_build_preprocessing_metadata(
                runtime.input_width,
                runtime.input_height,
                runtime.image_mean,
                runtime.image_std,
                runtime.id2label,
            ),
            license=runtime.license_name,
        ),
        prediction=AiDetectorPrediction(
            label=label,
            display_label=display_label,
            raw_class=raw_class,
            selected_class=selected_class,
            selected_class_id=selected_class_id,
            score=score,
            artificial_score=score,
            human_score=human_score,
            score_type="uncalibrated_model_score",
            decision_rule=decision_rule,
            threshold=reported_threshold,
        ),
        selected_class=selected_class,
        selected_class_id=selected_class_id,
        artificial_score=score,
        human_score=human_score,
        explanation=explanation,
        limitations=list(DEFAULT_AI_DETECTOR_LIMITATIONS),
        transformation_context=tx_ctx,
        ai_detector_interpretation=interp,
    )


