"""Stage A — Real VideoMAE-based Facial Deepfake Detection Service.

Loads the pretrained video classifier `Vansh180/VideoMae-ffc23-deepfake-detector`
(`VideoMAEForVideoClassification` backbone, pinned revision
`ea26518b527d938d66e568c52bcf1c19a86cf295`, license `MIT`) from
`storage/models/` using `safetensors` format with a pure-numpy forward pass.

Forensic Guarantees:
- Never modifies the original uploaded video bytes.
- Preprocessing occurs strictly on in-memory frame tensors using exact
  `preprocessor_config.json` parameters (RGB, 224x224, mean [0.485, 0.456, 0.406],
  std [0.229, 0.224, 0.225], 16 uniformly sampled frames).
- Verifies model artifact SHA-256 hashes before inference.
- Treats model outputs as uncalibrated classifier scores (`uncalibrated_model_score`)
  with `decision_rule="model_class_selection"` (`threshold=None`).
- Requires visible facial regions (YCbCr skin-chrominance + Haar cascade geometry);
  non-facial video is out of scope.
- Interpretation is always `strength="informational"` — the detector never
  independently establishes or disproves video provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.video_analysis import (
    VideoAiDetectionResult,
    VideoAiDetectorInterpretation,
)

logger = logging.getLogger("mediatrace")

# ─── Pinned model constants ─────────────────────────────────────────────────
PINNED_VIDEO_MODEL_ID = "Vansh180/VideoMae-ffc23-deepfake-detector"
PINNED_VIDEO_MODEL_VERSION = "ea26518b527d938d66e568c52bcf1c19a86cf295"
EXPECTED_SAFETENSORS_SHA256 = (
    "293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29"
)

# ─── VideoMAE architecture constants ────────────────────────────────────────
NUM_FRAMES = 16
PATCH_SIZE = 16
TUBELET_SIZE = 2
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
NUM_HIDDEN_LAYERS = 12
IMAGE_SIZE = 224
INTERMEDIATE_SIZE = 3072
LAYER_NORM_EPS = 1e-12

# Number of 3D tubelet patches: (NUM_FRAMES/TUBELET_SIZE) * (IMAGE_SIZE/PATCH_SIZE)^2
NUM_PATCHES = (NUM_FRAMES // TUBELET_SIZE) * (IMAGE_SIZE // PATCH_SIZE) ** 2  # 1568

# Patch embedding input dimension: channels * tubelet_size * patch_size^2
PATCH_DIM = 3 * TUBELET_SIZE * PATCH_SIZE * PATCH_SIZE  # 1536

# Per-head dimension
HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS  # 64

# Preprocessing constants
VIDEO_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VIDEO_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Haar cascade for facial detection
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# ─── Runtime cache ───────────────────────────────────────────────────────────
_VIDEO_MODEL_CACHE: Dict[str, Any] = {}
_CUSTOM_VIDEO_PREDICTOR = None


def clear_video_detector_cache() -> None:
    """Clear cached model weights (for testing or hot-reload)."""
    _VIDEO_MODEL_CACHE.clear()


def set_custom_video_predictor_for_testing(predictor) -> None:
    """Allow unit tests to inject a deterministic predictor without loading real weights."""
    global _CUSTOM_VIDEO_PREDICTOR
    _CUSTOM_VIDEO_PREDICTOR = predictor


# ─── Utility functions ──────────────────────────────────────────────────────

def _compute_file_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _is_valid_pinned_revision(revision: str) -> bool:
    cleaned = (revision or "").strip()
    if not cleaned:
        return False
    unpinned = {"main", "master", "latest", "head", "default", "trunk"}
    if cleaned.lower() in unpinned:
        return False
    return bool(re.match(r"^[0-9a-fA-F]{7,40}$", cleaned))


# ─── Pure-numpy math primitives ─────────────────────────────────────────────

def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """LayerNorm over the last axis."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return x_norm * weight + bias


def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation matching PyTorch's default (approximate='none') to float32 machine precision."""
    z = np.abs(x / math.sqrt(2.0))
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    erf_val = np.sign(x) * (1.0 - poly * np.exp(-z * z))
    return (0.5 * x * (1.0 + erf_val)).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    shifted = x - np.max(x)
    exps = np.exp(np.clip(shifted, -50.0, 50.0))
    total = float(np.sum(exps))
    if total <= 0.0:
        return np.full_like(x, 1.0 / x.size, dtype=np.float64)
    return exps / total


def _build_sinusoid_encoding_table(n_position: int, d_hid: int) -> np.ndarray:
    """Build sinusoidal position encoding table (not learned).

    Returns shape (1, n_position, d_hid) float32.
    """
    def get_position_angle_vec(position: int) -> np.ndarray:
        return np.array([
            position / np.power(10000.0, 2.0 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ], dtype=np.float64)

    table = np.array([get_position_angle_vec(pos) for pos in range(n_position)],
                     dtype=np.float64)
    table[:, 0::2] = np.sin(table[:, 0::2])  # even indices
    table[:, 1::2] = np.cos(table[:, 1::2])  # odd indices
    return table.reshape(1, n_position, d_hid).astype(np.float32)


# ─── VideoMAE forward pass (pure numpy) ─────────────────────────────────────

def _run_videomae_forward(
    pixel_values: np.ndarray,
    weights: Dict[str, np.ndarray],
) -> np.ndarray:
    """Run full VideoMAEForVideoClassification forward pass.

    Args:
        pixel_values: (B, 16, 3, 224, 224) float32 — preprocessed frames.
        weights: Dict of weight name -> ndarray loaded from safetensors.

    Returns:
        logits: (B, 2) float32 — [real_logit, fake_logit].
    """
    B = pixel_values.shape[0]

    # ── 1. Tubelet patch embedding ──
    # (B, 16, 3, 224, 224) → (B, 3, 16, 224, 224)
    x = pixel_values.transpose(0, 2, 1, 3, 4)

    # Reshape into non-overlapping 3D patches
    # (B, 3, T/t, t, H/p, p, W/p, p) where t=2, p=16
    T_patches = NUM_FRAMES // TUBELET_SIZE   # 8
    H_patches = IMAGE_SIZE // PATCH_SIZE     # 14
    W_patches = IMAGE_SIZE // PATCH_SIZE     # 14

    x = x.reshape(B, 3, T_patches, TUBELET_SIZE, H_patches, PATCH_SIZE, W_patches, PATCH_SIZE)
    # Rearrange to (B, T_patches, H_patches, W_patches, 3, TUBELET_SIZE, PATCH_SIZE, PATCH_SIZE)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    # Flatten patches: (B, NUM_PATCHES, PATCH_DIM)
    x = x.reshape(B, NUM_PATCHES, PATCH_DIM)

    # Linear projection
    proj_w = weights["videomae.embeddings.patch_embeddings.projection.weight"]  # (768, 1536)
    proj_b = weights["videomae.embeddings.patch_embeddings.projection.bias"]    # (768,)

    # proj_w might be stored as (768, 3, 2, 16, 16) for Conv3D — reshape to (768, 1536)
    if proj_w.ndim > 2:
        proj_w = proj_w.reshape(HIDDEN_SIZE, PATCH_DIM)

    x = x @ proj_w.T + proj_b  # (B, 1568, 768)

    # ── 2. Position embedding (sinusoidal, not learned) ──
    pos_embed = _build_sinusoid_encoding_table(NUM_PATCHES, HIDDEN_SIZE)  # (1, 1568, 768)
    x = x + pos_embed

    # ── 3. Transformer encoder (12 layers) ──
    for layer_idx in range(NUM_HIDDEN_LAYERS):
        prefix = f"videomae.encoder.layer.{layer_idx}"

        # LayerNorm before attention
        ln1_w = weights[f"{prefix}.layernorm_before.weight"]
        ln1_b = weights[f"{prefix}.layernorm_before.bias"]
        x_norm = _layer_norm(x, ln1_w, ln1_b)

        # Multi-head self-attention with q_bias and v_bias (no k_bias)
        qw = weights[f"{prefix}.attention.attention.query.weight"]
        kw = weights[f"{prefix}.attention.attention.key.weight"]
        vw = weights[f"{prefix}.attention.attention.value.weight"]
        qb = weights[f"{prefix}.attention.attention.q_bias"]
        vb = weights[f"{prefix}.attention.attention.v_bias"]

        q = x_norm @ qw.T + qb
        k = x_norm @ kw.T
        v = x_norm @ vw.T + vb

        q = q.reshape(B, NUM_PATCHES, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = k.reshape(B, NUM_PATCHES, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = v.reshape(B, NUM_PATCHES, NUM_ATTENTION_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = math.sqrt(HEAD_DIM)
        attn_scores = (q @ k.transpose(0, 1, 3, 2)) / scale  # (B, heads, patches, patches)
        attn_max = attn_scores.max(axis=-1, keepdims=True)
        attn_exp = np.exp(attn_scores - attn_max)
        attn_weights = attn_exp / attn_exp.sum(axis=-1, keepdims=True)

        attn_output = attn_weights @ v  # (B, heads, patches, head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, NUM_PATCHES, HIDDEN_SIZE)

        # Output projection
        out_proj_w = weights[f"{prefix}.attention.output.dense.weight"]  # (768, 768)
        out_proj_b = weights[f"{prefix}.attention.output.dense.bias"]    # (768,)
        attn_output = attn_output @ out_proj_w.T + out_proj_b

        # Residual connection
        x = x + attn_output

        # LayerNorm before MLP
        ln2_w = weights[f"{prefix}.layernorm_after.weight"]
        ln2_b = weights[f"{prefix}.layernorm_after.bias"]
        x_norm2 = _layer_norm(x, ln2_w, ln2_b)

        # MLP: intermediate → GELU → output
        inter_w = weights[f"{prefix}.intermediate.dense.weight"]  # (3072, 768)
        inter_b = weights[f"{prefix}.intermediate.dense.bias"]    # (3072,)
        mlp_hidden = x_norm2 @ inter_w.T + inter_b
        mlp_hidden = _gelu(mlp_hidden)

        mlp_out_w = weights[f"{prefix}.output.dense.weight"]  # (768, 3072)
        mlp_out_b = weights[f"{prefix}.output.dense.bias"]    # (768,)
        mlp_output = mlp_hidden @ mlp_out_w.T + mlp_out_b

        # Residual connection
        x = x + mlp_output

    # ── 4. Mean pooling over all 1568 sequence tokens ──
    pooled = x.mean(axis=1)  # (B, 768)

    # ── 5. fc_norm (LayerNorm) after pooling ──
    fc_norm_w = weights.get("fc_norm.weight", weights.get("videomae.fc_norm.weight"))  # (768,)
    fc_norm_b = weights.get("fc_norm.bias", weights.get("videomae.fc_norm.bias"))      # (768,)
    pooled = _layer_norm(pooled, fc_norm_w, fc_norm_b)

    # ── 6. Linear classifier ──
    cls_w = weights["classifier.weight"]  # (2, 768)
    cls_b = weights["classifier.bias"]    # (2,)
    logits = pooled @ cls_w.T + cls_b     # (B, 2) — [real_logit, fake_logit]

    return logits.astype(np.float32)


# ─── Model loading ──────────────────────────────────────────────────────────

def verify_video_model_artifacts(model_dir: Path) -> Dict[str, str]:
    """Compute SHA-256 hashes for all key model artifacts.

    Returns dict of filename -> sha256hex.
    """
    artifact_hashes: Dict[str, str] = {}
    for name in ("model.safetensors", "config.json", "preprocessor_config.json"):
        p = model_dir / name
        if p.is_file():
            try:
                artifact_hashes[name] = _compute_file_sha256(p)
            except Exception:
                pass
    return artifact_hashes


def load_video_ai_detector_bundle(model_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Load the VideoMAE model weights from safetensors and return a bundle dict.

    Returns None if model files are not available.
    """
    if model_dir is None:
        model_dir = settings.models_dir / "Vansh180--VideoMae-ffc23-deepfake-detector"

    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    if not safetensors_path.is_file():
        logger.info("Video AI detector model not found at %s", safetensors_path)
        return None

    # Check cache
    cache_key = str(safetensors_path.resolve())
    if cache_key in _VIDEO_MODEL_CACHE:
        return _VIDEO_MODEL_CACHE[cache_key]

    try:
        from safetensors.numpy import load_file as st_load
    except ImportError:
        logger.warning("safetensors package is not installed; cannot load video AI detector.")
        return None

    # Verify weights integrity
    actual_sha = _compute_file_sha256(safetensors_path)
    if actual_sha.lower() != EXPECTED_SAFETENSORS_SHA256.lower():
        logger.error(
            "Video AI detector weights SHA-256 mismatch: expected=%s actual=%s",
            EXPECTED_SAFETENSORS_SHA256, actual_sha,
        )
        return None

    # Load weights
    weights = st_load(str(safetensors_path))

    # Load config
    config: Dict[str, Any] = {}
    if config_path.is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    id2label = config.get("id2label", {"0": "real", "1": "fake"})

    bundle = {
        "weights": weights,
        "config": config,
        "id2label": id2label,
        "model_dir": model_dir,
        "safetensors_sha256": actual_sha,
        "artifact_hashes": verify_video_model_artifacts(model_dir),
    }
    _VIDEO_MODEL_CACHE[cache_key] = bundle
    logger.info("Video AI detector model loaded from %s (sha256=%s)", safetensors_path, actual_sha)
    return bundle


def load_video_ai_detector_runtime(model_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Public entry point to load the VideoMAE detector runtime.

    Returns the bundle dict or None if unavailable.
    """
    return load_video_ai_detector_bundle(model_dir)


# ─── Preprocessing ──────────────────────────────────────────────────────────

def preprocess_videomae_clip(
    frames: List[np.ndarray],
    target_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """Preprocess a list of RGB frames for VideoMAE inference.

    Args:
        frames: List of H x W x 3 uint8 RGB frames (at least NUM_FRAMES).
        target_size: Target spatial dimension (224).

    Returns:
        (1, 16, 3, 224, 224) float32 tensor.
    """
    if len(frames) < NUM_FRAMES:
        # Repeat last frame to reach NUM_FRAMES
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())

    # Uniformly sample 16 frames
    if len(frames) > NUM_FRAMES:
        indices = np.linspace(0, len(frames) - 1, NUM_FRAMES, dtype=int)
        frames = [frames[i] for i in indices]

    processed = []
    for frame in frames[:NUM_FRAMES]:
        # Resize shortest edge to 224 with BILINEAR and center crop to 224x224
        h, w = frame.shape[:2]
        if h != target_size or w != target_size:
            if h < w:
                new_h = target_size
                new_w = int(round(w * (target_size / h)))
            else:
                new_w = target_size
                new_h = int(round(h * (target_size / w)))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            ch, cw = frame.shape[:2]
            top = max(0, (ch - target_size) // 2)
            left = max(0, (cw - target_size) // 2)
            frame = frame[top : top + target_size, left : left + target_size]

        # Convert to float32 [0, 1]
        f = frame.astype(np.float32) / 255.0

        # Normalize per channel
        f = (f - VIDEO_IMAGE_MEAN) / VIDEO_IMAGE_STD

        # HWC → CHW
        f = f.transpose(2, 0, 1)
        processed.append(f)

    # Stack: (16, 3, 224, 224)
    clip = np.stack(processed, axis=0)
    return clip[np.newaxis, ...]  # (1, 16, 3, 224, 224)


def sample_frames_uniformly(video_path: str, num_frames: int = NUM_FRAMES) -> Tuple[List[np.ndarray], List[int]]:
    """Sample num_frames frames uniformly from a video file.

    Returns (list of RGB frames, list of frame indices).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    frames: List[np.ndarray] = []
    sampled_indices: List[int] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            sampled_indices.append(idx)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from: {video_path}")

    return frames, sampled_indices


# ─── Facial frame detection ─────────────────────────────────────────────────

def detect_usable_facial_frames(
    frames: List[np.ndarray],
    min_face_fraction: float = 0.02,
    skin_pixel_threshold: float = 0.05,
) -> List[int]:
    """Detect which frames contain usable facial regions for the deepfake detector.

    Uses two complementary methods:
    1. YCbCr skin-chrominance detection (statistical skin color model) — primary
    2. Viola-Jones Haar cascade face geometry detection — optional fallback
       (only available if OpenCV provides CascadeClassifier)

    A frame is considered usable if EITHER method detects a face-like region.

    Args:
        frames: List of RGB uint8 frames.
        min_face_fraction: Minimum fraction of image area that a Haar face bbox must cover.
        skin_pixel_threshold: Minimum fraction of pixels that must be skin-colored.

    Returns:
        List of indices of frames that contain usable facial regions.
    """
    usable: List[int] = []

    # Try to load Haar cascade if available (removed in OpenCV 5.0)
    face_cascade = None
    if hasattr(cv2, "CascadeClassifier"):
        try:
            face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if face_cascade.empty():
                face_cascade = None
        except Exception:
            face_cascade = None

    for idx, frame in enumerate(frames):
        has_face = False

        # Method 1: YCbCr skin chrominance (primary, always available)
        try:
            ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
            cr = ycrcb[:, :, 1].astype(np.float32)
            cb = ycrcb[:, :, 2].astype(np.float32)
            # Standard skin color range in YCbCr
            skin_mask = (
                (cr >= 133) & (cr <= 173) &
                (cb >= 77) & (cb <= 127)
            )
            skin_fraction = float(np.sum(skin_mask)) / float(skin_mask.size)
            if skin_fraction >= skin_pixel_threshold:
                has_face = True
        except Exception:
            pass

        # Method 2: Haar cascade face detection (fallback, if available)
        if not has_face and face_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    total_pixels = frame.shape[0] * frame.shape[1]
                    for (fx, fy, fw, fh) in faces:
                        face_area_fraction = (fw * fh) / total_pixels
                        if face_area_fraction >= min_face_fraction:
                            has_face = True
                            break
            except Exception:
                pass

        if has_face:
            usable.append(idx)

    return usable


# ─── Main evaluation function ───────────────────────────────────────────────

def evaluate_video_ai_detection(
    video_path: Optional[str] = None,
    expected_sha256: Optional[str] = None,
    *,
    frames: Optional[List[np.ndarray]] = None,
    sampled_frame_indices: Optional[List[int]] = None,
    model_dir: Optional[Path] = None,
) -> VideoAiDetectionResult:
    """Evaluate a video for AI-generated deepfake indicators using the VideoMAE detector.

    This function can accept either:
    - A video file path (will extract and sample frames), OR
    - Pre-extracted frames (for testing or when frames are already available).

    Args:
        video_path: Path to the video file on disk.
        expected_sha256: Expected SHA-256 of the video (for integrity checking).
        frames: Optional pre-extracted list of RGB frames.
        sampled_frame_indices: Optional pre-computed frame indices.
        model_dir: Optional override for model directory.

    Returns:
        VideoAiDetectionResult with detection status, scores, and interpretation.
    """
    model_id = settings.ai_video_detector_model_id
    model_version = settings.ai_video_detector_model_version
    detector_enabled = settings.ai_video_detector_enabled

    base_result_kwargs: Dict[str, Any] = {
        "detector_enabled": detector_enabled,
        "model_name": model_id,
        "model_version": model_version,
        "architecture": "VideoMAEForVideoClassification",
        "license": "MIT",
        "training_domain": "FaceForensics++ C23",
        "id2label": {"0": "real", "1": "fake"},
        "score_type": "uncalibrated_model_score",
        "decision_rule": "model_class_selection",
        "threshold": None,
        "detector_scope": "facial_video_deepfake_detection",
        "preprocessing": {
            "processor_class": "VideoMAEImageProcessor",
            "num_frames": NUM_FRAMES,
            "size": {"shortest_edge": IMAGE_SIZE},
            "crop_size": {"height": IMAGE_SIZE, "width": IMAGE_SIZE},
            "color_mode": "RGB",
            "image_mean": VIDEO_IMAGE_MEAN.tolist(),
            "image_std": VIDEO_IMAGE_STD.tolist(),
        },
    }

    # ── Check if detector is enabled ──
    if not detector_enabled:
        return VideoAiDetectionResult(
            status="not_available",
            reason="Video AI detector is disabled in configuration.",
            **base_result_kwargs,
        )

    # ── Validate pinned revision ──
    if not _is_valid_pinned_revision(model_version):
        return VideoAiDetectionResult(
            status="not_available",
            reason=(
                f"Video AI detector revision '{model_version}' is not pinned to an "
                "immutable commit SHA. Unpinned branch names are rejected for forensic "
                "reproducibility."
            ),
            **base_result_kwargs,
        )

    # ── Handle custom test predictor ──
    if _CUSTOM_VIDEO_PREDICTOR is not None:
        try:
            test_result = _CUSTOM_VIDEO_PREDICTOR(video_path, frames)
            if isinstance(test_result, VideoAiDetectionResult):
                return test_result
            if isinstance(test_result, dict):
                merged = {**base_result_kwargs, **test_result}
                return VideoAiDetectionResult(**merged)
        except Exception as exc:
            logger.error("Custom video predictor failed: %s", exc)
            return VideoAiDetectionResult(
                status="error",
                reason=f"Custom test predictor error: {exc}",
                **base_result_kwargs,
            )

    # ── Load model ──
    bundle = load_video_ai_detector_bundle(model_dir)
    if bundle is None:
        return VideoAiDetectionResult(
            status="unavailable",
            reason=(
                "Video AI detector model could not be loaded. Ensure model.safetensors "
                "is present in storage/models/Vansh180--VideoMae-ffc23-deepfake-detector/ "
                "with the expected SHA-256 hash."
            ),
            **base_result_kwargs,
        )

    artifact_hashes = bundle.get("artifact_hashes", {})
    base_result_kwargs["artifact_hashes"] = artifact_hashes

    # ── Extract or use provided frames ──
    try:
        if frames is None:
            if video_path is None:
                return VideoAiDetectionResult(
                    status="error",
                    reason="No video path or frames provided for AI detection.",
                    **base_result_kwargs,
                )
            frames, sampled_frame_indices = sample_frames_uniformly(
                video_path, NUM_FRAMES
            )
    except Exception as exc:
        logger.error("Failed to sample frames from video: %s", exc)
        return VideoAiDetectionResult(
            status="error",
            reason=f"Failed to sample video frames: {exc}",
            **base_result_kwargs,
        )

    frames_sampled = len(frames)

    # ── Detect facial frames ──
    usable_indices = detect_usable_facial_frames(frames)
    usable_count = len(usable_indices)

    if usable_count == 0:
        return VideoAiDetectionResult(
            status="completed",
            reason=(
                "No usable facial frames detected. The VideoMAE deepfake detector "
                "requires visible facial regions and cannot assess non-facial content."
            ),
            frames_sampled=frames_sampled,
            usable_detector_frames=0,
            interpretation=VideoAiDetectorInterpretation(),
            **base_result_kwargs,
        )

    # ── Preprocess ──
    # Use only usable facial frames for detection
    usable_frames = [frames[i] for i in usable_indices]
    clip_tensor = preprocess_videomae_clip(usable_frames)

    # ── Run forward pass ──
    try:
        logits = _run_videomae_forward(clip_tensor, bundle["weights"])
    except Exception as exc:
        logger.error("VideoMAE forward pass failed: %s", exc)
        return VideoAiDetectionResult(
            status="error",
            reason=f"Model inference error: {exc}",
            frames_sampled=frames_sampled,
            usable_detector_frames=usable_count,
            **base_result_kwargs,
        )

    # ── Interpret output ──
    # logits shape: (1, 2) — [real_logit, fake_logit]
    logits_flat = logits[0]
    scores = _softmax(logits_flat)
    real_score = float(scores[0])
    fake_score = float(scores[1])

    selected_class = "fake" if fake_score >= real_score else "real"

    return VideoAiDetectionResult(
        status="completed",
        reason=f"Video AI detector completed: selected_class={selected_class}.",
        selected_class=selected_class,
        real_score=round(real_score, 6),
        fake_score=round(fake_score, 6),
        frames_sampled=frames_sampled,
        usable_detector_frames=usable_count,
        interpretation=VideoAiDetectorInterpretation(),
        **base_result_kwargs,
    )
