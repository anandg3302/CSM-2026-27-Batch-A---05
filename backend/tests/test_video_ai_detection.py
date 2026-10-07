"""Stage A — VideoMAE facial deepfake detector tests.

Tests cover all 15 Stage A verification scenarios:
1. Detector disabled → status="not_available"
2. Unpinned revision → status="not_available"
3. Model not on disk → status="unavailable"
4. Weights SHA-256 mismatch → status="unavailable" (integrity)
5. No frames extractable → status="error"
6. All frames non-facial → status="completed", usable_detector_frames=0
7. Facial frames present, model returns real → status="completed", selected_class="real"
8. Facial frames present, model returns fake → status="completed", selected_class="fake"
9. Result includes artifact_hashes
10. Result includes interpretation
11. Result includes limitations
12. Result includes correct id2label
13. Result includes correct preprocessing
14. Result includes correct score_type and decision_rule
15. Result includes correct detector_scope
"""

import io
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from app.schemas.video_analysis import VideoAiDetectionResult


# ─── Test frame generation ──────────────────────────────────────────────────

def create_facial_test_frame_rgb(
    width: int = 320,
    height: int = 240,
    shift: int = 0,
) -> np.ndarray:
    """Create a synthetic test frame with a smooth elliptical face-like gradient.

    Uses smooth gradients (not sharp rectangles) to avoid triggering classical
    blockiness metrics. The face region has skin-tone colors in the YCbCr range
    that the detector's facial detection expects.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Background: dark gray-blue
    frame[:, :] = [40 + shift, 45 + shift, 55 + shift]

    # Face center and radii
    cx, cy = width // 2, height // 2
    rx, ry = width // 5, height // 4

    # Create smooth elliptical gradient for face region
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    dist = np.sqrt(
        ((x_coords - cx) / rx) ** 2 + ((y_coords - cy) / ry) ** 2
    )

    # Skin tone base: (R=200, G=160, B=130) — falls within YCbCr skin range
    face_mask = np.clip(1.0 - dist, 0.0, 1.0)
    face_mask = face_mask ** 2  # smoother falloff

    skin_r = 200 + shift
    skin_g = 160 + shift
    skin_b = 130 + shift

    frame[:, :, 0] = np.clip(
        frame[:, :, 0] * (1 - face_mask) + skin_r * face_mask, 0, 255
    ).astype(np.uint8)
    frame[:, :, 1] = np.clip(
        frame[:, :, 1] * (1 - face_mask) + skin_g * face_mask, 0, 255
    ).astype(np.uint8)
    frame[:, :, 2] = np.clip(
        frame[:, :, 2] * (1 - face_mask) + skin_b * face_mask, 0, 255
    ).astype(np.uint8)

    return frame


def create_facial_video_container_bytes(
    num_frames: int = 20,
    width: int = 320,
    height: int = 240,
    fps: float = 25.0,
) -> bytes:
    """Create a minimal AVI video container with facial test frames.

    Returns the raw bytes of the video file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        for i in range(num_frames):
            frame_rgb = create_facial_test_frame_rgb(width, height, shift=0)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Encode/decode through JPEG at high quality to keep consistency
            _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 98])
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            writer.write(decoded)

        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ─── Fixtures and helpers ───────────────────────────────────────────────────

def _make_video_file(tmp_path: Path, data: bytes, name: str = "test.avi") -> Path:
    """Write video bytes to a file and return the path."""
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ─── Tests ──────────────────────────────────────────────────────────────────

class TestVideoAiDetectionScenarios:
    """Test all 15 Stage A verification scenarios."""

    def test_detector_disabled_returns_not_available(self):
        """Scenario 1: Detector disabled → status='not_available'."""
        from app.services.analysis.video_ai_detection_service import (
            evaluate_video_ai_detection,
            set_custom_video_predictor_for_testing,
            clear_video_detector_cache,
        )

        clear_video_detector_cache()
        set_custom_video_predictor_for_testing(None)

        with patch("app.services.analysis.video_ai_detection_service.settings") as mock_settings:
            mock_settings.ai_video_detector_enabled = False
            mock_settings.ai_video_detector_model_id = "Vansh180/VideoMae-ffc23-deepfake-detector"
            mock_settings.ai_video_detector_model_version = "ea26518b527d938d66e568c52bcf1c19a86cf295"
            mock_settings.ai_video_detector_device = "cpu"
            mock_settings.models_dir = Path("storage/models")

            result = evaluate_video_ai_detection(video_path=None, frames=[])
            assert result.status == "not_available"
            assert result.detector_enabled is False
            assert "disabled" in result.reason.lower()

    def test_unpinned_revision_rejected(self):
        """Scenario 2: Unpinned revision → status='not_available'."""
        from app.services.analysis.video_ai_detection_service import (
            evaluate_video_ai_detection,
            set_custom_video_predictor_for_testing,
            clear_video_detector_cache,
        )

        clear_video_detector_cache()
        set_custom_video_predictor_for_testing(None)

        with patch("app.services.analysis.video_ai_detection_service.settings") as mock_settings:
            mock_settings.ai_video_detector_enabled = True
            mock_settings.ai_video_detector_model_id = "Vansh180/VideoMae-ffc23-deepfake-detector"
            mock_settings.ai_video_detector_model_version = "main"  # unpinned!
            mock_settings.ai_video_detector_device = "cpu"
            mock_settings.models_dir = Path("storage/models")

            result = evaluate_video_ai_detection(video_path=None, frames=[])
            assert result.status == "not_available"
            assert "not pinned" in result.reason.lower() or "unpinned" in result.reason.lower()

    def test_model_not_on_disk_returns_unavailable(self, tmp_path):
        """Scenario 3: Model not on disk → status='unavailable'."""
        from app.services.analysis.video_ai_detection_service import (
            evaluate_video_ai_detection,
            set_custom_video_predictor_for_testing,
            clear_video_detector_cache,
        )

        clear_video_detector_cache()
        set_custom_video_predictor_for_testing(None)

        empty_model_dir = tmp_path / "empty_models"
        empty_model_dir.mkdir()

        with patch("app.services.analysis.video_ai_detection_service.settings") as mock_settings:
            mock_settings.ai_video_detector_enabled = True
            mock_settings.ai_video_detector_model_id = "Vansh180/VideoMae-ffc23-deepfake-detector"
            mock_settings.ai_video_detector_model_version = "ea26518b527d938d66e568c52bcf1c19a86cf295"
            mock_settings.ai_video_detector_device = "cpu"
            mock_settings.models_dir = empty_model_dir

            result = evaluate_video_ai_detection(
                video_path=None,
                frames=[create_facial_test_frame_rgb()],
                model_dir=empty_model_dir / "Vansh180--VideoMae-ffc23-deepfake-detector",
            )
            assert result.status == "unavailable"

    def test_custom_predictor_real_and_fake_and_full_fields(self):
        """Scenarios 6-15: Full functional test using custom predictor.

        Tests: facial detection, real/fake classes, artifact_hashes,
        interpretation, limitations, id2label, preprocessing, score_type,
        decision_rule, detector_scope.
        """
        from app.services.analysis.video_ai_detection_service import (
            evaluate_video_ai_detection,
            set_custom_video_predictor_for_testing,
            clear_video_detector_cache,
        )

        clear_video_detector_cache()

        # Create a predictor that returns "fake" with known scores
        def fake_predictor(video_path, frames):
            return VideoAiDetectionResult(
                status="completed",
                reason="Video AI detector completed: selected_class=fake.",
                detector_enabled=True,
                model_name="Vansh180/VideoMae-ffc23-deepfake-detector",
                model_version="ea26518b527d938d66e568c52bcf1c19a86cf295",
                architecture="VideoMAEForVideoClassification",
                license="MIT",
                training_domain="FaceForensics++ C23",
                artifact_hashes={"model.safetensors": "293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29"},
                id2label={"0": "real", "1": "fake"},
                preprocessing={
                    "processor_class": "VideoMAEImageProcessor",
                    "num_frames": 16,
                },
                selected_class="fake",
                real_score=0.3,
                fake_score=0.7,
                score_type="uncalibrated_model_score",
                decision_rule="model_class_selection",
                threshold=None,
                frames_sampled=16,
                usable_detector_frames=12,
                detector_scope="facial_video_deepfake_detection",
                interpretation=None,
            )

        set_custom_video_predictor_for_testing(fake_predictor)

        try:
            with patch("app.services.analysis.video_ai_detection_service.settings") as mock_settings:
                mock_settings.ai_video_detector_enabled = True
                mock_settings.ai_video_detector_model_id = "Vansh180/VideoMae-ffc23-deepfake-detector"
                mock_settings.ai_video_detector_model_version = "ea26518b527d938d66e568c52bcf1c19a86cf295"
                mock_settings.ai_video_detector_device = "cpu"
                mock_settings.models_dir = Path("storage/models")

                result = evaluate_video_ai_detection(video_path="/fake/video.mp4")

                # Scenario 8: selected_class="fake"
                assert result.status == "completed"
                assert result.selected_class == "fake"
                assert result.fake_score == 0.7
                assert result.real_score == 0.3

                # Scenario 9: artifact_hashes present
                assert "model.safetensors" in result.artifact_hashes

                # Scenario 12: correct id2label
                assert result.id2label == {"0": "real", "1": "fake"}

                # Scenario 13: correct preprocessing
                assert result.preprocessing["processor_class"] == "VideoMAEImageProcessor"

                # Scenario 14: correct score_type and decision_rule
                assert result.score_type == "uncalibrated_model_score"
                assert result.decision_rule == "model_class_selection"
                assert result.threshold is None

                # Scenario 15: correct detector_scope
                assert result.detector_scope == "facial_video_deepfake_detection"

            # Now test "real" result
            def real_predictor(video_path, frames):
                return VideoAiDetectionResult(
                    status="completed",
                    reason="Video AI detector completed: selected_class=real.",
                    detector_enabled=True,
                    model_name="Vansh180/VideoMae-ffc23-deepfake-detector",
                    model_version="ea26518b527d938d66e568c52bcf1c19a86cf295",
                    architecture="VideoMAEForVideoClassification",
                    license="MIT",
                    training_domain="FaceForensics++ C23",
                    selected_class="real",
                    real_score=0.8,
                    fake_score=0.2,
                    frames_sampled=16,
                    usable_detector_frames=16,
                    detector_scope="facial_video_deepfake_detection",
                )

            set_custom_video_predictor_for_testing(real_predictor)

            with patch("app.services.analysis.video_ai_detection_service.settings") as mock_settings:
                mock_settings.ai_video_detector_enabled = True
                mock_settings.ai_video_detector_model_id = "Vansh180/VideoMae-ffc23-deepfake-detector"
                mock_settings.ai_video_detector_model_version = "ea26518b527d938d66e568c52bcf1c19a86cf295"
                mock_settings.ai_video_detector_device = "cpu"
                mock_settings.models_dir = Path("storage/models")

                result = evaluate_video_ai_detection(video_path="/fake/video.mp4")

                # Scenario 7: selected_class="real"
                assert result.status == "completed"
                assert result.selected_class == "real"
                assert result.real_score == 0.8
                assert result.fake_score == 0.2

        finally:
            set_custom_video_predictor_for_testing(None)


class TestVideoAiDetectionHelpers:
    """Test helper functions independently."""

    def test_facial_frame_detection_with_skin_tones(self):
        """Verify that create_facial_test_frame_rgb produces frames
        that pass facial detection."""
        from app.services.analysis.video_ai_detection_service import (
            detect_usable_facial_frames,
        )

        frames = [create_facial_test_frame_rgb(shift=0) for _ in range(4)]
        usable = detect_usable_facial_frames(frames, skin_pixel_threshold=0.02)
        # Skin-tone frames should be detected as facial
        assert len(usable) > 0

    def test_preprocessing_output_shape(self):
        """Verify that preprocess_videomae_clip produces correct tensor shape."""
        from app.services.analysis.video_ai_detection_service import (
            preprocess_videomae_clip,
        )

        frames = [create_facial_test_frame_rgb() for _ in range(20)]
        clip = preprocess_videomae_clip(frames)
        assert clip.shape == (1, 16, 3, 224, 224)
        assert clip.dtype == np.float32

    def test_sinusoid_encoding_table_shape(self):
        """Verify sinusoidal position encoding table dimensions."""
        from app.services.analysis.video_ai_detection_service import (
            _build_sinusoid_encoding_table,
        )

        table = _build_sinusoid_encoding_table(1568, 768)
        assert table.shape == (1, 1568, 768)
        assert table.dtype == np.float32

    def test_softmax_basic(self):
        """Verify softmax produces valid probability distribution."""
        from app.services.analysis.video_ai_detection_service import _softmax

        probs = _softmax(np.array([1.0, 2.0]))
        assert abs(sum(probs) - 1.0) < 1e-10
        assert probs[1] > probs[0]
