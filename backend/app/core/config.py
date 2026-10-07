from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve repository root directory (MediaTrace/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Centralized configuration for the MediaTrace backend."""

    app_name: str = "MediaTrace API"
    app_env: str = "development"
    api_prefix: str = "/api"

    # Allowed origins for the local React/Vite frontend
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Storage paths for forensic pipelines (uploads, analysis results, generated reports, models)
    storage_root: Path = PROJECT_ROOT / "storage"
    uploads_dir: Path = PROJECT_ROOT / "storage" / "uploads"
    results_dir: Path = PROJECT_ROOT / "storage" / "results"
    reports_dir: Path = PROJECT_ROOT / "storage" / "reports"
    models_dir: Path = PROJECT_ROOT / "storage" / "models"

    # AI-Generated Image Detection Settings (Step 14.1 / Step 14.1.1)
    ai_image_detector_enabled: bool = True
    ai_image_detector_model_id: str = "umm-maybe/AI-image-detector"
    ai_image_detector_model_version: str = "c7e223baf11bc40528af364ba7bdea030ef42f9e"
    ai_image_detector_device: str = "cpu"
    ai_image_detector_threshold: Optional[float] = None

    # AI-Generated Video Detection Settings (Stage A — VideoMAE Deepfake Detector)
    ai_video_detector_enabled: bool = True
    ai_video_detector_model_id: str = "Vansh180/VideoMae-ffc23-deepfake-detector"
    ai_video_detector_model_version: str = "ea26518b527d938d66e568c52bcf1c19a86cf295"
    ai_video_detector_device: str = "cpu"

    # Maximum number of representative frames to sample during video forensics (Step 10)
    max_sample_frames: int = 12

    # Audio Forensics Settings (Step 11)
    max_audio_analysis_seconds: float = 300.0
    silence_threshold_amplitude: float = 0.02
    min_silence_duration_seconds: float = 0.25

    # Document Forensics Settings (Step 12)
    max_document_analysis_mb: int = 50
    max_document_preview_pages: int = 10
    max_extracted_text_chars: int = 5_000_000
    max_embedded_items: int = 100

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def model_post_init(self, __context) -> None:
        """Resolve any relative storage paths from .env against PROJECT_ROOT."""
        for attr in ("storage_root", "uploads_dir", "results_dir", "reports_dir", "models_dir"):
            val: Path = getattr(self, attr)
            if not val.is_absolute():
                cleaned = str(val).replace("\\", "/")
                while cleaned.startswith("../"):
                    cleaned = cleaned[3:]
                cleaned = cleaned.lstrip("./")
                setattr(self, attr, (PROJECT_ROOT / cleaned).resolve())

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def ensure_storage_directories(self) -> None:
        """Ensure required storage folders exist on disk."""
        for directory in (
            self.storage_root,
            self.uploads_dir,
            self.results_dir,
            self.reports_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
