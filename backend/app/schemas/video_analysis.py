from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

AllowedVideoAssessment = Literal[
    "no_strong_indicators",
    "potential_manipulation",
    "localized_temporal_anomaly",
    "possible_recompression",
    "insufficient_evidence",
]


class VideoForensicFinding(BaseModel):
    """Objective observation or conservative signal at the video or sampled-frame level."""

    scope: Literal["video_level", "sampled_frame"] = Field(
        ..., description="Distinguishes video-level findings from sampled-frame findings"
    )
    category: str = Field(..., description="Domain: container, metadata, compression, noise, or temporal")
    finding: str = Field(..., description="Neutral identifier or title of the observation")
    description: str = Field(..., description="Conservative explanation of the observed property")
    value: str = Field(..., description="Measured or observed value")
    strength: Literal["informational", "weak", "moderate"] = Field(default="informational")


class VideoMetadataInfo(BaseModel):
    """Technical container, stream, audio, and privacy-safe metadata extracted from the video."""

    container_format: Optional[str] = None
    codec: Optional[str] = None
    codec_profile: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_rate: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    pixel_format: Optional[str] = None
    color_space: Optional[str] = None
    rotation: Optional[int] = None
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channel_count: Optional[int] = None
    creation_timestamp: Optional[str] = None
    encoder_software: Optional[str] = None
    gps_present: bool = False
    gps_notice: Optional[str] = Field(
        default=None,
        description="Reported strictly as 'GPS metadata present' when GPS metadata exists; raw coordinates are never exposed",
    )


class SampledFrameItem(BaseModel):
    """Forensic metrics and findings for a single representative sampled video frame."""

    frame_number: int = Field(..., ge=1, description="1-based sample sequence index")
    frame_index: int = Field(..., ge=0, description="0-based frame index within the video stream")
    timestamp_seconds: float = Field(..., ge=0.0, description="Timestamp of the sampled frame in seconds")
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)
    color_mode: str = Field(default="RGB")
    mean_brightness: float = Field(..., ge=0.0, le=255.0)
    noise_level: float = Field(..., ge=0.0)
    edge_density: float = Field(..., ge=0.0)
    compression_blockiness: float = Field(..., ge=0.0)
    artifact: str = Field(
        ...,
        description="Relative artifact path under storage/results/<case_id>/video/ (e.g. storage/results/.../video/frame_0001.jpg)",
    )
    artifact_filename: str = Field(..., description="Filename of the saved frame artifact (e.g. frame_0001.jpg)")
    findings: List[VideoForensicFinding] = Field(default_factory=list)


class VideoFrameSamplingResult(BaseModel):
    """Bounded representative frame sampling results."""

    max_sample_frames: int = Field(..., ge=1)
    sample_count: int = Field(..., ge=0)
    sampling_strategy: str = Field(
        default="bounded_representative_intervals",
        description="Representative intervals (beginning, ~25%, ~50%, ~75%, end, plus duration-based samples up to MAX_SAMPLE_FRAMES)",
    )
    frames: List[SampledFrameItem] = Field(default_factory=list)


class TemporalConsistencySignal(BaseModel):
    """Frame-to-frame consistency observation across sampled frames."""

    type: str = Field(..., description="Signal type (e.g. noise_level_shift, compression_artifact_shift, brightness_discontinuity)")
    from_frame_index: int = Field(..., ge=0)
    to_frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    description: str
    strength: Literal["informational", "weak", "moderate"] = Field(default="weak")


class SuspiciousFrameItem(BaseModel):
    """Candidate frame flagged for review due to a localized temporal or compression anomaly."""

    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    finding: str = Field(default="localized_temporal_anomaly")
    status: Literal["potentially_suspicious"] = Field(default="potentially_suspicious")
    description: str
    artifact: Optional[str] = None
    artifact_filename: Optional[str] = None


class VideoTemporalAnalysisResult(BaseModel):
    """Conservative temporal consistency analysis across sampled frames."""

    status: Literal["potential_signal", "no_strong_signal", "insufficient_data"]
    finding: Optional[str] = None
    description: str
    signals: List[TemporalConsistencySignal] = Field(default_factory=list)
    suspicious_frames: List[SuspiciousFrameItem] = Field(default_factory=list)


class VideoRecompressionAnalysisResult(BaseModel):
    """Conservative recompression and encoding consistency evaluation."""

    status: Literal["possible_recompression", "no_strong_signal", "insufficient_data"]
    description: str
    indicators: List[str] = Field(default_factory=list)
    limitation: str = Field(
        default="Possible recompression is not equivalent to manipulation and frequently results from standard platform transcoding or compression."
    )


class SceneChangeItem(BaseModel):
    """Detected visual transition between consecutive sampled frames."""

    frame_index: int = Field(..., ge=0)
    timestamp_seconds: float = Field(..., ge=0.0)
    type: str = Field(default="visual_transition")
    difference_score: float = Field(..., ge=0.0)
    description: str = Field(
        default="Measurable visual transition observed between sampled frames; does not by itself establish an edit."
    )


class VideoSceneAnalysisResult(BaseModel):
    """Basic scene/cut transition analysis."""

    status: Literal["available", "not_available"]
    scene_changes: List[SceneChangeItem] = Field(default_factory=list)


class VideoAiDetectorInterpretation(BaseModel):
    """Conservative forensic interpretation of the video AI detector output."""

    what_detector_observes: str = Field(
        default="The detector evaluates sampled facial video frames for statistical patterns "
        "associated with face-swap deepfake generation (FaceForensics++ C23 domain)."
    )
    what_detector_cannot_prove: str = Field(
        default="A single model score cannot prove or disprove that a video is AI-generated, "
        "authentic, or unmanipulated. The detector is trained on a specific deepfake dataset "
        "and may not generalize to other generation methods or real-world conditions."
    )


class VideoAiDetectionResult(BaseModel):
    """Stage A — VideoMAE-based facial deepfake detection result."""

    status: Literal["not_available", "completed", "error", "unavailable"] = Field(
        default="not_available"
    )
    reason: str = Field(
        default="No trained AI-generated video detection model is currently configured."
    )
    detector_enabled: bool = Field(default=False)
    model_name: Optional[str] = Field(default=None)
    model_version: Optional[str] = Field(default=None)
    architecture: Optional[str] = Field(default=None)
    license: Optional[str] = Field(default=None)
    training_domain: Optional[str] = Field(default=None)
    artifact_hashes: Dict[str, str] = Field(default_factory=dict)
    id2label: Dict[str, str] = Field(
        default_factory=lambda: {"0": "real", "1": "fake"}
    )
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    selected_class: Optional[str] = Field(default=None)
    real_score: Optional[float] = Field(default=None)
    fake_score: Optional[float] = Field(default=None)
    score_type: str = Field(default="uncalibrated_model_score")
    decision_rule: str = Field(default="model_class_selection")
    threshold: Optional[float] = Field(default=None)
    frames_sampled: Optional[int] = Field(default=None)
    usable_detector_frames: Optional[int] = Field(default=None)
    detector_scope: Optional[str] = Field(default="facial_video_deepfake_detection")
    interpretation: Optional[VideoAiDetectorInterpretation] = Field(default=None)
    limitations: List[str] = Field(default_factory=lambda: [
        "This detector is trained on FaceForensics++ C23 face-swap deepfakes only.",
        "The model score is uncalibrated and is not a probability of AI generation.",
        "Absence of a 'fake' signal does not prove the video is authentic.",
        "Presence of a 'fake' signal does not prove the video is AI-generated.",
        "The detector requires visible facial regions; non-facial video is out of scope.",
        "Real-world transformations (compression, resizing) may alter detector output.",
    ])


class VideoForensicAssessment(BaseModel):
    """Conservative qualitative assessment of the video analysis."""

    status: AllowedVideoAssessment
    label: str
    description: str


class VideoAnalysisResult(BaseModel):
    """Structured result for Step 10 Video Forensic Analysis."""

    analyzed_at: str
    media_type: Literal["video"] = Field(default="video")
    filename: str
    format: str
    container: Optional[str] = None
    codec: Optional[str] = None
    codec_profile: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    frame_rate: Optional[float] = None
    frame_count: Optional[int] = None
    duration_seconds: Optional[float] = None
    bitrate: Optional[int] = None
    pixel_format: Optional[str] = None
    color_space: Optional[str] = None
    rotation: Optional[int] = None
    has_audio: bool = False
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    audio_channel_count: Optional[int] = None
    metadata: VideoMetadataInfo
    video_level_findings: List[VideoForensicFinding] = Field(default_factory=list)
    frame_sampling: VideoFrameSamplingResult
    temporal_analysis: VideoTemporalAnalysisResult
    recompression_analysis: VideoRecompressionAnalysisResult
    scene_analysis: VideoSceneAnalysisResult
    ai_generation: VideoAiDetectionResult = Field(default_factory=VideoAiDetectionResult)
    assessment: VideoForensicAssessment
    limitations: List[str] = Field(default_factory=list)
