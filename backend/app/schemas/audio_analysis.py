from typing import List, Literal, Optional
from pydantic import BaseModel, Field

AllowedAudioAssessment = Literal[
    "no_strong_indicators",
    "potential_audio_anomaly",
    "possible_discontinuity",
    "possible_recompression",
    "insufficient_evidence",
]


class AudioMetadataInfo(BaseModel):
    """Technical container, stream, and privacy-safe tag metadata extracted from the audio file."""

    format: Optional[str] = None
    container: Optional[str] = None
    codec: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    channel_layout: Optional[str] = None
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    encoder: Optional[str] = None
    software: Optional[str] = None
    creation_time: Optional[str] = None
    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    comment: Optional[str] = None
    gps_present: bool = False
    gps_notice: Optional[str] = Field(
        default=None,
        description="Reported strictly as 'GPS metadata present' when GPS/location metadata exists; raw coordinates are never exposed",
    )


class AudioWaveformAnalysis(BaseModel):
    """Conservative measurable waveform and amplitude statistics."""

    status: Literal["available", "insufficient_data"] = Field(default="available")
    peak_amplitude: Optional[float] = Field(default=None, ge=0.0)
    rms_level: Optional[float] = Field(default=None, ge=0.0)
    dynamic_range_db: Optional[float] = Field(default=None, ge=0.0)
    minimum_level: Optional[float] = None
    maximum_level: Optional[float] = None
    mean_level: Optional[float] = None
    clipping_sample_count: Optional[int] = Field(default=None, ge=0)
    analyzed_duration_seconds: Optional[float] = Field(default=None, ge=0.0)
    artifact: Optional[str] = Field(
        default=None,
        description="Relative artifact path under storage/results/<case_id>/audio/waveform.png",
    )
    artifact_filename: Optional[str] = Field(default=None, description="e.g. waveform.png")


class AudioSilenceRegion(BaseModel):
    """Low-amplitude / silence candidate region identified during bounded analysis."""

    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float = Field(..., ge=0.0)
    duration_seconds: float = Field(..., ge=0.0)
    type: Literal["low_amplitude_region", "silence_candidate"] = Field(
        default="low_amplitude_region"
    )
    mean_amplitude: Optional[float] = Field(default=None, ge=0.0)
    description: str = Field(
        default="Low-amplitude region observed; silence may be natural and does not prove audio deletion."
    )


class AudioSilenceAnalysis(BaseModel):
    """Bounded silence and low-amplitude gap analysis."""

    status: Literal["regions_detected", "no_regions_detected", "insufficient_data"]
    threshold_amplitude: float = Field(..., ge=0.0)
    min_silence_duration_seconds: float = Field(..., ge=0.0)
    region_count: int = Field(default=0, ge=0)
    regions: List[AudioSilenceRegion] = Field(default_factory=list)


class SpectralTransitionCandidate(BaseModel):
    """Candidate spectral characteristic shift within the analyzed audio."""

    timestamp_seconds: float = Field(..., ge=0.0)
    finding: str = Field(default="spectral_characteristic_shift")
    spectral_flux_score: float = Field(..., ge=0.0)
    description: str = Field(
        default="Measurable shift in short-time spectral characteristics; may result from legitimate sound transitions."
    )


class AudioSpectralAnalysis(BaseModel):
    """Conservative short-time Fourier transform (STFT) spectral analysis."""

    status: Literal["potential_signal", "no_strong_signal", "insufficient_data"]
    finding: Optional[str] = None
    description: str
    spectral_centroid_mean_hz: Optional[float] = Field(default=None, ge=0.0)
    spectral_rolloff_mean_hz: Optional[float] = Field(default=None, ge=0.0)
    spectral_flux_mean: Optional[float] = Field(default=None, ge=0.0)
    high_frequency_energy_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    candidate_transitions: List[SpectralTransitionCandidate] = Field(default_factory=list)
    artifact: Optional[str] = Field(
        default=None,
        description="Relative artifact path under storage/results/<case_id>/audio/spectrogram.png",
    )
    artifact_filename: Optional[str] = Field(default=None, description="e.g. spectrogram.png")


class AudioDiscontinuityCandidate(BaseModel):
    """Candidate abrupt waveform or spectral discontinuity for manual forensic review."""

    start_seconds: float = Field(..., ge=0.0)
    end_seconds: float = Field(..., ge=0.0)
    duration_seconds: float = Field(..., ge=0.0)
    finding: str = Field(default="possible_audio_discontinuity")
    status: Literal["potentially_suspicious"] = Field(default="potentially_suspicious")
    description: str = Field(
        default="A measurable abrupt change in waveform/spectral characteristics was detected."
    )
    artifact: Optional[str] = None
    artifact_filename: Optional[str] = None


class AudioDiscontinuityAnalysis(BaseModel):
    """Conservative discontinuity analysis across the audio stream."""

    status: Literal["potential_signal", "no_strong_signal", "insufficient_data"]
    description: str
    candidate_count: int = Field(default=0, ge=0)
    candidates: List[AudioDiscontinuityCandidate] = Field(default_factory=list)


class AudioRecompressionAnalysis(BaseModel):
    """Conservative transcoding and recompression indicator analysis."""

    status: Literal["possible_recompression", "no_strong_signal", "insufficient_data"]
    description: str
    indicators: List[str] = Field(default_factory=list)
    limitation: str = Field(
        default="Possible recompression is not equivalent to manipulation."
    )


class AudioAiDetectionResult(BaseModel):
    """Modular AI-generated audio detector status."""

    status: Literal["not_available"] = Field(default="not_available")
    reason: str = Field(
        default="No trained AI-generated audio detection model is currently configured."
    )


class AudioForensicAssessment(BaseModel):
    """Conservative qualitative forensic assessment of the audio analysis."""

    status: AllowedAudioAssessment
    label: str
    description: str


class AudioAnalysisResult(BaseModel):
    """Structured result for Step 11 Audio Forensic Analysis."""

    analyzed_at: str
    media_type: Literal["audio"] = Field(default="audio")
    filename: str
    format: str
    container: Optional[str] = None
    codec: Optional[str] = None
    duration_seconds: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bit_depth: Optional[int] = None
    bitrate: Optional[int] = None
    metadata: AudioMetadataInfo
    waveform_analysis: AudioWaveformAnalysis
    silence_analysis: AudioSilenceAnalysis
    spectral_analysis: AudioSpectralAnalysis
    discontinuity_analysis: AudioDiscontinuityAnalysis
    recompression_analysis: AudioRecompressionAnalysis
    ai_generation: AudioAiDetectionResult = Field(default_factory=AudioAiDetectionResult)
    assessment: AudioForensicAssessment
    limitations: List[str] = Field(default_factory=list)
