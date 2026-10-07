from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

AllowedSignalCategory = Literal[
    "integrity",
    "metadata",
    "compression",
    "ai_detector",
    "localization",
    "reference_comparison",
    "temporal",
    "structural",
]

AllowedSignalAvailability = Literal[
    "available",
    "unavailable",
    "not_evaluated",
]

AllowedSignalStrength = Literal[
    "informational",
    "weak",
    "moderate",
    "unavailable",
]

AllowedEvidentiaryRole = Literal[
    "preservation_evidence",
    "contextual_metadata",
    "classical_forensic_observation",
    "auxiliary_model_signal",
    "spatial_anomaly_observation",
    "comparative_difference_observation",
    "cross_analysis_correlation",
    "unavailable_signal",
]

AllowedIndependenceGroup = Literal[
    "file_integrity",
    "metadata_and_timestamps",
    "jpeg_and_compression",
    "pixel_localization",
    "reference_comparison",
    "ai_detector",
    "structural_container",
]

AllowedConsistencyValue = Literal[
    "consistent",
    "partially_consistent",
    "conflicting",
    "insufficient_information",
    "not_applicable",
]

AllowedSignalRelationshipValue = Literal[
    "supports_consistency",
    "partial_consistency",
    "independent",
    "conflict",
    "unavailable",
    "not_comparable",
]

AllowedEvidenceClusterCategory = Literal[
    "integrity_observation",
    "metadata_observation",
    "compression_observation",
    "ai_model_observation",
    "localization_observation",
    "comparison_observation",
    "cross_signal_consistency",
    "cross_signal_conflict",
]

AllowedOverallForensicAssessment = Literal[
    "insufficient_evidence",
    "no_significant_forensic_evidence",
    "isolated_forensic_signal",
    "multiple_consistent_forensic_observations",
    "mixed_or_conflicting_forensic_observations",
]


class ForensicSignalItem(BaseModel):
    """Explicit representation of a single forensic signal separating observation from interpretation."""

    signal_id: str = Field(..., description="Deterministic signal identifier (e.g. SIG-001)")
    signal_category: AllowedSignalCategory
    availability: AllowedSignalAvailability
    observation: str = Field(
        ...,
        description="Direct factual observation reported by the underlying service without speculative conclusions",
    )
    forensic_interpretation: str = Field(
        ...,
        description="Conservative forensic interpretation of what the observation means and what it does not establish",
    )
    strength: AllowedSignalStrength = Field(
        ...,
        description="Qualitative evidentiary strength ('informational', 'weak', 'moderate', or 'unavailable')",
    )
    evidentiary_role: AllowedEvidentiaryRole
    source_service: str = Field(..., description="Originating MediaTrace pipeline service")
    independence_group: AllowedIndependenceGroup = Field(
        ...,
        description="Explicit independence group preventing double-counting of signals from the same analytical source",
    )
    evidence_ids: List[str] = Field(
        default_factory=list,
        description="Correlated Step 7 EvidenceItem IDs (e.g. EV-001) associated with this signal",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Explicit forensic limitations applicable to this signal",
    )
    supporting_artifacts: List[str] = Field(
        default_factory=list,
        description="Available non-destructive visual or data artifacts supporting this observation",
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured uncalibrated/factual parameters preserved from the underlying analysis stage",
    )


class IndependenceGroupSummary(BaseModel):
    """Summary of signals sharing a single underlying forensic evidence source."""

    group_id: AllowedIndependenceGroup
    group_label: str
    available: bool
    signal_ids: List[str] = Field(default_factory=list)
    meaningful_observation_present: bool = Field(
        default=False,
        description="Whether this group contains an active forensic anomaly or model artificial selection",
    )
    highest_strength: AllowedSignalStrength = "unavailable"
    summary_observation: str
    independence_note: str = Field(
        ...,
        description="Explains why multiple outputs within this group represent a single evidence source and are not counted as multiple independent signals",
    )


class SignalRelationshipItem(BaseModel):
    """Explicit qualitative consistency/conflict relationship between two signal groups or categories."""

    signal_a: str
    signal_b: str
    relationship: AllowedSignalRelationshipValue
    consistency: AllowedConsistencyValue
    reason: str


class EvidenceClusterItem(BaseModel):
    """Qualitative evidence cluster grouping related signals without double-counting."""

    cluster_id: str
    cluster_category: AllowedEvidenceClusterCategory
    independence_group: str
    signals: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    observation: str
    interpretation: str
    independence_note: str


class TransformationContextSummary(BaseModel):
    """Documented transformation context affecting forensic signals and model scores."""

    documented: bool = False
    observed_flags: List[str] = Field(
        default_factory=list,
        description="Observed flags from {'dimension_change_observed', 'recompression_observed', 'processing_metadata_present', 'reference_difference_observed'}",
    )
    statement: str = Field(
        default="No prior transformation is documented in the available evidence."
    )


class ForensicDecisionResult(BaseModel):
    """Step 14.2 Multi-Signal Forensic Decision & Interpretation Framework result."""

    case_id: str
    status: Literal["available", "unavailable"] = "available"
    signal_inventory: List[ForensicSignalItem] = Field(default_factory=list)
    independence_groups: List[IndependenceGroupSummary] = Field(default_factory=list)
    consistency_classification: AllowedConsistencyValue = "insufficient_information"
    signal_relationships: List[SignalRelationshipItem] = Field(default_factory=list)
    evidence_clusters: List[EvidenceClusterItem] = Field(default_factory=list)
    transformation_context: TransformationContextSummary = Field(
        default_factory=TransformationContextSummary
    )
    overall_assessment: AllowedOverallForensicAssessment = "insufficient_evidence"
    overall_assessment_label: str = Field(
        default="Insufficient forensic evidence available"
    )
    interpretation: str = Field(
        ...,
        description="Conservative multi-signal forensic interpretation preserving uncertainty",
    )
    supported_conclusions: List[str] = Field(default_factory=list)
    unsupported_conclusions: List[str] = Field(default_factory=list)
    alternative_explanations: List[str] = Field(default_factory=list)
    additional_evidence_recommendations: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    generated_at: str
