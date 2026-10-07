from typing import Any, List, Literal
from pydantic import BaseModel, Field

AllowedEvidenceCategory = Literal[
    "metadata",
    "compression",
    "manipulation",
    "localization",
    "comparison",
    "integrity",
]

AllowedEvidenceStrength = Literal[
    "informational",
    "weak",
    "moderate",
]

AllowedEvidenceAssessment = Literal[
    "no_significant_evidence",
    "multiple_forensic_signals",
    "localized_visual_differences",
    "insufficient_evidence",
]


class EvidenceItem(BaseModel):
    """Single correlated forensic evidence observation."""

    id: str = Field(..., description="Sequential evidence identifier (e.g. EV-001)")
    category: AllowedEvidenceCategory
    type: str = Field(..., description="Machine-readable evidence subtype (e.g. software_metadata, cross_analysis_overlap)")
    title: str = Field(..., description="Concise human-readable title")
    description: str = Field(..., description="Neutral explanation of what was observed")
    value: Any = Field(..., description="Observed value or structured correlation payload")
    strength: AllowedEvidenceStrength = Field(..., description="Qualitative signal strength ('informational', 'weak', or 'moderate')")
    source: str = Field(..., description="Originating pipeline stage (e.g. step3_basic_forensics, cross_analysis)")


class EvidenceSummary(BaseModel):
    """Aggregate statistics of correlated evidence items."""

    total_evidence_items: int = Field(..., ge=0)
    informational: int = Field(..., ge=0)
    weak: int = Field(..., ge=0)
    moderate: int = Field(..., ge=0)
    analysis_stages_available: List[str] = Field(default_factory=list)


class EvidenceReport(BaseModel):
    """Structured Step 7 Evidence Correlation Report."""

    case_id: str = Field(..., description="Forensic case identifier")
    generated_at: str = Field(..., description="UTC ISO-8601 timestamp when the evidence report was generated")
    assessment: AllowedEvidenceAssessment
    assessment_label: str
    summary: EvidenceSummary
    items: List[EvidenceItem] = Field(default_factory=list)
    disclaimer: str = Field(
        default="Correlated forensic evidence connects observations across analysis stages and does not by itself prove intentional manipulation."
    )
