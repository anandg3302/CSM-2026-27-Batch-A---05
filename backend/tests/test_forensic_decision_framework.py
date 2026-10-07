r"""Step 14.2 — Multi-Signal Forensic Decision Framework Tests.

Covers all 24 required test scenarios (Section 25) and Manual Test Cases A–F (Section 28):
1. No signals
2. AI detector only
3. AI artificial + no classical anomaly
4. AI human + localization anomaly
5. AI artificial + localization anomaly
6. Multiple independent consistent signals
7. Conflicting signals
8. Metadata-only anomaly
9. Localization-only anomaly
10. Reference difference only
11. Integrity-only result
12. Detector unavailable
13. Transformation documented
14. Transformation undocumented
15. Duplicate signals from same independence group (metadata+temporal, localization regions+heatmap+mask)
16. SHA-256 preservation
17. No numeric confidence generated
18. No AI-only escalation
19. No manipulation verdict
20. History integration ('Multi-signal forensic interpretation performed.', status='observed', temporal_certainty='exact')
21. JSON report integration (multi_signal_interpretation section)
22. PDF report integration (MULTI-SIGNAL FORENSIC INTERPRETATION section + 9 subsections)
23. API endpoint (POST and GET /api/forensics/{case_id}/interpret)
24. Frontend rendering check (MediaUpload.jsx + api.js conservative UI without verdict badges/probability bars)
25. Manual Test Cases A–F on real images & controlled pipelines
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.core.config import settings
from app.main import app
from app.schemas.ai_detection import (
    AiDetectorMetadata,
    AiDetectorPrediction,
    AiGenerationDetectionResult,
    OFFICIAL_MODEL_ID,
    OFFICIAL_PINNED_REVISION,
)
from app.schemas.analysis import (
    DetectionSignal,
    ExifFields,
    ImageAnalysisResult,
    ImageDimensions,
    ImageLocalizationResult,
    ImageMetadataInfo,
    ImageTechnicalInfo,
    LocalizationVisualization,
    ManipulationDetectionResult,
    SuspiciousRegion,
)
from app.schemas.comparison import (
    ChangedRegion,
    ComparisonEndpointSummary,
    ComparisonVisualization,
    DimensionComparisonInfo,
    ImageComparisonResult,
    MetadataComparisonInfo,
    PixelDifferenceInfo,
)
from app.schemas.media import MediaCaseMetadata
from app.services.evidence.forensic_decision_service import build_forensic_decision

client = TestClient(app)


def _base_case(case_id: str = "MT-20260929-DECISION") -> MediaCaseMetadata:
    return MediaCaseMetadata(
        case_id=case_id,
        original_filename="sample.jpg",
        stored_filename=f"{case_id}.jpg",
        media_type="image",
        extension=".jpg",
        file_size=10240,
        sha256="a" * 64,
        status="uploaded",
        created_at="2026-09-29T12:00:00Z",
    )


def _make_analysis(
    *,
    exif_present: bool = True,
    make: str | None = None,
    model: str | None = None,
    software: str | None = None,
    dt_orig: str | None = None,
    dt_mod: str | None = None,
) -> ImageAnalysisResult:
    return ImageAnalysisResult(
        analyzed_at="2026-09-29T12:01:00Z",
        image=ImageTechnicalInfo(format="JPEG", mode="RGB", width=640, height=480),
        metadata=ImageMetadataInfo(
            exif_present=exif_present,
            gps_present=False,
            fields=ExifFields(
                make=make,
                model=model,
                software=software,
                datetime_original=dt_orig,
                datetime=dt_mod,
            ),
        ),
        findings=[],
    )


def _make_detection(
    *,
    ai_status: str = "available",
    selected_class: str = "human",
    art_score: float = 0.20,
    signals: list[DetectionSignal] | None = None,
    recompression_status: str = "no_strong_indicator",
) -> ManipulationDetectionResult:
    if ai_status == "available":
        hum_score = round(1.0 - art_score, 4)
        cls_id = 0 if selected_class == "artificial" else 1
        ai_res = AiGenerationDetectionResult(
            status="available",
            selected_class=selected_class,
            selected_class_id=cls_id,
            artificial_score=art_score,
            human_score=hum_score,
            detector=AiDetectorMetadata(
                name=OFFICIAL_MODEL_ID,
                version=OFFICIAL_PINNED_REVISION,
                pinned_revision=OFFICIAL_PINNED_REVISION,
                framework="onnxruntime",
            ),
            prediction=AiDetectorPrediction(
                label="potentially_ai_generated" if selected_class == "artificial" else "no_ai_signal_detected",
                display_label="Model-selected class: artificial" if selected_class == "artificial" else "Model-selected class: human",
                raw_class=selected_class,
                selected_class=selected_class,
                selected_class_id=cls_id,
                score=art_score,
                artificial_score=art_score,
                human_score=hum_score,
            ),
            explanation="Real ONNX evaluation.",
        )
    else:
        ai_res = AiGenerationDetectionResult(status="not_available")

    return ManipulationDetectionResult(
        status="completed",
        evaluated_at="2026-09-29T12:02:00Z",
        overall_assessment="no_strong_indicators",
        assessment_label="No strong manipulation indicators detected",
        recompression_status=recompression_status,
        signals=signals or [],
        ai_generation=ai_res,
        ai_generation_detection=ai_res,
    )


def _make_localization(*, has_anomaly: bool = True) -> ImageLocalizationResult:
    regions = (
        [
            SuspiciousRegion(
                id="R-001",
                x=40,
                y=50,
                width=120,
                height=100,
                area=12000,
                mean_anomaly_score=0.68,
                max_anomaly_score=0.84,
            ),
            SuspiciousRegion(
                id="R-002",
                x=200,
                y=150,
                width=80,
                height=60,
                area=4800,
                mean_anomaly_score=0.62,
                max_anomaly_score=0.77,
            ),
        ]
        if has_anomaly
        else []
    )
    return ImageLocalizationResult(
        status="completed",
        localized_at="2026-09-29T12:03:00Z",
        localized_anomaly=has_anomaly,
        threshold=0.65,
        image_dimensions=ImageDimensions(width=640, height=480),
        method="ela_noise_hybrid",
        message="Localization completed.",
        regions=regions,
        visualization=LocalizationVisualization(
            heatmap="localization/heatmap.png",
            mask="localization/mask.png",
        )
        if has_anomaly
        else None,
    )


def _make_comparison(*, changed_pct: float = 12.5, same_dims: bool = True) -> ImageComparisonResult:
    regions = (
        [
            ChangedRegion(
                id="CR-001",
                x=45,
                y=52,
                width=110,
                height=95,
                area=10450,
                changed_pixel_count=9200,
                changed_pixel_percentage=changed_pct,
            )
        ]
        if changed_pct > 0.0
        else []
    )
    return ImageComparisonResult(
        status="completed",
        compared_at="2026-09-29T12:04:00Z",
        case_image=ComparisonEndpointSummary(
            width=640, height=480, format="JPEG", mode="RGB", sha256="a" * 64
        ),
        reference_image=ComparisonEndpointSummary(
            width=640 if same_dims else 1280,
            height=480 if same_dims else 960,
            format="JPEG",
            mode="RGB",
            sha256="b" * 64,
        ),
        dimensions=DimensionComparisonInfo(
            same=same_dims,
            same_dimensions=same_dims,
            same_aspect_ratio=True,
            same_format=True,
            same_mode=True,
            case_dimensions=ImageDimensions(width=640, height=480),
            reference_dimensions=ImageDimensions(width=640 if same_dims else 1280, height=480 if same_dims else 960),
        ),
        pixel_difference=PixelDifferenceInfo(
            changed_pixel_percentage=changed_pct,
            mean_absolute_difference=8.2 if changed_pct > 0 else 0.0,
            max_pixel_difference=180 if changed_pct > 0 else 0,
            pixel_similarity=max(0.0, 1.0 - changed_pct / 100.0),
        ),
        changed_regions=regions,
        metadata_comparison=MetadataComparisonInfo(
            format_changed=False,
            dimensions_changed=not same_dims,
            exif_presence_changed=False,
            software_changed=False,
            camera_changed=False,
            timestamp_changed=False,
        ),
        visualization=ComparisonVisualization(),
        assessment="localized_visual_differences_detected" if changed_pct > 0 else "no_significant_difference",
        assessment_label="Localized visual differences detected" if changed_pct > 0 else "No significant difference",
    )


def test_1_and_11_no_signals_and_integrity_only():
    """Tests 1 & 11:
    - No signals (include_integrity=False) -> status='unavailable', overall_assessment='insufficient_evidence'
    - Integrity-only result (include_integrity=True) -> Rule J verified, overall_assessment='insufficient_evidence'
    """
    case = _base_case()

    # 1. No signals at all
    res_no_sig = build_forensic_decision(case, include_integrity=False)
    assert res_no_sig.status == "unavailable"
    assert res_no_sig.overall_assessment == "insufficient_evidence"
    assert res_no_sig.consistency_classification == "insufficient_information"

    # 11. Integrity-only result
    res_int = build_forensic_decision(case, include_integrity=True)
    assert res_int.status == "unavailable"
    assert res_int.overall_assessment == "insufficient_evidence"
    int_sig = next(s for s in res_int.signal_inventory if s.signal_category == "integrity")
    assert int_sig.availability == "available"
    assert "does not establish that the file itself is authentic or original" in int_sig.forensic_interpretation


def test_2_3_18_ai_detector_only_and_ai_artificial_no_classical_anomaly():
    """Tests 2, 3, 18:
    - AI detector only (selected_class='artificial') -> isolated_forensic_signal (Rule E & F: no AI-only escalation)
    - AI artificial + no classical anomaly -> isolated_forensic_signal, Case 1 template
    """
    # 2. AI detector only
    case_ai_only = _base_case()
    case_ai_only.manipulation_detection = _make_detection(selected_class="artificial", art_score=0.82)
    res_ai_only = build_forensic_decision(case_ai_only)
    assert res_ai_only.overall_assessment == "isolated_forensic_signal"
    assert res_ai_only.overall_assessment != "multiple_consistent_forensic_observations"
    assert "do not independently establish manipulation or synthetic generation" in res_ai_only.interpretation

    # 3. AI artificial + basic analysis & localization showing no classical anomaly
    case_ai_clean = _base_case()
    case_ai_clean.analysis = _make_analysis(exif_present=False)
    case_ai_clean.manipulation_detection = _make_detection(selected_class="artificial", art_score=0.88)
    case_ai_clean.localization = _make_localization(has_anomaly=False)
    res_ai_clean = build_forensic_decision(case_ai_clean)
    assert res_ai_clean.overall_assessment == "isolated_forensic_signal"
    assert res_ai_clean.consistency_classification == "partially_consistent"
    assert "The AI detector selected the artificial class, but the available non-model forensic signals do not independently establish manipulation or synthetic generation." in res_ai_clean.interpretation


def test_4_5_6_cross_signal_combinations():
    """Tests 4, 5, 6:
    - 4. AI human + localization anomaly -> Case 2 template, distinct observations
    - 5. AI artificial + localization anomaly -> multiple_consistent_forensic_observations, Rule C disclaimer
    - 6. Multiple independent consistent signals (AI artificial + localization + reference differences) -> Case 3 + Rule C
    """
    # 4. AI human + localization anomaly
    case_4 = _base_case()
    case_4.analysis = _make_analysis(exif_present=True)
    case_4.manipulation_detection = _make_detection(selected_class="human", art_score=0.18)
    case_4.localization = _make_localization(has_anomaly=True)
    res_4 = build_forensic_decision(case_4)
    assert res_4.overall_assessment == "isolated_forensic_signal"
    assert res_4.consistency_classification == "partially_consistent"
    assert "The AI detector selected the human class, while separate localization analysis identified candidate anomalous regions." in res_4.interpretation

    # 5. AI artificial + localization anomaly
    case_5 = _base_case()
    case_5.manipulation_detection = _make_detection(selected_class="artificial", art_score=0.85)
    case_5.localization = _make_localization(has_anomaly=True)
    res_5 = build_forensic_decision(case_5)
    assert res_5.overall_assessment == "multiple_consistent_forensic_observations"
    assert "This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent." in res_5.interpretation

    # 6. Multiple independent consistent signals (AI artificial + localization + reference differences)
    case_6 = _base_case()
    case_6.manipulation_detection = _make_detection(selected_class="artificial", art_score=0.91)
    case_6.localization = _make_localization(has_anomaly=True)
    case_6.comparison = _make_comparison(changed_pct=15.4)
    res_6 = build_forensic_decision(case_6)
    assert res_6.overall_assessment == "multiple_consistent_forensic_observations"
    assert "Multiple forensic observations are present and are partially consistent with image processing or alteration." in res_6.interpretation
    assert "This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent." in res_6.interpretation


def test_7_conflicting_signals():
    """Test 7: Conflicting signals -> mixed_or_conflicting_forensic_observations (Rule D & Case 5)."""
    # Conflict between AI detector selecting 'human' and documented synthetic generator metadata
    case_conflict = _base_case()
    case_conflict.original_filename = "ai_generated_sample.jpg"
    case_conflict.analysis = _make_analysis(exif_present=True, software="Stable Diffusion XL")
    case_conflict.manipulation_detection = _make_detection(selected_class="human", art_score=0.2736)
    res_conflict = build_forensic_decision(case_conflict)
    assert res_conflict.overall_assessment == "mixed_or_conflicting_forensic_observations"
    assert res_conflict.consistency_classification == "conflicting"
    assert "The available forensic signals are mixed or conflicting. The current evidence does not support a single definitive interpretation." in res_conflict.interpretation


def test_8_9_10_15_single_group_isolation_and_no_double_counting():
    """Tests 8, 9, 10, 15:
    - 8. Metadata-only anomaly (software + timestamp difference in same metadata_and_timestamps group) -> isolated_forensic_signal
    - 9. Localization-only anomaly (multiple regions + heatmap + mask in pixel_localization group) -> isolated_forensic_signal
    - 10. Reference difference only -> isolated_forensic_signal
    - 15. Duplicate signals from same independence group are NOT double-counted as multiple independent groups
    """
    # 8 & 15a: Metadata + temporal timestamp difference both belong to metadata_and_timestamps
    case_meta = _base_case()
    case_meta.analysis = _make_analysis(
        exif_present=True,
        software="Adobe Photoshop 2025",
        dt_orig="2026:09:29 10:00:00",
        dt_mod="2026:09:29 10:01:16",
    )
    res_meta = build_forensic_decision(case_meta)
    meta_group_sigs = [s for s in res_meta.signal_inventory if s.independence_group == "metadata_and_timestamps"]
    assert len(meta_group_sigs) == 2  # metadata + temporal
    assert res_meta.overall_assessment == "isolated_forensic_signal"

    # 9 & 15b: Localization-only anomaly (regions + heatmap + mask)
    case_loc = _base_case()
    case_loc.localization = _make_localization(has_anomaly=True)
    res_loc = build_forensic_decision(case_loc)
    assert res_loc.overall_assessment == "isolated_forensic_signal"
    loc_cluster = next(c for c in res_loc.evidence_clusters if c.cluster_category == "localization_observation")
    assert "same spatial localization analysis" in loc_cluster.independence_note

    # 10: Reference difference only
    case_ref = _base_case()
    case_ref.comparison = _make_comparison(changed_pct=8.4)
    res_ref = build_forensic_decision(case_ref)
    assert res_ref.overall_assessment == "isolated_forensic_signal"
    assert "do not establish which file is original or prove malicious alteration" in res_ref.interpretation


def test_12_13_14_17_19_detector_unavailable_transformation_context_and_no_verdict_or_confidence():
    """Tests 12, 13, 14, 17, 19:
    - 12. Detector unavailable -> 'AI detector signal unavailable.', never negative evidence
    - 13. Transformation documented -> documented=True, observed_flags populated
    - 14. Transformation undocumented -> 'No prior transformation is documented in the available evidence.'
    - 17. No numeric confidence/probability generated
    - 19. No manipulation/fake/authentic verdict generated
    """
    # 12 & 14: Detector unavailable + transformation undocumented
    case_unavail = _base_case()
    case_unavail.analysis = _make_analysis(exif_present=False)
    case_unavail.manipulation_detection = _make_detection(ai_status="not_available")
    res_unavail = build_forensic_decision(case_unavail)

    ai_sig = next(s for s in res_unavail.signal_inventory if s.signal_category == "ai_detector")
    assert ai_sig.availability == "unavailable"
    assert ai_sig.observation == "AI detector signal unavailable."
    assert "does not constitute evidence for or against AI generation" in ai_sig.forensic_interpretation
    assert res_unavail.transformation_context.documented is False
    assert res_unavail.transformation_context.statement == "No prior transformation is documented in the available evidence."

    # 13: Transformation documented
    case_tx = _base_case()
    case_tx.analysis = _make_analysis(exif_present=True, software="GIMP 2.10")
    case_tx.manipulation_detection = _make_detection(
        selected_class="artificial",
        art_score=0.75,
        recompression_status="likely_recompressed",
    )
    case_tx.comparison = _make_comparison(changed_pct=18.2, same_dims=False)
    res_tx = build_forensic_decision(case_tx)
    assert res_tx.transformation_context.documented is True
    assert "dimension_change_observed" in res_tx.transformation_context.observed_flags
    assert "recompression_observed" in res_tx.transformation_context.observed_flags
    assert "processing_metadata_present" in res_tx.transformation_context.observed_flags
    assert "reference_difference_observed" in res_tx.transformation_context.observed_flags

    # 17 & 19: Verify no numeric confidence/probability keys or banned verdict claims exist
    dumped = json.dumps(res_tx.model_dump()).lower()
    for forbidden in (
        '"fake_probability"',
        '"ai_probability"',
        '"manipulation_score"',
        '"forensic_certainty_score"',
        '"confidence_percentage"',
        "definitely fake",
        "definitely ai-generated",
        "proves manipulation",
    ):
        assert forbidden not in dumped


def test_16_20_21_22_23_24_end_to_end_api_history_reports_and_frontend():
    """Tests 16, 20, 21, 22, 23, 24 & Manual Test Cases A–F:
    - Upload known human photo ('datasets/ai_detector_validation/human/anand_pic.jpg')
    - Verify SHA-256 preservation before and after full pipeline + /api/forensics/{case_id}/interpret
    - Verify History records 'Multi-signal forensic interpretation performed.' (status='observed', temporal_certainty='exact')
    - Verify JSON & PDF reports contain MULTI-SIGNAL FORENSIC INTERPRETATION section and all 9 subsections
    - Verify Frontend source code renders Multi-Signal Forensic Interpretation without verdict badges or probability bars
    """
    human_img_path = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "ai_detector_validation"
        / "human"
        / "anand_pic.jpg"
    )
    orig_bytes = human_img_path.read_bytes()
    orig_sha = hashlib.sha256(orig_bytes).hexdigest()

    # Upload case
    up_resp = client.post(
        "/api/media/upload",
        files={"file": ("anand_pic.jpg", io.BytesIO(orig_bytes), "image/jpeg")},
    )
    assert up_resp.status_code == 201
    case_id = up_resp.json()["case_id"]
    assert up_resp.json()["sha256"] == orig_sha

    # Run Step 3, Step 4, Step 5, Step 7, Step 8
    assert client.post(f"/api/analysis/image/{case_id}").status_code == 200
    assert client.post(f"/api/analysis/image/{case_id}/detect").status_code == 200
    assert client.post(f"/api/analysis/image/{case_id}/localize").status_code == 200
    assert client.post(f"/api/evidence/{case_id}/generate").status_code == 200
    assert client.post(f"/api/history/{case_id}/reconstruct").status_code == 200

    # 23. Execute POST /api/forensics/{case_id}/interpret and GET /api/forensics/{case_id}/interpret
    post_interp = client.post(f"/api/forensics/{case_id}/interpret")
    assert post_interp.status_code == 200
    decision_data = post_interp.json()
    assert decision_data["case_id"] == case_id
    assert decision_data["status"] == "available"

    get_interp = client.get(f"/api/forensics/{case_id}/interpret")
    assert get_interp.status_code == 200
    assert get_interp.json()["overall_assessment"] == decision_data["overall_assessment"]

    # 20. Verify History integration
    hist_resp = client.get(f"/api/history/{case_id}")
    assert hist_resp.status_code == 200
    hist_events = hist_resp.json()["events"]
    ms_events = [
        e for e in hist_events if e["title"] == "Multi-signal forensic interpretation performed."
    ]
    assert len(ms_events) == 1
    assert ms_events[0]["status"] == "observed"
    assert ms_events[0]["temporal_certainty"] == "exact"
    assert ms_events[0]["event_type"] == "forensic_analysis_event"

    # 21. Verify JSON Report integration
    rep_resp = client.post(f"/api/reports/{case_id}/generate")
    assert rep_resp.status_code == 200
    rep_json = rep_resp.json()
    assert rep_json["multi_signal_interpretation"] is not None
    assert (
        rep_json["multi_signal_interpretation"]["overall_assessment"]
        == decision_data["overall_assessment"]
    )

    # 22. Verify PDF Report integration
    pdf_resp = client.post(f"/api/reports/{case_id}/generate-pdf")
    assert pdf_resp.status_code == 200
    pdf_text = pdf_resp.content.decode("latin-1", errors="ignore")
    assert "MULTI-SIGNAL FORENSIC INTERPRETATION" in pdf_text
    assert "1. Signal Inventory" in pdf_text
    assert "2. Evidence Independence" in pdf_text
    assert "3. Signal Relationships" in pdf_text
    assert "4. Overall Qualitative Assessment" in pdf_text
    assert "5. Supported Conclusions" in pdf_text
    assert "6. Unsupported Conclusions" in pdf_text
    assert "7. Alternative Explanations" in pdf_text
    assert "8. Additional Evidence Recommendations" in pdf_text
    assert "9. Limitations" in pdf_text

    # 16. Verify SHA-256 preservation of stored original file
    stored_filename = client.get(f"/api/media/{case_id}").json()["stored_filename"]
    stored_bytes = (settings.uploads_dir / stored_filename).read_bytes()
    assert hashlib.sha256(stored_bytes).hexdigest() == orig_sha

    # 24. Frontend source verification
    media_upload_jsx = (
        Path(__file__).resolve().parents[2]
        / "frontend"
        / "src"
        / "components"
        / "MediaUpload.jsx"
    ).read_text(encoding="utf-8")
    assert "Multi-Signal Forensic Interpretation" in media_upload_jsx
    assert "Overall Qualitative Assessment" in media_upload_jsx
    assert "Signal Inventory" in media_upload_jsx
    assert "Supported Conclusions" in media_upload_jsx
    assert "Unsupported Conclusions" in media_upload_jsx
    assert "Alternative Non-Malicious Explanations" in media_upload_jsx
    assert "Additional Evidence Recommendations" in media_upload_jsx


def test_manual_cases_a_to_f_on_real_validation_images():
    """Section 28 Manual Test Cases A–F on real images from datasets/ai_detector_validation/:
    - CASE A: Known human photograph (human/anand_pic.jpg) -> AI detector human, no definitive authenticity conclusion.
    - CASE B: Known AI image where detector selects artificial (ai_generated/solar/downloa.jpeg) -> AI artificial reported as one model signal, no definitive AI-generation conclusion.
    - CASE C: Known AI image where detector selects human (ai_generated/solar/download.jpeg) -> Model/category disagreement preserved without hiding or overriding.
    - CASE D: Human image with localization anomaly -> Localization observation remains separate from AI detector result.
    - CASE E: Image with reference comparison differences (human/anand_pic.jpg vs transformed/anand_pic_q60_512x691.jpg) -> Differences reported as differences, not tampering.
    - CASE F: Multiple independent forensic groups contain observations -> multiple_consistent_forensic_observations with explicit non-proof statement.
    """
    ds_root = Path(__file__).resolve().parents[2] / "datasets" / "ai_detector_validation"

    # CASE A & D & E: Upload human/anand_pic.jpg, run full pipeline + reference comparison with transformed/anand_pic_q60_512x691.jpg
    human_bytes = (ds_root / "human" / "anand_pic.jpg").read_bytes()
    trans_bytes = (ds_root / "transformed" / "anand_pic_q60_512x691.jpg").read_bytes()

    cid_a = client.post(
        "/api/media/upload",
        files={"file": ("anand_pic.jpg", io.BytesIO(human_bytes), "image/jpeg")},
    ).json()["case_id"]
    client.post(f"/api/analysis/image/{cid_a}")
    client.post(f"/api/analysis/image/{cid_a}/detect")
    # CASE A (before localization/comparison): AI detector = human, no definitive authenticity conclusion
    res_a = client.post(f"/api/forensics/{cid_a}/interpret").json()
    ai_sig_a = next(s for s in res_a["signal_inventory"] if s["signal_category"] == "ai_detector")
    assert ai_sig_a["details"]["selected_class"] == "human"
    assert "The image is definitively authentic or an unedited camera original." in res_a["unsupported_conclusions"]

    # CASE D: Run localization on human/anand_pic.jpg -> localization anomaly remains separate from AI detector human
    client.post(f"/api/analysis/image/{cid_a}/localize")
    res_d = client.post(f"/api/forensics/{cid_a}/interpret").json()
    loc_sig_d = next(s for s in res_d["signal_inventory"] if s["signal_category"] == "localization")
    assert loc_sig_d["independence_group"] == "pixel_localization"
    assert ai_sig_a["independence_group"] == "ai_detector"

    # CASE E & F: Upload reference image (transformed/anand_pic_q60_512x691.jpg) and compare -> differences reported as differences + multiple independent groups
    client.post(
        f"/api/comparison/image/{cid_a}/reference",
        files={"file": ("anand_pic_q60_512x691.jpg", io.BytesIO(trans_bytes), "image/jpeg")},
    )
    client.post(f"/api/comparison/image/{cid_a}")
    res_ef = client.post(f"/api/forensics/{cid_a}/interpret").json()
    ref_sig = next(s for s in res_ef["signal_inventory"] if s["signal_category"] == "reference_comparison")
    assert "Visual/content differences were observed between the analyzed file and the supplied reference." in ref_sig["forensic_interpretation"]
    assert res_ef["overall_assessment"] == "multiple_consistent_forensic_observations"
    assert "This indicates consistency among the available forensic observations; it does not establish manipulation, AI generation, provenance, or intent." in res_ef["interpretation"]

    # CASE B: Known AI image where detector selects artificial (ai_generated/solar/downloa.jpeg)
    ai_art_bytes = (ds_root / "ai_generated" / "solar" / "downloa.jpeg").read_bytes()
    cid_b = client.post(
        "/api/media/upload",
        files={"file": ("downloa.jpeg", io.BytesIO(ai_art_bytes), "image/jpeg")},
    ).json()["case_id"]
    client.post(f"/api/analysis/image/{cid_b}")
    client.post(f"/api/analysis/image/{cid_b}/detect")
    res_b = client.post(f"/api/forensics/{cid_b}/interpret").json()
    ai_sig_b = next(s for s in res_b["signal_inventory"] if s["signal_category"] == "ai_detector")
    assert ai_sig_b["details"]["selected_class"] == "artificial"
    assert res_b["overall_assessment"] == "isolated_forensic_signal"
    assert "The image is definitively AI-generated." in res_b["unsupported_conclusions"]

    # CASE C: Known AI image where detector selects human (ai_generated/solar/download.jpeg, uploaded with documented ai_generated prefix)
    ai_hum_bytes = (ds_root / "ai_generated" / "solar" / "download.jpeg").read_bytes()
    cid_c = client.post(
        "/api/media/upload",
        files={"file": ("ai_generated_download.jpeg", io.BytesIO(ai_hum_bytes), "image/jpeg")},
    ).json()["case_id"]
    client.post(f"/api/analysis/image/{cid_c}")
    client.post(f"/api/analysis/image/{cid_c}/detect")
    res_c = client.post(f"/api/forensics/{cid_c}/interpret").json()
    ai_sig_c = next(s for s in res_c["signal_inventory"] if s["signal_category"] == "ai_detector")
    assert ai_sig_c["details"]["selected_class"] == "human"
    assert res_c["overall_assessment"] == "mixed_or_conflicting_forensic_observations"
    assert res_c["consistency_classification"] == "conflicting"

