#!/usr/bin/env python3
"""Stage A.2.3 — Expanded Controlled Validation of Video AI Detectors.

Evaluates both independently verified video AI detectors:
1. VideoMAE (Vansh180/VideoMae-ffc23-deepfake-detector @ ea26518b527d938d66e568c52bcf1c19a86cf295)
2. RealForensics (ahaliassos/RealForensics @ a53e849407d2fd5be0f0a7eb3df7d1a846b5b803)

Against the expanded controlled validation cohort:
- 10 human samples (v1..v10)
- 10 manipulated samples (vs1..vs10)
- 6 transformed derivatives (3 human, 3 manipulated)
Total: 26 samples.

Forensic Principles:
- VALIDATION-ONLY stage. Zero changes to production code or APIs.
- Ground truth is never inferred from detector predictions or heuristics.
- Model scores are kept in their native uncalibrated formats:
  * VideoMAE: fake_score (uncalibrated probability of fake class)
  * RealForensics: uncalibrated scalar logit z
- No combined score, no fused probability, no threshold tuning.
- Out-of-domain engineering experiment on SDFVD samples; not a benchmark.
- Descriptive observations only.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BACKEND_DIR / "tests") not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR / "tests"))

from app.core.config import settings
from app.services.analysis.video_ai_detection_service import (
    EXPECTED_SAFETENSORS_SHA256,
    PINNED_VIDEO_MODEL_ID,
    PINNED_VIDEO_MODEL_VERSION,
    detect_usable_facial_frames,
    evaluate_video_ai_detection,
    load_video_ai_detector_runtime,
    verify_video_model_artifacts,
)
from fan_arch import FAN
from realforensics_arch import RealForensicsModel

EXPANDED_DATASET_DIR = PROJECT_ROOT / "datasets" / "ai_video_detector_validation" / "expanded"
EXPANDED_MANIFEST_PATH = EXPANDED_DATASET_DIR / "validation_manifest.json"

RF_MODELS_DIR = PROJECT_ROOT / "storage" / "models" / "realforensics-verification"
RF_CHECKPOINT_PATH = RF_MODELS_DIR / "realforensics_ff.pth"
RF_MEAN_FACE_PATH = RF_MODELS_DIR / "20words_mean_face.npy"
RF_FAN_WEIGHTS_PATH = RF_MODELS_DIR / "2DFAN4-11f355bf06.pth.tar"

EXPECTED_RF_CHECKPOINT_SHA256 = "843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d"
EXPECTED_RF_MEAN_FACE_SHA256 = "dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422"
EXPECTED_RF_FAN_SHA256 = "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"

STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
REQUIRED_RF_CLIP_FRAMES = 25
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_affine_transform(center: np.ndarray, scale: float, resolution: float = 256.0, invert: bool = False) -> np.ndarray:
    h = 200.0 * scale
    t = np.eye(3, dtype=np.float32)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)
    if invert:
        t = np.linalg.inv(t)
    return t


def crop_fan(img: np.ndarray, center: np.ndarray, scale: float, resolution: float = 256.0) -> np.ndarray:
    t = get_affine_transform(center, scale, resolution)
    return cv2.warpAffine(img, t[:2], (int(resolution), int(resolution)), flags=cv2.INTER_LINEAR)


def get_landmarks_from_heatmaps(hm: np.ndarray, center: np.ndarray, scale: float, resolution: float = 256.0) -> np.ndarray:
    C, H, W = hm.shape
    coords_64 = np.zeros((C, 2), dtype=np.float32)
    for c in range(C):
        m = hm[c]
        idx = np.argmax(m)
        y, x = np.unravel_index(idx, (H, W))
        px, py = float(x), float(y)
        if 0 < x < W - 1 and 0 < y < H - 1:
            dx = 0.25 * np.sign(m[y, x + 1] - m[y, x - 1])
            dy = 0.25 * np.sign(m[y + 1, x] - m[y - 1, x])
            px += dx
            py += dy
        coords_64[c] = [px, py]
    coords_256 = coords_64 * (resolution / float(H))
    t_inv = get_affine_transform(center, scale, resolution, invert=True)
    coords_homo = np.hstack([coords_256, np.ones((C, 1), dtype=np.float32)])
    return (t_inv @ coords_homo.T).T[:, :2]


def affine_transform_frame(
    frame: np.ndarray,
    landmarks: np.ndarray,
    reference: np.ndarray,
    stable_points: Tuple[int, ...] = STABLE_POINTS,
    target_size: Tuple[int, int] = (256, 256),
    reference_size: Tuple[int, int] = (256, 256),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    stable_reference = np.vstack([reference[x] for x in stable_points])
    stable_reference[:, 0] -= (reference_size[0] - target_size[0]) / 2.0
    stable_reference[:, 1] -= (reference_size[1] - target_size[1]) / 2.0

    transform = cv2.estimateAffinePartial2D(
        np.vstack([landmarks[x] for x in stable_points]),
        stable_reference,
        method=cv2.LMEDS,
    )[0]
    if transform is None:
        return None, None

    transformed_frame = cv2.warpAffine(
        frame,
        transform,
        dsize=(target_size[0], target_size[1]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    transformed_landmarks = (
        np.matmul(landmarks, transform[:, :2].transpose())
        + transform[:, 2].transpose()
    )
    return transformed_frame, transformed_landmarks


def cut_patch(img: np.ndarray, landmarks: np.ndarray, height: int = 75, width: int = 75) -> np.ndarray:
    center_x, center_y = np.mean(landmarks, axis=0)
    if center_y - height < 0:
        center_y = height
    if center_x - width < 0:
        center_x = width
    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    return np.copy(
        img[
            int(round(center_y) - round(height)) : int(round(center_y) + round(height)),
            int(round(center_x) - round(width)) : int(round(center_x) + round(width)),
        ]
    )


def recompute_and_save_summary(
    evaluated_records: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    rf_ckpt_sha: str,
    output_dir: Path,
) -> Dict[str, Any]:
    # Update status and relationship correctly for each record
    for r in evaluated_records:
        cat = r["category"]
        vmae = r["videomae"]
        rf = r["realforensics"]

        # Normalize VideoMAE status
        if vmae["status"] in ("completed", "evaluated") and vmae["fake_score"] is not None:
            vmae["status"] = "evaluated"
            vmae["warning"] = None
        elif vmae["status"] != "evaluated":
            vmae["status"] = "not_available"

        # Normalize RealForensics status
        if rf["uncalibrated_logit"] is not None:
            rf["status"] = "evaluated"
            rf["warning"] = None
        elif rf["status"] != "evaluated":
            rf["status"] = "not_available"

        # VideoMAE category relationship
        if cat == "transformed":
            vmae["category_relationship"] = "transformation_sensitivity_observed"
        elif vmae["status"] == "evaluated":
            if vmae["normalized_class"] == cat:
                vmae["category_relationship"] = "model_category_agreement"
            else:
                vmae["category_relationship"] = "model_category_disagreement"
        else:
            vmae["category_relationship"] = "not_available"

        # RealForensics category relationship
        if cat == "transformed":
            rf["category_relationship"] = "transformation_sensitivity_observed"
        elif rf["status"] == "evaluated":
            if rf["selected_class"] == cat:
                rf["category_relationship"] = "model_category_agreement"
            else:
                rf["category_relationship"] = "model_category_disagreement"
        else:
            rf["category_relationship"] = "not_available"

        # Dual relationship
        if cat == "transformed":
            r["dual_model_relationship"] = "evaluated_derivative"
        elif vmae["status"] != "evaluated" or rf["status"] != "evaluated":
            r["dual_model_relationship"] = "one_or_both_unavailable"
        elif vmae["category_relationship"] == "model_category_agreement" and rf["category_relationship"] == "model_category_agreement":
            r["dual_model_relationship"] = "both_agree_with_known_category"
        elif vmae["category_relationship"] == "model_category_disagreement" and rf["category_relationship"] == "model_category_disagreement":
            r["dual_model_relationship"] = "both_disagree_with_known_category"
        else:
            r["dual_model_relationship"] = "models_disagree"

    # Summary Statistics
    human_records = [r for r in evaluated_records if r["category"] == "human"]
    manip_records = [r for r in evaluated_records if r["category"] == "manipulated"]
    trans_records = [r for r in evaluated_records if r["category"] == "transformed"]

    vmae_human_agree = sum(1 for r in human_records if r["videomae"]["category_relationship"] == "model_category_agreement")
    vmae_manip_agree = sum(1 for r in manip_records if r["videomae"]["category_relationship"] == "model_category_agreement")

    rf_human_agree = sum(1 for r in human_records if r["realforensics"]["category_relationship"] == "model_category_agreement")
    rf_manip_agree = sum(1 for r in manip_records if r["realforensics"]["category_relationship"] == "model_category_agreement")

    comp_both_agree = sum(1 for r in evaluated_records if r["dual_model_relationship"] == "both_agree_with_known_category")
    comp_both_disagree = sum(1 for r in evaluated_records if r["dual_model_relationship"] == "both_disagree_with_known_category")
    comp_models_disagree = sum(1 for r in evaluated_records if r["dual_model_relationship"] == "models_disagree")
    comp_derivatives = sum(1 for r in evaluated_records if r["dual_model_relationship"] == "evaluated_derivative")

    # Transformation sensitivity
    trans_sensitivity = []
    records_by_fn = {r["filename"]: r for r in evaluated_records}
    for tr in trans_records:
        parent_fn = tr["source_video_filename"]
        parent_rec = records_by_fn.get(parent_fn)
        if not parent_rec:
            continue

        vmae_parent_score = parent_rec["videomae"]["fake_score"]
        vmae_trans_score = tr["videomae"]["fake_score"]
        vmae_delta = round(vmae_trans_score - vmae_parent_score, 6) if (vmae_parent_score is not None and vmae_trans_score is not None) else None
        vmae_class_preserved = (parent_rec["videomae"]["selected_class"] == tr["videomae"]["selected_class"])

        rf_parent_logit = parent_rec["realforensics"]["uncalibrated_logit"]
        rf_trans_logit = tr["realforensics"]["uncalibrated_logit"]
        rf_delta = round(rf_trans_logit - rf_parent_logit, 6) if (rf_parent_logit is not None and rf_trans_logit is not None) else None
        rf_class_preserved = (parent_rec["realforensics"]["selected_class"] == tr["realforensics"]["selected_class"])

        trans_sensitivity.append({
            "derivative_filename": tr["filename"],
            "parent_filename": parent_fn,
            "parent_category": tr["parent_category"],
            "transformation_type": tr["transformation_type"],
            "videomae": {
                "parent_score": vmae_parent_score,
                "derivative_score": vmae_trans_score,
                "score_delta": vmae_delta,
                "class_preserved": vmae_class_preserved,
            },
            "realforensics": {
                "parent_logit": rf_parent_logit,
                "derivative_logit": rf_trans_logit,
                "logit_delta": rf_delta,
                "class_preserved": rf_class_preserved,
            },
        })

    summary_data = {
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A.2.3",
        "dataset": {
            "name": "SDFVD (Small-scale Deepfake Forgery Video Dataset) — Expanded Controlled Cohort",
            "manifest_version": manifest.get("manifest_version", "2.0.0"),
            "role": "controlled_out_of_domain_validation",
            "total_samples": len(evaluated_records),
            "category_counts": {
                "human": len(human_records),
                "manipulated": len(manip_records),
                "transformed": len(trans_records),
            },
            "evaluated_count": len([r for r in evaluated_records if r["videomae"]["status"] == "evaluated" and r["realforensics"]["status"] == "evaluated"]),
            "unavailable_count": len([r for r in evaluated_records if r["videomae"]["status"] != "evaluated" or r["realforensics"]["status"] != "evaluated"]),
        },
        "detectors": [
            {
                "name": "VideoMAE",
                "repository": "Vansh180/VideoMae-ffc23-deepfake-detector",
                "revision": "ea26518b527d938d66e568c52bcf1c19a86cf295",
                "architecture": "VideoMAEForVideoClassification",
                "safetensors_sha256": EXPECTED_SAFETENSORS_SHA256,
                "score_representation": "uncalibrated_probability_fake_class",
                "decision_rule": "model_class_selection",
                "observations": {
                    "known_human_total": len(human_records),
                    "known_human_agreed": vmae_human_agree,
                    "known_human_disagreed": len(human_records) - vmae_human_agree,
                    "known_manipulated_total": len(manip_records),
                    "known_manipulated_agreed": vmae_manip_agree,
                    "known_manipulated_disagreed": len(manip_records) - vmae_manip_agree,
                },
            },
            {
                "name": "RealForensics",
                "repository": "ahaliassos/RealForensics",
                "revision": "a53e849407d2fd5be0f0a7eb3df7d1a846b5b803",
                "architecture": "csn_r101 + MeanLinear (norm_linear=True, scale=64.0)",
                "checkpoint": "realforensics_ff.pth",
                "checkpoint_sha256": rf_ckpt_sha,
                "score_representation": "uncalibrated_scalar_logit",
                "decision_rule": "scalar_logit_sign (z > 0 -> manipulated, z <= 0 -> human)",
                "observations": {
                    "known_human_total": len(human_records),
                    "known_human_agreed": rf_human_agree,
                    "known_human_disagreed": len(human_records) - rf_human_agree,
                    "known_manipulated_total": len(manip_records),
                    "known_manipulated_agreed": rf_manip_agree,
                    "known_manipulated_disagreed": len(manip_records) - rf_manip_agree,
                },
            },
        ],
        "comparative_observations": {
            "both_agree_with_known_category": comp_both_agree,
            "both_disagree_with_known_category": comp_both_disagree,
            "models_disagree": comp_models_disagree,
            "evaluated_derivative": comp_derivatives,
            "total_primary_samples": len(human_records) + len(manip_records),
        },
        "transformation_sensitivity": trans_sensitivity,
        "limitations": [
            "Controlled descriptive engineering validation on a small out-of-domain sample cohort (SDFVD: Pexels / Remaker AI); NOT a benchmark.",
            "Raw model scores are uncalibrated (uncalibrated probability for VideoMAE, uncalibrated scalar logit for RealForensics), NOT calibrated real-world probabilities.",
            "Both models were pretrained on FaceForensics++ C23 and evaluated on SDFVD out-of-domain samples.",
            "Model agreement does not prove authenticity or manipulation; disagreement does not prove failure.",
            "Precludes real-world generalization claims.",
        ],
    }

    full_output = {
        "dataset": summary_data["dataset"],
        "detectors": summary_data["detectors"],
        "results": evaluated_records,
        "comparative_observations": summary_data["comparative_observations"],
        "transformation_sensitivity": summary_data["transformation_sensitivity"],
        "limitations": summary_data["limitations"],
    }

    # 1. Write validation_results.json
    results_json_path = output_dir / "validation_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # 2. Write validation_summary.json
    summary_json_path = output_dir / "validation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 3. Write validation_results.csv
    results_csv_path = output_dir / "validation_results.csv"
    with open(results_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "filename",
            "category",
            "known_category",
            "sha256",
            "videomae_status",
            "videomae_fake_score",
            "videomae_selected_class",
            "videomae_category_rel",
            "realforensics_status",
            "realforensics_logit",
            "realforensics_selected_class",
            "realforensics_category_rel",
            "dual_model_relationship",
            "total_runtime_seconds",
        ])
        for r in evaluated_records:
            writer.writerow([
                r["sample_id"],
                r["filename"],
                r["category"],
                r["known_category"],
                r["sha256"],
                r["videomae"]["status"],
                r["videomae"]["fake_score"],
                r["videomae"]["normalized_class"],
                r["videomae"]["category_relationship"],
                r["realforensics"]["status"],
                r["realforensics"]["uncalibrated_logit"],
                r["realforensics"]["selected_class"],
                r["realforensics"]["category_relationship"],
                r["dual_model_relationship"],
                r.get("total_runtime_seconds", 0.0),
            ])

    # 4. Write validation_summary.md
    summary_md_path = output_dir / "validation_summary.md"
    md_lines = [
        "# MediaTrace Stage A.2.3 — Expanded Controlled Validation Summary",
        "",
        f"**Execution Timestamp (UTC):** {summary_data['validation_timestamp_utc']}  ",
        f"**Dataset:** {summary_data['dataset']['name']}  ",
        f"**Cohort Size:** {summary_data['dataset']['total_samples']} samples ({summary_data['dataset']['category_counts']['human']} human, {summary_data['dataset']['category_counts']['manipulated']} manipulated, {summary_data['dataset']['category_counts']['transformed']} transformed derivatives)  ",
        f"**Stage:** `A.2.3` | **Status:** `COMPLETE`",
        "",
        "---",
        "",
        "## 1. Per-Sample Detailed Observations",
        "",
        "| Sample | Category | VMAE Norm Class | VMAE Score | VMAE Agree? | RF Class | RF Logit | RF Agree? | Dual Relationship |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in evaluated_records:
        v_cls = r["videomae"]["normalized_class"] or "N/A"
        v_sc = f"{r['videomae']['fake_score']:.4f}" if r["videomae"]["fake_score"] is not None else "N/A"
        v_ag = "Yes" if r["videomae"]["category_relationship"] == "model_category_agreement" else ("No" if r["videomae"]["category_relationship"] == "model_category_disagreement" else "N/A")
        
        r_cls = r["realforensics"]["selected_class"] or "N/A"
        r_sc = f"{r['realforensics']['uncalibrated_logit']:+.4f}" if r["realforensics"]["uncalibrated_logit"] is not None else "N/A"
        r_ag = "Yes" if r["realforensics"]["category_relationship"] == "model_category_agreement" else ("No" if r["realforensics"]["category_relationship"] == "model_category_disagreement" else "N/A")

        md_lines.append(f"| `{r['filename']}` | {r['category']} | {v_cls} | {v_sc} | {v_ag} | {r_cls} | {r_sc} | {r_ag} | `{r['dual_model_relationship']}` |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Model Performance Breakdown (Descriptive)",
        "",
        f"### VideoMAE (`Vansh180/VideoMae-ffc23-deepfake-detector`)",
        f"- **Known Human Samples (n={len(human_records)}):** {vmae_human_agree} agreed ({len(human_records) - vmae_human_agree} selected fake)",
        f"- **Known Manipulated Samples (n={len(manip_records)}):** {vmae_manip_agree} agreed ({len(manip_records) - vmae_manip_agree} selected real)",
        "",
        f"### RealForensics (`realforensics_ff.pth`)",
        f"- **Known Human Samples (n={len(human_records)}):** {rf_human_agree} agreed ({len(human_records) - rf_human_agree} selected manipulated)",
        f"- **Known Manipulated Samples (n={len(manip_records)}):** {rf_manip_agree} agreed ({len(manip_records) - rf_manip_agree} selected human)",
        "",
        "---",
        "",
        "## 3. Comparative Signal Relationship Matrix (n=20 primary)",
        "",
        f"| Relationship Category | Count | Percentage |",
        f"| :--- | :--- | :--- |",
        f"| `both_agree_with_known_category` | {comp_both_agree} | {comp_both_agree / 20 * 100:.1f}% |",
        f"| `both_disagree_with_known_category` | {comp_both_disagree} | {comp_both_disagree / 20 * 100:.1f}% |",
        f"| `models_disagree` | {comp_models_disagree} | {comp_models_disagree / 20 * 100:.1f}% |",
        "",
        "---",
        "",
        "## 4. Transformation Robustness (n=6 derivatives)",
        "",
        "| Derivative | Parent | Type | VMAE Parent -> Deriv | VMAE Delta | VMAE Preserved | RF Parent -> Deriv | RF Delta | RF Preserved |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for ts in trans_sensitivity:
        v_p = f"{ts['videomae']['parent_score']:.4f}"
        v_d = f"{ts['videomae']['derivative_score']:.4f}"
        v_delta = f"{ts['videomae']['score_delta']:+.4f}"
        v_pres = "Yes" if ts["videomae"]["class_preserved"] else "No"

        r_p = f"{ts['realforensics']['parent_logit']:+.4f}"
        r_d = f"{ts['realforensics']['derivative_logit']:+.4f}"
        r_delta = f"{ts['realforensics']['logit_delta']:+.4f}"
        r_pres = "Yes" if ts["realforensics"]["class_preserved"] else "No"

        md_lines.append(f"| `{ts['derivative_filename']}` | `{ts['parent_filename']}` | {ts['transformation_type']} | {v_p} -> {v_d} | {v_delta} | {v_pres} | {r_p} -> {r_d} | {r_delta} | {r_pres} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 5. Methodological Safeguards & Limitations",
        "",
        "- **Out-of-Domain Engineering Experiment:** Evaluated on SDFVD (Pexels / Remaker AI); not a scientific benchmark or generalized real-world accuracy claim.",
        "- **Uncalibrated Model Scores:** VideoMAE outputs uncalibrated fake class probabilities; RealForensics outputs raw unbounded scalar logits.",
        "- **Signal Diversity Observation:** RealForensics exhibited a different output pattern from VideoMAE on the controlled validation samples, including different classifications on several known-human and known-manipulated videos. This demonstrates signal diversity in this experiment but does not establish detector reliability or forensic validity.",
        "- **Zero Production Changes:** No production detection services, forensic frameworks, API endpoints, or UI were altered.",
    ])

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSummary update complete! Artifacts written to {output_dir}")
    print(f"  Results JSON: {results_json_path}")
    print(f"  Summary JSON: {summary_json_path}")
    print(f"  Results CSV:  {results_csv_path}")
    print(f"  Summary MD:   {summary_md_path}")

    return summary_data


def run_expanded_validation(
    manifest_path: Path = EXPANDED_MANIFEST_PATH,
    output_dir: Path = EXPANDED_DATASET_DIR,
    recompute_summary_only: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    rf_ckpt_sha = compute_file_sha256(RF_CHECKPOINT_PATH)

    if recompute_summary_only:
        existing_results_path = output_dir / "validation_results.json"
        if existing_results_path.exists():
            print(f"Recomputing summary and comparative categories from existing {existing_results_path}...")
            with open(existing_results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return recompute_and_save_summary(data["results"], manifest, rf_ckpt_sha, output_dir)

    print("--- 1. Verifying Model Integrity ---")
    vmae_dir = settings.models_dir / "Vansh180--VideoMae-ffc23-deepfake-detector"
    vmae_hashes = verify_video_model_artifacts(vmae_dir)
    assert vmae_hashes.get("model.safetensors", "").lower() == EXPECTED_SAFETENSORS_SHA256.lower()

    assert RF_CHECKPOINT_PATH.exists(), f"RealForensics checkpoint missing: {RF_CHECKPOINT_PATH}"
    assert RF_MEAN_FACE_PATH.exists(), f"Mean face template missing: {RF_MEAN_FACE_PATH}"
    assert RF_FAN_WEIGHTS_PATH.exists(), f"2D-FAN weights missing: {RF_FAN_WEIGHTS_PATH}"

    rf_mean_face_sha = compute_file_sha256(RF_MEAN_FACE_PATH)
    rf_fan_sha = compute_file_sha256(RF_FAN_WEIGHTS_PATH)

    assert rf_ckpt_sha.lower() == EXPECTED_RF_CHECKPOINT_SHA256.lower()
    assert rf_mean_face_sha.lower() == EXPECTED_RF_MEAN_FACE_SHA256.lower()
    assert rf_fan_sha.lower() == EXPECTED_RF_FAN_SHA256.lower()

    fan = FAN(num_modules=4)
    fan.load_state_dict(torch.load(RF_FAN_WEIGHTS_PATH, map_location="cpu", weights_only=True))
    fan.eval()

    rf_model = RealForensicsModel()
    rf_model.load_state_dict(torch.load(RF_CHECKPOINT_PATH, map_location="cpu", weights_only=False), strict=True)
    rf_model.eval()

    rf_mean_face = np.load(RF_MEAN_FACE_PATH)
    print("All model artifacts verified successfully.")

    print(f"--- 2. Loading Manifest: {manifest_path} ---")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples = manifest["samples"]
    print(f"Total samples to evaluate: {len(samples)}")

    evaluated_records: List[Dict[str, Any]] = []

    for idx, s in enumerate(samples, 1):
        fn = s["filename"]
        cat = s["category"]
        video_path = output_dir / cat / fn
        print(f"[{idx}/{len(samples)}] Evaluating {fn} ({cat})...", flush=True)

        # -----------------------------
        # Detector 1: VideoMAE
        # -----------------------------
        t0_vmae = time.perf_counter()
        vmae_res = evaluate_video_ai_detection(str(video_path))
        vmae_runtime = round(time.perf_counter() - t0_vmae, 3)

        vmae_status = "evaluated" if (vmae_res.status == "completed" and getattr(vmae_res, "usable_detector_frames", 1) > 0) else ("evaluated" if vmae_res.status == "evaluated" else vmae_res.status)
        vmae_warning = vmae_res.reason if vmae_status != "evaluated" else None
        vmae_fake_score = vmae_res.fake_score
        vmae_real_score = vmae_res.real_score
        vmae_selected_class = vmae_res.selected_class
        vmae_normalized_class = "manipulated" if vmae_selected_class == "fake" else ("human" if vmae_selected_class == "real" else None)

        if cat == "transformed":
            vmae_cat_rel = "transformation_sensitivity_observed"
        elif vmae_status == "evaluated":
            if vmae_normalized_class == cat:
                vmae_cat_rel = "model_category_agreement"
            else:
                vmae_cat_rel = "model_category_disagreement"
        else:
            vmae_cat_rel = "not_available"

        # -----------------------------
        # Detector 2: RealForensics
        # -----------------------------
        t0_rf = time.perf_counter()
        rf_status = "evaluated"
        rf_warning = None
        rf_logit = None
        rf_selected_class = None
        rf_cat_rel = None

        cap = cv2.VideoCapture(str(video_path))
        frames_bgr = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames_bgr.append(frame)
        cap.release()
        total_frames = len(frames_bgr)

        if total_frames < REQUIRED_RF_CLIP_FRAMES:
            rf_status = "not_available"
            rf_warning = f"Insufficient frames: {total_frames} < {REQUIRED_RF_CLIP_FRAMES}"
        else:
            start_idx = (total_frames - REQUIRED_RF_CLIP_FRAMES) // 2
            selected_rf_indices = list(range(start_idx, start_idx + REQUIRED_RF_CLIP_FRAMES))
            selected_rf_frames = [frames_bgr[i] for i in selected_rf_indices]

            h_f, w_f = selected_rf_frames[0].shape[:2]
            center = np.array([w_f / 2.0, h_f / 2.0], dtype=np.float32)
            scale = max(w_f, h_f) / 200.0 * 0.8

            frame_landmarks: List[np.ndarray] = []
            align_failed = False

            for f in selected_rf_frames:
                crop_img = crop_fan(f, center, scale, 256.0)
                crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                inp = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
                with torch.no_grad():
                    hm = fan(inp)[-1][0].numpy()
                lms = get_landmarks_from_heatmaps(hm, center, scale, 256.0)
                frame_landmarks.append(lms)

            window_margin = 12
            smoothed_landmarks: List[np.ndarray] = []
            for f_idx in range(REQUIRED_RF_CLIP_FRAMES):
                margin = min(window_margin // 2, f_idx, REQUIRED_RF_CLIP_FRAMES - 1 - f_idx)
                window_pts = [frame_landmarks[x] for x in range(f_idx - margin, f_idx + margin + 1)]
                sm = np.mean(window_pts, axis=0)
                sm += frame_landmarks[f_idx].mean(axis=0) - sm.mean(axis=0)
                smoothed_landmarks.append(sm)

            patches: List[np.ndarray] = []
            for f_idx in range(REQUIRED_RF_CLIP_FRAMES):
                f = selected_rf_frames[f_idx]
                sm = smoothed_landmarks[f_idx]
                warped_f, warped_lms = affine_transform_frame(f, sm, rf_mean_face)
                if warped_f is None:
                    align_failed = True
                    rf_warning = f"Affine transform failed at frame index {selected_rf_indices[f_idx]}"
                    break

                patch = cut_patch(warped_f, warped_lms[15:68], height=75, width=75)
                ph, pw = patch.shape[:2]
                ch_t = (ph - 140) // 2
                ch_l = (pw - 140) // 2
                crop140 = patch[ch_t : ch_t + 140, ch_l : ch_l + 140]
                res112 = cv2.resize(crop140, (112, 112), interpolation=cv2.INTER_LINEAR)
                rgb112 = cv2.cvtColor(res112, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                norm112 = (rgb112 - IMAGENET_MEAN) / IMAGENET_STD
                patches.append(norm112.transpose(2, 0, 1))

            if align_failed or len(patches) != REQUIRED_RF_CLIP_FRAMES:
                rf_status = "not_available"
                if not rf_warning:
                    rf_warning = f"Alignment yielded {len(patches)}/{REQUIRED_RF_CLIP_FRAMES} frames"
            else:
                clip_np = np.stack(patches, axis=1)
                clip_tensor = torch.from_numpy(clip_np).float().unsqueeze(0)
                with torch.no_grad():
                    rf_logit = float(rf_model(clip_tensor).item())

                rf_selected_class = "manipulated" if rf_logit > 0 else "human"
                if cat == "transformed":
                    rf_cat_rel = "transformation_sensitivity_observed"
                elif rf_selected_class == cat:
                    rf_cat_rel = "model_category_agreement"
                else:
                    rf_cat_rel = "model_category_disagreement"

        rf_runtime = round(time.perf_counter() - t0_rf, 3)

        # Dual-Model Comparison Relationship
        if cat == "transformed":
            dual_rel = "evaluated_derivative"
        elif vmae_status != "evaluated" or rf_status != "evaluated":
            dual_rel = "one_or_both_unavailable"
        elif vmae_cat_rel == "model_category_agreement" and rf_cat_rel == "model_category_agreement":
            dual_rel = "both_agree_with_known_category"
        elif vmae_cat_rel == "model_category_disagreement" and rf_cat_rel == "model_category_disagreement":
            dual_rel = "both_disagree_with_known_category"
        else:
            dual_rel = "models_disagree"

        rec = {
            "sample_id": s["sample_id"],
            "filename": fn,
            "category": cat,
            "known_category": s.get("ground_truth_category", cat),
            "parent_category": s.get("parent_category"),
            "source_video_filename": s.get("source_video_filename"),
            "transformation_type": s.get("transformation_type"),
            "sha256": s["sha256"],
            "file_size_bytes": s["file_size_bytes"],
            "resolution": s["resolution"],
            "fps": s["fps"],
            "frame_count": s["frame_count"],
            "duration_seconds": s["duration_seconds"],
            "videomae": {
                "status": vmae_status,
                "fake_score": vmae_fake_score,
                "real_score": vmae_real_score,
                "selected_class": vmae_selected_class,
                "normalized_class": vmae_normalized_class,
                "category_relationship": vmae_cat_rel,
                "warning": vmae_warning,
                "runtime_seconds": vmae_runtime,
            },
            "realforensics": {
                "status": rf_status,
                "uncalibrated_logit": round(rf_logit, 6) if rf_logit is not None else None,
                "selected_class": rf_selected_class,
                "category_relationship": rf_cat_rel,
                "warning": rf_warning,
                "runtime_seconds": rf_runtime,
            },
            "dual_model_relationship": dual_rel,
            "total_runtime_seconds": round(vmae_runtime + rf_runtime, 3),
        }
        evaluated_records.append(rec)
        print(f"  VMAE: {vmae_normalized_class} (score={vmae_fake_score}) | RF: {rf_selected_class} (logit={rf_logit}) | Rel: {dual_rel}", flush=True)

    return recompute_and_save_summary(evaluated_records, manifest, rf_ckpt_sha, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expanded Video AI Detector Validation")
    parser.add_argument("--manifest", type=Path, default=EXPANDED_MANIFEST_PATH, help="Path to manifest JSON")
    parser.add_argument("--output-dir", type=Path, default=EXPANDED_DATASET_DIR, help="Path to output directory")
    parser.add_argument("--recompute-summary-only", action="store_true", help="Recompute summary and comparative categories from existing results")
    args = parser.parse_args()

    run_expanded_validation(args.manifest, args.output_dir, recompute_summary_only=args.recompute_summary_only)
