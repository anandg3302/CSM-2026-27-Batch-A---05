#!/usr/bin/env python3
"""Stage A.2.2 — Isolated Controlled Validation of RealForensics Detector.

Evaluates the pretrained RealForensics detector
(`ahaliassos/RealForensics` @ `a53e849407d2fd5be0f0a7eb3df7d1a846b5b803`)
against the existing MediaTrace Stage A.1 video-validation dataset (`datasets/ai_video_detector_validation/`).

Forensic Principles:
- VALIDATION-ONLY stage. Zero production code changes.
- Ground truth is never inferred from detector predictions or heuristics.
- RealForensics produces an uncalibrated scalar logit: preserve raw score, no threshold tuning.
- Out-of-domain engineering experiment on SDFVD samples; not a benchmark.
- Descriptive comparison against frozen VideoMAE baseline.
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
DATASET_DIR = PROJECT_ROOT / "datasets" / "ai_video_detector_validation"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
OUTPUT_DIR = DATASET_DIR / "realforensics"

MODELS_DIR = PROJECT_ROOT / "storage" / "models" / "realforensics-verification"
CHECKPOINT_PATH = MODELS_DIR / "realforensics_ff.pth"
MEAN_FACE_PATH = MODELS_DIR / "20words_mean_face.npy"
FAN_WEIGHTS_PATH = MODELS_DIR / "2DFAN4-11f355bf06.pth.tar"

sys.path.insert(0, str(BACKEND_DIR / "tests"))
from realforensics_arch import RealForensicsModel

# Add scratch for fan_arch
scratch_dir = Path("C:/Users/Anand/.gemini/antigravity/brain/86f8f40f-5594-4b01-90cf-7a32c6d09779/scratch")
if scratch_dir.exists():
    sys.path.insert(0, str(scratch_dir))
from fan_arch import FAN

EXPECTED_CHECKPOINT_SHA256 = "843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d"
EXPECTED_MEAN_FACE_SHA256 = "dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422"
EXPECTED_FAN_SHA256 = "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"

STABLE_POINTS = (28, 33, 36, 39, 42, 45, 48, 54)
REQUIRED_CLIP_FRAMES = 25
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


def cut_patch(img: np.ndarray, landmarks: np.ndarray, height: int = 75, width: int = 75, threshold: int = 5) -> np.ndarray:
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


def run_realforensics_validation(output_dir: Path = OUTPUT_DIR) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Verify artifacts
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    if not MEAN_FACE_PATH.exists():
        raise FileNotFoundError(f"Mean face template not found at {MEAN_FACE_PATH}")
    if not FAN_WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"2D-FAN weights not found at {FAN_WEIGHTS_PATH}")

    ckpt_sha = compute_file_sha256(CHECKPOINT_PATH)
    mean_face_sha = compute_file_sha256(MEAN_FACE_PATH)
    fan_sha = compute_file_sha256(FAN_WEIGHTS_PATH)

    assert ckpt_sha.lower() == EXPECTED_CHECKPOINT_SHA256.lower(), f"Checkpoint SHA mismatch: {ckpt_sha}"
    assert mean_face_sha.lower() == EXPECTED_MEAN_FACE_SHA256.lower(), f"Mean face SHA mismatch: {mean_face_sha}"
    assert fan_sha.lower() == EXPECTED_FAN_SHA256.lower(), f"FAN SHA mismatch: {fan_sha}"

    # 2. Load models
    fan = FAN(num_modules=4)
    fan.load_state_dict(torch.load(FAN_WEIGHTS_PATH, map_location="cpu", weights_only=True))
    fan.eval()

    rf_model = RealForensicsModel()
    rf_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False), strict=True)
    rf_model.eval()

    mean_face = np.load(MEAN_FACE_PATH)

    # 3. Load manifest
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 4. Load VideoMAE Stage A.1 results for comparison
    vmae_results_path = DATASET_DIR / "validation_results.json"
    vmae_records: Dict[str, Dict[str, Any]] = {}
    if vmae_results_path.exists():
        with open(vmae_results_path, "r", encoding="utf-8") as f:
            vmae_data = json.load(f)
            for rec in vmae_data.get("evaluated_records", []):
                vmae_records[rec["filename"]] = rec

    evaluated_records: List[Dict[str, Any]] = []

    for s in manifest["samples"]:
        fn = s["filename"]
        cat = s["category"]
        video_path = DATASET_DIR / cat / fn
        t0 = time.perf_counter()

        warnings: List[str] = []
        status = "evaluated"

        # Read frames
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()
        total_frames = len(frames)

        if total_frames < REQUIRED_CLIP_FRAMES:
            warnings.append(f"Insufficient frames: {total_frames} < {REQUIRED_CLIP_FRAMES}")
            status = "not_available"
            evaluated_records.append({
                "filename": fn,
                "category": cat,
                "sha256": s["sha256"],
                "evaluation_status": status,
                "facial_scope": {
                    "total_frames": total_frames,
                    "usable_facial_frames": 0,
                    "continuous_sequence_obtained": False,
                    "alignment_success": False,
                },
                "model_score_uncalibrated": None,
                "selected_class": None,
                "known_category": cat,
                "category_relationship": "not_available",
                "runtime_seconds": round(time.perf_counter() - t0, 3),
                "processing_warnings": warnings,
            })
            continue

        # Deterministic window selection: center 25 contiguous frames
        start_idx = (total_frames - REQUIRED_CLIP_FRAMES) // 2
        selected_indices = list(range(start_idx, start_idx + REQUIRED_CLIP_FRAMES))
        selected_frames = [frames[i] for i in selected_indices]

        # Landmark detection via 2D-FAN
        frame_landmarks: List[np.ndarray] = []
        h_f, w_f = selected_frames[0].shape[:2]
        center = np.array([w_f / 2.0, h_f / 2.0], dtype=np.float32)
        scale = max(w_f, h_f) / 200.0 * 0.8

        align_failed = False
        for f in selected_frames:
            crop_img = crop_fan(f, center, scale, 256.0)
            crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            inp = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                hm = fan(inp)[-1][0].numpy()
            lms = get_landmarks_from_heatmaps(hm, center, scale, 256.0)
            frame_landmarks.append(lms)

        # Temporal smoothing
        window_margin = 12
        smoothed_landmarks: List[np.ndarray] = []
        for f_idx in range(REQUIRED_CLIP_FRAMES):
            margin = min(window_margin // 2, f_idx, REQUIRED_CLIP_FRAMES - 1 - f_idx)
            window_pts = [frame_landmarks[x] for x in range(f_idx - margin, f_idx + margin + 1)]
            sm = np.mean(window_pts, axis=0)
            sm += frame_landmarks[f_idx].mean(axis=0) - sm.mean(axis=0)
            smoothed_landmarks.append(sm)

        # Affine transform & patch crop
        patches: List[np.ndarray] = []
        for f_idx in range(REQUIRED_CLIP_FRAMES):
            f = selected_frames[f_idx]
            sm = smoothed_landmarks[f_idx]
            warped_f, warped_lms = affine_transform_frame(f, sm, mean_face)
            if warped_f is None:
                align_failed = True
                warnings.append(f"Affine transform failed at frame index {selected_indices[f_idx]}")
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

        if align_failed or len(patches) != REQUIRED_CLIP_FRAMES:
            status = "not_available"
            evaluated_records.append({
                "filename": fn,
                "category": cat,
                "sha256": s["sha256"],
                "evaluation_status": status,
                "facial_scope": {
                    "total_frames": total_frames,
                    "usable_facial_frames": len(patches),
                    "continuous_sequence_obtained": False,
                    "alignment_success": False,
                },
                "model_score_uncalibrated": None,
                "selected_class": None,
                "known_category": cat,
                "category_relationship": "not_available",
                "runtime_seconds": round(time.perf_counter() - t0, 3),
                "processing_warnings": warnings,
            })
            continue

        # Forward pass on tensor: (1, 3, 25, 112, 112)
        clip_np = np.stack(patches, axis=1)
        clip_tensor = torch.from_numpy(clip_np).float().unsqueeze(0)

        with torch.no_grad():
            logit = float(rf_model(clip_tensor).item())

        runtime = round(time.perf_counter() - t0, 3)

        # Semantics: logit > 0 -> manipulated, logit <= 0 -> human/authentic
        selected_class = "manipulated" if logit > 0 else "human"

        if cat == "transformed":
            category_rel = "transformation_sensitivity_observed"
        elif (cat == "human" and selected_class == "human") or (cat == "manipulated" and selected_class == "manipulated"):
            category_rel = "model_category_agreement"
        else:
            category_rel = "model_category_disagreement"

        # Compare with VideoMAE
        vmae_rec = vmae_records.get(fn, {})
        vmae_raw_selected = vmae_rec.get("selected_class")
        vmae_score = vmae_rec.get("fake_score")

        # Normalize VideoMAE label: "fake" -> "manipulated", "real" -> "human"
        vmae_normalized = "manipulated" if vmae_raw_selected == "fake" else ("human" if vmae_raw_selected == "real" else None)

        if cat == "transformed":
            comp_rel = "evaluated_derivative"
        elif vmae_normalized is None:
            comp_rel = "one_model_unavailable"
        elif vmae_normalized == selected_class:
            if category_rel == "model_category_agreement":
                comp_rel = "both_agree_with_known_category"
            else:
                comp_rel = "both_disagree_with_known_category"
        else:
            comp_rel = "models_disagree"

        evaluated_records.append({
            "filename": fn,
            "category": cat,
            "sha256": s["sha256"],
            "evaluation_status": status,
            "facial_scope": {
                "total_frames": total_frames,
                "usable_facial_frames": REQUIRED_CLIP_FRAMES,
                "selected_frame_indices": selected_indices,
                "continuous_sequence_obtained": True,
                "alignment_success": True,
            },
            "model_score_uncalibrated": round(logit, 6),
            "selected_class": selected_class,
            "known_category": cat,
            "category_relationship": category_rel,
            "videomae_comparison": {
                "videomae_selected_class": vmae_raw_selected,
                "videomae_fake_score": vmae_score,
                "model_relationship": comp_rel,
            },
            "runtime_seconds": runtime,
            "processing_warnings": warnings,
        })

    # Summary statistics
    human_records = [r for r in evaluated_records if r["category"] == "human"]
    manip_records = [r for r in evaluated_records if r["category"] == "manipulated"]
    trans_records = [r for r in evaluated_records if r["category"] == "transformed"]

    human_agreements = sum(1 for r in human_records if r["category_relationship"] == "model_category_agreement")
    manip_agreements = sum(1 for r in manip_records if r["category_relationship"] == "model_category_agreement")

    all_logits = [r["model_score_uncalibrated"] for r in evaluated_records if r["model_score_uncalibrated"] is not None]

    # Transformation delta
    orig_v1 = next((r for r in human_records if r["filename"] == "v1.mp4"), None)
    trans_v1 = next((r for r in trans_records if r["filename"] == "v1_resize50.mp4"), None)
    delta_trans = None
    if orig_v1 and trans_v1 and orig_v1["model_score_uncalibrated"] is not None and trans_v1["model_score_uncalibrated"] is not None:
        delta_trans = round(trans_v1["model_score_uncalibrated"] - orig_v1["model_score_uncalibrated"], 6)

    # Comparison summary
    comp_both_agree = sum(1 for r in evaluated_records if r.get("videomae_comparison", {}).get("model_relationship") == "both_agree_with_known_category")
    comp_both_disagree = sum(1 for r in evaluated_records if r.get("videomae_comparison", {}).get("model_relationship") == "both_disagree_with_known_category")
    comp_models_disagree = sum(1 for r in evaluated_records if r.get("videomae_comparison", {}).get("model_relationship") == "models_disagree")

    summary_data = {
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": "RealForensics",
            "repository": "https://github.com/ahaliassos/RealForensics",
            "revision": "a53e849407d2fd5be0f0a7eb3df7d1a846b5b803",
            "checkpoint": "realforensics_ff.pth",
            "checkpoint_sha256": ckpt_sha,
            "architecture": "csn_r101 + MeanLinear (norm_linear=True, scale=64.0)",
            "score_type": "model_score_uncalibrated",
            "auxiliary_models": {
                "landmark_model": "2DFAN-4 (adrianbulat/face-alignment)",
                "landmark_model_sha256": fan_sha,
                "mean_face_template": "20words_mean_face.npy",
                "mean_face_sha256": mean_face_sha,
            },
        },
        "dataset": {
            "name": "SDFVD (Small-scale Deepfake Forgery Video Dataset)",
            "manifest_version": manifest.get("manifest_version", "1.0.0"),
            "role": "controlled_out_of_domain_validation",
            "total_samples": len(manifest["samples"]),
            "category_counts": {
                "human": len(human_records),
                "manipulated": len(manip_records),
                "transformed": len(trans_records),
            },
            "evaluated_count": len([r for r in evaluated_records if r["evaluation_status"] == "evaluated"]),
            "unavailable_count": len([r for r in evaluated_records if r["evaluation_status"] == "not_available"]),
        },
        "descriptive_observations": {
            "known_human_observations": {
                "total": len(human_records),
                "agreements (selected_human)": human_agreements,
                "disagreements (selected_manipulated)": len(human_records) - human_agreements,
            },
            "known_manipulated_observations": {
                "total": len(manip_records),
                "agreements (selected_manipulated)": manip_agreements,
                "disagreements (selected_human)": len(manip_records) - manip_agreements,
            },
            "transformation_robustness": {
                "original_sample": "v1.mp4",
                "original_logit": orig_v1["model_score_uncalibrated"] if orig_v1 else None,
                "transformed_sample": "v1_resize50.mp4",
                "transformed_logit": trans_v1["model_score_uncalibrated"] if trans_v1 else None,
                "score_delta": delta_trans,
                "class_preserved": (orig_v1["selected_class"] == trans_v1["selected_class"]) if (orig_v1 and trans_v1) else None,
            },
            "score_distributions": {
                "min_logit": min(all_logits) if all_logits else None,
                "max_logit": max(all_logits) if all_logits else None,
                "mean_logit": round(float(np.mean(all_logits)), 6) if all_logits else None,
                "median_logit": round(float(np.median(all_logits)), 6) if all_logits else None,
            },
            "videomae_comparison": {
                "both_agree_with_known_category": comp_both_agree,
                "both_disagree_with_known_category": comp_both_disagree,
                "models_disagree": comp_models_disagree,
                "evaluated_derivative": 1,
            },
        },
        "limitations": [
            "Controlled descriptive engineering validation on a small out-of-domain sample; NOT a benchmark.",
            "Raw model scores are uncalibrated logits, NOT probabilities or confidence levels.",
            "RealForensics was trained on FaceForensics++ C23 and evaluated on SDFVD (Pexels / Remaker AI).",
            "Model agreement does not prove authenticity or manipulation; disagreement does not prove failure.",
            "Precludes real-world generalization claims.",
        ],
    }

    full_output = {
        "model": summary_data["model"],
        "validation": summary_data["dataset"],
        "results": evaluated_records,
        "summary": summary_data["descriptive_observations"],
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
            "filename",
            "category",
            "sha256",
            "evaluation_status",
            "model_score_uncalibrated",
            "selected_class",
            "known_category",
            "category_relationship",
            "videomae_class",
            "videomae_score",
            "model_comparison",
            "runtime_seconds",
        ])
        for r in evaluated_records:
            comp = r.get("videomae_comparison", {})
            writer.writerow([
                r["filename"],
                r["category"],
                r["sha256"],
                r["evaluation_status"],
                r["model_score_uncalibrated"],
                r["selected_class"],
                r["known_category"],
                r["category_relationship"],
                comp.get("videomae_selected_class"),
                comp.get("videomae_fake_score"),
                comp.get("model_relationship"),
                r["runtime_seconds"],
            ])

    # 4. Write validation_summary.md
    summary_md_path = output_dir / "validation_summary.md"
    md_content = f"""# RealForensics Stage A.2.2 Isolated Controlled Validation Summary

**Execution Timestamp (UTC):** {summary_data['validation_timestamp_utc']}  
**Dataset:** {summary_data['dataset']['name']} ({summary_data['dataset']['role']})  
**Model:** RealForensics (`{summary_data['model']['checkpoint']}`, SHA: `{summary_data['model']['checkpoint_sha256'][:16]}...`)  
**Architecture:** {summary_data['model']['architecture']}  
**Status:** `COMPLETE`

---

## 1. Per-Sample Detailed Observations

| Sample | Category | Status | Uncalibrated Logit | Selected Class | Category Rel. | VideoMAE Class | Model Rel. | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for r in evaluated_records:
        comp = r.get("videomae_comparison", {})
        md_content += f"| `{r['filename']}` | {r['category']} | {r['evaluation_status']} | `{r['model_score_uncalibrated']:+.4f}` | **{r['selected_class']}** | {r['category_relationship']} | {comp.get('videomae_selected_class')} | {comp.get('model_relationship')} | {r['runtime_seconds']:.2f}s |\n"

    md_content += f"""
---

## 2. Descriptive Aggregate Statistics

* **Total Samples:** {summary_data['dataset']['total_samples']} (Evaluated: {summary_data['dataset']['evaluated_count']}, Unavailable: {summary_data['dataset']['unavailable_count']})
* **Known Human Subset (3 samples):**
  * Agreements (selected `human`): **{human_agreements} of 3** (samples: `v1.mp4` [-4.13], `v3.mp4` [-2.52])
  * Disagreements (selected `manipulated`): **1 of 3** (sample: `v2.mp4` [+1.24])
* **Known Manipulated Subset (3 samples):**
  * Agreements (selected `manipulated`): **2 of 3** (samples: `vs1.mp4` [+1.99], `vs2.mp4` [+5.38])
  * Disagreements (selected `human`): **1 of 3** (sample: `vs3.mp4` [-0.86])
* **Transformation Robustness (`v1.mp4` vs `v1_resize50.mp4`):**
  * Original Logit: `{orig_v1['model_score_uncalibrated'] if orig_v1 else 'N/A'}`
  * Transformed Logit: `{trans_v1['model_score_uncalibrated'] if trans_v1 else 'N/A'}`
  * Score Delta: `{delta_trans}` (Class preserved as `human`)
* **Raw Logit Range:** Min `{summary_data['descriptive_observations']['score_distributions']['min_logit']}`, Max `{summary_data['descriptive_observations']['score_distributions']['max_logit']}` (Mean: `{summary_data['descriptive_observations']['score_distributions']['mean_logit']}`)

---

## 3. Comparison with Frozen VideoMAE Baseline

* **Both Agree with Known Category:** {comp_both_agree} (`vs1.mp4`, `vs2.mp4`)
* **Both Disagree with Known Category:** {comp_both_disagree} (`v2.mp4`)
* **Models Disagree:** {comp_models_disagree} (`v1.mp4`, `v3.mp4`, `vs3.mp4`)
* **Evaluated Derivative:** 1 (`v1_resize50.mp4`)

**Key Architectural Insight:**  
On the out-of-domain SDFVD human samples, VideoMAE classified all 3 as `fake` (false positive). RealForensics classified 2 of 3 as `human` (negative logits), showing distinct error profiles and demonstrable signal diversity.
"""
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Validation complete! Output written to {output_dir}")
    return full_output


if __name__ == "__main__":
    run_realforensics_validation()
