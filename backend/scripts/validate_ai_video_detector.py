#!/usr/bin/env python3
"""Stage A.1 — Controlled Video AI/Deepfake Detector Validation Harness.

Evaluates the production VideoMAE deepfake detector
(`Vansh180/VideoMae-ffc23-deepfake-detector` @ `ea26518b527d938d66e568c52bcf1c19a86cf295`)
against independently documented ground-truth samples in `datasets/ai_video_detector_validation/`.

Forensic Principles:
- Controlled engineering validation experiment only; NOT a scientific benchmark or accuracy claim.
- Ground truth is never inferred from detector predictions, metadata, or heuristics.
- Videos are never fabricated or downloaded without independent provenance.
- Non-facial video samples are classified as `not_available` (out of facial scope) and excluded from model agreement counts.
- Outputs include per-video JSON, CSV, summary JSON, and Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Project root resolution
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATASET_DIR = PROJECT_ROOT / "datasets" / "ai_video_detector_validation"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
RESULTS_DIR = DATASET_DIR / "results"

import sys
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.services.analysis.video_ai_detection_service import (
    EXPECTED_SAFETENSORS_SHA256,
    PINNED_VIDEO_MODEL_ID,
    PINNED_VIDEO_MODEL_VERSION,
    evaluate_video_ai_detection,
    load_video_ai_detector_runtime,
    verify_video_model_artifacts,
)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TARGET_COUNTS = {
    "human": 3,
    "manipulated": 3,
    "transformed": 1,
}

VALIDATION_LIMITATIONS = [
    "Controlled descriptive engineering validation harness, not a benchmark or general-world accuracy claim.",
    "Known categories come exclusively from independent ground-truth documentation; never inferred from detector output.",
    "Model scores are uncalibrated model-derived scores (score_type='uncalibrated_model_score') with decision_rule='model_class_selection' and threshold=null.",
    "Detector scope is restricted to facial video deepfakes (FaceForensics++ C23 domain); non-facial content is out of scope.",
    "Transformed derivatives evaluate transformation sensitivity and are reported separately.",
    "The detector alone never escalates forensic assessments and does not prove authenticity or provenance.",
]


class VideoValidationHarnessError(ValueError):
    """Raised when validation manifest, dataset files, or configuration are invalid."""


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash in 1MB chunks."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def inspect_video_properties(video_path: Path) -> Dict[str, Any]:
    """Inspect basic technical properties of a video file safely."""
    info: Dict[str, Any] = {
        "file_size_bytes": video_path.stat().st_size,
        "resolution": "unknown",
        "duration_seconds": None,
        "frame_count": None,
        "fps": None,
        "container": video_path.suffix.lstrip(".").lower(),
    }
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if w > 0 and h > 0:
                info["resolution"] = f"{w}x{h}"
            if fc > 0 and fps > 0:
                info["frame_count"] = fc
                info["fps"] = round(fps, 2)
                info["duration_seconds"] = round(fc / fps, 2)
        cap.release()
    except Exception:
        pass
    return info


def verify_model_integrity_for_validation() -> Dict[str, Any]:
    """Verify local model weights and configuration before running validation."""
    model_dir = settings.models_dir / "Vansh180--VideoMae-ffc23-deepfake-detector"
    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"

    if not safetensors_path.is_file():
        raise VideoValidationHarnessError(
            f"Video AI detector model weights not found at {safetensors_path}."
        )

    actual_sha = compute_file_sha256(safetensors_path)
    if actual_sha.lower() != EXPECTED_SAFETENSORS_SHA256.lower():
        raise VideoValidationHarnessError(
            f"Model weights SHA-256 mismatch: expected={EXPECTED_SAFETENSORS_SHA256}, actual={actual_sha}"
        )

    artifact_hashes = verify_video_model_artifacts(model_dir)
    return {
        "model_id": PINNED_VIDEO_MODEL_ID,
        "model_version": PINNED_VIDEO_MODEL_VERSION,
        "weights_sha256": actual_sha,
        "artifact_hashes": artifact_hashes,
        "model_dir": str(model_dir),
    }


def load_validation_manifest(manifest_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """Load and validate manifest.json."""
    if not manifest_path.is_file():
        raise VideoValidationHarnessError(f"Validation manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise VideoValidationHarnessError("manifest.json must contain a top-level JSON object")

    return data


def scan_category_videos(category_dir: Path) -> List[Path]:
    """Safely scan a category directory for valid video files."""
    if not category_dir.is_dir():
        return []
    videos = []
    for f in sorted(category_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS and not f.name.startswith("."):
            videos.append(f)
    return videos


def evaluate_single_sample(
    video_path: Path,
    category: str,
    manifest_entry: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate one video file using the production detector."""
    sha256_hash = compute_file_sha256(video_path)
    tech_info = inspect_video_properties(video_path)

    start_time = time.perf_counter()
    detection = evaluate_video_ai_detection(video_path=str(video_path), expected_sha256=sha256_hash)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Determine agreement status
    ground_truth = (manifest_entry.get("ground_truth_category") or category).lower()
    selected_class = (detection.selected_class or "").lower()
    detector_status = detection.status

    if detection.usable_detector_frames == 0 or detector_status != "completed":
        agreement_status = "not_applicable"
        effective_status = "not_available"
    elif ground_truth == "human":
        agreement_status = "agreement" if selected_class == "real" else "model/category disagreement observed"
        effective_status = "completed"
    elif ground_truth == "manipulated":
        agreement_status = "agreement" if selected_class == "fake" else "model/category disagreement observed"
        effective_status = "completed"
    else:
        agreement_status = "evaluated_derivative"
        effective_status = "completed"

    selected_class_id = 1 if selected_class == "fake" else (0 if selected_class == "real" else None)

    return {
        "filename": video_path.name,
        "category": category,
        "sha256": sha256_hash,
        "file_size_bytes": tech_info["file_size_bytes"],
        "resolution": tech_info["resolution"],
        "duration_seconds": tech_info["duration_seconds"],
        "fps": tech_info["fps"],
        "container": tech_info["container"],
        "detector_status": effective_status,
        "selected_class": detection.selected_class,
        "selected_class_id": selected_class_id,
        "real_score": detection.real_score,
        "fake_score": detection.fake_score,
        "score_type": detection.score_type,
        "decision_rule": detection.decision_rule,
        "threshold": detection.threshold,
        "frames_sampled": detection.frames_sampled,
        "usable_detector_frames": detection.usable_detector_frames,
        "detector_scope": detection.detector_scope,
        "model_repository": detection.model_name or PINNED_VIDEO_MODEL_ID,
        "model_revision": detection.model_version or PINNED_VIDEO_MODEL_VERSION,
        "model_artifact_sha": detection.artifact_hashes.get("model.safetensors", EXPECTED_SAFETENSORS_SHA256),
        "preprocessing_version": "videomae_shortest224_bilinear_centercrop224_rgb_imagenet",
        "agreement_status": agreement_status,
        "reason": detection.reason if effective_status == "not_available" else None,
        "processing_time_ms": elapsed_ms,
        "ground_truth_basis": manifest_entry.get("ground_truth_basis") if manifest_entry else None,
        "source_description": manifest_entry.get("source_description") if manifest_entry else None,
    }


def run_controlled_validation(
    dataset_dir: Path = DATASET_DIR,
    manifest_path: Path = MANIFEST_PATH,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the complete controlled validation harness."""
    model_provenance = verify_model_integrity_for_validation()
    manifest = load_validation_manifest(manifest_path)

    # Build manifest lookup by filename
    manifest_samples = {s.get("filename"): s for s in manifest.get("samples", [])}

    # Discover videos across categories
    categories = ["human", "manipulated", "transformed"]
    discovered: Dict[str, List[Path]] = {}
    for cat in categories:
        cat_path = dataset_dir / cat
        discovered[cat] = scan_category_videos(cat_path)

    # Count samples
    counts = {cat: len(vids) for cat, vids in discovered.items()}
    is_complete = (
        counts["human"] >= TARGET_COUNTS["human"]
        and counts["manipulated"] >= TARGET_COUNTS["manipulated"]
        and counts["transformed"] >= TARGET_COUNTS["transformed"]
    )
    dataset_status = "complete" if is_complete else "incomplete"

    evaluated_records: List[Dict[str, Any]] = []

    # Evaluate each video
    for cat in categories:
        for vid_path in discovered[cat]:
            manifest_entry = manifest_samples.get(vid_path.name)
            record = evaluate_single_sample(vid_path, cat, manifest_entry)
            evaluated_records.append(record)

    # Compile descriptive observations
    obs_by_cat: Dict[str, Dict[str, int]] = {
        "known_human": {"evaluated": 0, "selected_real": 0, "selected_fake": 0, "not_available": 0},
        "known_manipulated": {"evaluated": 0, "selected_real": 0, "selected_fake": 0, "not_available": 0},
        "transformed": {"evaluated": 0, "selected_real": 0, "selected_fake": 0, "not_available": 0},
    }

    disagreements: List[Dict[str, Any]] = []

    for r in evaluated_records:
        cat = r["category"]
        key = f"known_{cat}" if cat in {"human", "manipulated"} else "transformed"
        obs = obs_by_cat[key]
        obs["evaluated"] += 1

        if r["detector_status"] == "not_available":
            obs["not_available"] += 1
        elif r["selected_class"] == "real":
            obs["selected_real"] += 1
        elif r["selected_class"] == "fake":
            obs["selected_fake"] += 1

        if r["agreement_status"] == "model/category disagreement observed":
            disagreements.append({
                "filename": r["filename"],
                "category": r["category"],
                "selected_class": r["selected_class"],
                "real_score": r["real_score"],
                "fake_score": r["fake_score"],
            })

    timestamp = datetime.now(timezone.utc).isoformat()

    summary_data: Dict[str, Any] = {
        "validation_timestamp": timestamp,
        "dataset_name": manifest.get("dataset_name", "MediaTrace Controlled Video AI Detector Validation Dataset"),
        "manifest_version": manifest.get("manifest_version", "1.0.0"),
        "dataset_status": dataset_status,
        "target_counts": TARGET_COUNTS,
        "observed_counts": counts,
        "total_samples_evaluated": len(evaluated_records),
        "model_provenance": model_provenance,
        "controlled_validation_observations": obs_by_cat,
        "disagreements": disagreements,
        "disagreement_count": len(disagreements),
        "limitations": VALIDATION_LIMITATIONS,
    }

    # Destination directories
    target_dirs = [dataset_dir]
    if output_dir:
        target_dirs.append(output_dir)
    else:
        results_folder = dataset_dir / "results"
        results_folder.mkdir(parents=True, exist_ok=True)
        target_dirs.append(results_folder)

    for out_d in target_dirs:
        out_d.mkdir(parents=True, exist_ok=True)

        # 1. validation_results.json
        with open(out_d / "validation_results.json", "w", encoding="utf-8") as f:
            json.dump({"evaluated_records": evaluated_records, "summary": summary_data}, f, indent=2)

        # 2. validation_results.csv
        csv_path = out_d / "validation_results.csv"
        csv_fields = [
            "filename", "category", "sha256", "file_size_bytes", "resolution", "duration_seconds",
            "fps", "container", "detector_status", "selected_class", "selected_class_id",
            "real_score", "fake_score", "score_type", "decision_rule", "threshold",
            "frames_sampled", "usable_detector_frames", "detector_scope", "model_repository",
            "model_revision", "model_artifact_sha", "preprocessing_version", "agreement_status",
            "reason", "processing_time_ms", "ground_truth_basis",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for r in evaluated_records:
                writer.writerow(r)

        # 3. validation_summary.json
        with open(out_d / "validation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # 4. validation_summary.md
        md_content = generate_markdown_summary(summary_data, evaluated_records)
        with open(out_d / "validation_summary.md", "w", encoding="utf-8") as f:
            f.write(md_content)

    return summary_data


def generate_markdown_summary(summary: Dict[str, Any], records: List[Dict[str, Any]]) -> str:
    """Generate structured markdown summary report."""
    lines = [
        "# Controlled Video AI Detector Validation Summary",
        "",
        f"- **Validation Timestamp**: `{summary['validation_timestamp']}`",
        f"- **Dataset Status**: **`{summary['dataset_status'].upper()}`**",
        f"- **Model Repository**: `{summary['model_provenance']['model_id']}`",
        f"- **Pinned Revision**: `{summary['model_provenance']['model_version']}`",
        f"- **Model Weights SHA-256**: `{summary['model_provenance']['weights_sha256']}`",
        "",
        "## 1. Dataset Sample Inventory",
        "",
        "| Category | Target Count | Observed Count | Status |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for cat in ("human", "manipulated", "transformed"):
        t_cnt = summary["target_counts"][cat]
        o_cnt = summary["observed_counts"][cat]
        st = "Satisfied" if o_cnt >= t_cnt else "Incomplete"
        lines.append(f"| `{cat}` | {t_cnt} | {o_cnt} | {st} |")

    lines.extend([
        "",
        "## 2. Controlled Validation Observations",
        "",
        "| Category Group | Evaluated | Selected Real | Selected Fake | Not Available (Out of Scope) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for grp, vals in summary["controlled_validation_observations"].items():
        lines.append(f"| `{grp}` | {vals['evaluated']} | {vals['selected_real']} | {vals['selected_fake']} | {vals['not_available']} |")

    lines.extend([
        "",
        "## 3. Disagreements",
        "",
    ])
    if summary["disagreements"]:
        lines.append("| Filename | Category | Selected Class | Real Score | Fake Score |")
        lines.append("| :--- | :--- | :--- | :--- | :--- |")
        for d in summary["disagreements"]:
            lines.append(f"| `{d['filename']}` | `{d['category']}` | `{d['selected_class']}` | {d['real_score']} | {d['fake_score']} |")
    else:
        lines.append("No model/category disagreements observed in evaluated samples.")

    lines.extend([
        "",
        "## 4. Evaluated Sample Records",
        "",
    ])
    if records:
        lines.append("| Filename | Category | SHA-256 | Status | Class | Real | Fake | Usable Frames | Agreement |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
        for r in records:
            sha_short = r["sha256"][:12] + "..."
            lines.append(
                f"| `{r['filename']}` | `{r['category']}` | `{sha_short}` | `{r['detector_status']}` | "
                f"`{r['selected_class'] or '-'}` | {r['real_score'] or '-'} | {r['fake_score'] or '-'} | "
                f"{r['usable_detector_frames']} | `{r['agreement_status']}` |"
            )
    else:
        lines.append("No video samples currently populated in dataset directories.")

    lines.extend([
        "",
        "## 5. Forensic Limitations",
        "",
    ])
    for lim in summary["limitations"]:
        lines.append(f"- {lim}")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="MediaTrace Video AI Detector Validation Harness")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR, help="Path to validation dataset directory")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="Path to manifest.json")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory for reports")
    args = parser.parse_args()

    print("=" * 60)
    print("MediaTrace Stage A.1 — Video AI Detector Validation Harness")
    print("=" * 60)

    try:
        summary = run_controlled_validation(
            dataset_dir=args.dataset_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
        )
        print(f"\nValidation completed successfully.")
        print(f"Dataset status: {summary['dataset_status'].upper()}")
        print(f"Observed counts: {summary['observed_counts']}")
        print(f"Disagreements: {summary['disagreement_count']}")
    except Exception as exc:
        print(f"\nValidation harness failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
