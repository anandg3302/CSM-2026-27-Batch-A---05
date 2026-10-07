r"""Step 14.1.5 — Controlled AI-Detector Transformation Robustness Testing.

Evaluates how the existing MediaTrace AI image detector (`umm-maybe/AI-image-detector`
at pinned revision `c7e223baf11bc40528af364ba7bdea030ef42f9e`) behaves when known
human and AI-generated images are transformed by common deterministic operations:
- original (exact byte copy)
- resize_50 (50% aspect-ratio preserving bicubic resize, JPEG Q=95)
- resize_25 (25% aspect-ratio preserving bicubic resize, JPEG Q=95)
- jpeg_q90 (same dimensions, JPEG re-encode at quality=90)
- jpeg_q70 (same dimensions, JPEG re-encode at quality=70)
- jpeg_q50 (same dimensions, JPEG re-encode at quality=50)

IMPORTANT:
- Do NOT modify the AI detector model, preprocessing, threshold, or score semantics.
- This is a small controlled descriptive engineering experiment, NOT a benchmark or
  scientific claim of accuracy, robustness, reliability, or forensic validity.
- Source categories come strictly from `datasets/ai_detector_validation/manifest.json`
  and are never rewritten from detector predictions.

Usage:
    cd backend
    $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/test_ai_detector_transformation_robustness.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Optional

from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

__test__ = False

from app.services.analysis import image_ai_detection_service
from scripts.validate_ai_image_detector import (
    ValidationHarnessError,
    load_and_validate_manifest,
    verify_model_integrity_for_validation,
)


TRANSFORMATION_ORDER = [
    "original",
    "resize_50",
    "resize_25",
    "jpeg_q90",
    "jpeg_q70",
    "jpeg_q50",
]

MAX_ALLOWED_DIMENSION = 16384
MAX_ALLOWED_PIXELS = 64_000_000

ROBUSTNESS_LIMITATIONS = [
    "This is a small controlled engineering experiment.",
    "Results are descriptive and are not a benchmark.",
    "Transformations are limited to those actually tested (original, resize_50, resize_25, jpeg_q90, jpeg_q70, jpeg_q50).",
    "Results cannot establish detector accuracy or real-world reliability.",
    "JPEG recompression/resizing can change detector outputs.",
    "The detector score remains uncalibrated (uncalibrated_model_score).",
    "The detector is not a universal AI-image/deepfake detector.",
    "Model behavior may differ for other generators, image domains, resolutions, screenshots, crops, or compression pipelines.",
    "No provenance or authenticity conclusion can be made from the detector alone.",
]


def _ensure_safe_subpath(child: Path, parent: Path) -> Path:
    resolved_child = child.resolve()
    resolved_parent = parent.resolve()
    try:
        resolved_child.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValidationHarnessError(
            f"Security violation: path '{resolved_child}' escaped allowed directory '{resolved_parent}'."
        ) from exc
    return resolved_child


def _create_single_derivative(
    source_bytes: bytes,
    source_ext: str,
    orig_width: int,
    orig_height: int,
    transformation: str,
) -> tuple[bytes, int, int, str, Dict[str, Any]]:
    """Deterministically create a derivative in memory and return (bytes, width, height, format, params)."""
    if transformation == "original":
        with Image.open(io.BytesIO(source_bytes)) as im:
            fmt = str(im.format or source_ext.lstrip(".").upper())
            return (
                source_bytes,
                orig_width,
                orig_height,
                fmt,
                {
                    "operation": "exact_byte_copy",
                    "preserve_bytes": True,
                    "preserve_dimensions": True,
                },
            )

    with Image.open(io.BytesIO(source_bytes)) as im:
        rgb = im.convert("RGB")

        if transformation == "resize_50":
            new_w = max(1, int(round(orig_width * 0.50)))
            new_h = max(1, int(round(orig_height * 0.50)))
            transformed = rgb.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
            jpeg_quality = 95
            params: Dict[str, Any] = {
                "operation": "resize",
                "scale_factor": 0.50,
                "preserve_aspect_ratio": True,
                "resampling": "Image.Resampling.BICUBIC",
                "output_format": "JPEG",
                "jpeg_quality": jpeg_quality,
            }
        elif transformation == "resize_25":
            new_w = max(1, int(round(orig_width * 0.25)))
            new_h = max(1, int(round(orig_height * 0.25)))
            transformed = rgb.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
            jpeg_quality = 95
            params = {
                "operation": "resize",
                "scale_factor": 0.25,
                "preserve_aspect_ratio": True,
                "resampling": "Image.Resampling.BICUBIC",
                "output_format": "JPEG",
                "jpeg_quality": jpeg_quality,
            }
        elif transformation == "jpeg_q90":
            new_w, new_h = orig_width, orig_height
            transformed = rgb
            jpeg_quality = 90
            params = {
                "operation": "jpeg_recompress",
                "preserve_dimensions": True,
                "output_format": "JPEG",
                "jpeg_quality": 90,
            }
        elif transformation == "jpeg_q70":
            new_w, new_h = orig_width, orig_height
            transformed = rgb
            jpeg_quality = 70
            params = {
                "operation": "jpeg_recompress",
                "preserve_dimensions": True,
                "output_format": "JPEG",
                "jpeg_quality": 70,
            }
        elif transformation == "jpeg_q50":
            new_w, new_h = orig_width, orig_height
            transformed = rgb
            jpeg_quality = 50
            params = {
                "operation": "jpeg_recompress",
                "preserve_dimensions": True,
                "output_format": "JPEG",
                "jpeg_quality": 50,
            }
        else:
            raise ValidationHarnessError(f"Unsupported transformation '{transformation}'.")

        buf = io.BytesIO()
        transformed.save(buf, format="JPEG", quality=jpeg_quality)
        out_bytes = buf.getvalue()

    # Verify generated JPEG is valid and matches expected dimensions
    with Image.open(io.BytesIO(out_bytes)) as verify_im:
        verify_w, verify_h = verify_im.size
        verify_fmt = str(verify_im.format or "JPEG")
        verify_im.verify()

    return out_bytes, verify_w, verify_h, verify_fmt, params


def _compute_group_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    art_scores = [float(r["artificial_score"]) for r in items]
    hum_scores = [float(r["human_score"]) for r in items]
    art_deltas = [float(r["artificial_score_delta"]) for r in items]
    hum_deltas = [float(r["human_score_delta"]) for r in items]
    sel_art = sum(1 for r in items if r["selected_class"] == "artificial")
    sel_hum = sum(1 for r in items if r["selected_class"] == "human")
    class_changed = sum(1 for r in items if r["selected_class_changed"] is True)
    disagreements = sum(1 for r in items if r["source_vs_model_disagreement"] is True)

    return {
        "sample_count": len(items),
        "selected_artificial_count": sel_art,
        "selected_human_count": sel_hum,
        "selected_class_changed_count": class_changed,
        "source_vs_model_disagreement_count": disagreements,
        "artificial_score_min": round(min(art_scores), 4) if art_scores else None,
        "artificial_score_max": round(max(art_scores), 4) if art_scores else None,
        "artificial_score_mean": round(statistics.mean(art_scores), 4) if art_scores else None,
        "artificial_score_median": round(statistics.median(art_scores), 4) if art_scores else None,
        "human_score_min": round(min(hum_scores), 4) if hum_scores else None,
        "human_score_max": round(max(hum_scores), 4) if hum_scores else None,
        "human_score_mean": round(statistics.mean(hum_scores), 4) if hum_scores else None,
        "human_score_median": round(statistics.median(hum_scores), 4) if hum_scores else None,
        "artificial_score_delta_min": round(min(art_deltas), 4) if art_deltas else None,
        "artificial_score_delta_max": round(max(art_deltas), 4) if art_deltas else None,
        "artificial_score_delta_mean": round(statistics.mean(art_deltas), 4) if art_deltas else None,
        "artificial_score_delta_median": round(statistics.median(art_deltas), 4) if art_deltas else None,
        "human_score_delta_min": round(min(hum_deltas), 4) if hum_deltas else None,
        "human_score_delta_max": round(max(hum_deltas), 4) if hum_deltas else None,
        "human_score_delta_mean": round(statistics.mean(hum_deltas), 4) if hum_deltas else None,
        "human_score_delta_median": round(statistics.median(hum_deltas), 4) if hum_deltas else None,
    }


def _write_robustness_csv(csv_path: Path, results: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "derivative_id",
        "source_id",
        "source_category",
        "source_filename",
        "source_sha256",
        "derivative_filename",
        "derivative_relative_path",
        "derivative_sha256",
        "transformation",
        "original_width",
        "original_height",
        "derivative_width",
        "derivative_height",
        "derivative_format",
        "status",
        "selected_class",
        "baseline_selected_class",
        "selected_class_changed",
        "class_change_description",
        "artificial_score",
        "baseline_artificial_score",
        "artificial_score_delta",
        "human_score",
        "baseline_human_score",
        "human_score_delta",
        "source_vs_model_disagreement",
        "disagreement_observation",
        "score_type",
        "decision_rule",
        "threshold",
        "model_repository",
        "pinned_model_revision",
        "model_architecture",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _generate_markdown_summary(
    results_payload: Dict[str, Any],
) -> str:
    comp = results_payload["dataset_composition"]
    mi = results_payload["model_integrity"]
    results = results_payload["results"]
    grouped = results_payload["grouped_descriptive_results"]
    class_changes = results_payload["observed_class_changes"]
    disagreements = results_payload["observed_disagreements"]

    lines: List[str] = [
        "# Step 14.1.5 — Controlled AI-Detector Transformation Robustness Summary",
        "",
        "> **Descriptive Engineering Experiment Notice**: This document reports descriptive transformation results (`observed score shift` and `observed class change`) for a small controlled sample. It is **not** a scientific benchmark and does not claim accuracy, precision, recall, robustness, or forensic validity.",
        "",
        "## 1. Dataset Composition & Model Metadata",
        "",
        f"- **Source Manifest**: `{comp['source_manifest']}`",
        f"- **Known Human Source Images**: `{comp['human_source_count']}`",
        f"- **Known AI-Generated Source Images**: `{comp['ai_generated_source_count']}`",
        f"- **Total Source Images Selected**: `{comp['total_source_images']}`",
        f"- **Transformations per Source Image**: `{len(TRANSFORMATION_ORDER)}` (`{', '.join(TRANSFORMATION_ORDER)}`)",
        f"- **Total Evaluated Files (Originals + Transformed Derivatives)**: `{comp['total_evaluated_files']}` (`{comp['original_copies_count']}` originals + `{comp['transformed_derivatives_count']}` transformed derivatives)",
        f"- **Model Repository**: `{mi['model_repository']}`",
        f"- **Pinned Model Revision**: `{mi['pinned_revision']}`",
        f"- **Model Architecture**: `{mi['architecture']}`",
        f"- **PyTorch Checkpoint SHA-256**: `{mi['pytorch_model_sha256']}`",
        f"- **Local ONNX Artifact SHA-256**: `{mi['model_onnx_sha256']}`",
        f"- **Score Semantics**: `score_type = uncalibrated_model_score`, `decision_rule = model_class_selection`, `threshold = null`",
        "",
        "## 2. Transformations Tested",
        "",
        "| Transformation | Description | Resampling / Encoding Parameters |",
        "| :--- | :--- | :--- |",
        "| `original` | Exact byte copy of source image into `transformation_robustness/` | Byte-for-byte identical copy (`SHA-256` preserved) |",
        "| `resize_50` | 50% aspect-ratio preserving downscale | `Image.Resampling.BICUBIC`, saved as `JPEG` (`quality=95`) |",
        "| `resize_25` | 25% aspect-ratio preserving downscale | `Image.Resampling.BICUBIC`, saved as `JPEG` (`quality=95`) |",
        "| `jpeg_q90` | Same pixel dimensions, re-encoded at JPEG Q=90 | `JPEG` (`quality=90`) |",
        "| `jpeg_q70` | Same pixel dimensions, re-encoded at JPEG Q=70 | `JPEG` (`quality=70`) |",
        "| `jpeg_q50` | Same pixel dimensions, re-encoded at JPEG Q=50 | `JPEG` (`quality=50`) |",
        "",
        "## 3. Per-Image Descriptive Results & Observed Score Shifts",
        "",
        "| Source ID | Source Category | Transformation | Dimensions | Selected Class | Artificial Score | $\\Delta$ Artificial | Human Score | $\\Delta$ Human | Class Changed vs Original | Source/Model Disagreement |",
        "| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |",
    ]

    for r in results:
        dims = f"{r['derivative_width']}x{r['derivative_height']}"
        lines.append(
            f"| `{r['source_id']}` | `{r['source_category']}` | `{r['transformation']}` | `{dims}` | "
            f"`{r['selected_class']}` | `{r['artificial_score']:.4f}` | `{r['artificial_score_delta']:+.4f}` | "
            f"`{r['human_score']:.4f}` | `{r['human_score_delta']:+.4f}` | "
            f"`{str(r['selected_class_changed']).lower()}` | `{str(r['source_vs_model_disagreement']).lower()}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Grouped Descriptive Results by Source Category & Transformation",
            "",
            "### 4.1 Known Human Source Images (`source_category = human`)",
            "",
            "| Transformation | Samples | Selected `artificial` | Selected `human` | Class Changed Count | Artificial Score (`min` / `max` / `mean` / `median`) | $\\Delta$ Artificial Mean |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
    )

    for t_name in TRANSFORMATION_ORDER:
        st = grouped["human"][t_name]
        lines.append(
            f"| `{t_name}` | `{st['sample_count']}` | `{st['selected_artificial_count']}` | "
            f"`{st['selected_human_count']}` | `{st['selected_class_changed_count']}` | "
            f"`{st['artificial_score_min']}` / `{st['artificial_score_max']}` / `{st['artificial_score_mean']}` / `{st['artificial_score_median']}` | "
            f"`{st['artificial_score_delta_mean']:+.4f}` |"
        )

    lines.extend(
        [
            "",
            "### 4.2 Known AI-Generated Source Images (`source_category = ai_generated`)",
            "",
            "| Transformation | Samples | Selected `artificial` | Selected `human` | Class Changed Count | Artificial Score (`min` / `max` / `mean` / `median`) | $\\Delta$ Artificial Mean |",
            "| :--- | ---: | ---: | ---: | ---: | :--- | ---: |",
        ]
    )

    for t_name in TRANSFORMATION_ORDER:
        st = grouped["ai_generated"][t_name]
        lines.append(
            f"| `{t_name}` | `{st['sample_count']}` | `{st['selected_artificial_count']}` | "
            f"`{st['selected_human_count']}` | `{st['selected_class_changed_count']}` | "
            f"`{st['artificial_score_min']}` / `{st['artificial_score_max']}` / `{st['artificial_score_mean']}` / `{st['artificial_score_median']}` | "
            f"`{st['artificial_score_delta_mean']:+.4f}` |"
        )

    lines.extend(
        [
            "",
            "## 5. Observed Class Changes Relative to Original",
            "",
            f"- **Total Derivatives with Observed Class Change (`selected_class_changed = true`)**: `{class_changes['count']}`",
        ]
    )
    if class_changes["items"]:
        for item in class_changes["items"]:
            lines.append(
                f"  - `{item['derivative_id']}` (`source_id={item['source_id']}`, `source_category={item['source_category']}`, "
                f"`transformation={item['transformation']}`): `{item['baseline_selected_class']}` $\\rightarrow$ `{item['selected_class']}` "
                f"(`artificial_score`: `{item['baseline_artificial_score']:.4f}` $\\rightarrow$ `{item['artificial_score']:.4f}`, "
                f"$\\Delta = {item['artificial_score_delta']:+.4f}`)"
            )
    else:
        lines.append("  - None observed in this run.")

    lines.extend(
        [
            "",
            "## 6. Observed Source-Category vs. Model-Output Disagreements",
            "",
            f"- **Total Evaluations with Source/Model Disagreement (`source_vs_model_disagreement = true`)**: `{disagreements['count']}`",
        ]
    )
    for d in disagreements["items"]:
        lines.append(
            f"  - `{d['derivative_id']}` (`source_id={d['source_id']}`, `source_category={d['source_category']}`, "
            f"`transformation={d['transformation']}`): `model_selected_class={d['selected_class']}` "
            f"(`artificial_score={d['artificial_score']:.4f}`, `human_score={d['human_score']:.4f}`) — `{d['disagreement_observation']}`"
        )

    lines.extend(
        [
            "",
            "## 7. Explicit Limitations",
            "",
        ]
    )
    for lim in ROBUSTNESS_LIMITATIONS:
        lines.append(f"- {lim}")

    lines.append("")
    return "\n".join(lines)


def run_transformation_robustness_experiment(
    manifest_path: Optional[Path] = None,
    output_base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the Step 14.1.5 Controlled AI-Detector Transformation Robustness Experiment."""
    repo_root = Path(__file__).resolve().parents[2]
    resolved_manifest = (
        manifest_path.resolve()
        if manifest_path is not None
        else (repo_root / "datasets" / "ai_detector_validation" / "manifest.json").resolve()
    )
    dataset_root = (
        output_base_dir.resolve()
        if output_base_dir is not None
        else resolved_manifest.parent.resolve()
    )
    robustness_dir = _ensure_safe_subpath(dataset_root / "transformation_robustness", dataset_root)

    # 1. Verify production model integrity
    model_integrity = verify_model_integrity_for_validation()

    # 2. Load source manifest and select human + ai_generated source images
    manifest_info = load_and_validate_manifest(resolved_manifest)
    source_samples = [
        s for s in manifest_info["samples"] if s["known_category"] in {"human", "ai_generated"}
    ]
    if not source_samples:
        raise ValidationHarnessError(
            "No 'human' or 'ai_generated' source samples found in validation manifest."
        )

    manifest_derivatives: List[Dict[str, Any]] = []
    detector_results: List[Dict[str, Any]] = []

    for src in source_samples:
        src_path: Path = _ensure_safe_subpath(src["resolved_path"], resolved_manifest.parent)
        src_bytes_before = src_path.read_bytes()
        src_sha256_before = hashlib.sha256(src_bytes_before).hexdigest()
        if src_sha256_before != src["initial_sha256"]:
            raise ValidationHarnessError(f"Source file SHA-256 changed before experiment: {src_path}")

        with Image.open(io.BytesIO(src_bytes_before)) as src_im:
            orig_w, orig_h = src_im.size
            if (
                orig_w <= 0
                or orig_h <= 0
                or orig_w > MAX_ALLOWED_DIMENSION
                or orig_h > MAX_ALLOWED_DIMENSION
                or (orig_w * orig_h) > MAX_ALLOWED_PIXELS
            ):
                raise ValidationHarnessError(
                    f"Image dimensions ({orig_w}x{orig_h}) exceed safety bounds for '{src['id']}'."
                )

        cat_folder = "ai" if src["known_category"] == "ai_generated" else "human"
        src_stem = src_path.stem
        src_ext = src_path.suffix.lower()

        # Store per-source results temporarily so we can compute deltas against 'original'
        src_eval_rows: List[Dict[str, Any]] = []

        for t_name in TRANSFORMATION_ORDER:
            sub_dir = _ensure_safe_subpath(robustness_dir / cat_folder / t_name, robustness_dir)
            sub_dir.mkdir(parents=True, exist_ok=True)

            if t_name == "original":
                deriv_filename = f"{src['id']}_{src_stem}{src_ext}"
            else:
                deriv_filename = f"{src['id']}_{src_stem}_{t_name}.jpg"

            deriv_path = _ensure_safe_subpath(sub_dir / deriv_filename, robustness_dir)
            if deriv_path.resolve() == src_path.resolve():
                raise ValidationHarnessError("Refusing to overwrite source image path.")

            deriv_bytes, d_w, d_h, d_fmt, t_params = _create_single_derivative(
                source_bytes=src_bytes_before,
                source_ext=src_ext,
                orig_width=orig_w,
                orig_height=orig_h,
                transformation=t_name,
            )
            deriv_path.write_bytes(deriv_bytes)
            deriv_sha256 = hashlib.sha256(deriv_bytes).hexdigest()

            if t_name == "original" and deriv_sha256 != src_sha256_before:
                raise ValidationHarnessError(
                    f"Original copy SHA-256 mismatch for '{src['id']}'."
                )

            created_ts = datetime.now(timezone.utc).isoformat()
            deriv_rel_path = deriv_path.relative_to(dataset_root).as_posix()
            derivative_id = f"{src['id']}__{t_name}"

            manifest_entry = {
                "derivative_id": derivative_id,
                "source_id": src["id"],
                "source_filename": src_path.name,
                "source_relative_path": src["path"],
                "source_category": src["known_category"],
                "source_sha256": src_sha256_before,
                "derivative_filename": deriv_filename,
                "derivative_relative_path": deriv_rel_path,
                "derivative_sha256": deriv_sha256,
                "transformation": t_name,
                "transformation_parameters": t_params,
                "original_width": orig_w,
                "original_height": orig_h,
                "derivative_width": d_w,
                "derivative_height": d_h,
                "derivative_format": d_fmt,
                "creation_timestamp": created_ts,
            }
            manifest_derivatives.append(manifest_entry)

            # Run production AI detector against derivative
            ai_res = image_ai_detection_service.detect_ai_generated_image(
                deriv_path,
                expected_sha256=deriv_sha256,
            )
            if ai_res.status != "available" or ai_res.prediction is None or ai_res.detector is None:
                raise ValidationHarnessError(
                    f"Production AI detector returned status='{ai_res.status}' on '{deriv_rel_path}'."
                )

            # Verify derivative & source files were never mutated by inference
            if hashlib.sha256(deriv_path.read_bytes()).hexdigest() != deriv_sha256:
                raise ValidationHarnessError(f"Derivative modified during inference: {deriv_path}")
            if hashlib.sha256(src_path.read_bytes()).hexdigest() != src_sha256_before:
                raise ValidationHarnessError(f"Source file modified during inference: {src_path}")

            art_score = round(float(ai_res.prediction.score), 4)
            hum_score = round(float(1.0 - art_score), 4)
            sel_class = str(ai_res.prediction.raw_class)

            is_disagreement = (
                (src["known_category"] == "human" and sel_class == "artificial")
                or (src["known_category"] == "ai_generated" and sel_class == "human")
            )
            disagreement_note = (
                "model/category disagreement observed" if is_disagreement else None
            )

            src_eval_rows.append(
                {
                    "derivative_id": derivative_id,
                    "source_id": src["id"],
                    "source_category": src["known_category"],
                    "source_filename": src_path.name,
                    "source_sha256": src_sha256_before,
                    "derivative_filename": deriv_filename,
                    "derivative_relative_path": deriv_rel_path,
                    "derivative_sha256": deriv_sha256,
                    "transformation": t_name,
                    "transformation_parameters": t_params,
                    "original_width": orig_w,
                    "original_height": orig_h,
                    "derivative_width": d_w,
                    "derivative_height": d_h,
                    "derivative_format": d_fmt,
                    "model_repository": ai_res.detector.model_repository or ai_res.detector.name,
                    "pinned_model_revision": ai_res.detector.pinned_revision,
                    "model_architecture": ai_res.detector.architecture,
                    "preprocessing": (
                        ai_res.detector.preprocessing.model_dump()
                        if ai_res.detector.preprocessing is not None
                        else {}
                    ),
                    "artificial_score": art_score,
                    "human_score": hum_score,
                    "selected_class": sel_class,
                    "score_type": ai_res.prediction.score_type,
                    "decision_rule": ai_res.prediction.decision_rule,
                    "threshold": ai_res.prediction.threshold,
                    "status": ai_res.status,
                    "source_vs_model_disagreement": is_disagreement,
                    "disagreement_observation": disagreement_note,
                }
            )

        # Compute deltas relative to this source image's 'original' evaluation
        baseline_row = next(r for r in src_eval_rows if r["transformation"] == "original")
        base_art = baseline_row["artificial_score"]
        base_hum = baseline_row["human_score"]
        base_cls = baseline_row["selected_class"]

        for r in src_eval_rows:
            r["baseline_artificial_score"] = base_art
            r["baseline_human_score"] = base_hum
            r["baseline_selected_class"] = base_cls
            r["artificial_score_delta"] = round(float(r["artificial_score"] - base_art), 4)
            r["human_score_delta"] = round(float(r["human_score"] - base_hum), 4)
            r["selected_class_changed"] = bool(r["selected_class"] != base_cls)
            r["class_change_description"] = (
                f"observed class change: {base_cls} -> {r['selected_class']}"
                if r["selected_class_changed"]
                else "no class change observed"
            )
            detector_results.append(r)

    # Grouped descriptive summaries by source_category and transformation
    grouped_descriptive: Dict[str, Dict[str, Any]] = {
        "human": {},
        "ai_generated": {},
        "all_sources": {},
    }
    for t_name in TRANSFORMATION_ORDER:
        h_items = [
            r
            for r in detector_results
            if r["source_category"] == "human" and r["transformation"] == t_name
        ]
        ai_items = [
            r
            for r in detector_results
            if r["source_category"] == "ai_generated" and r["transformation"] == t_name
        ]
        all_items = [r for r in detector_results if r["transformation"] == t_name]
        grouped_descriptive["human"][t_name] = _compute_group_summary(h_items)
        grouped_descriptive["ai_generated"][t_name] = _compute_group_summary(ai_items)
        grouped_descriptive["all_sources"][t_name] = _compute_group_summary(all_items)

    class_change_items = [r for r in detector_results if r["selected_class_changed"] is True]
    disagreement_items = [r for r in detector_results if r["source_vs_model_disagreement"] is True]

    human_src_count = sum(1 for s in source_samples if s["known_category"] == "human")
    ai_src_count = sum(1 for s in source_samples if s["known_category"] == "ai_generated")

    manifest_output_path = _ensure_safe_subpath(
        dataset_root / "transformation_robustness_manifest.json", dataset_root
    )
    results_json_path = _ensure_safe_subpath(
        dataset_root / "transformation_robustness_results.json", dataset_root
    )
    results_csv_path = _ensure_safe_subpath(
        dataset_root / "transformation_robustness_results.csv", dataset_root
    )
    summary_md_path = _ensure_safe_subpath(
        dataset_root / "transformation_robustness_summary.md", dataset_root
    )

    generated_at = datetime.now(timezone.utc).isoformat()

    manifest_payload = {
        "schema_version": "1.0",
        "experiment": "step_14_1_5_transformation_robustness",
        "generated_at": generated_at,
        "source_manifest": str(resolved_manifest),
        "transformations": list(TRANSFORMATION_ORDER),
        "source_image_count": len(source_samples),
        "derivative_count": len(manifest_derivatives),
        "derivatives": manifest_derivatives,
    }

    results_payload = {
        "report_type": "descriptive transformation result",
        "experiment": "step_14_1_5_transformation_robustness",
        "generated_at": generated_at,
        "dataset_composition": {
            "source_manifest": str(resolved_manifest),
            "human_source_count": human_src_count,
            "ai_generated_source_count": ai_src_count,
            "total_source_images": len(source_samples),
            "transformations_tested": list(TRANSFORMATION_ORDER),
            "original_copies_count": len(source_samples),
            "transformed_derivatives_count": len(manifest_derivatives) - len(source_samples),
            "total_evaluated_files": len(manifest_derivatives),
        },
        "model_integrity": model_integrity,
        "grouped_descriptive_results": grouped_descriptive,
        "observed_class_changes": {
            "count": len(class_change_items),
            "items": class_change_items,
        },
        "observed_disagreements": {
            "count": len(disagreement_items),
            "items": disagreement_items,
        },
        "results": detector_results,
        "limitations": list(ROBUSTNESS_LIMITATIONS),
    }

    manifest_output_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    results_json_path.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    _write_robustness_csv(results_csv_path, detector_results)
    summary_md_path.write_text(_generate_markdown_summary(results_payload), encoding="utf-8")

    return {
        "manifest_payload": manifest_payload,
        "results_payload": results_payload,
        "output_files": {
            "transformation_robustness_manifest_json": str(manifest_output_path),
            "transformation_robustness_results_json": str(results_json_path),
            "transformation_robustness_results_csv": str(results_csv_path),
            "transformation_robustness_summary_md": str(summary_md_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 14.1.5 — Controlled AI-Detector Transformation Robustness Testing CLI"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to source manifest.json (default: datasets/ai_detector_validation/manifest.json)",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=None,
        help="Base output directory (default: datasets/ai_detector_validation/)",
    )
    args = parser.parse_args()

    out = run_transformation_robustness_experiment(
        manifest_path=args.manifest,
        output_base_dir=args.output_base_dir,
    )
    rp = out["results_payload"]
    comp = rp["dataset_composition"]

    print("==========================================================================")
    print("MediaTrace AI Image Detector — Transformation Robustness Experiment (14.1.5)")
    print("==========================================================================")
    print(f"report_type          : {rp['report_type']}")
    print(f"source images        : {comp['total_source_images']} (human={comp['human_source_count']}, ai_generated={comp['ai_generated_source_count']})")
    print(f"transformations      : {', '.join(comp['transformations_tested'])}")
    print(f"total evaluated files: {comp['total_evaluated_files']} ({comp['original_copies_count']} originals + {comp['transformed_derivatives_count']} transformed)")
    print(f"observed class change: {rp['observed_class_changes']['count']}")
    print(f"source/model disagree: {rp['observed_disagreements']['count']}")
    print("--------------------------------------------------------------------------")
    for r in rp["results"]:
        print(
            f"[{r['derivative_id']:22s}] ({r['source_category']:12s}) "
            f"sel={r['selected_class']:10s} | art={r['artificial_score']:.4f} (delta={r['artificial_score_delta']:+.4f}) | "
            f"cls_chg={str(r['selected_class_changed']):5s} | disagree={str(r['source_vs_model_disagreement']):5s}"
        )
    print("--------------------------------------------------------------------------")
    for k, v in out["output_files"].items():
        print(f"  {k}: {v}")
    print("==========================================================================")


if __name__ == "__main__":
    main()
