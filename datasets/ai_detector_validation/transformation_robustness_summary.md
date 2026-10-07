# Step 14.1.5 — Controlled AI-Detector Transformation Robustness Summary

> **Descriptive Engineering Experiment Notice**: This document reports descriptive transformation results (`observed score shift` and `observed class change`) for a small controlled sample. It is **not** a scientific benchmark and does not claim accuracy, precision, recall, robustness, or forensic validity.

## 1. Dataset Composition & Model Metadata

- **Source Manifest**: `C:\Users\Anand\finalPro\datasets\ai_detector_validation\manifest.json`
- **Known Human Source Images**: `1`
- **Known AI-Generated Source Images**: `3`
- **Total Source Images Selected**: `4`
- **Transformations per Source Image**: `6` (`original, resize_50, resize_25, jpeg_q90, jpeg_q70, jpeg_q50`)
- **Total Evaluated Files (Originals + Transformed Derivatives)**: `24` (`4` originals + `20` transformed derivatives)
- **Model Repository**: `umm-maybe/AI-image-detector`
- **Pinned Model Revision**: `c7e223baf11bc40528af364ba7bdea030ef42f9e`
- **Model Architecture**: `SwinForImageClassification`
- **PyTorch Checkpoint SHA-256**: `f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf`
- **Local ONNX Artifact SHA-256**: `693911a555891bac487c9e24daf4f2adcef258eaa7304d8d344ecc96b9af7b2e`
- **Score Semantics**: `score_type = uncalibrated_model_score`, `decision_rule = model_class_selection`, `threshold = null`

## 2. Transformations Tested

| Transformation | Description | Resampling / Encoding Parameters |
| :--- | :--- | :--- |
| `original` | Exact byte copy of source image into `transformation_robustness/` | Byte-for-byte identical copy (`SHA-256` preserved) |
| `resize_50` | 50% aspect-ratio preserving downscale | `Image.Resampling.BICUBIC`, saved as `JPEG` (`quality=95`) |
| `resize_25` | 25% aspect-ratio preserving downscale | `Image.Resampling.BICUBIC`, saved as `JPEG` (`quality=95`) |
| `jpeg_q90` | Same pixel dimensions, re-encoded at JPEG Q=90 | `JPEG` (`quality=90`) |
| `jpeg_q70` | Same pixel dimensions, re-encoded at JPEG Q=70 | `JPEG` (`quality=70`) |
| `jpeg_q50` | Same pixel dimensions, re-encoded at JPEG Q=50 | `JPEG` (`quality=50`) |

## 3. Per-Image Descriptive Results & Observed Score Shifts

| Source ID | Source Category | Transformation | Dimensions | Selected Class | Artificial Score | $\Delta$ Artificial | Human Score | $\Delta$ Human | Class Changed vs Original | Source/Model Disagreement |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| `human_001` | `human` | `original` | `731x987` | `human` | `0.2974` | `+0.0000` | `0.7026` | `+0.0000` | `false` | `false` |
| `human_001` | `human` | `resize_50` | `366x494` | `human` | `0.4207` | `+0.1233` | `0.5793` | `-0.1233` | `false` | `false` |
| `human_001` | `human` | `resize_25` | `183x247` | `artificial` | `0.6711` | `+0.3737` | `0.3289` | `-0.3737` | `true` | `true` |
| `human_001` | `human` | `jpeg_q90` | `731x987` | `human` | `0.3297` | `+0.0323` | `0.6703` | `-0.0323` | `false` | `false` |
| `human_001` | `human` | `jpeg_q70` | `731x987` | `human` | `0.1185` | `-0.1789` | `0.8815` | `+0.1789` | `false` | `false` |
| `human_001` | `human` | `jpeg_q50` | `731x987` | `human` | `0.0988` | `-0.1986` | `0.9012` | `+0.1986` | `false` | `false` |
| `ai_001` | `ai_generated` | `original` | `517x415` | `artificial` | `0.7490` | `+0.0000` | `0.2510` | `+0.0000` | `false` | `false` |
| `ai_001` | `ai_generated` | `resize_50` | `258x208` | `artificial` | `0.8851` | `+0.1361` | `0.1149` | `-0.1361` | `false` | `false` |
| `ai_001` | `ai_generated` | `resize_25` | `129x104` | `artificial` | `0.8939` | `+0.1449` | `0.1061` | `-0.1449` | `false` | `false` |
| `ai_001` | `ai_generated` | `jpeg_q90` | `517x415` | `artificial` | `0.7167` | `-0.0323` | `0.2833` | `+0.0323` | `false` | `false` |
| `ai_001` | `ai_generated` | `jpeg_q70` | `517x415` | `artificial` | `0.7331` | `-0.0159` | `0.2669` | `+0.0159` | `false` | `false` |
| `ai_001` | `ai_generated` | `jpeg_q50` | `517x415` | `artificial` | `0.6439` | `-0.1051` | `0.3561` | `+0.1051` | `false` | `false` |
| `ai_002` | `ai_generated` | `original` | `192x152` | `artificial` | `0.9770` | `+0.0000` | `0.0230` | `+0.0000` | `false` | `false` |
| `ai_002` | `ai_generated` | `resize_50` | `96x76` | `artificial` | `0.9278` | `-0.0492` | `0.0722` | `+0.0492` | `false` | `false` |
| `ai_002` | `ai_generated` | `resize_25` | `48x38` | `artificial` | `0.8240` | `-0.1530` | `0.1760` | `+0.1530` | `false` | `false` |
| `ai_002` | `ai_generated` | `jpeg_q90` | `192x152` | `artificial` | `0.9772` | `+0.0002` | `0.0228` | `-0.0002` | `false` | `false` |
| `ai_002` | `ai_generated` | `jpeg_q70` | `192x152` | `artificial` | `0.9835` | `+0.0065` | `0.0165` | `-0.0065` | `false` | `false` |
| `ai_002` | `ai_generated` | `jpeg_q50` | `192x152` | `artificial` | `0.9850` | `+0.0080` | `0.0150` | `-0.0080` | `false` | `false` |
| `ai_003` | `ai_generated` | `original` | `510x408` | `human` | `0.2736` | `+0.0000` | `0.7264` | `+0.0000` | `false` | `true` |
| `ai_003` | `ai_generated` | `resize_50` | `255x204` | `artificial` | `0.7008` | `+0.4272` | `0.2992` | `-0.4272` | `true` | `false` |
| `ai_003` | `ai_generated` | `resize_25` | `128x102` | `artificial` | `0.7510` | `+0.4774` | `0.2490` | `-0.4774` | `true` | `false` |
| `ai_003` | `ai_generated` | `jpeg_q90` | `510x408` | `human` | `0.3085` | `+0.0349` | `0.6915` | `-0.0349` | `false` | `true` |
| `ai_003` | `ai_generated` | `jpeg_q70` | `510x408` | `human` | `0.2721` | `-0.0015` | `0.7279` | `+0.0015` | `false` | `true` |
| `ai_003` | `ai_generated` | `jpeg_q50` | `510x408` | `human` | `0.3211` | `+0.0475` | `0.6789` | `-0.0475` | `false` | `true` |

## 4. Grouped Descriptive Results by Source Category & Transformation

### 4.1 Known Human Source Images (`source_category = human`)

| Transformation | Samples | Selected `artificial` | Selected `human` | Class Changed Count | Artificial Score (`min` / `max` / `mean` / `median`) | $\Delta$ Artificial Mean |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| `original` | `1` | `0` | `1` | `0` | `0.2974` / `0.2974` / `0.2974` / `0.2974` | `+0.0000` |
| `resize_50` | `1` | `0` | `1` | `0` | `0.4207` / `0.4207` / `0.4207` / `0.4207` | `+0.1233` |
| `resize_25` | `1` | `1` | `0` | `1` | `0.6711` / `0.6711` / `0.6711` / `0.6711` | `+0.3737` |
| `jpeg_q90` | `1` | `0` | `1` | `0` | `0.3297` / `0.3297` / `0.3297` / `0.3297` | `+0.0323` |
| `jpeg_q70` | `1` | `0` | `1` | `0` | `0.1185` / `0.1185` / `0.1185` / `0.1185` | `-0.1789` |
| `jpeg_q50` | `1` | `0` | `1` | `0` | `0.0988` / `0.0988` / `0.0988` / `0.0988` | `-0.1986` |

### 4.2 Known AI-Generated Source Images (`source_category = ai_generated`)

| Transformation | Samples | Selected `artificial` | Selected `human` | Class Changed Count | Artificial Score (`min` / `max` / `mean` / `median`) | $\Delta$ Artificial Mean |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: |
| `original` | `3` | `2` | `1` | `0` | `0.2736` / `0.977` / `0.6665` / `0.749` | `+0.0000` |
| `resize_50` | `3` | `3` | `0` | `1` | `0.7008` / `0.9278` / `0.8379` / `0.8851` | `+0.1714` |
| `resize_25` | `3` | `3` | `0` | `1` | `0.751` / `0.8939` / `0.823` / `0.824` | `+0.1564` |
| `jpeg_q90` | `3` | `2` | `1` | `0` | `0.3085` / `0.9772` / `0.6675` / `0.7167` | `+0.0009` |
| `jpeg_q70` | `3` | `2` | `1` | `0` | `0.2721` / `0.9835` / `0.6629` / `0.7331` | `-0.0036` |
| `jpeg_q50` | `3` | `2` | `1` | `0` | `0.3211` / `0.985` / `0.65` / `0.6439` | `-0.0165` |

## 5. Observed Class Changes Relative to Original

- **Total Derivatives with Observed Class Change (`selected_class_changed = true`)**: `3`
  - `human_001__resize_25` (`source_id=human_001`, `source_category=human`, `transformation=resize_25`): `human` $\rightarrow$ `artificial` (`artificial_score`: `0.2974` $\rightarrow$ `0.6711`, $\Delta = +0.3737`)
  - `ai_003__resize_50` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=resize_50`): `human` $\rightarrow$ `artificial` (`artificial_score`: `0.2736` $\rightarrow$ `0.7008`, $\Delta = +0.4272`)
  - `ai_003__resize_25` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=resize_25`): `human` $\rightarrow$ `artificial` (`artificial_score`: `0.2736` $\rightarrow$ `0.7510`, $\Delta = +0.4774`)

## 6. Observed Source-Category vs. Model-Output Disagreements

- **Total Evaluations with Source/Model Disagreement (`source_vs_model_disagreement = true`)**: `5`
  - `human_001__resize_25` (`source_id=human_001`, `source_category=human`, `transformation=resize_25`): `model_selected_class=artificial` (`artificial_score=0.6711`, `human_score=0.3289`) — `model/category disagreement observed`
  - `ai_003__original` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=original`): `model_selected_class=human` (`artificial_score=0.2736`, `human_score=0.7264`) — `model/category disagreement observed`
  - `ai_003__jpeg_q90` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=jpeg_q90`): `model_selected_class=human` (`artificial_score=0.3085`, `human_score=0.6915`) — `model/category disagreement observed`
  - `ai_003__jpeg_q70` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=jpeg_q70`): `model_selected_class=human` (`artificial_score=0.2721`, `human_score=0.7279`) — `model/category disagreement observed`
  - `ai_003__jpeg_q50` (`source_id=ai_003`, `source_category=ai_generated`, `transformation=jpeg_q50`): `model_selected_class=human` (`artificial_score=0.3211`, `human_score=0.6789`) — `model/category disagreement observed`

## 7. Explicit Limitations

- This is a small controlled engineering experiment.
- Results are descriptive and are not a benchmark.
- Transformations are limited to those actually tested (original, resize_50, resize_25, jpeg_q90, jpeg_q70, jpeg_q50).
- Results cannot establish detector accuracy or real-world reliability.
- JPEG recompression/resizing can change detector outputs.
- The detector score remains uncalibrated (uncalibrated_model_score).
- The detector is not a universal AI-image/deepfake detector.
- Model behavior may differ for other generators, image domains, resolutions, screenshots, crops, or compression pipelines.
- No provenance or authenticity conclusion can be made from the detector alone.
