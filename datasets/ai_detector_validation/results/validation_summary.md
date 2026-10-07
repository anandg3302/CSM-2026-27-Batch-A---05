# MediaTrace AI Image Detector — Controlled Validation Summary (Step 14.1.7)

> **Controlled Engineering Validation Statement**: This report documents a controlled descriptive engineering validation exercise for the MediaTrace AI image detector (`umm-maybe/AI-image-detector`). It is **not** a claim of benchmark accuracy, real-world reliability, or forensic model validation.

## 1. Dataset Composition
- **Dataset Completeness Status**: `incomplete` (Categories below target sample count (human, ai_generated) are reported descriptively as incomplete without fabricating or downloading synthetic images.)
- **Known-Human Source Images**: `1` (Target: `10`)
- **Known-AI Source Images**: `3` (Target: `10`)
- **Unique Source SHA-256 Digests**: `4`
- **Transformed Derivatives (Tracked Separately)**: `1` (Excluded from source human-vs-AI counts)

## 2. Provenance Summary
- **GPS Coordinates Exposed**: `False`
- **Known-Human Provenance**: `count=1`, `known_camera_source=1`, `unknown_source=0`, `devices={"realme 10 Pro 5G (RMX3660)": 1}`, `formats={"JPEG": 1}`, `dimensions={"731x987": 1}`
- **Known-AI Provenance**: `count=3`, `known_generator=0`, `unknown_generator=3`, `generators={"unknown": 3}`, `formats={"JPEG": 3}`, `dimensions={"517x415": 1, "192x152": 1, "510x408": 1}`

## 3. Model Configuration
- **Repository**: `umm-maybe/AI-image-detector`
- **Pinned Revision**: `c7e223baf11bc40528af364ba7bdea030ef42f9e`
- **Architecture**: `SwinForImageClassification`
- **License**: `Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)`
- **Class Mapping**: `0 = artificial`, `1 = human`
- **Score Type**: `uncalibrated_model_score`
- **Decision Rule**: `model_class_selection`
- **Threshold**: `null`
- **Preprocessing**: RGB, resize `224x224` (`BICUBIC`), `[0,255] -> [0.0,1.0]`, ImageNet mean `[0.485, 0.456, 0.406]` / std `[0.229, 0.224, 0.225]`, `NCHW float32`

## 4. Known-Human Results
| Sample ID | Filename | Format | Dimensions | Camera / Source | Selected Class | Artificial Score | Human Score | Disagreement |
|---|---|---|---|---|---|---|---|---|
| `human_001` | `anand_pic.jpg` | `JPEG` | `731x987` | `realme 10 Pro 5G (RMX3660)` | `human` | `0.2974` | `0.7026` | `False` |

## 5. Known-AI Results
| Sample ID | Filename | Format | Dimensions | Generator | Selected Class | Artificial Score | Human Score | Disagreement |
|---|---|---|---|---|---|---|---|---|
| `ai_001` | `downlo.jpeg` | `JPEG` | `517x415` | `unknown` | `artificial` | `0.7490` | `0.2510` | `False` |
| `ai_002` | `downloa.jpeg` | `JPEG` | `192x152` | `unknown` | `artificial` | `0.9770` | `0.0230` | `False` |
| `ai_003` | `download.jpeg` | `JPEG` | `510x408` | `unknown` | `human` | `0.2736` | `0.7264` | `True` |

## 6. Model/Category Disagreements
- **Total Disagreements Observed**: `1`
  - **`ai_003` (`download.jpeg`)**: known_category=`ai_generated`, selected_class=`human`, artificial_score=`0.2736`, human_score=`0.7264`, dimensions=`510x408`, format=`JPEG`, observation=`model/category disagreement observed`.

## 7. Transformation Experiment Reference (Kept Separate)
- **Manifest Transformed Derivatives (`transformed/`)**: `1` sample(s) evaluated separately and excluded from source human-vs-AI counts.
  - `transformed_001` (parent=`human_001`, parent_sha256=`fd98d202253fe5e7f643384687bf4c96b78f4c67b1a931023f342538508c0f2b`): selected_class=`human`, artificial_score=`0.0785`, human_score=`0.9215`.
- **Step 14.1.5 Controlled Transformation Robustness Experiment**: Maintained separately in `datasets/ai_detector_validation/transformation_robustness_results.json` and `transformation_robustness_summary.md` (24 records across 4 source images × 6 conditions).

## 8. Descriptive Statistics (`controlled dataset descriptive counts`)
- **Label**: `controlled dataset descriptive counts`
- **Known-Human (`n=1`)**: selected_human=`1`, selected_artificial=`0`, artificial_score min/max/mean/median=`0.2974` / `0.2974` / `0.2974` / `0.2974`, human_score min/max/mean/median=`0.7026` / `0.7026` / `0.7026` / `0.7026`
- **Known-AI (`n=3`)**: selected_artificial=`2`, selected_human=`1`, artificial_score min/max/mean/median=`0.2736` / `0.977` / `0.6665` / `0.749`, human_score min/max/mean/median=`0.023` / `0.7264` / `0.3335` / `0.251`

## 9. Dataset Quality Warnings & Limitations
### Dataset Quality Warnings (Informational)
- **[`human_count_below_target`]**: Known-human source sample count (1) is below the target of 10; dataset is marked incomplete and does not support statistical generalizations.
- **[`ai_count_below_target`]**: Known-AI source sample count (3) is below the target of 10; dataset is marked incomplete and does not support statistical generalizations.
- **[`limited_known_generator_diversity`]**: Only 0 distinct known AI generator(s) explicitly identified across 3 AI sample(s); generator model diversity is limited.
- **[`single_human_source_device`]**: Only 1 human camera/device source(s) represented (['realme 10 Pro 5G (RMX3660)']).
- **[`unknown_generator_or_camera_provenance`]**: 3 of 4 source sample(s) have 'unknown' specific generator or camera model provenance.
- **[`dataset_dominated_by_single_source`]**: Source dataset is dominated by a single source description (3/4 samples: 'Known AI-generated image supplied by developer under datasets/ai_detector_validation/ai_generated/solar/').

### Forensic & Validation Limitations
- This is a controlled descriptive engineering validation harness, not a scientific or forensic benchmark.
- Known categories come exclusively from manifest.json and are never inferred from model predictions, EXIF metadata, or visual heuristics.
- Model outputs are uncalibrated model-derived scores (uncalibrated_model_score) with decision_rule='model_class_selection' and threshold=null.
- This small convenience sample is not a representative statistical benchmark; results must not be generalized to arbitrary internet imagery or newer generators.
- Transformed images are reported separately and excluded from the source-level human-vs-AI descriptive confusion counts.
- No authenticity, provenance, benchmark accuracy, or manipulation probability conclusion is generated from detector scores or classical forensic observations.

## 10. Explicit Controlled Engineering Validation Statement
This validation run is a **controlled descriptive engineering experiment** using locally available, provenance-documented files. It does **not** establish production accuracy, benchmark accuracy, forensic reliability, or calibrated performance for `umm-maybe/AI-image-detector`.
