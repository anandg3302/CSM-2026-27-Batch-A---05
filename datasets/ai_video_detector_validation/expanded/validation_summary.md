# MediaTrace Stage A.2.3 — Expanded Controlled Validation Summary

**Execution Timestamp (UTC):** 2026-09-30T12:09:51.725155+00:00  
**Dataset:** SDFVD (Small-scale Deepfake Forgery Video Dataset) — Expanded Controlled Cohort  
**Cohort Size:** 26 samples (10 human, 10 manipulated, 6 transformed derivatives)  
**Stage:** `A.2.3` | **Status:** `COMPLETE`

---

## 1. Per-Sample Detailed Observations

| Sample | Category | VMAE Norm Class | VMAE Score | VMAE Agree? | RF Class | RF Logit | RF Agree? | Dual Relationship |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v1.mp4` | human | manipulated | 0.7482 | No | human | -4.1347 | Yes | `models_disagree` |
| `v2.mp4` | human | manipulated | 0.8595 | No | manipulated | +1.2441 | No | `both_disagree_with_known_category` |
| `v3.mp4` | human | manipulated | 0.9216 | No | human | -2.5179 | Yes | `models_disagree` |
| `v4.mp4` | human | manipulated | 0.5224 | No | human | -2.7173 | Yes | `models_disagree` |
| `v5.mp4` | human | manipulated | 0.6348 | No | manipulated | +1.9748 | No | `both_disagree_with_known_category` |
| `v6.mp4` | human | manipulated | 0.8576 | No | manipulated | +1.8113 | No | `both_disagree_with_known_category` |
| `v7.mp4` | human | manipulated | 0.9341 | No | human | -0.2008 | Yes | `models_disagree` |
| `v8.mp4` | human | manipulated | 0.9663 | No | manipulated | +5.3446 | No | `both_disagree_with_known_category` |
| `v9.mp4` | human | manipulated | 0.7924 | No | manipulated | +0.5005 | No | `both_disagree_with_known_category` |
| `v10.mp4` | human | manipulated | 0.9436 | No | manipulated | +5.3309 | No | `both_disagree_with_known_category` |
| `vs1.mp4` | manipulated | manipulated | 0.8115 | Yes | manipulated | +1.9942 | Yes | `both_agree_with_known_category` |
| `vs2.mp4` | manipulated | manipulated | 0.8984 | Yes | manipulated | +5.3826 | Yes | `both_agree_with_known_category` |
| `vs3.mp4` | manipulated | manipulated | 0.9064 | Yes | human | -0.8573 | No | `models_disagree` |
| `vs4.mp4` | manipulated | human | 0.4801 | No | manipulated | +2.7986 | Yes | `models_disagree` |
| `vs5.mp4` | manipulated | manipulated | 0.7094 | Yes | manipulated | +9.5728 | Yes | `both_agree_with_known_category` |
| `vs6.mp4` | manipulated | manipulated | 0.9030 | Yes | manipulated | +1.7614 | Yes | `both_agree_with_known_category` |
| `vs7.mp4` | manipulated | manipulated | 0.9417 | Yes | manipulated | +7.2406 | Yes | `both_agree_with_known_category` |
| `vs8.mp4` | manipulated | manipulated | 0.9251 | Yes | manipulated | +5.6189 | Yes | `both_agree_with_known_category` |
| `vs9.mp4` | manipulated | manipulated | 0.8121 | Yes | human | -0.0822 | No | `models_disagree` |
| `vs10.mp4` | manipulated | manipulated | 0.9658 | Yes | manipulated | +8.0995 | Yes | `both_agree_with_known_category` |
| `v1_resize50.mp4` | transformed | manipulated | 0.7404 | N/A | human | -4.7611 | N/A | `evaluated_derivative` |
| `v1_resize75.mp4` | transformed | manipulated | 0.7672 | N/A | human | -5.0491 | N/A | `evaluated_derivative` |
| `v2_resize50.mp4` | transformed | manipulated | 0.8879 | N/A | manipulated | +2.2342 | N/A | `evaluated_derivative` |
| `vs1_resize50.mp4` | transformed | manipulated | 0.8234 | N/A | manipulated | +3.1409 | N/A | `evaluated_derivative` |
| `vs1_resize75.mp4` | transformed | manipulated | 0.8130 | N/A | manipulated | +0.4295 | N/A | `evaluated_derivative` |
| `vs2_resize50.mp4` | transformed | manipulated | 0.9155 | N/A | manipulated | +1.1290 | N/A | `evaluated_derivative` |

---

## 2. Model Performance Breakdown (Descriptive)

### VideoMAE (`Vansh180/VideoMae-ffc23-deepfake-detector`)
- **Known Human Samples (n=10):** 0 agreed (10 selected fake)
- **Known Manipulated Samples (n=10):** 9 agreed (1 selected real)

### RealForensics (`realforensics_ff.pth`)
- **Known Human Samples (n=10):** 4 agreed (6 selected manipulated)
- **Known Manipulated Samples (n=10):** 8 agreed (2 selected human)

---

## 3. Comparative Signal Relationship Matrix (n=20 primary)

| Relationship Category | Count | Percentage |
| :--- | :--- | :--- |
| `both_agree_with_known_category` | 7 | 35.0% |
| `both_disagree_with_known_category` | 6 | 30.0% |
| `models_disagree` | 7 | 35.0% |

---

## 4. Transformation Robustness (n=6 derivatives)

| Derivative | Parent | Type | VMAE Parent -> Deriv | VMAE Delta | VMAE Preserved | RF Parent -> Deriv | RF Delta | RF Preserved |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v1_resize50.mp4` | `v1.mp4` | spatial_downscale_50pct | 0.7482 -> 0.7404 | -0.0078 | Yes | -4.1347 -> -4.7611 | -0.6264 | Yes |
| `v1_resize75.mp4` | `v1.mp4` | spatial_downscale_75pct | 0.7482 -> 0.7672 | +0.0190 | Yes | -4.1347 -> -5.0491 | -0.9144 | Yes |
| `v2_resize50.mp4` | `v2.mp4` | spatial_downscale_50pct | 0.8595 -> 0.8879 | +0.0284 | Yes | +1.2441 -> +2.2342 | +0.9901 | Yes |
| `vs1_resize50.mp4` | `vs1.mp4` | spatial_downscale_50pct | 0.8115 -> 0.8234 | +0.0119 | Yes | +1.9942 -> +3.1409 | +1.1467 | Yes |
| `vs1_resize75.mp4` | `vs1.mp4` | spatial_downscale_75pct | 0.8115 -> 0.8130 | +0.0015 | Yes | +1.9942 -> +0.4295 | -1.5647 | Yes |
| `vs2_resize50.mp4` | `vs2.mp4` | spatial_downscale_50pct | 0.8984 -> 0.9155 | +0.0171 | Yes | +5.3826 -> +1.1290 | -4.2535 | Yes |

---

## 5. Methodological Safeguards & Limitations

- **Out-of-Domain Engineering Experiment:** Evaluated on SDFVD (Pexels / Remaker AI); not a scientific benchmark or generalized real-world accuracy claim.
- **Uncalibrated Model Scores:** VideoMAE outputs uncalibrated fake class probabilities; RealForensics outputs raw unbounded scalar logits.
- **Signal Diversity Observation:** RealForensics exhibited a different output pattern from VideoMAE on the controlled validation samples, including different classifications on several known-human and known-manipulated videos. This demonstrates signal diversity in this experiment but does not establish detector reliability or forensic validity.
- **Zero Production Changes:** No production detection services, forensic frameworks, API endpoints, or UI were altered.
