# RealForensics Stage A.2.2 Isolated Controlled Validation Summary

**Execution Timestamp (UTC):** 2026-09-30T11:23:55.297656+00:00  
**Dataset:** SDFVD (Small-scale Deepfake Forgery Video Dataset) (controlled_out_of_domain_validation)  
**Model:** RealForensics (`realforensics_ff.pth`, SHA: `843b86f29fc98c67...`)  
**Architecture:** csn_r101 + MeanLinear (norm_linear=True, scale=64.0)  
**Status:** `COMPLETE`

---

## 1. Per-Sample Detailed Observations

| Sample | Category | Status | Uncalibrated Logit | Selected Class | Category Rel. | VideoMAE Class | Model Rel. | Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v1.mp4` | human | evaluated | `-4.1347` | **human** | model_category_agreement | fake | models_disagree | 23.22s |
| `v2.mp4` | human | evaluated | `+1.2441` | **manipulated** | model_category_disagreement | fake | both_disagree_with_known_category | 20.22s |
| `v3.mp4` | human | evaluated | `-2.5179` | **human** | model_category_agreement | fake | models_disagree | 19.21s |
| `vs1.mp4` | manipulated | evaluated | `+1.9942` | **manipulated** | model_category_agreement | fake | both_agree_with_known_category | 20.42s |
| `vs2.mp4` | manipulated | evaluated | `+5.3826` | **manipulated** | model_category_agreement | fake | both_agree_with_known_category | 19.70s |
| `vs3.mp4` | manipulated | evaluated | `-0.8573` | **human** | model_category_disagreement | fake | models_disagree | 19.27s |
| `v1_resize50.mp4` | transformed | evaluated | `-4.7611` | **human** | transformation_sensitivity_observed | fake | evaluated_derivative | 21.10s |

---

## 2. Descriptive Aggregate Statistics

* **Total Samples:** 7 (Evaluated: 7, Unavailable: 0)
* **Known Human Subset (3 samples):**
  * Agreements (selected `human`): **2 of 3** (samples: `v1.mp4` [-4.13], `v3.mp4` [-2.52])
  * Disagreements (selected `manipulated`): **1 of 3** (sample: `v2.mp4` [+1.24])
* **Known Manipulated Subset (3 samples):**
  * Agreements (selected `manipulated`): **2 of 3** (samples: `vs1.mp4` [+1.99], `vs2.mp4` [+5.38])
  * Disagreements (selected `human`): **1 of 3** (sample: `vs3.mp4` [-0.86])
* **Transformation Robustness (`v1.mp4` vs `v1_resize50.mp4`):**
  * Original Logit: `-4.134709`
  * Transformed Logit: `-4.761127`
  * Score Delta: `-0.626418` (Class preserved as `human`)
* **Raw Logit Range:** Min `-4.761127`, Max `5.38255` (Mean: `-0.521467`)

---

## 3. Comparison with Frozen VideoMAE Baseline

* **Both Agree with Known Category:** 2 (`vs1.mp4`, `vs2.mp4`)
* **Both Disagree with Known Category:** 1 (`v2.mp4`)
* **Models Disagree:** 3 (`v1.mp4`, `v3.mp4`, `vs3.mp4`)
* **Evaluated Derivative:** 1 (`v1_resize50.mp4`)

**Key Architectural Insight:**  
On the out-of-domain SDFVD human samples, VideoMAE classified all 3 as `fake` (false positive). RealForensics classified 2 of 3 as `human` (negative logits), showing distinct error profiles and demonstrable signal diversity.
