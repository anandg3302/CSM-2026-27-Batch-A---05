# Controlled Video AI Detector Validation Summary

- **Validation Timestamp**: `2026-09-30T09:16:59.555090+00:00`
- **Dataset Status**: **`COMPLETE`**
- **Model Repository**: `Vansh180/VideoMae-ffc23-deepfake-detector`
- **Pinned Revision**: `ea26518b527d938d66e568c52bcf1c19a86cf295`
- **Model Weights SHA-256**: `293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29`

## 1. Dataset Sample Inventory

| Category | Target Count | Observed Count | Status |
| :--- | :--- | :--- | :--- |
| `human` | 3 | 3 | Satisfied |
| `manipulated` | 3 | 3 | Satisfied |
| `transformed` | 1 | 1 | Satisfied |

## 2. Controlled Validation Observations

| Category Group | Evaluated | Selected Real | Selected Fake | Not Available (Out of Scope) |
| :--- | :--- | :--- | :--- | :--- |
| `known_human` | 3 | 0 | 3 | 0 |
| `known_manipulated` | 3 | 0 | 3 | 0 |
| `transformed` | 1 | 0 | 1 | 0 |

## 3. Disagreements

| Filename | Category | Selected Class | Real Score | Fake Score |
| :--- | :--- | :--- | :--- | :--- |
| `v1.mp4` | `human` | `fake` | 0.251844 | 0.748156 |
| `v2.mp4` | `human` | `fake` | 0.140484 | 0.859516 |
| `v3.mp4` | `human` | `fake` | 0.078377 | 0.921623 |

## 4. Evaluated Sample Records

| Filename | Category | SHA-256 | Status | Class | Real | Fake | Usable Frames | Agreement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `v1.mp4` | `human` | `d4f6698fee42...` | `completed` | `fake` | 0.251844 | 0.748156 | 16 | `model/category disagreement observed` |
| `v2.mp4` | `human` | `f53c36e8b625...` | `completed` | `fake` | 0.140484 | 0.859516 | 16 | `model/category disagreement observed` |
| `v3.mp4` | `human` | `26680561f761...` | `completed` | `fake` | 0.078377 | 0.921623 | 16 | `model/category disagreement observed` |
| `vs1.mp4` | `manipulated` | `40544991df2a...` | `completed` | `fake` | 0.188454 | 0.811546 | 16 | `agreement` |
| `vs2.mp4` | `manipulated` | `a2feff0ec068...` | `completed` | `fake` | 0.101619 | 0.898381 | 16 | `agreement` |
| `vs3.mp4` | `manipulated` | `f189db47c5bd...` | `completed` | `fake` | 0.093644 | 0.906356 | 16 | `agreement` |
| `v1_resize50.mp4` | `transformed` | `d3c104b56649...` | `completed` | `fake` | 0.259604 | 0.740396 | 16 | `evaluated_derivative` |

## 5. Forensic Limitations

- Controlled descriptive engineering validation harness, not a benchmark or general-world accuracy claim.
- Known categories come exclusively from independent ground-truth documentation; never inferred from detector output.
- Model scores are uncalibrated model-derived scores (score_type='uncalibrated_model_score') with decision_rule='model_class_selection' and threshold=null.
- Detector scope is restricted to facial video deepfakes (FaceForensics++ C23 domain); non-facial content is out of scope.
- Transformed derivatives evaluate transformation sensitivity and are reported separately.
- The detector alone never escalates forensic assessments and does not prove authenticity or provenance.
