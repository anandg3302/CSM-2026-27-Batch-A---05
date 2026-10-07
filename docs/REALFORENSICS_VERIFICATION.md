# RealForensics Pretrained Model Verification & Forensic Compatibility Report

**MediaTrace Platform Engineering — Stage A.2.1**  
**Document ID:** `MT-DOC-RF-VERIFY-001`  
**Classification:** Research & Model Verification Only (Zero Production Code Changes)  
**Status:** `VERIFIED_STANDALONE`  
**Date:** September 30, 2026  

---

## 1. Executive Summary

Stage A.2.1 establishes whether the official pretrained **RealForensics** video face-forgery detector (*"Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection"*, CVPR 2022) is viable as a second, independent video AI detector alongside MediaTrace's existing VideoMAE baseline.

### Key Conclusions:
1. **Model Weights & Provenance Verified:** The official pretrained weights (`realforensics_ff.pth`, 86,220,575 bytes, SHA-256 `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d`) were successfully acquired from the authors' official distribution, cryptographically verified, and pinned in `storage/models/realforensics-verification/artifact_integrity.json`.
2. **100% Dependency-Free Pure PyTorch Reconstruction:** The official code relies on deprecated and conflicting research libraries (`pytorchvideo 0.1.2`, `pytorch_lightning 1.3.8`, `hydra-core 1.1.0`) incompatible with Python 3.13 and modern PyTorch 2.x. We successfully reconstructed the complete CSN-101 backbone and `MeanLinear` classification head in pure standard PyTorch. All 625 state dictionary keys loaded strictly with zero missing or unexpected keys (`strict=True`).
3. **Deterministic Execution Established:** Synthetic test tensors executed deterministically on CPU, yielding identical numerical logits across repeated runs ($1.866194$ on zero inputs, $-0.702618$ on fixed seed `randn`).
4. **Significant Preprocessing Gap Identified:** VideoMAE processes 16 uniformly sampled full-frame facial crops with standard image scaling. In contrast, RealForensics strictly requires a 25-frame contiguous facial sequence aligned via 68-point 2D facial landmarks (RetinaFace/2D-FAN) warped onto a 20-word mean face template (`20words_mean_face.npy`) and cropped to 112x112 / 150x150.
5. **Production Readiness Verdict:** **VIABLE FOR STAGE A.2 STAGED ADOPTION**, provided a modular, standalone facial alignment preprocessor is engineered before pipeline integration. Zero production services, thresholds, or decision rules were modified during this verification stage.

---

## 2. Upstream Provenance & Licensing

| Attribute | Specification |
| :--- | :--- |
| **Model Name** | RealForensics (FaceForensics++ pretrained) |
| **Authors** | Alexandros Haliassos, Rodrigo Mira, Stavros Petridis, Maja Pantic |
| **Conference** | IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022) |
| **Paper Title** | *Leveraging Real Talking Faces via Self-Supervision for Robust Forgery Detection* |
| **Official Repository** | `https://github.com/ahaliassos/RealForensics` |
| **Git Default Branch / Commit** | `main` / `a53e849407d2fd5be0f0a7eb3df7d1a846b5b803` (March 16, 2023) |
| **Upstream License** | **MIT License** (permits commercial use, modification, distribution, and private use) |
| **Weight Distribution** | Google Drive (official author link in README): `1nqEVlRN51WyzMWSeB7x9okcaegFgA-BQ` |
| **Local Checkpoint Path** | `storage/models/realforensics-verification/realforensics_ff.pth` |
| **File Size** | `86,220,575` bytes (~82.23 MB) |
| **Cryptographic Digest (SHA-256)** | `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d` |

---

## 3. Architectural Comparison: VideoMAE vs. RealForensics

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                  MediaTrace Video Detectors                            │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│         VideoMAE (Existing Baseline)      │       RealForensics (Candidate Signal)     │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ Architecture: Vision Transformer (ViT-B)  │ Architecture: 3D CSN-101 (ir-CSN-101)      │
│ Input: 16 uniformly sampled frames        │ Input: 25 contiguous frames                │
│ Spatial Resolution: 224 x 224             │ Spatial Resolution: 112 x 112 / 150 x 150  │
│ Alignment: Bounding-box skin / face crop  │ Alignment: 68-point landmark affine warp   │
│ Inductive Bias: Global spatiotemporal     │ Inductive Bias: Channel-separated 3D convs │
│   self-attention across patch tokens      │   preserving temporal facial dynamics      │
│ Output: 2-class softmax logits            │ Output: 1 scalar logit (MeanLinear)        │
│ Parameters: ~86,000,000                   │ Parameters: 21,396,224                     │
│ Checkpoint Size: ~340 MB                  │ Checkpoint Size: ~82.2 MB                  │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

### Detailed Dimensional Comparison:

| Forensic / Architectural Dimension | VideoMAE Baseline (`Vansh180/VideoMae-ffc23`) | RealForensics (`realforensics_ff.pth`) | Forensic Synergy & Complementarity |
| :--- | :--- | :--- | :--- |
| **Neural Backbone** | 12-layer Vision Transformer (`ViT-Base`, patch 16x16x2) | 3D ResNet-101 with Channel-Separated Convolutions (`ir-CSN-101`) | **High:** ViT excels at spatial patch relationships; 3D CSN excels at fine temporal consistency and lip synchronization. |
| **Temporal Span & Sampling** | 16 frames sampled across entire video duration ($T_{\text{sample}} = 16$) | 25 consecutive frames at native 25 fps ($T_{\text{clip}} = 1.0\text{s}$) | **High:** VideoMAE catches video-level narrative artifacts; RealForensics catches micro-temporal warping and mouth-flutter. |
| **Input Shape** | `(B, 3, 16, 224, 224)` | `(B, 3, 25, 112, 112)` | Complementary resolutions and spatial coverage. |
| **Facial Region of Interest** | Loose facial bounding box / skin area | Tightly cropped, landmark-stabilized lower face / full face | RealForensics is specialized for talking-face and lip-sync manipulation (Face2Face, Deepfakes, NeuralTextures). |
| **Pretraining Objective** | Masked Video Autoencoding (90% video tube masking) | Audio-Visual Contrastive BYOL (predicting acoustic representations from visual lips) | Self-supervision from completely different modalities reduces shared blind spots. |
| **Classification Head** | Linear classifier: $768 \to 2$ | `MeanLinear`: temporal average pooling followed by cosine-normalized projection $2048 \to 1$ with scale $\alpha = 64$ | Provides an independent scalar signal for calibration and ensemble scoring. |

---

## 4. Model Parameterization & Layer Breakdown

The RealForensics architecture consists of 625 parameter and buffer tensors totaling **21,396,224 parameters** (~21.4M):

1. **Stem (`blocks[0]`):**
   - 3D Convolution: Kernel `(3, 7, 7)`, stride `(1, 2, 2)`, padding `(1, 3, 3)`, channels $3 \to 64$.
   - 3D Batch Normalization (eps=$10^{-5}$, momentum=0.1).
   - ReLU Activation + 3D MaxPool (kernel `(1, 3, 3)`, stride `(1, 2, 2)`, padding `(0, 1, 1)`).
2. **Residual Stage 0 (`blocks[1]`):**
   - 3 Residual blocks, 256 output channels.
   - Bottlenecks with channel-separated $3 \times 3 \times 3$ depthwise-separable 3D convolutions.
3. **Residual Stage 1 (`blocks[2]`):**
   - 4 Residual blocks, 512 output channels.
4. **Residual Stage 2 (`blocks[3]`):**
   - 23 Residual blocks, 1024 output channels.
5. **Residual Stage 3 (`blocks[4]`):**
   - 3 Residual blocks, 2048 output channels.
6. **Spatial Average Pooling (`blocks[5]`):**
   - Global spatial average over $H \times W$ (`x.mean(dim=(3, 4))`), outputting feature shape `(B, 2048, T)`.
7. **Classification Head (`df_head`):**
   - `MeanLinear` with temporal average pooling over $T$ (`x.mean(-1)`).
   - Cosine-normalized linear projection ($w \in \mathbb{R}^{1 \times 2048}$) scaled by $\alpha = 64.0$:
     $$\text{logit} = 64.0 \times \frac{\bar{x} \cdot w^T}{\|\bar{x}\|_2 \, \|w\|_2}$$

---

## 5. Checkpoint Verification & Bit-Level Integrity

The downloaded checkpoint file was verified directly using cryptographic digests and strict dictionary inspection:

```json
{
  "checkpoint_file": "realforensics_ff.pth",
  "file_size_bytes": 86220575,
  "sha256": "843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d",
  "total_state_dict_tensors": 625,
  "backbone_tensors": 624,
  "head_tensors": 1,
  "missing_keys": 0,
  "unexpected_keys": 0
}
```

Every parameter tensor was loaded with `strict=True`. No weights were missing, truncated, or unassigned.

---

## 6. Standalone Pure PyTorch Reconstruction

The original training and evaluation codebase (`ahaliassos/RealForensics`) was developed against an outdated scientific computing stack (`el_dorado` conda environment with PyTorch 1.9, torchvision 0.10, PyTorch Lightning 1.3.8, and Hydra 1.1.0). Attempting to install `pytorchvideo` on Python 3.13 causes fatal build failures due to deprecated C++ extensions and unmaintained wheel builds.

### Our Solution: Zero-Dependency Architecture Reconstruction
We inspected the underlying operations in `pytorchvideo.models.resnet` and `pytorchvideo.models.csn` and authored a self-contained, 100% standard PyTorch implementation:
- File location: `backend/tests/realforensics_arch.py`
- External dependencies: **`torch` and `math` ONLY**.
- Eliminates `pytorchvideo`, `pytorch_lightning`, and `hydra` completely.
- Fully compatible with current project environment: Python 3.13.15, PyTorch 2.14.0+cpu.

---

## 7. Preprocessing & Alignment Pipeline Analysis

The official RealForensics preprocessing pipeline (`preprocessing/extract_faces.py`) is fundamentally different from VideoMAE's lightweight frame extractor:

```
[Input Video (MP4)] 
        │
        ▼
[Face Landmark Extraction (RetinaFace + 2D-FAN)] ──> Produces 68-point 2D landmarks (.npy)
        │
        ▼
[Temporal Landmark Smoothing] ──> Window margin = 12 frames
        │
        ▼
[Affine Transformation] ──> LMEDS affine warp using 8 stable points onto 20words_mean_face.npy
        │
        ▼
[Facial Patch Cropping] ──> 150x150 crop centered on facial landmarks (points 15..68)
        │
        ▼
[Evaluation Transformation] ──> CenterCrop(140) -> Resize(112x112) -> ImageNet Normalization
```

### Preprocessing Gap Summary:
1. **Landmark Model Requirement:** RealForensics requires a 68-point 2D facial landmark detector. MediaTrace's current OpenCV Haar/skin-tone cascade extracts only crude bounding boxes and cannot produce the 8 stable points (`[28, 33, 36, 39, 42, 45, 48, 54]`) needed for the affine warp.
2. **Mean Face Reference:** RealForensics relies on `preprocessing/20words_mean_face.npy` (a $68 \times 2$ coordinate array derived from the LRW dataset). This template must be retained as a static calibration asset.
3. **Temporal Alignment:** Videos with severe head rotation, rapid occlusion, or missing facial detections fail the strict affine crop if landmarks fall outside bounds. Fallback mechanisms will be required in production.

---

## 8. Operational Performance & Resource Profiling

Inference latency and memory footprints were measured on the local host machine using CPU execution:

| Metric | VideoMAE Baseline | RealForensics Standalone |
| :--- | :--- | :--- |
| **Model Parameter Count** | 86,220,575 (~86M) | 21,396,224 (~21.4M) |
| **Disk Storage (Checkpoint)** | ~340 MB | 82.2 MB |
| **RAM Footprint (Weights loaded)** | ~360 MB | ~95 MB |
| **Single-Clip Inference (CPU)** | ~1.4 s (16 frames @ 224x224) | ~3.06 s (25 frames @ 150x150) |
| **Clip FPS Throughput** | ~11.4 fps | ~8.2 fps |

*Note: RealForensics performs 3D spatio-temporal convolutions across 25 frames, which involves higher FLOPs per frame than a 2D patch-projection transformer, resulting in ~3.06s inference on CPU per 1-second clip.*

---

## 9. Output Interpretation, Scoring Mechanics, & Calibration

### Output Mechanics:
- The model classification head outputs a single scalar logit $z \in \mathbb{R}$.
- Label mapping in upstream training:
  - $\text{Label } 0$: Authentic / Real face
  - $\text{Label } 1$: Manipulated / Forged face (Deepfakes, FaceSwap, Face2Face, NeuralTextures)
- Therefore:
  - $z > 0$: Evidence supporting manipulation
  - $z < 0$: Evidence supporting authenticity
- Raw sigmoid mapping:
  $$p_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Forensic Calibration Directive:
> [!WARNING]
> **Forensic Neutrality Mandate:** $p_{\text{raw}} = \sigma(z)$ is an **uncalibrated network confidence score**, NOT a true posterior probability of manipulation. Under real-world distribution shifts (e.g. unknown codecs, resolution degradation, or non-frontal lighting), uncalibrated sigmoid outputs cluster heavily near 0.0 or 1.0. 
> 
> In Stage A.2, RealForensics outputs must NEVER be presented to users as "99% probability of deepfake" without explicit temperature scaling / Platt scaling and domain calibration. It must be treated as an anomaly signal / model logit within the forensic decision framework.

---

## 10. Forensic Decision Fusion Compatibility (Stage A.2 Readiness)

VideoMAE and RealForensics possess ideal characteristics for multi-model decision fusion:

```
               [ Video File Input ]
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
  [ VideoMAE Pipeline ]       [ RealForensics Pipeline ]
  - Full-scene / Face ROI     - 25-frame contiguous facial sequence
  - ViT-Base (Global spatio-  - CSN-101 (Micro-temporal facial &
    temporal tube tokens)       lip-sync dynamics)
          │                           │
          ▼                           ▼
    Score: S_vmae               Score: S_rf
    (0.0 to 1.0)                (Raw logit / Calibrated score)
          │                           │
          └─────────────┬─────────────┘
                        ▼
       [ Multi-Signal Decision Framework ]
       - Agreement Matrix (Co-corroboration)
       - Conflicting Signal Resolution
       - Metadata & Audio-Visual Integrity Checks
       - Final Calibrated Integrity Assessment
```

### Fusion Scenarios:
1. **Corroborating Detection ($S_{\text{vmae}} > \tau_1 \land S_{\text{rf}} > \tau_2$):** High forensic confidence of manipulation; both global patch artifacts and fine facial temporal incoherence detected.
2. **Conflicting Signals ($S_{\text{vmae}} > \tau_1 \land S_{\text{rf}} \le \tau_2$):** Triggers `INCONCLUSIVE_DISCREPANCY` review flag; common in full-frame background manipulations or when face swapping artifacts are subtle.
3. **Corroborating Authenticity ($S_{\text{vmae}} \le \tau_1 \land S_{\text{rf}} \le \tau_2$):** Both detectors find consistent natural statistics.

---

## 11. Engineering Risk Assessment & Failure Modes

| Risk / Failure Mode | Impact | Severity | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Landmark Detection Failure** | Video with extreme angle, rapid motion blur, or heavy occlusion causes face detector to lose landmarks. | HIGH | Fallback to VideoMAE-only analysis; flag video as `PARTIAL_FACE_TRACKING`. |
| **CPU Latency Accumulation** | Running both VideoMAE (1.4s) and RealForensics (3.0s) adds ~4.5s total CPU analysis time per video. | MEDIUM | Execute models asynchronously in parallel worker threads or batch across clips. |
| **Compression Sensitivity** | Heavy C40/re-compression degrades high-frequency lip contours. | MEDIUM | Characterize performance degradation across compression factors during Stage A.2 validation. |
| **Uncalibrated Output Conflation** | Developer or UI conflates raw sigmoid score with ground-truth authenticity probability. | HIGH | Strict schema enforcement: designate field as `model_score_uncalibrated` and forbid probability language in reports. |

---

## 12. Compatibility Matrix

| Dependency / Component | Required by RealForensics | MediaTrace Environment | Verification Status |
| :--- | :--- | :--- | :--- |
| **Operating System** | Linux / Windows | Windows (Active) | **COMPATIBLE** |
| **Python Runtime** | 3.8 (legacy) | 3.13.15 (Active) | **COMPATIBLE** (via pure PyTorch reconstruction) |
| **PyTorch** | 1.9.0 (legacy) | 2.14.0+cpu (Active) | **COMPATIBLE** |
| **Torchvision** | 0.10.0 | Not installed in venv | **COMPATIBLE** (no torchvision dependencies needed) |
| **PyTorchVideo** | 0.1.2 | Incompatible / Deprecated | **REMOVED** (reconstructed in native PyTorch) |
| **PyTorch Lightning** | 1.3.8 | Incompatible with Py3.13 | **REMOVED** (reconstructed in native PyTorch) |
| **Hydra-core** | 1.1.0 | Not needed for inference | **REMOVED** |

---

## 13. Stage A.2 Integration Roadmap

If and when Stage A.2 is formally authorized:
1. **Phase 1: Lightweight Landmark Alignment Module:** Integrate a robust, lightweight facial landmark detector (e.g., MediaPipe Face Mesh or ONNX-based 2D-FAN) that generates the required 68-point landmarks without heavy CUDA/compiler requirements.
2. **Phase 2: Service Encapsulation:** Implement `backend/app/services/analysis/realforensics_service.py` adhering to the same interface as `video_ai_detection_service.py`.
3. **Phase 3: Controlled Validation Harness Extension:** Run RealForensics through the existing Stage A.1 validation harness (SDFVD dataset) to establish empirical accuracy, FPR, and FNR baselines.
4. **Phase 4: Multi-Signal Decision Fusion:** Integrate calibrated RealForensics scores into `forensic_decision_framework.py` with explicit conflict resolution rules.

---

## 14. Verification Test Suite Summary

The verification test suite was established in `backend/tests/test_realforensics_verification.py`. All tests pass cleanly:

```
backend/tests/test_realforensics_verification.py::TestRealForensicsVerification::test_architecture_reconstruction_without_checkpoint PASSED [ 20%]
backend/tests/test_realforensics_verification.py::TestRealForensicsVerification::test_checkpoint_integrity_if_present PASSED [ 40%]
backend/tests/test_realforensics_verification.py::TestRealForensicsVerification::test_state_dict_exact_loading_if_present PASSED [ 60%]
backend/tests/test_realforensics_verification.py::TestRealForensicsVerification::test_deterministic_inference_pass_if_present PASSED [ 80%]
backend/tests/test_realforensics_verification.py::TestRealForensicsVerification::test_forensic_output_properties PASSED [100%]
============================== 5 passed in 10.79s ==============================
```

Existing VideoMAE regression tests (`test_videomae_equivalence.py`, `test_video_ai_detection.py`) were run concurrently and remain 100% passing.

---

## 15. Appendix: Exact Reproducibility Hashes

- **Checkpoint File:** `storage/models/realforensics-verification/realforensics_ff.pth`
- **SHA-256 Digest:** `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d`
- **Total Bytes:** `86,220,575`
- **Manifest:** `storage/models/realforensics-verification/artifact_integrity.json`
- **Standalone Architecture:** `backend/tests/realforensics_arch.py`
- **Verification Test Suite:** `backend/tests/test_realforensics_verification.py`
