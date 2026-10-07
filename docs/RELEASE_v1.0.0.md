# MediaTrace v1.0.0 — Official Release Manifest & Audit Summary

- **Release Version**: `v1.0.0`
- **Application Version**: `MEDIATRACE_APPLICATION_VERSION = "1.0.0"`
- **Release Date**: `2026-09-30`
- **Release Status**: `RELEASE CANDIDATE (v1.0.0-rc1) — PASS WITH WARNINGS`

---

## 1. Supported Media Types & Forensic Pipeline Scope

| Modality | Supported Extensions | Implemented Forensic & Audit Capabilities |
| :--- | :--- | :--- |
| **Image** | `.jpg`, `.jpeg`, `.png`, `.webp` | SHA-256 byte integrity, container/EXIF inspection (with raw GPS coordinate redaction), JPEG quantization/subsampling analysis, ELA, noise consistency, blockiness, pinned Swin AI-image detector (`umm-maybe/AI-image-detector`), spatial anomaly localization (`heatmap.png`, `overlay.png`, `regions.png`), suspect-vs-reference comparison (`difference_map.png`, `comparison_overlay.png`), cross-module evidence correlation, Step 14.2 multi-signal forensic interpretation, Step 8 possible edit-history reconstruction, Step 14.3 reproducibility snapshot (`MTS-*`), JSON & PDF forensic reports |
| **Video** | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v` | Container/stream inspection, codec/FPS/resolution/bitrate extraction, bounded representative frame sampling (`MAX_SAMPLE_FRAMES=12`), frame-level ELA & noise metrics (`frame_*.jpg`, `frame_*_ela.jpg`), temporal delta consistency, scene transition & recompression checks, experimental VideoMAE deepfake detector (`Vansh180/VideoMae-ffc23-deepfake-detector`), experimental RealForensics detector (`realforensics_ff.pth`), evidence/reproducibility & reporting |
| **Audio** | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.wma` | Container/codec/bitrate/sample-rate inspection, ID3/Vorbis/MP4 tag & encoder extraction, RMS/peak/crest-factor/clipping/DC-offset waveform metrics, silence/gap segmentation, STFT spectral analysis, frame-to-frame discontinuity & splice candidate flagging (`waveform.png`, `spectrogram.png`, `discontinuity_*.png`), evidence/reproducibility & reporting |
| **Document** | `.pdf`, `.docx`, `.odt`, `.txt`, `.rtf`, `.doc` | Container/MIME signature checks, PDF incremental update (`%%EOF` / `/Prev`) inspection, DOCX/ODT safe ZIP inspection (decompression-bomb protected), track-changes (`w:ins` / `w:del`) & annotation detection, embedded image/script/macro detection, timestamp consistency analysis, rendered PDF page previews (`page_*.png`), evidence/reproducibility & reporting |

---

## 2. Pinned AI Models & Checkpoint Integrity

### A. Auxiliary AI Image Detector
- **Repository**: `umm-maybe/AI-image-detector`
- **Pinned Revision**: `c7e223baf11bc40528af364ba7bdea030ef42f9e`
- **Architecture**: `SwinForImageClassification`
- **License**: `CC BY-ND 4.0`
- **Class Mapping**: `{"0": "artificial", "1": "human"}`
- **Score Semantics**: `score_type = "uncalibrated_model_score"`, `decision_rule = "model_class_selection"`, `threshold = null`
- **Verified Local Checkpoint**: `storage/models/umm-maybe--AI-image-detector/pytorch_model.bin` (`347,596,479` bytes, SHA-256: `f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf`)

### B. Experimental Video Detector 1: VideoMAE
- **Repository**: `Vansh180/VideoMae-ffc23-deepfake-detector` (`VideoMAE-DeepFake-Detector-v1`)
- **Pinned Revision**: `ea26518b527d938d66e568c52bcf1c19a86cf295`
- **Architecture**: `VideoMAEForVideoClassification`
- **License**: `MIT`
- **Training Domain**: `FaceForensics++ C23`
- **Score Semantics**: `score_type = "uncalibrated_model_score"`, `decision_rule = "model_class_selection"`, `threshold = null`, `strength = "informational"`
- **Verified Local Checkpoint**: `storage/models/Vansh180--VideoMae-ffc23-deepfake-detector/model.safetensors` (`347,381,256` bytes, SHA-256: `293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29`)

### C. Experimental Video Detector 2: RealForensics
- **Repository / Checkpoint**: `realforensics_ff.pth` (Official pretrained FaceForensics weights)
- **License**: Research / Academic (FaceForensics++ & RealForensics license terms)
- **Architecture**: `ResNet-18 3D + Transformer Temporal Encoder`
- **Score Semantics**: Raw uncalibrated logit, no threshold tuning, no score fusion, landmark-dependent
- **Verified Local Checkpoint**: `storage/models/realforensics-verification/realforensics_ff.pth` (`86,220,575` bytes, SHA-256: `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d`)

---

## 3. Verification & Quality Gate Summary

- **Backend Automated Test Suite**: `122 passed, 0 failed, 6 warnings` (`pytest backend/tests`)
- **Frontend Production Build**: `PASSED` (`vite build`, `dist/` bundle generated cleanly in 1.17s)
- **Health Check (`GET /api/health`)**: `{"status": "ok", "service": "MediaTrace API", "version": "1.0.0"}` (`HTTP 200`)
- **System Version Check (`GET /api`)**: `{"service": "MediaTrace API", "version": "1.0.0", "supported_modalities": ["image", "video", "audio", "document"]}` (`HTTP 200`)
- **Multimodal E2E Pipelines**: Image, Video, Audio, Document pipelines pass with full SHA-256 byte immutability and report generation (JSON + PDF).
- **Security & Privacy Status**:
  - Path traversal, null-byte filenames, and malformed filenames sanitized or rejected.
  - Raw GPS coordinates strictly redacted from API responses and reports (`gps_metadata_present` boolean and `gps_notice` text only).
  - Decompression bombs (Pillow 64 MP ceiling, DOCX/ZIP size bounds) safely handled.
  - Sanitized 400/404/500 error responses without exposing internal stack traces or filesystem paths.

---

## 4. Mandatory Forensic Limitations & Non-Proof Boundaries

1. **Experimental Video AI Detectors**: VideoMAE and RealForensics are experimental forensic signals provided for auxiliary screening. They are not production-authoritative.
2. **Lack of Generalization**: Their controlled validation on the Small-scale Deepfake Forgery Video Dataset (SDFVD) does not establish out-of-domain generalization or cross-generator reliability.
3. **Cross-Dataset Validation Status**: Independent cross-dataset validation (Stage A.3) is currently **deferred** because the official DeepFake Detection Challenge (DFDC) dataset access requires web-browser competition rule acceptance not accessible in the current environment.
4. **Out-of-Domain Baseline Cohort**: The 20-sample SDFVD cohort consists of Pexels-sourced web clips swapped via Remaker AI, which represents an out-of-domain distribution relative to the models' FaceForensics++ C23 training data.
5. **Image AI Detector Domain Shift**: `umm-maybe/AI-image-detector` was trained on 2022-era artistic generators; it is sensitive to resizing, cropping, and JPEG recompression and does not generalize across modern photorealistic diffusion pipelines.
6. **No Autonomous Provenance Verdicts**: No current AI detector establishes authenticity, manipulation, provenance, or intent by itself. AI model signals always carry `strength: "informational"`, belong to independence group `ai_detector`, and cannot escalate overall forensic assessments.
7. **Classical Forensic Restraint**: Classical forensic signals (metadata tags, ELA, quantization tables, noise profiles, frame deltas, audio discontinuities) are also not standalone proof of manipulation, as benign editing, transcoding, or distribution workflows frequently produce similar artifacts.
