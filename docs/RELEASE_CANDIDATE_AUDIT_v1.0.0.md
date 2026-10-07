# MediaTrace v1.0.0 — Release-Candidate Audit Report

- **Target Release**: `v1.0.0`
- **Application Version**: `1.0.0`
- **Audit Timestamp**: `2026-09-30T19:10:00+05:30`
- **Audit Type**: Full System Production Hardening & Release-Candidate Audit (Stage B.1)
- **Host Platform**: Windows 11 (`win32`, Python `3.13.15`, Node `v20.18.0`)

---

## 1. Executive Result

```text
================================================================================
STAGE B.1 AUDIT STATUS: PASS WITH WARNINGS
================================================================================
```

The MediaTrace v1.0.0 codebase has successfully passed all functional, security, privacy, multimodal end-to-end, concurrency, reproducibility, report generation, and automated regression quality gates.

No release blockers exist. Non-blocking warnings are documented below regarding deferred cross-dataset video validation (Stage A.3) and host-level Docker daemon availability.

---

## 2. Comprehensive Quality Gate & Audit Results

### 2.1 Backend Regression Suite
- **Command**: `pytest backend/tests -q`
- **Result**: **122 passed, 0 failed, 6 warnings** in 110.98s
- **Test Modules Verified**:
  - `test_ai_detector_interpretation_hardening.py` (6 tests)
  - `test_ai_detector_transformation_robustness.py` (2 tests)
  - `test_ai_detector_validation.py` (6 tests)
  - `test_ai_image_detection.py` (8 tests)
  - `test_analysis.py` (4 tests)
  - `test_audio_analysis.py` (3 tests)
  - `test_comparison.py` (4 tests)
  - `test_detection.py` (5 tests)
  - `test_document_analysis.py` (2 tests)
  - `test_end_to_end.py` (2 tests)
  - `test_evidence.py` (3 tests)
  - `test_forensic_decision_framework.py` (8 tests)
  - `test_forensic_reproducibility_audit.py` (3 tests)
  - `test_health.py` (2 tests)
  - `test_history.py` (4 tests)
  - `test_localization.py` (5 tests)
  - `test_media.py` (6 tests)
  - `test_realforensics_validation.py` (8 tests)
  - `test_realforensics_verification.py` (5 tests)
  - `test_release_hardening_phase_a.py` (2 tests)
  - `test_reports.py` (3 tests)
  - `test_stage_a3_candidates_documentation.py` (3 tests)
  - `test_video_ai_detection.py` (8 tests)
  - `test_video_ai_detector_expanded_validation.py` (6 tests)
  - `test_video_analysis.py` (3 tests)
  - `test_videomae_equivalence.py` (4 tests)
  - `test_release_candidate_audit.py` (7 tests)

### 2.2 Frontend Production Build
- **Command**: `npm run build` (inside `frontend/`)
- **Result**: **PASSED** (`vite v5.4.21 building for production... ✓ built in 1.17s`)
- **Generated Bundle**:
  - `dist/index.html`: `0.42 kB` (gzip: `0.28 kB`)
  - `dist/assets/index-CSpvbFNw.css`: `10.07 kB` (gzip: `2.49 kB`)
  - `dist/assets/index-BTOwEZIR.js`: `238.97 kB` (gzip: `63.19 kB`)
- **Errors / Lints**: Zero build errors or unhandled bundle warnings.

### 2.3 Clean Startup Verification
- **API Health**: `GET /api/health` -> HTTP 200 `{"status": "ok", "service": "MediaTrace API", "version": "1.0.0"}`
- **API Root**: `GET /api` -> HTTP 200 `{"message": "...", "service": "MediaTrace API", "version": "1.0.0", "supported_modalities": ["image", "video", "audio", "document"]}`
- **Storage Subsystems**: `storage/uploads`, `storage/results`, `storage/reports`, and `storage/models` directories auto-provision cleanly via lifespan startup handler.

### 2.4 Multimodal End-to-End Regression
All four supported modalities were exercised from upload through report generation:
1. **Image**: Upload -> EXIF/container inspection -> JPEG analysis -> ELA/noise -> Swin AI detector -> spatial localization -> reference comparison -> evidence correlation -> multi-signal interpretation -> edit-history reconstruction -> JSON & PDF reports. Upload bytes remained 100% immutable (SHA-256 verified).
2. **Video**: Upload -> container/stream extraction -> frame sampling -> ELA/noise analysis -> VideoMAE detection -> RealForensics verification -> evidence correlation -> JSON & PDF reports. Upload bytes remained immutable.
3. **Audio**: Upload -> container inspection -> RMS/clipping/silence segmentation -> STFT spectral analysis -> discontinuity flagging -> waveform/spectrogram artifacts -> JSON & PDF reports. Upload bytes remained immutable.
4. **Document**: Upload -> MIME/signature check -> PDF structure & incremental update analysis -> DOCX/ODT safe ZIP parsing -> text extraction -> page rendering -> JSON & PDF reports. Upload bytes remained immutable.

### 2.5 Security Audit
- **Path Traversal**: POSIX (`../../etc/passwd.jpg`) and Windows (`..\..\windows\system.jpg`) path traversals are stripped to safe leaf basenames. Absolute paths are discarded.
- **Null Bytes**: Both raw null bytes (`\x00`) and URL-encoded null bytes (`%00`) are rejected with HTTP 400 Bad Request.
- **Resource Limits**:
  - Max upload size limits enforced (20 MB for images, 100 MB for videos, 50 MB for audio and documents).
  - Image decompression bomb ceiling enforced via Pillow (`Image.MAX_IMAGE_PIXELS = 64_000_000`).
  - Document decompression bomb ceiling enforced via ZIP threshold checks (`MAX_DECOMPRESSED_DOCUMENT_SIZE_MB`).
  - Bounded video frame sampling (`MAX_SAMPLE_FRAMES = 12`).
  - Bounded audio processing duration (`MAX_AUDIO_ANALYSIS_SECONDS = 300.0`).
- **Document Security**:
  - XML external entity (XXE) and entity expansion attacks blocked in DOCX/ODT parser.
  - PDF JavaScript, launch actions, and macros are flagged and not executed.
- **API Sanitization**:
  - Unhandled 500 exceptions return sanitized JSON `{"detail": "An internal error occurred while processing the forensic request."}` without exposing Python tracebacks, secrets, or internal filesystem paths.

### 2.6 Privacy Verification
- **GPS Coordinates**: Raw EXIF GPS coordinates (latitude, longitude, altitude) are strictly redacted in all layers:
  - `metadata.gps_present: true/false`
  - `metadata.gps_info: "GPS metadata present"` (or `null`)
  - `media.gps_metadata_present: true/false`
  - `media.gps_notice: "GPS metadata present"` (or `null`)
  - PDF report table renders "GPS Metadata Presence: GPS metadata present" or "Not present".
  - Logs, case JSON, API responses, and reports contain zero numeric coordinates.

### 2.7 Forensic Reproducibility Audit
- Every case maintains a deterministic `ForensicAnalysisSnapshot` (`MTS-YYYYMMDD-XXXXXXXX`).
- Snapshots include canonical `snapshot_sha256`, input byte verification, analysis execution records (`ANL-*`), artifact registry (`MTA-*`), model provenance, and append-only system audit events (`AUD-*`).
- Re-analyzing the same preserved input bytes under identical configuration produces bit-identical forensic observations.

### 2.8 AI Detector Boundary Audit
- **Image AI Detector (`umm-maybe/AI-image-detector`)**:
  - Reports `score_type = "uncalibrated_model_score"`.
  - Default `threshold = null`, `decision_rule = "model_class_selection"`.
  - Belongs to `ai_detector` independence group; cannot escalate overall forensic assessment.
- **VideoMAE (`Vansh180/VideoMae-ffc23-deepfake-detector`)**:
  - Output is strictly `strength = "informational"`.
  - No autonomous manipulation verdict.
  - No fake confidence percentages.
- **RealForensics (`realforensics_ff.pth`)**:
  - Kept as an independent, uncalibrated logit signal.
  - No score fusion or score averaging with VideoMAE.
  - Landmark dependency and domain boundaries explicitly preserved.

### 2.9 Forensic Language Audit
- Audited codebase and frontend for misleading claims of certainty.
- Verified that findings use conservative forensic terms: "observed signal", "anomaly", "possible processing", "supported inference", "evidence-ordered hypothesis", "uncalibrated model output", "insufficient evidence".
- Verified that "proof of manipulation", "proof of authenticity", "guaranteed original", and calibrated probability percentages are strictly avoided.

### 2.10 Report Generation Audit
- JSON and PDF reports verified across all 4 modalities.
- `application_version: "1.0.0"` reported in top-level JSON and rendered in PDF header.
- Original SHA-256 hash verified and displayed.
- PDFs verified using `pypdf.PdfReader` for page integrity and text content.

### 2.11 Concurrency & Basic Stability Test
- 6 concurrent multimodal workflows executed simultaneously via `ThreadPoolExecutor(max_workers=4)`.
- Verified 6 unique case IDs, 6 unique report IDs, zero filename collisions, zero cross-case artifact leakage, and zero data corruption.

### 2.12 Docker & Deployment Audit
- `backend/Dockerfile` verified (Python 3.11-slim, system libraries, non-root storage paths).
- `frontend/Dockerfile` verified (Node 20-alpine multi-stage build, Nginx 1.27-alpine runtime).
- `docker-compose.yml` verified and corrected (`VITE_API_BASE_URL` aligned to `/api`).
- *Note*: Docker CLI is not installed on the host Windows machine, so live container startup could not be executed locally. Container configurations were audited statically.

### 2.13 Frozen Baseline Verification
- Stage A.2.3 expanded validation artifacts verified intact and unchanged:
  - `datasets/ai_video_detector_validation/expanded/validation_manifest.json` (26 samples, SHA verified)
  - `datasets/ai_video_detector_validation/expanded/validation_results.json`
  - `datasets/ai_video_detector_validation/expanded/validation_summary.json`
  - `datasets/ai_video_detector_validation/expanded/validation_results.csv`
  - `datasets/ai_video_detector_validation/expanded/validation_summary.md`
  - `docs/VIDEO_AI_DETECTOR_VALIDATION_A2.3.md`
- Baseline A.1 / A.2.2 validation artifacts verified intact and unchanged.

---

## 3. Fixed Issues

1. **API Health Endpoint Version Reporting**:
   - `app/schemas/system.py` and `app/services/system_service.py` updated to include `version: str = MEDIATRACE_APPLICATION_VERSION` in `HealthResponse`, ensuring `GET /api/health` directly returns `"version": "1.0.0"`.
2. **Report Application Version Field**:
   - `app/schemas/report.py` and `app/services/reporting/report_service.py` updated to include top-level `application_version: str = "1.0.0"` in `ForensicReport`, matching audit specification.
3. **Docker Orchestration API URL Mismatch**:
   - Fixed `VITE_API_BASE_URL` in `docker-compose.yml` and `frontend/Dockerfile` from `http://localhost:8000/api/v1` to `http://localhost:8000/api`, preventing frontend 404 routing failures in containerized deployments.
4. **Regression Audit Test Suite Added**:
   - Implemented `backend/tests/test_release_candidate_audit.py` with 7 comprehensive regression tests covering health, path traversal, null bytes, GPS redaction, report versioning, AI detector boundaries, concurrency isolation, and sanitized error responses.

---

## 4. Remaining Warnings (Non-Blocking)

1. **Deferred Stage A.3 Cross-Dataset Validation**:
   - Access to the official DeepFake Detection Challenge (DFDC) dataset requires interactive Kaggle web browser competition rules acceptance. Because the headless environment cannot complete this, Stage A.3 remains deferred. This does not block the v1.0.0 core release candidate.
2. **Host Docker CLI Not Available**:
   - The local Windows host environment does not have Docker Desktop / CLI installed. Container configurations were verified via static analysis and syntax validation, but live multi-container deployment testing was not performed on this machine.
3. **Upstream Library Deprecation Warnings**:
   - 6 non-breaking deprecation warnings (`StarletteDeprecationWarning` regarding `HTTP_422_UNPROCESSABLE_ENTITY` alias and `httpx` in TestClient; Pillow `mode` deprecation in tests) emitted by third-party packages.

---

## 5. Mandatory Forensic Limitations

1. **VideoMAE & RealForensics Are Experimental Signals**: Both detectors are auxiliary screening models. Neither model is production-authoritative.
2. **Lack of Demonstrated Cross-Dataset Generalization**: Controlled validation on 20 SDFVD samples (plus 6 derivatives) demonstrated substantial sensitivity to domain shifts and out-of-domain compression. Generalization across diverse real-world generators is unproven.
3. **Out-of-Domain Baseline Cohort**: SDFVD consists of Pexels web videos swapped using Remaker AI, which differs significantly from the models' FaceForensics++ C23 training domain.
4. **Image AI Detector Domain Shift**: `umm-maybe/AI-image-detector` was trained on 2022-era artistic generators and is sensitive to resizing, cropping, and JPEG recompression.
5. **No Autonomous Provenance Verdicts**: No AI detector in MediaTrace establishes authenticity, manipulation, provenance, or intent by itself. AI detector evidence is strictly informational (`strength: "informational"`).
6. **Classical Forensic Restraint**: Metadata inconsistencies, ELA variations, JPEG quantization grids, and audio discontinuities can arise from benign sharing, transcoding, or storage and do not constitute definitive proof of tampering.

---

## 6. Release Blockers

```text
NONE. There are zero release blockers.
```

---

## 7. Model Provenance & Checkpoint Hashes

| Model | Component | Checkpoint File | File Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :--- | :--- | :--- |
| **Swin AI Image Detector** | PyTorch Weights | `storage/models/umm-maybe--AI-image-detector/pytorch_model.bin` | 347,596,479 | `f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf` |
| **Swin AI Image Detector** | ONNX Export | `storage/models/umm-maybe--AI-image-detector/model.onnx` | 358,286,809 | `693911a555891bac487c9e24daf4f2adcef258eaa7304d8d344ecc96b9af7b2e` |
| **VideoMAE Deepfake Detector** | SafeTensors Weights | `storage/models/Vansh180--VideoMae-ffc23-deepfake-detector/model.safetensors` | 347,381,256 | `293c668a5d3289d3162902d8cac6687cd11551cf2358768bad126de32b6d7f29` |
| **RealForensics Detector** | PyTorch Checkpoint | `storage/models/realforensics-verification/realforensics_ff.pth` | 86,220,575 | `843b86f29fc98c67132a6f98180a35bfc617882a32642b8b80d571944c2e032d` |
| **2D-FAN Landmark Model** | Alignment Weights | `storage/models/realforensics-verification/2DFAN4-11f355bf06.pth.tar` | 96,530,683 | `3e30f1d53381fa20f7f3408a6552945d8b284b3917a22ff32824dfcaee5bf72a` |

---

## 8. Final Recommendation

MediaTrace meets all technical, forensic, security, privacy, and reproducibility criteria for **v1.0.0 Release Candidate 1 (`v1.0.0-rc1`)**.

The status is designated **PASS WITH WARNINGS** solely due to the external dependency on Kaggle web authentication for deferred Stage A.3 cross-dataset validation and local absence of the Docker daemon. The application is robust, verified, and hardened for deployment.
