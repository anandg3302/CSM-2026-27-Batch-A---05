# MediaTrace

**AI-Based Detection and Reconstruction of Edits in Digital Media**  
*Multi-Modal Digital Media Forensics, Evidence Correlation, Possible Edit-History Reconstruction, and Forensic Reporting Platform*

---

## 1. Project Overview

**MediaTrace** is a conservative, chain-of-custody-preserving digital media forensic investigation system supporting **Image**, **Video**, **Audio**, and **Document** evidence. It is engineered around strict forensic principles:

1. **Evidence Immutability**: Uploaded files in `storage/uploads/` are verified with SHA-256 cryptographic hashes before and after every analysis stage and are never modified in place.
2. **Conservative Signal Interpretation**: Technical metadata, compression/re-encoding signals, spatial/temporal anomalies, and structural revisions are reported as **forensic indicators for analyst review**, never overstated as definitive proof of manipulation.
3. **Explicit ML / AI Detector Boundaries**: No fake or random AI-generation confidence scores are ever produced when a trained modality detector is not loaded (`status: "not_available"`).
4. **Privacy & Safety**: Raw GPS coordinates are redacted from API responses (`gps_present: true` notice only), filenames are sanitized against path traversal, and XML/ZIP containers are protected against decompression bombs (`MAX_DECOMPRESSED_DOCUMENT_SIZE_MB`).

---

## 2. System Architecture

```text
MediaTrace/
├── backend/                  # FastAPI + Python 3.11+ forensic backend
│   ├── app/
│   │   ├── api/routes/       # REST API endpoints (/api/*)
│   │   ├── core/             # Typed configuration, version registry & audit sanitization
│   │   ├── schemas/          # Strict Pydantic v2 forensic & reproducibility schemas
│   │   └── services/
│   │       ├── analysis/     # Image, Video, Audio, Document, Manipulation & Localization
│   │       ├── comparison/   # Suspect vs Reference image comparison engine
│   │       ├── evidence/     # Evidence Correlation, Multi-Signal Decision & Reproducibility
│   │       ├── history/      # Conservative Possible Edit-History Reconstruction
│   │       └── reporting/    # Deterministic JSON & multi-page PDF report generator
│   ├── tests/                # Pytest suite (unit, integration, security, reproducibility, and E2E)
│   └── Dockerfile            # Production backend container definition
├── frontend/                 # React 18 + Vite 5 interactive forensic workbench
│   ├── src/
│   │   ├── components/       # Upload, Case Viewer, Comparison, Evidence, History, Reproducibility & Report panels
│   │   └── services/         # Typed API client services
│   └── Dockerfile            # Multi-stage Nginx production frontend container
├── ml/                       # Pluggable ML model adapter registry
├── storage/                  # Isolated chain-of-custody storage
│   ├── uploads/              # Immutable original uploaded files + reference files
│   ├── results/              # Atomic JSON case state records (MT-*.json) & immutable artifact archives
│   ├── reports/              # Generated JSON & PDF forensic reports
│   └── models/               # Local pinned AI detector weights & ONNX artifacts (git-ignored)
├── docker-compose.yml        # Full-stack multi-container orchestration
└── .env.example              # Reference configuration template
```

---

## 3. Supported Media Types & Forensic Capabilities

| Modality | Supported Extensions | Key Forensic Capabilities |
| :--- | :--- | :--- |
| **Image** | `.jpg`, `.jpeg`, `.png`, `.webp` | SHA-256 integrity, container/EXIF inspection, ELA (Error Level Analysis), noise consistency, blockiness, chroma subsampling, pinned Swin AI-image detector (`umm-maybe/AI-image-detector`), spatial anomaly localization heatmaps, suspect-vs-reference image comparison, evidence correlation, multi-signal forensic interpretation, possible edit-history reconstruction, reproducibility snapshot (`MTS-*`), JSON & PDF forensic reports |
| **Video** | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v` | Container/stream inspection, codec/FPS/resolution/bitrate extraction, bounded representative frame sampling, frame-level ELA & noise metrics, temporal delta consistency, scene transition & recompression checks, evidence/reproducibility & reporting |
| **Audio** | `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`, `.opus`, `.wma` | Container/codec/bitrate/sample-rate inspection, ID3/Vorbis/MP4 tag & encoder extraction, RMS/peak/crest-factor/clipping/DC-offset waveform metrics, silence/gap segmentation, STFT spectral analysis, frame-to-frame discontinuity & splice candidate flagging, waveform & spectrogram PNG artifacts, evidence/reproducibility & reporting |
| **Document** | `.pdf`, `.docx`, `.odt`, `.txt`, `.rtf`, `.doc` | Container/MIME signature checks, PDF incremental update (`%%EOF` / `/Prev`) inspection, DOCX/ODT safe ZIP inspection (with decompression-bomb protection), track-changes (`w:ins` / `w:del`) & annotation detection, embedded image/script/macro detection, timestamp consistency analysis, rendered PDF page previews, evidence/reproducibility & reporting |

---

## 4. Installation & Local Development

### Prerequisites
- **Python**: `3.11+`
- **Node.js**: `18+` (or `20+`)

### 1. Environment Configuration
Copy `.env.example` to `.env` in the repository root (optional; safe defaults are built in):
```bash
cp .env.example .env
```

### 2. Backend Setup & Run
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- API Root & Health Check: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 3. Frontend Setup & Run
```powershell
cd frontend
npm install
npm run dev
```
- Web Workbench UI: `http://localhost:5173`

---

## 5. Testing & Production Build Verification

### Run Full Backend Test Suite
```powershell
cd backend
.\.venv\Scripts\pytest -q
```

### Run Frontend Production Build
```powershell
cd frontend
npm run build
```

---

## 6. Docker Deployment

Launch both the FastAPI backend (`:8000`) and Nginx production frontend (`:5173`) with persistent storage mounted:
```bash
docker compose up --build -d
```

---

## 7. Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `MediaTrace API` | Application title |
| `APP_ENV` | `development` | Runtime environment (`development` / `production`) |
| `API_PREFIX` | `/api` | Base route prefix for all API endpoints |
| `STORAGE_ROOT` | `storage` | Root storage directory (resolved relative to repository root) |
| `UPLOADS_DIR` | `storage/uploads` | Immutable uploaded media & reference files directory |
| `RESULTS_DIR` | `storage/results` | Case JSON state records & derived visual artifacts directory |
| `REPORTS_DIR` | `storage/reports` | Generated JSON & PDF forensic reports directory |
| `MODELS_DIR` | `storage/models` | Local directory for pinned AI detector weights & ONNX runtime artifacts (git-ignored) |
| `AI_IMAGE_DETECTOR_ENABLED` | `true` | Enable Step 14.1 pretrained AI-generated image detector |
| `AI_IMAGE_DETECTOR_MODEL_ID` | `umm-maybe/AI-image-detector` | Pretrained detector model identifier |
| `AI_IMAGE_DETECTOR_MODEL_VERSION` | `c7e223baf11bc40528af364ba7bdea030ef42f9e` | Pinned immutable Hugging Face revision commit SHA (`main` is rejected) |
| `AI_IMAGE_DETECTOR_DEVICE` | `cpu` | Inference device (`cpu` default, optional `cuda`) |
| `AI_IMAGE_DETECTOR_THRESHOLD` | `null` (`None`) | Unset by default (`decision_rule="model_class_selection"`) |
| `MAX_SAMPLE_FRAMES` | `12` | Maximum representative frames sampled per video |
| `MAX_AUDIO_ANALYSIS_SECONDS` | `300` | Maximum audio duration decoded for waveform/STFT analysis |
| `MAX_DOCUMENT_ANALYSIS_MB` | `50` | Maximum document file size in MB |
| `MAX_DOCUMENT_PREVIEW_PAGES` | `10` | Maximum rendered PDF page previews |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed CORS origins |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000/api` | Frontend API base URL |

---

## 8. Step 14.1 & 14.1.1 — Real AI-Generated Image Detection (Hardened)

### Purpose
Step 14.1 / 14.1.1 adds an independent, pretrained neural vision detector layer for **AI-generated image detection** alongside MediaTrace's existing classical image forensics (EXIF, JPEG quantization tables, ELA, noise consistency, and spatial localization). Classical forensic findings and AI detector outputs are reported in separate sections and are never merged into a fabricated numerical probability.

> **Important Forensic Principle**:  
> *"AI-generation detection is a model-derived forensic signal (`uncalibrated_model_score`) and is not, by itself, proof of provenance or authenticity."*

### Pretrained Detector Specification (Official Hugging Face Source of Truth)
- **Model Identifier**: `umm-maybe/AI-image-detector`
- **Official Source URL**: [https://huggingface.co/umm-maybe/AI-image-detector/tree/c7e223baf11bc40528af364ba7bdea030ef42f9e](https://huggingface.co/umm-maybe/AI-image-detector/tree/c7e223baf11bc40528af364ba7bdea030ef42f9e)
- **Pinned Revision / Commit SHA**: `c7e223baf11bc40528af364ba7bdea030ef42f9e` (mutable branch names such as `main` are rejected for forensic reproducibility)
- **License**: `Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)` ([https://creativecommons.org/licenses/by-nd/4.0/](https://creativecommons.org/licenses/by-nd/4.0/))
- **Architecture**: `SwinForImageClassification` (`model_type: "swin"`)
- **Upstream Format vs. Runtime Conversion**:
  - **Upstream Repository Files**: `config.json`, `preprocessor_config.json`, `pytorch_model.bin` (the upstream Hugging Face repository does **not** publish an upstream `model.onnx`).
  - **MediaTrace Runtime Pipeline**: `upstream PyTorch weights (pytorch_model.bin, SwinForImageClassification) -> MediaTrace local ONNX export (opset 14) -> ONNX Runtime inference` (`onnx_is_upstream: false`), or local `safetensors` via `transformers` (`local_files_only=True`, `trust_remote_code=False`).
- **Exact In-Memory Preprocessing (`preprocessor_config.json`)**:
  - **Color Mode**: `RGB`
  - **Resize**: `224x224` (`do_resize: true`, `size: 224`, direct resize without center crop)
  - **Resampling Filter**: `Image.Resampling.BICUBIC` (`resample: 3` in `ViTFeatureExtractor`)
  - **Value Scaling**: `[0, 255] uint8 -> [0.0, 1.0] float32`
  - **Normalization Mean**: `[0.485, 0.456, 0.406]`
  - **Normalization Std**: `[0.229, 0.224, 0.225]`
  - **Tensor Layout**: `NCHW (1, 3, 224, 224) float32`
  - **Class Ordering (`id2label`)**: `{"0": "artificial", "1": "human"}`
  - Preprocessing runs exclusively on an in-memory copy; the original uploaded image file on disk is never resized, re-encoded, or modified (`SHA-256` before == `SHA-256` after).

### Model Provisioning, Local ONNX Export & Runtime Verification (`storage/models/`)
Large model weights are never stored in Git (`storage/models/*` is ignored in `.gitignore`), and MediaTrace never silently downloads weights during normal API requests.

1. **Download Pinned `pytorch_model.bin` & Export Local `model.onnx` (Idempotent)**:
   ```powershell
   cd backend
   $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/setup_ai_image_detector.py --verify
   ```
   - Downloads `pytorch_model.bin` (`347,596,479` bytes, SHA-256 `f4c14ed23eb1d65b1c6ca7b163f8f91bebcf3a4d8c275296c850c9f1bbd01daf`) from pinned revision `c7e223baf11bc40528af364ba7bdea030ef42f9e` into `storage/models/umm-maybe--AI-image-detector/`.
   - Deterministically exports `model.onnx` (`358,286,809` bytes, SHA-256 `693911a555891bac487c9e24daf4f2adcef258eaa7304d8d344ecc96b9af7b2e`, `onnx_is_upstream: false`), inspects input/output tensor metadata (`[batch_size, 3, 224, 224]` `float32` $\rightarrow$ `[batch_size, 2]` `float32`), and writes `artifact_integrity.json` and `config.json`.
   - Idempotent: skips re-downloading or re-exporting when SHA-256 hashes match the recorded artifacts.

2. **Run Standalone Real ONNX Runtime Verification**:
   ```powershell
   cd backend
   $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/verify_ai_image_detector.py
   ```

3. **Run AI Detector & Full Backend Test Suite**:
   ```powershell
   cd backend
   .\.venv\Scripts\pytest tests/test_ai_image_detection.py -v
   .\.venv\Scripts\pytest -v
   ```
If the model weights are not present in `storage/models/`, if `AI_IMAGE_DETECTOR_ENABLED=false`, if an unpinned revision (such as `main`) is configured, or if SHA-256 / ONNX tensor shape checks fail, MediaTrace gracefully returns `status: "not_available"` without inventing a score.

### Decision Rule, Score Calibration & Known Limitations
- **Decision Rule (`model_class_selection`)**: By default (`AI_IMAGE_DETECTOR_THRESHOLD` unset / `threshold: null`), MediaTrace uses `decision_rule: "model_class_selection"` (selecting the predicted model class `artificial` vs `human` via `argmax` over model scores) rather than claiming `0.5` is a MediaTrace-validated forensic threshold.
- **Uncalibrated Model Score**: `score_type` is always `"uncalibrated_model_score"`. Model scores are raw classifier softmax activations and are **not** calibrated forensic probabilities.
- **Model Limitations**:
  - `umm-maybe/AI-image-detector` was published in October 2022 as a proof-of-concept classifier trained and evaluated on artistic AI-generated imagery from older generators (such as VQGAN+CLIP and early diffusion/GAN pipelines).
  - It is not a universal AI-image detector across modern photorealistic diffusion/autoregressive generators and is not a dedicated deepfake or local photo-manipulation detector.
  - Classifier activations are sensitive to domain shift, JPEG recompression, resizing, cropping, screenshots, and post-processing.
  - A negative output (`no_ai_signal_detected`) does not prove that an image is human-created or authentic.
  - AI detector output is one supporting forensic signal among MediaTrace's classical metadata, compression, localization, and reference-comparison evidence sources.

### Step 14.1.3 — Controlled Multi-Image Descriptive Validation Harness
- **Engineering Characterization (Not Scientific Validation)**: `backend/scripts/validate_ai_image_detector.py` provides a controlled, reproducible validation harness over `datasets/ai_detector_validation/manifest.json`.
- **Key Rules & Limitations**:
  - This is a controlled descriptive validation harness, **not** a scientific or forensic model validation benchmark.
  - Known categories (`human`, `ai_generated`, `transformed`) come exclusively from `manifest.json` and are never inferred from model predictions, EXIF headers, filenames, or visual heuristics.
  - Model outputs are uncalibrated model-derived scores (`uncalibrated_model_score`, `decision_rule="model_class_selection"`, `threshold=null`) and are never reported as accuracy, precision, recall, F1, AUC, or calibrated forensic probabilities.
  - The convenience sample is not a representative statistical benchmark, and results must not be generalized to arbitrary internet images.
  - Transformed images (`transformed/`) are analyzed and reported separately and excluded from the human-vs-AI confusion count table.
  - No authenticity, provenance, or combined numerical probability conclusion is generated.
- **Run Controlled Validation CLI**:
  ```powershell
  cd backend
  $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/validate_ai_image_detector.py
  $env:PYTHONPATH="."; .\.venv\Scripts\python scripts/validate_ai_image_detector.py --manifest ../datasets/ai_detector_validation/manifest.json --skip-forensics
  ```
  Writes `validation_results.json`, `validation_summary.json`, and `validation_results.csv` to `storage/results/ai_detector_validation/`.

---

## 8B. Stage A — Pretrained Video AI / Deepfake Detection (`Vansh180/VideoMae-ffc23-deepfake-detector`)

### Pretrained Video Detector Specification
- **Model Repository**: `Vansh180/VideoMae-ffc23-deepfake-detector` (`VideoMAE-DeepFake-Detector-v1`)
- **Pinned Immutable Revision (Commit SHA)**: `ea26518b527d938d66e568c52bcf1c19a86cf295`
- **Architecture**: `VideoMAEForVideoClassification` (`model_type: "videomae"`)
- **License**: `MIT`
- **Documented Training Domain**: `FaceForensics++ C23` (`detector_scope = "facial_video_deepfake_detection"`)
- **Documented Classes (`id2label`)**: `{"0": "real", "1": "fake"}`
- **Preprocessing (`VideoMAEImageProcessor`)**:
  - Uniformly sample `16` RGB frames across the clip
  - Resize shortest edge to `224` (`BILINEAR`, `resample=2`) and center-crop `224 x 224`
  - Rescale `[0, 255] -> [0.0, 1.0]` and normalize with `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`
- **Score Semantics**: `score_type = "uncalibrated_model_score"`, `decision_rule = "model_class_selection"`, `threshold = null` (`None`).
- **Facial-Video Scope Enforcement**: Because the model is trained on FaceForensics++ C23 facial video clips, MediaTrace verifies whether sampled frames contain suitable facial content before running inference. If suitable facial content is not present (`usable_detector_frames = 0`), the detector returns `status = "not_available"` with an explicit reason rather than forcing non-face video through the facial classifier.
- **Conservative Evidence & Interpretation**: Video AI detector outputs always carry `strength = "informational"` (`type = "ai_detection"`, `source = "video_ai_detector"`), belong to independence group `ai_detector`, and never independently escalate `video_analysis.assessment` or `forensic_decision.overall_assessment`.

### Provisioning & Controlled Video Validation CLI
```powershell
cd backend
$env:PYTHONPATH="."; .\.venv\Scripts\python scripts/setup_ai_video_detector.py --verify
$env:PYTHONPATH="."; .\.venv\Scripts\python scripts/validate_ai_video_detector.py
.\.venv\Scripts\pytest tests/test_video_ai_detection.py -v
```

---

## 8C. Stage A.2 — Experimental Video AI Detector 2: RealForensics (`realforensics_ff.pth`)

### Pretrained RealForensics Specification
- **Model Checkpoint**: `realforensics_ff.pth` (FaceForensics++ pretrained weights, SHA-256: `843b86f29fc98c67...`)
- **Architecture**: `CSN-R101` 3D ResNet backbone + temporal self-supervision head
- **Landmark Preprocessing**: 25-frame facial sequence aligned via 68-point 2D-FAN landmarks (`2DFAN4-11f355bf06.pth.tar`) warped to 20-word mean face template (`20words_mean_face.npy`)
- **Score Semantics**: Raw uncalibrated logit; no score calibration; no threshold tuning; no score fusion with VideoMAE.
- **Stage A.2.3 Dual-Detector Validation**: Validated alongside VideoMAE across 26 SDFVD samples (10 human + 10 manipulated + 6 derivatives). Both models demonstrated domain sensitivity to web video compression.
- **Independence & Scope**: Kept as an independent auxiliary signal. Cross-dataset generalization is **not** established.

---

## 9. Step 14.2 & 14.3 — Multi-Signal Forensic Decision & Case Reproducibility

### Step 14.2 — Multi-Signal Forensic Decision Framework
- **Endpoints**:
  - `POST /api/forensics/{case_id}/interpret`
  - `GET /api/forensics/{case_id}/interpret`
- Groups correlated indicators into explicit **Independence Groups** so multiple outputs from the same underlying evidence source are never double-counted, classifies pairwise signal consistency/disagreement, and separates **Supported Conclusions** from **Unsupported Conclusions** and **Alternative Explanations** without fabricating numerical probabilities.

### Step 14.3 — Forensic Case Reproducibility & Audit Trail
- **Endpoints**:
  - `POST /api/forensics/{case_id}/reproducibility/check`
  - `GET /api/forensics/{case_id}/reproducibility`
- Produces an auditable `ForensicAnalysisSnapshot` (`MTS-YYYYMMDD-XXXXXXXX`) with canonical `snapshot_sha256`, input byte integrity verification, analysis execution registry (`ANL-YYYYMMDD-XXXXXXXX`), versioned immutable artifact registry (`MTA-YYYYMMDD-XXXXXXXX`), AI model provenance (`umm-maybe/AI-image-detector`, `c7e223baf11bc40528af364ba7bdea030ef42f9e`), evidence dependency graph, and system audit events (`AUD-YYYYMMDD-XXXXXXXX`) sanitized against local path, secret, and raw GPS leakage.

---

## 10. Mandatory Forensic Limitations & Release Boundaries

> **Forensic Notice & Operational Boundaries**:
> 1. **Application Version**: MediaTrace is version `1.0.0` (Release Candidate state: `v1.0.0-rc1`).
> 2. **No Forensic Certainty**: MediaTrace extracts technical, statistical, structural, and metadata indicators. It makes **zero claims of forensic certainty**.
> 3. **Non-Proof Semantics**: Neither AI model outputs, metadata anomalies, compression/quantization grids, spatial localization heatmaps, reference comparisons, multi-signal consistency, nor reconstructed histories constitute definitive proof of authenticity, manipulation, provenance, exact chronology, or malicious intent.
> 4. **Experimental Video AI Detectors**: VideoMAE and RealForensics are experimental, auxiliary forensic signals. Neither model is production-authoritative.
> 5. **Uncalibrated Model Scores**: All detector scores (`umm-maybe`, VideoMAE, RealForensics) are uncalibrated classifier outputs. MediaTrace does not generate fake confidence percentages or calibrated manipulation probabilities.
> 6. **No Generalization Established**: Controlled validation on 26 SDFVD samples does not establish detector generalization across arbitrary real-world generators or unseen compression conditions.
> 7. **Deferred Cross-Dataset Validation**: Stage A.3 independent cross-dataset validation (DFDC) is currently deferred because access requires web-browser competition-rule acceptance that cannot be automated via API token.
> 8. **Classical Forensic Restraint**: Metadata inconsistencies, ELA variations, and audio discontinuities can arise from benign re-encoding or distribution and are not standalone proof of manipulation.
> 9. **Possible Edit-History Sequences**: Reconstructed edit histories represent evidence-ordered hypothetical sequences (`sequence_type: "evidence_ordered_hypothesis"`), not confirmed historical timelines.
> 10. **Deployment Verification**: Docker deployment configurations are statically verified; live runtime container testing was not performed on audit hosts lacking Docker CLI.


