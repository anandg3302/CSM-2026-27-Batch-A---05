# Official FaceForensics++ Access & Staging Instructions

## Current Access Status
- **Status:** PENDING OFFICIAL ACCESS REQUEST
- **Official Script:** Not yet supplied in the repository.
- **Dataset Acquired:** 0 files (pending authorized download).

## Step-by-Step Developer Procedure

### 1. Request Official Access
1. Visit the official FaceForensics++ repository / project portal:
   - Official Repository: `https://github.com/ondyari/FaceForensics`
2. Submit the official Google Form application requested by the authors:
   - Provide institutional/organizational contact info and agree to the Terms of Use (non-commercial research only, no re-distribution).
3. Wait for approval email containing the official download instructions and credentials/script.

### 2. Obtain & Place the Official Download Script
1. Upon receiving official access, place the official download script in this directory or project root:
   - `datasets/external_sources/faceforensicspp_c23/download_faceforensics.py`
2. Inspect the script to verify official origin and available arguments.

### 3. Download the Small Validation Subset Only
Do **NOT** download the complete FaceForensics++ dataset (~hundreds of gigabytes to terabytes).
Download only the minimum subset required for controlled engineering validation:
- **3 Original Sequences (C23 compression):**
  - Dataset: `original_sequences`
  - Compression: `c23`
  - Type: `videos`
- **3 Manipulated Sequences (C23 compression):**
  - Dataset: `manipulated_sequences`
  - Method: `Deepfakes` (or `Face2Face` / `FaceSwap` / `NeuralTextures`)
  - Compression: `c23`
  - Type: `videos`

Example invocation using official script parameters (adjust according to the official script received):
```bash
python download_faceforensics.py datasets/external_sources/faceforensicspp_c23/ -d original_sequences -c c23 -t videos --num_videos 3
python download_faceforensics.py datasets/external_sources/faceforensicspp_c23/ -d Deepfakes -c c23 -t videos --num_videos 3
```

### 4. Verification and Ingestion into MediaTrace Validation Dataset
Once downloaded:
1. Verify source file integrity and calculate SHA-256 for all 6 videos.
2. Complete provenance records adhering to `provenance_template.json`.
3. Ingest files byte-for-byte into `datasets/ai_video_detector_validation/`:
   - 3 Original files -> `datasets/ai_video_detector_validation/human/`
   - 3 Manipulated files -> `datasets/ai_video_detector_validation/manipulated/`
4. Confirm `source_sha256 == validation_sha256`.
5. Update `datasets/ai_video_detector_validation/manifest.json`.
