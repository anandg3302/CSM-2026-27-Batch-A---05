# Kaggle DeepFake Detection Challenge (DFDC) Sample Staging Directory

## 1. Dataset Overview & Source
- **Dataset Name:** DeepFake Detection Challenge (DFDC) Sample / Validation Material
- **Official Source:** Kaggle Competitions
- **Competition Identifier:** `deepfake-detection-challenge` (Official URL: `https://www.kaggle.com/c/deepfake-detection-challenge`)
- **Staging Location:** `datasets/external_sources/dfdc_kaggle_sample/`
- **Acquisition Date:** 2026-09-30 (Staging & access preparation)

## 2. Purpose & Non-Benchmark Status
- **Intended Use:** Controlled engineering characterization and functional validation of the MediaTrace VideoMAE detector (`Vansh180/VideoMae-ffc23-deepfake-detector`).
- **Domain Distinction:** The production detector was trained on FaceForensics++ C23, whereas DFDC constitutes a different evaluation domain.
- **Strict Limitation:** This controlled validation is strictly an internal engineering verification dataset. It does **not** establish scientific benchmark performance, general real-world detector accuracy, calibrated detection probability, or universal deepfake detection efficacy.

## 3. Mandatory Licensing & Terms of Use
- **Kaggle Competition Rules:** Access and download of the DFDC dataset require accepting the official competition rules and maintaining an active Kaggle account with appropriate competition participation privileges.
- **No Redistribution:** MediaTrace does not distribute, host, or mirror DFDC video media. Authorized developers must acquire the dataset directly from Kaggle using official API credentials.
- **Preservation & Integrity:** Original downloaded archives and extracted video files must be preserved byte-for-byte in their original containers without alteration, recompression, or resizing. SHA-256 checksums must be recorded upon download and verified before ingestion.

## 4. Controlled Validation Target Subset
- **Small Sample Only:** Do not download the full multi-hundred-gigabyte competition dataset.
- **Required Ingestion Subset:**
  - 3 Documented REAL facial videos (`metadata.json` label == "REAL")
  - 3 Documented FAKE facial videos (`metadata.json` label == "FAKE", preserving original reference field if declared)
  - Total: 6 videos
- **Ground Truth Determination:** Ground truth must be derived strictly from official `metadata.json` annotations, never from filenames, visual appearance, or detector predictions.
