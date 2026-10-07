# FaceForensics++ C23 External Dataset Staging Directory

## Overview & Purpose
This staging directory is reserved for holding officially acquired **FaceForensics++ (FF++) C23** benchmark video media for controlled engineering validation of MediaTrace's facial deepfake detector (`Vansh180/VideoMae-ffc23-deepfake-detector`).

## Mandatory Terms of Use and Access Requirements
- **Official Access Required:** FaceForensics++ access must be requested and authorized through the official project agreement (Technical University of Munich / Google / FaceForensics team). Once approved, researchers receive official download access.
- **No Unofficial Mirrors:** Unofficial mirrors, torrents, or third-party scraping are strictly prohibited.
- **No Redistribution:** MediaTrace does not distribute, redistribute, or publicly host FaceForensics++ media. All benchmark material staged here must be obtained directly by authorized developers/researchers.
- **Respect License & Agreement:** All usage must respect the terms stipulated in the FaceForensics++ Terms of Use and academic licensing agreements.

## Preservation & Integrity Policy
- **Byte Preservation:** Original downloaded source files must remain completely unmodified and preserved in their original state. Do not re-encode, recompress, resize, or alter containers/streams.
- **Pre-Ingestion SHA-256 Verification:** Complete cryptographic SHA-256 digests must be computed and verified before any sample is copied to `datasets/ai_video_detector_validation/`.
- **Preserve Directory Hierarchy:** Preserve the official dataset directory structure within this staging directory (e.g., `original_sequences/youtube/c23/videos/` and `manipulated_sequences/Deepfakes/c23/videos/`).
- **No Fabricated Data:** Placeholder or synthetically generated video files must never be placed in this staging area.

## Intended Controlled Validation Subset (Small Subset Only)
To ensure lightweight, controlled engineering validation without downloading the full multi-terabyte dataset:
- **Human / Original:** Exactly 3 original C23 videos (`original_sequences/youtube/c23/videos/`)
- **Manipulated:** Exactly 3 manipulated C23 videos from a single documented manipulation method (e.g., `manipulated_sequences/Deepfakes/c23/videos/`)
- Total source videos: **6 files**

## Training Domain Limitation
- **Validation Domain:** `FaceForensics++ C23`
- **Notice:** This controlled validation uses material from the same broad dataset/domain used to train the selected VideoMAE detector and therefore does not establish independent real-world generalization. It must **never** be cited or reported as external benchmark validation, real-world accuracy, or independent performance evaluation.
