# MediaTrace — Stage A.3: Independent Cross-Dataset Video AI Detector Validation — Candidate Discovery Report

**MediaTrace Platform Engineering — Stage A.3 Discovery**  
**Document ID:** `MT-DOC-DATASET-DISCOVERY-A3`  
**Classification:** Research, Architecture & Candidate Discovery Only (Zero Downloads, Zero Code Modifications)  
**Status:** `DISCOVERY COMPLETE — AWAITING DOWNLOAD APPROVAL`  
**Date:** September 30, 2026  

---

## 1. Executive Summary & Objective

The objective of Stage A.3 is to identify and thoroughly document suitable candidate datasets for the next stage of controlled video deepfake validation.

In Stage A.1, A.2.2, and A.2.3, the platform evaluated:
1. **VideoMAE** (`Vansh180/VideoMae-ffc23-deepfake-detector` @ commit `ea26518b527d938d66e568c52bcf1c19a86cf295`)
2. **RealForensics** (`ahaliassos/RealForensics` @ commit `a53e849407d2fd5be0f0a7eb3df7d1a846b5b803`)

against a controlled cohort of 26 samples from the **SDFVD (Small-scale Deepfake Forgery Video Dataset)**.

The empirical findings from Stage A.2.3 demonstrated **signal diversity** (35% model disagreement, 35% mutual ground-truth agreement), but also highlighted domain-specific behaviors:
* VideoMAE predicted the `fake` class for all 10 authentic human stock videos in SDFVD.
* RealForensics predicted the `human` class for 4 of 10 human samples and 2 of 10 manipulated samples.

To evaluate whether these behaviors are artifacts of the SDFVD domain (Pexels stock footage + Remaker AI face swaps) or reflect broader detector characteristics, an **independent cross-dataset validation** is required.

> [!IMPORTANT]
> **Hard Scope Boundary:**  
> This document represents a **DISCOVERY-ONLY** step. No dataset archives, video files, or checkpoints have been downloaded. No production code, models, or decision frameworks have been modified. Execution is paused pending explicit user approval.

---

## 2. Frozen Stage A.2.3 Baseline Verification

The existing validation artifacts remain strictly frozen as the established baseline:

* Baseline manifest: `datasets/ai_video_detector_validation/expanded/validation_manifest.json` (v2.0.0, 26 samples, verified SHA-256 digests)
* Baseline results: `datasets/ai_video_detector_validation/expanded/validation_results.json` (26 evaluated records)
* Baseline summary: `datasets/ai_video_detector_validation/expanded/validation_summary.json` & `.md`
* Baseline documentation: `docs/VIDEO_AI_DETECTOR_VALIDATION_A2.3.md`
* Stage A.1 results: `datasets/ai_video_detector_validation/validation_results.json` (7 evaluated records)
* Stage A.2.2 results: `datasets/ai_video_detector_validation/realforensics/validation_results.json` (7 evaluated records)

None of the above files will be overwritten or altered during Stage A.3.

---

## 3. Independence Criteria Relative to SDFVD

A valid cross-dataset candidate for Stage A.3 must satisfy strict orthogonality criteria relative to the frozen baseline:

| Dimension | SDFVD Baseline | Stage A.3 Independence Requirement |
| :--- | :--- | :--- |
| **Source Video Population** | Pexels stock video collection (commercial stock actors, stylized lighting) | Natural conversational videos, journalistic interviews, or multi-angle laboratory actor recordings |
| **Real Video Capture** | Stock camera setups, staged b-roll, high color grading | Varied camera capture devices, natural conversational movement, real-world compression |
| **Manipulation Method** | Remaker AI (commercial closed-source web face-swap API) | Known open-source deepfake autoencoders (DeepFaceLab, Faceswap), 3D graphics (Face2Face, NeuralTextures), or generative diffusion/VAE frameworks |
| **Generation Pipeline** | Single pipeline (Remaker AI single-pass face replacement) | Diverse generation architectures with documented parameterization and blending methods |
| **Compression Profile** | Re-encoded H.264 MP4 with uniform bitrates | Documented multiple compression tiers (raw, high-bitrate c23, low-bitrate c40, or realistic web transmission loss) |
| **Provenance Documentation** | Published Mendeley Data repository (DOI: `10.17632/bcmkfgct2s.1`) | Peer-reviewed academic conference publication with verified metadata and ground-truth mapping |

---

## 4. Candidate Dataset Comparison

The following neutral comparison summarizes the 5 evaluated candidate datasets:

| Candidate | Ground Truth Availability | Manipulation Diversity | Independent of SDFVD | Access Mechanism | License | Selective Download Feasibility | Main Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Celeb-DF (v2)** | Yes (deterministic list of test videos & directory splits) | High (enhanced DeepFake synthesis with reduced boundary artifacts) | **Yes** (celebrity YouTube interviews vs Pexels; custom DeepFake pipeline vs Remaker AI) | Academic request form (Google Form / email to authors) | Non-commercial research ("as is", proprietary) | Moderate (author provides package links; requires selective extraction) | Requires manual form submission and waiting for author email approval |
| **FaceForensics++ (FF++)** | Yes (exact pairing files and manipulation type folders) | Very High (4 methods: Deepfakes, Face2Face, FaceSwap, NeuralTextures) | **Yes** (YouTube source vs Pexels; 4 distinct algorithms vs Remaker AI) | Academic request form (TUM Google Form to obtain token) | Non-commercial academic research | High (`download_ff.py` script supports `--dataset` and `--compression` flags) | **In-Domain Training Dataset:** Both VideoMAE and RealForensics were trained on FF++ C23; evaluates training fidelity rather than cross-dataset generalization |
| **DFDC (DeepFake Detection Challenge)** | Yes (documented in `metadata.json` for train/val chunks; test set is unlabelled) | Very High (multiple commercial & research face-swap architectures, puppetry, voice swap) | **Yes** (studio actor recordings vs Pexels; multiple synthesis models vs Remaker AI) | Kaggle API / Meta AI (requires Kaggle account and competition rules acceptance) | Custom Non-Commercial Research License | Moderate (Kaggle CLI can download individual training chunks, e.g., part 0) | Full dataset is massive (470+ GB, 100k+ videos); competition rules agreement required |
| **DeeperForensics-1.0** | Yes (explicit mapping lists and binary ground truth) | High (DF-VAE framework with 7 real-world perturbation types) | **Yes** (100 paid actors in multi-angle studio vs Pexels; DF-VAE vs Remaker AI) | Academic request form (Google Form / email to SenseTime authors) | Non-commercial research only | Low to Moderate (large multi-part archives; requires custom extraction) | Very large overall volume (~300 GB); requires formal application form |
| **WildDeepfake** | Yes (binary folder splits `real` / `fake`) | High (uncontrolled real-world internet deepfakes) | **Yes** (in-the-wild web collection vs Pexels; heterogeneous tools vs Remaker AI) | Direct public download (hosted on Hugging Face and GitHub) | Listed as Apache-2.0 on Hugging Face (subject to web source terms) | High (Hugging Face datasets library or direct file selection) | Many samples are pre-cropped face sequences rather than intact full-frame videos; exact synthesis provenance is unrecorded |

---

## 5. In-Depth Candidate Evaluations

### Candidate 1: Celeb-DF (v2)

* **Official Dataset Name:** Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics (Version 2)
* **Official Source & Repository:** `https://github.com/yuezunli/celeb-deepfakeforensics`
* **Paper / Project Citation:** Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, Siwei Lyu. *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
* **Dataset Purpose:** Developed to overcome the low visual quality and obvious boundary artifacts present in earlier deepfake datasets (such as UADFV and early FaceForensics), providing high visual quality face swaps with reduced color mismatches.
* **Real-Video Category:** `Celeb-real` (590 video sequences from YouTube of celebrity interviews) and `YouTube-real` (300 additional high-quality YouTube clips).
* **Manipulated / Fake Category:** `Celeb-synthesis` (5,639 synthesized DeepFake videos generated by swapping celebrity faces).
* **Manipulation Techniques:** DeepFake face-swapping using an improved encoder-decoder architecture with modified affine warp blending and resolution enhancement.
* **Approximate Scale:** 6,529 videos total; standard test split includes 518 videos (`List_of_testing_videos.txt`). Download size is approximately 10.8 GB compressed.
* **Ground-Truth Availability:** Fully documented in `List_of_testing_videos.txt` with unambiguous binary labels and paired source identification.
* **Licensing:** Released for non-commercial research purposes "as is". Users agree to terms upon requesting access. Redistribution is strictly restricted.
* **Access Requirements:** Requires filling out an official Google Form / Tencent Form via the GitHub repository or emailing `deepfakeforensics@gmail.com`. Upon review, the authors provide temporary download links.
* **Authentication Requirements:** Form submission requires an institutional/academic affiliation or research statement.
* **Selective Download Possibility:** The dataset is distributed as split archives or Google Drive/Baidu links. Selective download of specific files is only possible if individual file links or partial archives are provided.
* **Detector Compatibility:**
  * **VideoMAE:** Excellent. Videos are standard MP4 containers with high-resolution, centered facial footage, easily sampled into 16 frames.
  * **RealForensics:** Excellent. Continuous interview sequences provide smooth temporal motion across 25 contiguous frames, well-suited for 2D-FAN landmark tracking and affine alignment.
* **Domain Differences from SDFVD:** Celebrities in dynamic interview settings (varying posture, lighting, speech motion) vs staged Pexels stock footage; custom DeepFake autoencoder vs Remaker AI web service.
* **Candidate Classification:** **`Potentially Suitable`** (High scientific and forensic value; constrained only by manual form-approval gating).

---

### Candidate 2: FaceForensics++ (FF++)

* **Official Dataset Name:** FaceForensics++: Learning to Detect Manipulated Facial Images
* **Official Source & Repository:** `https://github.com/ondyari/FaceForensics`
* **Paper / Project Citation:** Andreas Rössler, Davide Cozzolino, Luisa Verdoliva, Christian Riess, Justus Thies, Matthias Nießner. *FaceForensics++: Learning to Detect Manipulated Facial Images*, IEEE International Conference on Computer Vision (ICCV), 2019.
* **Dataset Purpose:** Comprehensive benchmark for evaluating automated face forgery detection against multiple manipulation techniques and compression levels.
* **Real-Video Category:** `original_sequences/youtube` (1,000 real videos from YouTube, H.264 compressed).
* **Manipulated / Fake Category:** 4 distinct manipulation sub-datasets:
  * `manipulated_sequences/Deepfakes` (1,000 videos, autoencoder-based face replacement)
  * `manipulated_sequences/Face2Face` (1,000 videos, facial reenactment via RGB tracking)
  * `manipulated_sequences/FaceSwap` (1,000 videos, graphics-based facial replacement)
  * `manipulated_sequences/NeuralTextures` (1,000 videos, neural rendering with patch-based loss)
* **Manipulation Techniques:** Deepfakes, Face2Face, FaceSwap, NeuralTextures.
* **Approximate Scale:** 5,000 videos per compression level (`raw`, `c23` high quality, `c40` low quality). Total ~15,000 videos.
* **Ground-Truth Availability:** Complete, authoritative frame-by-frame and video-level ground truth with explicit source-target pairing.
* **Licensing:** Academic research use only. Commercial use and public redistribution are strictly prohibited.
* **Access Requirements:** Requires filling out an official Google Form to request the download script (`download_ff.py`) and API access.
* **Authentication Requirements:** Download script requires an approved access token parameter.
* **Selective Download Possibility:** The official `download_ff.py` script natively supports selecting specific subsets (e.g., `--dataset Deepfakes --compression c23 --type videos --num_videos 10`), making selective download technically straightforward once approved.
* **Detector Compatibility:**
  * **VideoMAE:** Native compatibility (VideoMAE was trained on FF++ C23).
  * **RealForensics:** Native compatibility (RealForensics was trained on FF++ C23).
* **Domain Differences from SDFVD:** 4 diverse manipulation techniques and YouTube journalistic footage vs single Remaker AI pipeline on Pexels stock video.
* **Candidate Classification:** **`Potentially Suitable (with In-Domain Limitation)`**  
  * *Methodological Caveat:* Because both VideoMAE and RealForensics were pretrained on FaceForensics++ C23, evaluating against FF++ measures **in-domain training fidelity and calibration**, rather than true out-of-domain cross-dataset generalization. It remains valuable for validating whether the models perform as published when in-domain, but cannot prove cross-dataset robustness.

---

### Candidate 3: DFDC (DeepFake Detection Challenge) / Kaggle Sample

* **Official Dataset Name:** DeepFake Detection Challenge (DFDC)
* **Official Source & Repository:** `https://ai.meta.com/datasets/dfdc/`, Kaggle: `https://www.kaggle.com/c/deepfake-detection-challenge`
* **Paper / Project Citation:** Brian Dolhansky et al. *The DeepFake Detection Challenge (DFDC) Dataset*, arXiv:2006.07397 (CVPR 2020 workshop).
* **Dataset Purpose:** Created by Meta AI, AWS, Microsoft, and the Partnership on AI to provide an unprecedentedly large and diverse dataset of deepfakes with diverse demographics, environments, and synthesis techniques.
* **Real-Video Category:** Videos filmed with 3,426 diverse paid consenting actors in varied indoor and outdoor lighting environments.
* **Manipulated / Fake Category:** Manipulated videos generated using multiple state-of-the-art face-swap, facial reenactment, and voice modification pipelines.
* **Manipulation Techniques:** Deepfakes (autoencoders), FSGAN, style-transfer facial reenactment, and audio deepfakes.
* **Approximate Scale:** 124,000+ total videos across 50 chunked archives (`dfdc_train_part_0` to `part_49`), totaling over 470 GB.
* **Ground-Truth Availability:** Each training chunk includes a `metadata.json` mapping each MP4 filename to its `label` (`REAL` or `FAKE`) and `original` source video filename. (Note: The competition's separate `test_videos` folder is unlabelled, but all training chunks are fully labelled).
* **Licensing:** Custom Non-Commercial Research License. Redistribution prohibited.
* **Access Requirements:** Requires a registered Kaggle account and explicit user agreement to the competition rules on the Kaggle website.
* **Authentication Requirements:** Requires Kaggle API token (`kaggle.json`) or web login.
* **Selective Download Possibility:** Through the Kaggle API, individual chunk zip files can be targeted (e.g., `dfdc_train_part_0.zip` is ~10 GB and contains ~2,000 labelled videos). Selective extraction of specific MP4s from the zip stream via HTTP range requests is technically feasible.
* **Detector Compatibility:**
  * **VideoMAE:** Good. Requires facial gating because some videos feature side profiles or body movement.
  * **RealForensics:** Moderate to Good. Requires filtering for sequences where 2D-FAN can maintain tracking across 25 consecutive frames.
* **Domain Differences from SDFVD:** Studio/field actors with diverse ethnicity and lighting vs polished stock footage; multiple synthesis methods vs single Remaker AI tool.
* **Candidate Classification:** **`Suitable (Recommended for Cross-Dataset Generalization)`**  
  * Independent provenance, diverse demographics, multiple synthesis techniques, and explicit `REAL`/`FAKE` ground-truth metadata in training chunks.

---

### Candidate 4: DeeperForensics-1.0

* **Official Dataset Name:** DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection
* **Official Source & Repository:** `https://github.com/EndlessSora/DeeperForensics-1.0`
* **Paper / Project Citation:** Liming Jiang, Ren Li, Wayne Wu, Chen Qian, Chen Change Loy. *DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection*, IEEE CVPR 2020.
* **Dataset Purpose:** Designed to simulate real-world deepfake scenarios by combining high-quality deepfake autoencoding with 7 types of real-world perturbations (JPEG compression, Gaussian noise, blur, color saturation, etc.).
* **Real-Video Category:** 1,000 source videos collected from 100 paid actors in controlled multi-camera studio conditions.
* **Manipulated / Fake Category:** 60,000 manipulated videos generated using the DeepFake Variational Auto-Encoder (DF-VAE).
* **Manipulation Techniques:** DF-VAE end-to-end framework with adversarial training.
* **Approximate Scale:** 60,000 videos, ~300 GB.
* **Ground-Truth Availability:** Full ground-truth pairing list files provided in the repository.
* **Licensing:** Non-commercial research only.
* **Access Requirements:** Requires submitting an academic request form to the authors (`deeperforensics@gmail.com`).
* **Authentication Requirements:** Academic/institutional verification.
* **Selective Download Possibility:** Very low. The dataset is distributed across large multi-part archives hosted on Google Drive / Baidu, making single-file selective download difficult without downloading multi-gigabyte parts.
* **Detector Compatibility:** High. Controlled studio head poses facilitate facial tracking.
* **Domain Differences from SDFVD:** Studio-recorded actors under controlled lighting vs web stock footage; DF-VAE with perturbation modeling vs Remaker AI.
* **Candidate Classification:** **`Potentially Suitable (Constrained by Download Packaging)`**  
  * High scientific quality, but large multi-part archive packaging makes selective download of a small controlled cohort technically difficult.

---

### Candidate 5: WildDeepfake

* **Official Dataset Name:** WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection
* **Official Source & Repository:** `https://github.com/OpenTAI/wild-deepfake`, Hugging Face: `xingjunm/WildDeepfake`
* **Paper / Project Citation:** Bojia Zi, Minghao Fang, Chengyu Jin, et al. *WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection*, ACM Multimedia 2020.
* **Dataset Purpose:** Testing deepfake detectors against diverse "in-the-wild" deepfake videos harvested directly from the internet.
* **Real-Video Category:** Real internet video sequences.
* **Manipulated / Fake Category:** Internet deepfake videos from various platforms.
* **Manipulation Techniques:** Unknown / heterogeneous web deepfake creation software.
* **Approximate Scale:** 7,314 face sequences extracted from 707 videos.
* **Ground-Truth Availability:** Binary labels (`real` / `fake`) provided by directory organization.
* **Licensing:** Listed as Apache-2.0 on Hugging Face (subject to original content terms).
* **Access Requirements:** Publicly downloadable from Hugging Face / GitHub.
* **Authentication Requirements:** None.
* **Selective Download Possibility:** Very High (single files or small folders downloadable via Hugging Face API/Git LFS).
* **Detector Compatibility:** Moderate to Low. Many distributions of WildDeepfake provide tightly cropped face sequences or normalized face squares rather than standard intact video streams, which may interfere with VideoMAE's full-frame context and 2D-FAN's whole-frame affine transformation pipeline.
* **Domain Differences from SDFVD:** Internet meme / social media deepfakes vs curated stock video.
* **Candidate Classification:** **`Potentially Suitable (Subject to Preprocessing Verification)`**  
  * Easy public access, but cropped face format may violate the detectors' validated full-frame input assumptions.

---

## 6. Candidate Suitability Assessment Summary

Based on rigorous assessment across access feasibility, ground truth reliability, detector compatibility, and independence from SDFVD:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Candidate Assessment Summary                                                           │
├─────────────────────┬───────────────────────┬──────────────────────────────────────────┤
│ Candidate Dataset   │ Suitability Status    │ Factual Decision Rationale               │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ DFDC                │ Suitable              │ Completely independent of SDFVD; highly  │
│ (Kaggle chunk part) │                       │ diverse; ground-truth metadata available;│
│                     │                       │ selective chunk extraction possible.     │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ Celeb-DF (v2)       │ Potentially Suitable  │ Outstanding visual quality and pairing;  │
│                     │                       │ fully independent; requires waiting for  │
│                     │                       │ author email approval.                   │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ FaceForensics++     │ Potentially Suitable  │ Authoritative; well-structured; BUT is   │
│ (FF++ C23)          │ (In-Domain Only)      │ the training domain of both models; does │
│                     │                       │ not test cross-dataset generalization.   │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ DeeperForensics-1.0 │ Potentially Suitable  │ High quality; large multi-part archives  │
│                     │                       │ make selective small cohort extraction   │
│                     │                       │ technically burdensome.                  │
├─────────────────────┼───────────────────────┼──────────────────────────────────────────┤
│ WildDeepfake        │ Potentially Suitable  │ Publicly accessible; pre-cropped format  │
│                     │ (Format Caveat)       │ risks violating full-frame assumptions.  │
└─────────────────────┴───────────────────────┴──────────────────────────────────────────┘
```

---

## 7. Proposed Stage A.3 Controlled Validation Protocol

If approved to proceed to execution, Stage A.3 will strictly adhere to the following experimental protocol:

### 7.1 Target Validation Cohort
* **Primary Sample Target:**
  * **At least 10 human facial videos**
  * **At least 10 manipulated facial videos**
* **Secondary Transformed Derivatives ($n=6$):**
  * 3 human derivatives (spatial downscaling 50% and 75%)
  * 3 manipulated derivatives (spatial downscaling 50% and 75%)
  * Evaluated only **after** the primary cohort is completed.
* **Total Cohort Size:** 20 primary samples + 6 derivatives = **26 samples**.

### 7.2 Preprocessing Invariance
* **VideoMAE Pipeline:**
  * 16 uniformly sampled frames across video duration.
  * Aspect-preserving resize of shortest edge to 224, center crop to $224 \times 224$, ImageNet normalization.
  * Facial gating via YCbCr chrominance and Haar cascade.
* **RealForensics Pipeline:**
  * Contiguous 25-frame sequence centered at video midpoint.
  * 68-point 2D-FAN landmark detection per frame with $w=12$ temporal smoothing.
  * LMEDS 2D affine warp onto `20words_mean_face.npy`.
  * Lower-face patch cropping ($140 \to 112 \times 112$), ImageNet normalization.

### 7.3 Score Recording & Evaluation Rules
* **No Score Fusion:** Both detector outputs are recorded in their native format:
  * VideoMAE: $p_\text{fake} \in [0, 1]$
  * RealForensics: $z \in \mathbb{R}$
* **No Threshold Tuning:** Decision boundaries remain frozen ($p_\text{fake} > 0.5$, $z > 0$).
* **Comparative Taxonomy Preserved:**
  * `both_agree_with_known_category`
  * `both_disagree_with_known_category`
  * `models_disagree`
  * `evaluated_derivative`
  * `one_or_both_unavailable`

---

## 8. Reproducibility & Provenance Requirements for Future Ingestion

When a candidate is formally approved for download, the ingestion script must automatically record:

1. **Official Source Details:** Source dataset name, publication venue, DOI/URL, release version.
2. **License Metadata:** Precise license identifier, terms of use, and non-commercial restrictions.
3. **Archive Identification:** Source archive filename, total byte size, and full SHA-256 hash.
4. **Per-File Provenance:**
   * Original dataset relative path
   * File byte size and SHA-256 digest
   * Native video container, codec, resolution, frame rate, frame count, duration
   * Ground-truth category (`human` or `manipulated`) and ground-truth basis
   * Sampling and selection rationale
5. **Model Checkpoints:** Pinned commit SHA-256 for VideoMAE (`293c668a...`) and RealForensics (`843b86f2...`).
6. **Manifest & Provenance Files:**
   * `datasets/ai_video_detector_validation/stage_a3/validation_manifest.json`
   * `datasets/external_sources/<dataset_name>/provenance.json`

---

## 9. Conclusion & Decision Checkpoint

Dataset discovery is **complete**. The candidates have been documented with neutral, factual criteria.

```
CRITICAL RULE: STOP AFTER DATASET DISCOVERY.
DO NOT DOWNLOAD ANY VIDEO MEDIA.
WAIT FOR EXPLICIT USER APPROVAL OF ONE CANDIDATE BEFORE PROCEEDING TO DOWNLOAD.
```
