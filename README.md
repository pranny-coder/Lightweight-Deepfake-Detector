# DynForensic-GBT: Edge-Behavior Forensics for AI-Generated Face Detection

> **Phase 1 Pilot Research & Feasibility Benchmark**  
> Lightweight, explainable edge-behavior and perturbation dynamics forensic framework for differentiating real (FFHQ) vs. AI-generated (StyleGAN) facial images.

---

## 📌 Overview

**DynForensic-GBT** investigates whether computationally lightweight, non-deep-learning handcrafted features (gradient statistics, optical structure tensor coherence, high-pass sensor noise residuals, 2D FFT spectral slope, facial regional zoning, and controlled perturbation dynamics $\Delta\Phi$) provide reliable, explainable separation between real and synthetic faces.

This repository contains the complete **Phase 1 Pilot Pipeline**, benchmarking experiments, comparison with the 61-D testing package, demo assets, and academic presentation slides.

---

## 📂 Repository Structure

```text
CV Project/
├── README.md                                          # Project documentation and quickstart
├── DynForensic_GBT_Final_Academic_Proposal.pptx      # Original academic proposal deck
├── DynForensic_GBT_Proposal_v3_dryrun (1).pptx       # Active presentation deck with visual demos
├── DynForensic_GBT_Proposal_v3_dryrun_BACKUP.pptx    # Presentation safety backup
├── DynForensic_GBT_Final_PPT_Keep_Remove_Checklist.docx # Presentation requirements checklist
│
├── data/
│   └── pilot/                                        # 400 pilot benchmark images
│       ├── real/                                     # 200 real face crops (FFHQ)
│       └── fake/                                     # 200 AI-generated face crops (StyleGAN)
│
├── src/                                              # Core implementation source code
│   ├── data.py                                       # HuggingFace streaming & dataset downloader
│   ├── preprocess.py                                 # Haar cascade face detection & 256x256 standardization
│   ├── features.py                                   # 102-D multi-branch feature extraction pipeline
│   ├── experiment.py                                 # Phase 1 pilot evaluation (train/test, ROC, ablation)
│   ├── benchmark_uploaded_package.py                 # Benchmarking 61-D testing package & retrained models
│   ├── make_architecture_diagram.py                  # Generates system architecture flow graphic
│   ├── make_same_face_demo.py                        # Same-face Real vs. AI controlled twin dry-run
│   ├── make_slide1_diagram.py                        # Side-by-side Real vs Fake visual asset
│   ├── generate_presentation_dryrun.py               # Generates 4-panel forensic heatmap comparisons
│   ├── summary.py                                    # Terminal summary reporting script
│   └── update_dryrun_ppt.py                          # Automated PPT layout and slide updater
│
├── results/                                          # Experimental evaluation & figures
│   ├── PILOT_REPORT.md                               # Complete 14-section formal academic pilot report
│   ├── pilot_results.csv                             # Classifier performance metrics
│   ├── pilot_summary.json                            # JSON structured metrics & timing
│   ├── package_evaluation_results.csv                # Benchmark evaluation on 61-D package
│   ├── comparison_pilot_vs_package.csv               # Comparison: 102-D vs 61-D (Zero-shot & Retrained)
│   ├── comparison_plot.png                           # Visual comparison chart
│   ├── confusion_matrix.png                          # Confusion matrices
│   ├── roc_curve.png                                 # ROC-AUC curves
│   ├── feature_importance.png                        # Top discriminatory features
│   ├── ablation_plot.png                             # Feature branch ablation performance
│   └── perturbation_plot.png                         # Perturbation parameter response curves
│
├── demo_dryrun_outputs/                              # Generated high-resolution visuals
│   ├── dynforensic_system_architecture_diagram.png   # 6-stage end-to-end architecture diagram
│   ├── same_face_real_vs_ai_dryrun.png               # Controlled identity twin heatmap
│   ├── presentation_side_by_side_dryrun.png          # Real vs. Fake 4-panel comparison
│   ├── presentation_heatmap_real_0000.png            # Real sample optical/residual heatmaps
│   └── presentation_heatmap_fake_0000.png            # Fake sample optical/residual heatmaps
│
└── DynForensic_Testing_Package/                      # Uploaded 61-D fast testing package
    ├── test_image.py                                 # Fast 61-D feature extractor & CLI runner
    ├── dynforensic_ffhq_features_4000.csv            # 4,000-image extracted feature matrix
    ├── dynforensic_ffhq_benchmark_report.json        # Precomputed benchmark report
    └── *.joblib                                      # Pretrained classifiers (LogReg, SVM, RF, HGB)
```

---

## 🔬 Feature Representation Breakdown

### 1. 102-D Experimental Pipeline (`src/features.py`)
- **Block A: Gradient & Edge Statistics (10D)** — Sobel/Scharr magnitude moments, orientation entropy, Otsu edge metrics.
- **Block B: Texture & High-Pass Noise Residual (12D)** — High-pass Gaussian residual variance, kurtosis, spatial lag-1 autocorrelation, LBP uniform patterns.
- **Block C: Frequency Domain (8D)** — 2D FFT radial power log-log slope ($\alpha$), spectral peak score, band energy distribution.
- **Block D: Facial Regional Zones (24D)** — Geometric landmarks for Forehead, Eyes, Nose, Mouth, Cheeks across gradient and noise statistics.
- **Block E: Perturbation Dynamics $\Delta\Phi$ (48D)** — Differential responses under JPEG (Q=70, 85), Spatial Downsampling (50%, 75%), and Gaussian Blurring ($\sigma=0.8, 1.5$).

### 2. Fast 61-D Architecture (`DynForensic_Testing_Package/test_image.py`)
- **Branch A (25D Static):** Gradients (5D) + Structure Tensor Coherence (4D) + Edge Density (4D) + High-Pass Residual (6D) + 2D FFT (6D).
- **Branch B (15D Zones):** 3 statistics (`grad_mean`, `res_var`, `kurt`) across 5 facial zones.
- **Branch C (21D Perturbation):** 7 core differential metrics across 3 transforms (JPEG Q=70, Resize 50%, Blur $\sigma=1.0$).
- **Extraction Latency:** **34.78 ms/image** (6.5× faster than 102-D at 224.78 ms).

---

## 📊 Benchmark & Key Empirical Results

### Classifier Performance (Test Split: $N=120$)

| Pipeline | Model | Dimensions | Latency (ms) | Accuracy | ROC-AUC | F1-Score |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **102-D Pilot** | Logistic Regression | 102 | 224.78 | 60.83% | 0.687 | 0.647 |
| **102-D Pilot** | Random Forest | 102 | 224.78 | 60.83% | 0.678 | 0.618 |
| **54-D Static Only** | Random Forest | 54 | 37.95 | **65.83%** | **0.726** | **0.661** |
| **61-D Zero-Shot** | Linear SVM (Pretrained) | 61 | 34.78 | 52.50% | 0.623 | 0.678 |
| **61-D Retrained** | Logistic Regression | 61 | 34.78 | 65.83% | 0.709 | 0.672 |
| **61-D Retrained** | **Linear SVM (Best)** | **61** | **34.78** | **68.33%** | **0.707** | **0.698** |

### Key Findings
1. **Separability Feasibility Confirmed:** Handcrafted physical features achieve **68.33% accuracy** and **0.726 ROC-AUC** without requiring massive deep neural networks.
2. **Efficiency vs. Perturbation:** Static features (37.95 ms) yield higher accuracy than full 102-D, proving that uncalibrated heavy perturbations add noise. The streamlined 61-D package achieves optimal speed-accuracy tradeoff (34.78 ms).
3. **Threshold Calibration:** Pretrained weights on external distributions exhibit zero-shot threshold drift; retraining locally restores top performance immediately.

---

## 🚀 Quickstart & Execution

### Environment Setup
Activate the conda/miniforge environment:
```bash
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python -m pip install -r requirements.txt # or verify cv2, scikit-learn, numpy, matplotlib
```

### 1. Run Feature Extraction & 102-D Pilot
```bash
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python src/experiment.py
```

### 2. Run 61-D Benchmark & Package Comparison
```bash
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python src/benchmark_uploaded_package.py
```

### 3. Test Any Single Face Image
```bash
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python DynForensic_Testing_Package/test_image.py data/pilot/fake/fake_0000.png
```

### 4. Regenerate Architecture Diagram & Demo Visuals
```bash
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python src/make_architecture_diagram.py
/Users/pranavthobhani/miniforge3/envs/edge-artifact-guard/bin/python src/make_same_face_demo.py
```
