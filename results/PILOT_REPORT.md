# PILOT REPORT: Edge-Behavior Forensics for AI-Generated Face Detection

## DynForensic-GBT — 1-Hour Pilot Experiment

**Date**: 2026-09-26  
**Status**: Completed Successfully

---

## 1. Dataset

| Property | Value |
|---|---|
| **Source** | `TheKernel01/140k-Real-and-Fake-Faces` (Hugging Face Hub) |
| **Acquisition** | Streaming mode (`streaming=True`), first 400 interleaved samples |
| **Split Used** | `train` |
| **Real Images** | 200 (FFHQ / Flickr-Faces-HQ) |
| **Fake Images** | 200 (StyleGAN) |
| **Total Images** | 400 |
| **Labels** | ClassLabel: 0 = real, 1 = fake |
| **Generator Field** | ClassLabel: 0 = Real, 1 = StyleGAN |
| **Storage** | `data/pilot/real/` and `data/pilot/fake/` (PNG, RGB) |

---

## 2. Experimental Setup

This is a **1-hour feasibility pilot experiment** that tests whether a compact ~100-D forensic feature representation provides measurable separation between real human faces (FFHQ) and AI-generated faces (StyleGAN) from the pilot dataset.

**Scope limitations:**
- Single generator family (StyleGAN) vs. single real source (FFHQ).
- 400 images total (200 per class) — a small-scale pilot.
- No deep learning classifiers or pretrained networks were used.
- No cross-generator evaluation was performed.
- Results are specific to this dataset distribution.

---

## 3. Preprocessing

For every image:
1. Load RGB from disk.
2. Detect the main face using **OpenCV Haar Cascade** (`haarcascade_frontalface_default.xml`).
3. **Fallback detectors**: Profile face cascade → conservative center crop (85% of shorter dimension).
4. Crop with 10% margin around detected face bounding box.
5. Resize to 256×256 (bicubic/area interpolation).
6. Convert to grayscale for feature extraction.
7. Identical preprocessing for real and fake images.

### Face Detection Results

| Detector | Count | Percentage |
|---|---|---|
| Primary (frontal Haar) | 384 | 96.0% |
| Fallback (profile Haar) | 3 | 0.75% |
| Fallback (center crop) | 13 | 3.25% |
| **Total failures** | **0** | **0%** |

All 400 images were successfully preprocessed. 13 images required the center-crop fallback because neither the frontal nor profile Haar cascade detected a face. This is expected for images with unusual poses or heavy occlusions. No images were silently discarded.

---

## 4. Feature Definitions

### Block A — Edge / Gradient (10 features)
| Feature | Description |
|---|---|
| `edge_density` | Ratio of Canny edge pixels to total pixels |
| `mean_grad_mag` | Mean Sobel gradient magnitude |
| `std_grad_mag` | Standard deviation of Sobel gradient magnitude |
| `grad_kurtosis` | Kurtosis of gradient magnitude distribution |
| `scharr_grad_mean` | Mean Scharr gradient magnitude |
| `scharr_grad_std` | Std of Scharr gradient magnitude |
| `edge_strength_var` | Variance of gradient magnitude along edge pixels |
| `edge_continuity` | Fraction of edge pixels with ≥2 connected neighbors |
| `mean_grad_dir` | Mean absolute gradient orientation angle |
| `otsu_threshold` | Otsu's binarization threshold on grayscale face |

### Block B — Texture / Residual (12 features)
| Feature | Description |
|---|---|
| `local_tex_mean` | Mean of 5×5 local standard deviation map |
| `local_tex_var` | Variance of local standard deviation map |
| `local_tex_std` | Std of local standard deviation map |
| `residual_energy` | Mean squared Gaussian-subtraction residual |
| `residual_std` | Std of Gaussian-subtraction residual |
| `high_pass_residual_var` | Variance of high-pass residual |
| `residual_lag1_corr` | Spatial autocorrelation of residual at lag 1 |
| `laplacian_energy` | Mean squared Laplacian response |
| `laplacian_kurtosis` | Kurtosis of Laplacian response |
| `lbp_mean` | Mean of 8-neighbor LBP map |
| `lbp_std` | Std of LBP map |
| `lbp_entropy` | Shannon entropy of LBP histogram (64 bins) |

### Block C — Frequency (8 features)
| Feature | Description |
|---|---|
| `low_freq_energy` | Normalized power in low-frequency band (<15% radius) |
| `mid_freq_energy` | Normalized power in mid-frequency band (15–50% radius) |
| `high_freq_energy` | Normalized power in high-frequency band (>50% radius) |
| `high_freq_ratio` | High-frequency energy / total energy |
| `spectral_peak_score` | Max/mean ratio in high-frequency band |
| `spectral_slope` | Log-log radial power spectral slope |
| `spectral_centroid` | Radial center of mass of power spectrum |
| `spectral_energy_skew` | Skewness of radial energy profile |

### Block D — Facial Zones (24 features)
Canonical zones on aligned 256×256 crop: **forehead, left eye, right eye, nose, mouth, cheeks**. For each zone, 4 statistics are computed:
- `zone_{z}_edge_density` — Canny edge ratio within zone
- `zone_{z}_mean_grad` — Mean Sobel gradient magnitude within zone
- `zone_{z}_residual_energy` — Gaussian residual energy within zone
- `zone_{z}_lbp_entropy` — LBP entropy within zone

6 zones × 4 features = 24 features.

> **Note**: MediaPipe FaceMesh was unavailable in the runtime environment (Python 3.14 / MediaPipe 1.0.1 API incompatibility). Instead, geometrically-defined canonical face zones were used on the aligned 256×256 face crops. This is a valid approximation for FFHQ/StyleGAN images, which are roughly aligned by design.

### Block E — Perturbation Response (48 features)
Four controlled transformations:
1. **JPEG Q70** compression
2. **Resize 50%** (128→256 bilinear upscale)
3. **Gaussian blur** (σ=1.0, ksize=5)
4. **Center crop 80%** and resize back to 256×256

For each transformation T, delta features are computed:

**ΔΦ = Φ(I) − Φ(T(I))**

10 tracked static features × 4 perturbations = 40 individual deltas + 4 × 2 summary stats (mean, std) = **48 perturbation features**.

### Total Feature Dimension: **102**

| Block | Features |
|---|---|
| A: Edge/Gradient | 10 |
| B: Texture/Residual | 12 |
| C: Frequency | 8 |
| D: Facial Zones | 24 |
| E: Perturbation ΔΦ | 48 |
| **Total** | **102** |

---

## 5. Classifiers

| Classifier | Configuration |
|---|---|
| **Logistic Regression** | `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)` |
| **Random Forest** | `RandomForestClassifier(n_estimators=100, random_state=42)`, raw features |

- Data split: **70% train (280)** / **30% test (120)**, stratified, `random_state=42`.
- No deep learning classifiers, CNNs, ViTs, CLIP, or DINO were used.

---

## 6. Ablation Results

| Feature Set | # Features | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 1. Edge only | 10 | Logistic Regression | 0.633 | 0.633 | 0.633 | 0.633 | 0.673 |
| 1. Edge only | 10 | Random Forest | 0.592 | 0.585 | 0.633 | 0.608 | 0.613 |
| 2. Texture/residual only | 12 | Logistic Regression | 0.600 | 0.586 | 0.683 | 0.631 | 0.657 |
| 2. Texture/residual only | 12 | Random Forest | 0.617 | 0.613 | 0.633 | 0.623 | 0.659 |
| 3. Frequency only | 8 | Logistic Regression | 0.600 | 0.588 | 0.667 | 0.625 | 0.611 |
| 3. Frequency only | 8 | Random Forest | 0.558 | 0.564 | 0.517 | 0.539 | 0.598 |
| 4. Face zones only | 24 | Logistic Regression | 0.600 | 0.581 | 0.717 | 0.642 | 0.633 |
| 4. Face zones only | 24 | Random Forest | 0.633 | 0.638 | 0.617 | 0.627 | 0.685 |
| 5. Edge + Texture | 22 | Logistic Regression | 0.617 | 0.603 | 0.683 | 0.641 | 0.654 |
| 5. Edge + Texture | 22 | Random Forest | 0.575 | 0.582 | 0.533 | 0.557 | 0.614 |
| 6. Edge + Texture + Freq | 30 | Logistic Regression | **0.650** | 0.645 | 0.667 | 0.656 | **0.695** |
| 6. Edge + Texture + Freq | 30 | Random Forest | 0.575 | 0.585 | 0.517 | 0.549 | 0.636 |
| 7. Static (A+B+C+D) | 54 | Logistic Regression | 0.633 | 0.605 | **0.767** | **0.676** | 0.695 |
| 7. Static (A+B+C+D) | 54 | Random Forest | **0.658** | **0.656** | 0.667 | 0.661 | **0.726** |
| 8. Full (Static+Perturbation) | 102 | Logistic Regression | 0.608 | 0.589 | 0.717 | 0.647 | 0.687 |
| 8. Full (Static+Perturbation) | 102 | Random Forest | 0.608 | 0.603 | 0.633 | 0.618 | 0.678 |

### Observations
- On this pilot dataset, individual feature blocks achieved above-chance separation (50% baseline), with measured accuracies in the 55–66% range.
- The Static (A+B+C+D) set with Random Forest achieved the highest measured accuracy (65.8%) and ROC-AUC (0.726) in this pilot.
- Adding 48 perturbation features to the 54 static features did not improve classification on this pilot dataset; this may indicate insufficient training samples for the higher-dimensional representation.
- Logistic Regression consistently showed better recall, while Random Forest showed better precision in most configurations.

---

## 7. Perturbation Robustness Results

| Perturbation | Model | Clean Acc | Perturbed Acc | Δ Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| JPEG Q70 | Random Forest | 0.608 | 0.600 | −0.008 | 0.510 | 0.683 |
| JPEG Q70 | Logistic Regression | 0.608 | **0.658** | **+0.050** | 0.650 | 0.659 |
| Resize 50% | Random Forest | 0.608 | 0.492 | −0.117 | 0.032 | 0.452 |
| Resize 50% | Logistic Regression | 0.608 | 0.525 | −0.083 | 0.678 | 0.711 |
| Gaussian blur σ=1 | Random Forest | 0.608 | 0.517 | −0.092 | 0.065 | 0.466 |
| Gaussian blur σ=1 | Logistic Regression | 0.608 | 0.533 | −0.075 | 0.682 | 0.703 |
| Center crop 80% | Random Forest | 0.608 | 0.492 | −0.117 | 0.315 | 0.521 |
| Center crop 80% | Logistic Regression | 0.608 | 0.592 | −0.017 | 0.707 | 0.638 |

### Observations
- JPEG compression at Q70 had minimal impact on classification accuracy. Logistic Regression performance marginally improved, which may indicate a test-set noise effect on this small sample.
- Resize 50% and Gaussian blur degraded Random Forest accuracy substantially (to near chance), while Logistic Regression was more robust.
- These results suggest that the Random Forest classifier relied more heavily on high-frequency features that are destroyed by downsampling and blurring, whereas Logistic Regression's standardized linear combination was less sensitive to these specific perturbations.

---

## 8. Feature Importance

### Top 15 Features by Random Forest Importance (MDI)

| Rank | Feature | Family | RF Importance | LR Coefficient |
|---|---|---|---|---|
| 1 | `zone_forehead_residual_energy` | Facial Zones (D) | 0.0360 | −0.030 |
| 2 | `delta_jpeg70_edge_density` | Perturbation (E) | 0.0233 | +0.419 |
| 3 | `delta_jpeg70_zone_forehead_mean_grad` | Perturbation (E) | 0.0219 | +0.191 |
| 4 | `delta_crop80_lbp_entropy` | Perturbation (E) | 0.0188 | +0.720 |
| 5 | `zone_nose_residual_energy` | Facial Zones (D) | 0.0161 | −0.596 |
| 6 | `delta_jpeg70_zone_left_eye_mean_grad` | Perturbation (E) | 0.0158 | −0.017 |
| 7 | `zone_cheeks_mean_grad` | Facial Zones (D) | 0.0158 | −0.983 |
| 8 | `zone_forehead_mean_grad` | Facial Zones (D) | 0.0156 | +0.198 |
| 9 | `high_freq_ratio` | Frequency (C) | 0.0144 | −0.136 |
| 10 | `delta_resize50_zone_forehead_mean_grad` | Perturbation (E) | 0.0140 | −0.226 |
| 11 | `zone_cheeks_residual_energy` | Facial Zones (D) | 0.0138 | +1.132 |
| 12 | `zone_nose_lbp_entropy` | Facial Zones (D) | 0.0136 | +0.253 |
| 13 | `zone_forehead_edge_density` | Facial Zones (D) | 0.0134 | −0.631 |
| 14 | `delta_crop80_zone_left_eye_mean_grad` | Perturbation (E) | 0.0134 | +0.240 |
| 15 | `lbp_mean` | Texture/Residual (B) | 0.0133 | +0.059 |

**Family distribution in Top 15**: Facial Zones (D): 7, Perturbation (E): 6, Frequency (C): 1, Texture/Residual (B): 1.

On this pilot dataset, face-zone features and perturbation-response features were the most informative individual features according to the Random Forest importance measure. This is consistent with the hypothesis that forensic differences may be localized to specific facial regions and that transformation-response patterns may contain complementary information.

---

## 9. Efficiency Results

| Metric | Value |
|---|---|
| Static feature extraction (A+B+C+D) | 37.9 ms/image |
| Perturbation feature extraction (E) | 186.8 ms/image |
| Total feature extraction | 224.8 ms/image |
| Random Forest inference | 3.58 ms/image |
| Logistic Regression inference | 0.09 ms/image |
| **Total processing (RF)** | **228.4 ms/image** |
| **Total processing (LR)** | **224.9 ms/image** |
| Number of scalar features | 102 |
| RF model size | 649.2 KB |
| LR model size | 1.5 KB |

> **Note**: These benchmarks were measured on a single CPU using the `edge-artifact-guard` conda environment with Python 3.11. They represent this specific experimental setup and should not be claimed as general deployment performance.

---

## 10. Failure Cases

- **Face detection fallbacks**: 16/400 images (4%) required fallback detection (3 profile, 13 center crop). No images were discarded.
- **MediaPipe FaceMesh**: Unavailable due to Python 3.14 / MediaPipe 1.0.1 API changes (`mp.solutions` removed in v1.0+). Geometrically-defined canonical face zones were used instead.
- **LightGBM / Linear SVM**: Not available in the runtime environment. The pilot used Logistic Regression and Random Forest only.
- **No images produced NaN or Inf features**: All 400 feature vectors were finite and complete.

---

## 11. Limitations

1. **Single generator family**: Only StyleGAN-generated faces were tested. Results cannot be assumed to hold for diffusion models (DALL-E, Midjourney, Stable Diffusion) or other GAN architectures.
2. **Single real source**: Only FFHQ photographs were used. Performance on other real-face distributions is unknown.
3. **Small sample size**: 400 images (200 per class) with a 120-image test set. Variance of performance estimates is high.
4. **No cross-generator evaluation**: The experiment trains and tests on the same generator family. This does not test generalization to unseen generators.
5. **Aligned face crops**: FFHQ and StyleGAN images are roughly pre-aligned, which may inflate face-zone feature utility.
6. **No shortcut audit**: Potential metadata, format, or JPEG-grid leakage was not systematically tested.
7. **Canonical zones, not landmark-guided**: Face zones used fixed bounding boxes rather than per-image landmarks from MediaPipe.
8. **CPU benchmarks are environment-specific**: Timing measurements are specific to this hardware and software configuration.

---

## 12. What This Experiment Actually Demonstrates

On this pilot dataset of 200 FFHQ real faces and 200 StyleGAN-generated faces:

- A compact 102-D representation combining edge/gradient, texture/residual, frequency, face-zone, and perturbation-response features provided **above-chance but moderate separation** between the two classes (best measured accuracy: 65.8%, ROC-AUC: 0.726).
- Face-zone features and perturbation-response delta features appeared among the most informative individual features according to Random Forest importance, providing preliminary evidence that localized and transformation-response statistics may contain discriminative information.
- The full pipeline processes images at approximately 225 ms/image on CPU, dominated by perturbation feature extraction.

## 13. What This Experiment Does NOT Demonstrate

- This experiment does **NOT** demonstrate that AI-generated faces are inherently or universally different from real faces.
- This experiment does **NOT** demonstrate generalization to unseen generator families.
- This experiment does **NOT** demonstrate state-of-the-art detection performance.
- This experiment does **NOT** claim that synthetic faces always exhibit unstable edges or collapse under perturbation.
- This experiment does **NOT** prove that these features work for all generators or all post-processing conditions.
- The moderate accuracy levels (~60–66%) suggest that the signal in this representation, while present, is not strongly discriminative on this specific pilot sample. This could be due to insufficient training data, feature design limitations, or the inherent difficulty of the task with purely classical features.

---

## 14. Recommended Next Experiment

1. **Scale up**: Use the full 140k dataset (or a larger balanced subset, e.g. 5000 per class) to reduce variance and potentially improve classifier performance.
2. **Cross-generator evaluation**: Train on StyleGAN, test on a held-out diffusion-model generator family.
3. **Install MediaPipe FaceMesh**: Use per-image facial landmarks for zone alignment rather than fixed canonical zones.
4. **Add classifiers**: Include Linear SVM and LightGBM for multi-classifier ablation as specified in the DynForensic-GBT checklist.
5. **Shortcut audit**: Verify that no metadata, format, or compression-artifact leakage is driving classification.
6. **Feature selection**: Apply feature selection (e.g. mutual information, recursive elimination) to reduce the 102-D representation to a more compact discriminative subset.
7. **SHAP analysis**: Add SHAP-based interpretability for per-feature contribution analysis (if feasible within time constraints).

---

## Output Files

| File | Description |
|---|---|
| `results/pilot_results.csv` | Full model metrics on test set |
| `results/pilot_summary.json` | Complete experiment summary in JSON |
| `results/feature_importance.csv` | All 102 features with RF importance and LR coefficients |
| `results/ablation_results.csv` | All ablation set × model performance metrics |
| `results/perturbation_results.csv` | Perturbation robustness analysis |
| `results/confusion_matrix.png` | Confusion matrices for LR and RF |
| `results/roc_curve.png` | ROC curves for full 102-D representation |
| `results/feature_importance.png` | Top 15 feature importance bar chart |
| `results/ablation_plot.png` | Ablation accuracy comparison |
| `results/perturbation_plot.png` | Perturbation robustness comparison |
