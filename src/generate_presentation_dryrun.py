import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent
PKG_DIR = PROJECT_ROOT / "DynForensic_Testing_Package"
DATA_DIR = PROJECT_ROOT / "data" / "pilot"
RESULTS_DIR = PROJECT_ROOT / "results"
DEMO_DIR = PROJECT_ROOT / "demo_dryrun_outputs"
DEMO_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PKG_DIR))
import test_image

def generate_comparative_dryrun():
    print("Generating Presentation Dry Run Demonstration...")
    
    # 1. Load Pretrained Models from package
    pretrained_models = {
        "Logistic Regression": joblib.load(PKG_DIR / "dynforensic_ffhq_logistic_regression.joblib"),
        "Linear SVM": joblib.load(PKG_DIR / "dynforensic_ffhq_linear_svm.joblib"),
        "HistGradientBoosting": joblib.load(PKG_DIR / "dynforensic_ffhq_histgradientboosting.joblib"),
        "Random Forest": joblib.load(PKG_DIR / "dynforensic_ffhq_random_forest.joblib")
    }

    # 2. Train calibrated/retrained models on pilot train split to get accurate predictions on pilot images
    df_sample = pd.read_csv(PKG_DIR / "dynforensic_ffhq_features_4000.csv", nrows=1)
    df_cols = [c for c in df_sample.columns if c not in {"path", "filename", "label", "latency_ms"}]

    # Load 2 representative images from dataset:
    # 1 Real face (FFHQ)
    real_img_path = DATA_DIR / "real" / "real_0000.png"
    # 1 Fake face (StyleGAN)
    fake_img_path = DATA_DIR / "fake" / "fake_0000.png"

    for img_path, ground_truth in [(real_img_path, "AUTHENTIC REAL FACE"), (fake_img_path, "AI-GENERATED FACE (StyleGAN)")]:
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        feats, maps = test_image.extract_features(rgb)
        x_vec = np.array([feats.get(c, 0.0) for c in df_cols]).reshape(1, -1)
        
        # Individual classifier probabilities
        probs = {}
        for mname, mobj in pretrained_models.items():
            if hasattr(mobj, "predict_proba"):
                probs[mname] = float(mobj.predict_proba(x_vec)[0, 1])
            else:
                probs[mname] = 0.5
        
        p_mean = float(np.mean(list(probs.values())))
        verdict = "AI-GENERATED FACE" if p_mean >= 0.50 else "AUTHENTIC REAL FACE"
        conf = p_mean if p_mean >= 0.50 else (1.0 - p_mean)

        # Plot 4-panel visual forensic heatmap
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=250)
        is_fake_ground_truth = "StyleGAN" in ground_truth
        tag_color = "#991B1B" if is_fake_ground_truth else "#065F46"
        
        fig.suptitle(f"Dry Run Sample: {img_path.name}  |  Ground Truth: [{ground_truth}]\n"
                     f"Physical Structure Coherence: {feats['S__structure_coh_mean']:.3f}  |  Residual Var: {feats['S__res_var']:.6f}  |  Perturbation Shift |ΔG|: {maps['diff_mag'].mean():.4f}",
                     fontsize=11.5, fontweight='bold', color=tag_color, y=1.05)

        # Panel 1: RGB
        axes[0].imshow(maps["rgb"])
        axes[0].set_title(f"1. Standardized Face (256x256)\n{img_path.name}", fontsize=10, fontweight='bold')
        axes[0].axis('off')

        # Panel 2: Structure Coherence Map
        im1 = axes[1].imshow(maps["coherence"], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f"2. Optical Structure Coherence\nMean: {feats['S__structure_coh_mean']:.3f} (Lens vs Latent)", fontsize=10, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: High-Pass Residual Map
        im2 = axes[2].imshow(maps["residual"], cmap='seismic', vmin=-0.10, vmax=0.10)
        axes[2].set_title(f"3. High-Pass Sensor Residual\nStd: {feats['S__res_std']:.4f} (Shot Noise vs Ringing)", fontsize=10, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Perturbation Differential Map |ΔG|
        im3 = axes[3].imshow(maps["diff_mag"], cmap='turbo')
        axes[3].set_title(f"4. Perturbation Response |ΔΦ|\nMean Shift: {maps['diff_mag'].mean():.4f} (JPEG Q70)", fontsize=10, fontweight='bold')
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_file = DEMO_DIR / f"presentation_heatmap_{img_path.stem}.png"
        fig.savefig(out_file, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_file}")

    # Generate a Side-by-Side Comparison Slide Graphic
    fig_comp, axes_comp = plt.subplots(2, 4, figsize=(15, 7.5), dpi=250)
    fig_comp.suptitle("DynForensic Physical Feature Response: Real (FFHQ) vs. AI-Generated (StyleGAN)", 
                      fontsize=14, fontweight='bold', y=0.98)

    for row_idx, (img_path, row_title) in enumerate([(real_img_path, "Real Human Face (FFHQ)"), (fake_img_path, "AI-Generated Human Face (StyleGAN)")]):
        bgr = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feats, maps = test_image.extract_features(rgb)

        axes_comp[row_idx, 0].imshow(maps["rgb"])
        axes_comp[row_idx, 0].set_title(f"Input: {row_title}", fontsize=9.5, fontweight='bold')
        axes_comp[row_idx, 0].axis('off')

        axes_comp[row_idx, 1].imshow(maps["coherence"], cmap='viridis', vmin=0, vmax=1)
        axes_comp[row_idx, 1].set_title(f"Structure Coherence (Mean={feats['S__structure_coh_mean']:.3f})", fontsize=9.5, fontweight='bold')
        axes_comp[row_idx, 1].axis('off')

        axes_comp[row_idx, 2].imshow(maps["residual"], cmap='seismic', vmin=-0.10, vmax=0.10)
        axes_comp[row_idx, 2].set_title(f"Sensor Residual (Std={feats['S__res_std']:.4f})", fontsize=9.5, fontweight='bold')
        axes_comp[row_idx, 2].axis('off')

        axes_comp[row_idx, 3].imshow(maps["diff_mag"], cmap='turbo')
        axes_comp[row_idx, 3].set_title(f"Perturbation Shift |ΔΦ| (Mean={maps['diff_mag'].mean():.4f})", fontsize=9.5, fontweight='bold')
        axes_comp[row_idx, 3].axis('off')

    plt.tight_layout()
    comparison_summary_img = DEMO_DIR / "presentation_side_by_side_dryrun.png"
    fig_comp.savefig(comparison_summary_img, bbox_inches='tight')
    plt.close(fig_comp)
    print(f"Saved side-by-side summary: {comparison_summary_img}")

if __name__ == "__main__":
    generate_comparative_dryrun()
