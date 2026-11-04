#!/usr/bin/env python3
"""Final terminal summary for the DynForensic-GBT pilot experiment (Section 18)."""

import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def print_summary():
    summary_path = os.path.join(RESULTS_DIR, "pilot_summary.json")
    with open(summary_path) as f:
        s = json.load(f)

    d = s["dataset"]
    feat = s["features"]
    eff = s["efficiency"]
    models = {m["model"]: m for m in s["models"]}

    lr = models["Logistic Regression"]
    rf = models["Random Forest"]

    print()
    print("=" * 65)
    print("  DynForensic-GBT  —  PILOT EXPERIMENT SUMMARY")
    print("=" * 65)

    print(f"""
Dataset:
  Source:       {d['name']}
  Real:         {d['real_images']}
  Fake:         {d['fake_images']}
  Total:        {d['total_images']}
  Detection:    primary={d['face_detection']['primary']}, fallback_profile={d['face_detection']['fallback_profile']}, fallback_center={d['face_detection']['fallback_center']}

Features:
  Edge (Block A):          {feat['edge_features_count']}
  Texture (Block B):       {feat['texture_features_count']}
  Frequency (Block C):     {feat['frequency_features_count']}
  Facial Zones (Block D):  {feat['landmark_face_zones_count']}  ({feat['landmark_status']})
  Perturbation (Block E):  {feat['perturbation_features_count']}
  Total:                   {feat['total_features_count']}

Models (Full 102-D Representation, Test Set):
                         Logistic Regression    Random Forest
  Accuracy:              {lr['accuracy']:.4f}                 {rf['accuracy']:.4f}
  Precision:             {lr['precision']:.4f}                 {rf['precision']:.4f}
  Recall:                {lr['recall']:.4f}                 {rf['recall']:.4f}
  F1:                    {lr['f1']:.4f}                 {rf['f1']:.4f}
  ROC-AUC:               {lr['roc_auc']:.4f}                 {rf['roc_auc']:.4f}

CPU Efficiency:
  Static feature extraction:       {eff['static_feature_time_ms']:.1f} ms/image
  Perturbation feature extraction: {eff['perturbation_feature_time_ms']:.1f} ms/image
  Total feature extraction:        {eff['total_feature_extraction_time_ms']:.1f} ms/image
  RF inference:                    {eff['rf_inference_time_ms']:.2f} ms/image
  LR inference:                    {eff['lr_inference_time_ms']:.3f} ms/image
  Total per image (RF):            {eff['total_processing_time_rf_ms']:.1f} ms
  Total per image (LR):            {eff['total_processing_time_lr_ms']:.1f} ms
  Model size (RF):                 {eff['rf_model_size_kb']:.1f} KB
  Model size (LR):                 {eff['lr_model_size_kb']:.1f} KB
""")

    print("-" * 65)
    print("PIPELINE STATUS")
    print("-" * 65)
    print("  ✅  Pilot pipeline ran successfully.")
    print("  ✅  Block A (Edge/Gradient) — 10 features — Sobel, Scharr, Canny, orientation")
    print("  ✅  Block B (Texture/Residual) — 12 features — local variance, LBP, Gaussian/Laplacian residuals")
    print("  ✅  Block C (Frequency) — 8 features — FFT radial bands, spectral slope, peak score")
    print("  ✅  Block D (Facial Zones) — 24 features — forehead, eyes, nose, mouth, cheeks (canonical zones)")
    print("  ✅  Block E (Perturbation ΔΦ) — 48 features — JPEG70, resize50, blur1, crop80 deltas")
    print("  ⚠️   MediaPipe FaceMesh: unavailable (v1.0.1 API changed); canonical zones used instead")
    print("  ⚠️   LightGBM / Linear SVM: not installed; omitted from classifier ablation")
    print()

    print("RESULT FILES:")
    results_files = [
        "pilot_results.csv",
        "pilot_summary.json",
        "feature_importance.csv",
        "ablation_results.csv",
        "perturbation_results.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "feature_importance.png",
        "ablation_plot.png",
        "perturbation_plot.png",
        "PILOT_REPORT.md"
    ]
    for rf in results_files:
        path = os.path.join(RESULTS_DIR, rf)
        exists = "✅" if os.path.exists(path) else "❌"
        size = ""
        if os.path.exists(path):
            sz = os.path.getsize(path)
            if sz > 1024:
                size = f"  ({sz/1024:.1f} KB)"
            else:
                size = f"  ({sz} B)"
        print(f"  {exists}  results/{rf}{size}")

    print()
    print("=" * 65)
    print("  All results saved to: results/")
    print("=" * 65)
    print()

if __name__ == "__main__":
    print_summary()
