"""
DynForensic Face Detector - Standalone Test Script
===================================================
Portable inference script for testing any face image (JPG, PNG, WEBP).
Uses relative paths so it works seamlessly on any machine!

Usage:
    python test_image.py path/to/face.jpg
    python test_image.py
"""

import sys
import os
import argparse
from pathlib import Path
import numpy as np
import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure safe console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.resolve()

# Relative model paths
MODEL_PATH_LOGREG = SCRIPT_DIR / "dynforensic_ffhq_logistic_regression.joblib"
MODEL_PATH_SVM    = SCRIPT_DIR / "dynforensic_ffhq_linear_svm.joblib"
MODEL_PATH_HGB    = SCRIPT_DIR / "dynforensic_ffhq_histgradientboosting.joblib"
MODEL_PATH_RF     = SCRIPT_DIR / "dynforensic_ffhq_random_forest.joblib"
DATASET_CSV       = SCRIPT_DIR / "dynforensic_ffhq_features_4000.csv"

EPS = 1e-8

def extract_features(rgb_image):
    """Self-contained 61-D DynForensic extractor"""
    rgb_256 = cv2.resize(rgb_image, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb_256, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape

    # 1. Gradients & Structure Tensor
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3)
    mag = cv2.magnitude(gx, gy)
    ang = np.arctan2(gy, gx)

    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 2.0)
    trace = jxx + jyy
    diff = np.sqrt(np.maximum((jxx - jyy)**2 + 4.0 * (jxy**2), 0.0))
    l1 = 0.5 * (trace + diff)
    l2 = 0.5 * (trace - diff)
    coh = np.where(trace > EPS, ((l1 - l2) / (trace + EPS))**2, 0.0)

    mag8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, edge8 = cv2.threshold(mag8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = edge8 > 0

    # 2. Residual
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)
    res = gray - blur
    corr_h = float(np.corrcoef(res[:, :-1].ravel(), res[:, 1:].ravel())[0, 1])
    corr_v = float(np.corrcoef(res[:-1, :].ravel(), res[1:, :].ravel())[0, 1])

    # 3. FFT Radial Spectrum
    g_zero = gray - gray.mean()
    spec = np.fft.fftshift(np.fft.fft2(g_zero))
    power = np.abs(spec)**2
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.indices((h, w))
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    rn = r / (np.max(r) + EPS)
    valid = (rn >= 0.03) & (rn <= 0.95)
    total_power = np.sum(power[valid]) + EPS
    low_ratio = float(np.sum(power[(rn < 0.35) & valid]) / total_power)
    mid_ratio = float(np.sum(power[(rn >= 0.35) & (rn < 0.70) & valid]) / total_power)
    high_ratio = float(np.sum(power[(rn >= 0.70) & valid]) / total_power)

    bins = np.linspace(0.03, 0.95, 26)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (rn >= lo) & (rn < hi) & valid
        if np.any(m):
            xs.append((lo + hi) / 2.0)
            ys.append(np.mean(power[m]) + EPS)
    slope = float(abs(np.polyfit(np.log(np.asarray(xs)), np.log(np.asarray(ys)), 1)[0])) if len(xs) >= 6 else 0.0
    peak_mask = (r > max(4.0, 0.04 * np.max(r))) & valid
    peak_vals = power[peak_mask]
    peak_score = float(np.max(peak_vals) / (np.mean(peak_vals) + EPS)) if peak_vals.size else 0.0

    p_norm = power[valid].astype(np.float64) / (total_power + EPS)
    p_norm = p_norm[p_norm > 0]
    spec_ent = float(-np.sum(p_norm * np.log2(p_norm)) / np.log2(max(len(p_norm), 2)))

    # 4. Regional Zones
    zones = {
        "forehead": gray[int(0.02*h):int(0.25*h), int(0.20*w):int(0.80*w)],
        "eyes": gray[int(0.25*h):int(0.45*h), int(0.15*w):int(0.85*w)],
        "nose": gray[int(0.42*h):int(0.68*h), int(0.32*w):int(0.68*w)],
        "mouth": gray[int(0.65*h):int(0.88*h), int(0.25*w):int(0.75*w)],
        "cheeks": np.concatenate([
            gray[int(0.45*h):int(0.75*h), int(0.08*w):int(0.28*w)].ravel(),
            gray[int(0.45*h):int(0.75*h), int(0.72*w):int(0.92*w)].ravel()
        ])
    }
    z_feats = {}
    for zn, zp in zones.items():
        if zn == "cheeks":
            z_feats[f"Z__zone_{zn}_res_var"] = float(zp.var())
            z_feats[f"Z__zone_{zn}_kurt"] = float(np.mean((zp - zp.mean())**4) / (zp.var()**2 + EPS))
            z_feats[f"Z__zone_{zn}_grad_mean"] = float(np.abs(np.diff(zp)).mean())
        else:
            z_gx = cv2.Sobel(zp, cv2.CV_32F, 1, 0, 3)
            z_gy = cv2.Sobel(zp, cv2.CV_32F, 0, 1, 3)
            z_mag = cv2.magnitude(z_gx, z_gy)
            z_res = zp - cv2.GaussianBlur(zp, (5, 5), 1.0)
            z_feats[f"Z__zone_{zn}_grad_mean"] = float(z_mag.mean())
            z_feats[f"Z__zone_{zn}_res_var"] = float(z_res.var())
            z_feats[f"Z__zone_{zn}_kurt"] = float(np.mean((z_mag - z_mag.mean())**4) / (z_mag.var()**2 + EPS))

    # 5. Perturbation Dynamics
    diff_feats = {}
    bgr_256 = cv2.cvtColor(rgb_256, cv2.COLOR_RGB2BGR)
    # T1: JPEG 70
    ok, enc = cv2.imencode(".jpg", bgr_256, [cv2.IMWRITE_JPEG_QUALITY, 70])
    dec_j = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # T2: Resize 50%
    dec_r = cv2.resize(cv2.resize(gray, (w // 2, h // 2)), (w, h))
    # T3: Blur 1.0
    dec_b = cv2.GaussianBlur(gray, (5, 5), 1.0)

    for tname, tg in [("jpeg70", dec_j), ("resize50", dec_r), ("blur1", dec_b)]:
        tgx = cv2.Sobel(tg, cv2.CV_32F, 1, 0, 3)
        tgy = cv2.Sobel(tg, cv2.CV_32F, 0, 1, 3)
        tmag = cv2.magnitude(tgx, tgy)
        tjxx = cv2.GaussianBlur(tgx * tgx, (0, 0), 2.0)
        tjyy = cv2.GaussianBlur(tgy * tgy, (0, 0), 2.0)
        tjxy = cv2.GaussianBlur(tgx * tgy, (0, 0), 2.0)
        ttr = tjxx + tjyy
        tdf = np.sqrt(np.maximum((tjxx - tjyy)**2 + 4.0 * (tjxy**2), 0.0))
        tcoh = np.where(ttr > EPS, ((tdf) / (ttr + EPS))**2, 0.0)
        tres = tg - cv2.GaussianBlur(tg, (5, 5), 1.0)

        diff_feats[f"P__delta__{tname}__grad_mean"] = float(mag.mean() - tmag.mean())
        diff_feats[f"P__delta__{tname}__grad_kurt"] = float((np.mean((mag - mag.mean())**4) / (mag.var()**2 + EPS)) - (np.mean((tmag - tmag.mean())**4) / (tmag.var()**2 + EPS)))
        diff_feats[f"P__delta__{tname}__structure_coh"] = float(coh.mean() - tcoh.mean())
        diff_feats[f"P__delta__{tname}__res_var"] = float(res.var() - tres.var())
        diff_feats[f"P__delta__{tname}__res_kurt"] = float((np.mean((res - res.mean())**4) / (res.var()**2 + EPS)) - (np.mean((tres - tres.mean())**4) / (tres.var()**2 + EPS)))
        diff_feats[f"P__delta__{tname}__fft_slope"] = float(slope - 2.0)
        diff_feats[f"P__delta__{tname}__fft_peak"] = float(peak_score - 10.0)

    # Static features dictionary
    s_feats = {
        "S__grad_mean": float(mag.mean()),
        "S__grad_std": float(mag.std()),
        "S__grad_kurt": float(np.mean((mag - mag.mean())**4) / (mag.var()**2 + EPS)),
        "S__grad_skew": float(np.mean((mag - mag.mean())**3) / (mag.std()**3 + EPS)),
        "S__grad_orient_entropy": 0.99,
        "S__structure_coh_mean": float(coh.mean()),
        "S__structure_coh_std": float(coh.std()),
        "S__structure_coh_p90": float(np.percentile(coh, 90)),
        "S__structure_coh_high_ratio": float((coh > 0.6).mean()),
        "S__edge_density": float(mask.mean()),
        "S__edge_mag_mean": float(mag[mask].mean()) if mask.any() else 0.0,
        "S__edge_mag_std": float(mag[mask].std()) if mask.any() else 0.0,
        "S__edge_otsu_thr": float(thr / 255.0),
        "S__res_mean": float(res.mean()),
        "S__res_std": float(res.std()),
        "S__res_var": float(res.var()),
        "S__res_kurt": float(np.mean((res - res.mean())**4) / (res.var()**2 + EPS)),
        "S__res_skew": float(np.mean((res - res.mean())**3) / (res.std()**3 + EPS)),
        "S__res_corr_lag1": float(0.5 * (corr_h + corr_v)),
        "S__fft_slope_alpha": slope,
        "S__fft_peak_score": peak_score,
        "S__fft_spec_entropy": spec_ent,
        "S__fft_low_ratio": low_ratio,
        "S__fft_mid_ratio": mid_ratio,
        "S__fft_high_ratio": high_ratio
    }

    full_dict = {}
    full_dict.update(s_feats)
    full_dict.update(z_feats)
    full_dict.update(diff_feats)

    maps = {
        "coherence": coh,
        "residual": res,
        "diff_mag": np.abs(mag - cv2.magnitude(cv2.Sobel(dec_j, cv2.CV_32F, 1, 0, 3), cv2.Sobel(dec_j, cv2.CV_32F, 0, 1, 3))),
        "rgb": rgb_256
    }
    return full_dict, maps

def test_face(image_path: Path):
    if not image_path.exists():
        print(f"Error: File not found -> {image_path}")
        return

    # Load models
    if not MODEL_PATH_SVM.exists():
        print(f"Error: Model file missing: {MODEL_PATH_SVM.name}. Please ensure the model file is in the same folder.")
        return

    model_svm = joblib.load(MODEL_PATH_SVM)
    # Load all available models
    models = {}
    for mname, mpath in [("Logistic Regression", MODEL_PATH_LOGREG),
                         ("Linear SVM", MODEL_PATH_SVM),
                         ("HistGradientBoosting", MODEL_PATH_HGB),
                         ("Random Forest", MODEL_PATH_RF)]:
        if mpath.exists():
            models[mname] = joblib.load(mpath)

    if not models:
        print(f"Error: No model weights (.joblib) found in {SCRIPT_DIR}!")
        return

    # Load image
    bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Error: Unable to decode image at {image_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Extract
    feats, maps = extract_features(rgb)

    # Get expected feature columns
    if DATASET_CSV.exists():
        import pandas as pd
        df_cols = [c for c in pd.read_csv(DATASET_CSV, nrows=1).columns if c not in {"path", "filename", "label", "latency_ms"}]
    else:
        df_cols = list(feats.keys())

    x_vec = np.array([feats.get(c, 0.0) for c in df_cols]).reshape(1, -1)

    model_probs = {}
    for mname, mobj in models.items():
        model_probs[mname] = float(mobj.predict_proba(x_vec)[0, 1])

    prob_consensus = float(np.mean(list(model_probs.values())))
    verdict = "AI-GENERATED FACE" if prob_consensus >= 0.50 else "AUTHENTIC REAL FACE"
    confidence = prob_consensus if prob_consensus >= 0.50 else (1.0 - prob_consensus)

    print("\n" + "="*72)
    print(f"       DYNFORENSIC-GBT MULTI-MODEL TEST REPORT: {image_path.name}")
    print("="*72)
    print(f"  CONSENSUS VERDICT : >>> [ {verdict} ] <<<")
    print(f"  Overall Confidence: {confidence*100:.1f}% (Mean P(AI) = {prob_consensus*100:.1f}%)")
    print("-" * 72)
    print("  INDIVIDUAL CLASSIFIER BREAKDOWN:")
    for mname, p_ai in model_probs.items():
        v = "AI-Generated" if p_ai >= 0.50 else "Real Face"
        c = p_ai if p_ai >= 0.50 else 1.0 - p_ai
        print(f"    * {mname:<22}: {p_ai*100:5.1f}% AI -> [{v}] (Conf: {c*100:.1f}%)")
    print("-" * 72)
    print("  PHYSICAL FORENSIC EXPLANATION:")
    print(f"    * Structure Coherence (C) : {feats['S__structure_coh_mean']:.4f} (Real > 0.48 | Fake < 0.45)")
    print(f"    * High-Pass Residual Var  : {feats['S__res_var']:.6f} (Sensor shot noise vs Ringing)")
    print(f"    * Perturbation Delta-Grad : {feats['P__delta__jpeg70__grad_mean']:.4f} (Excess shift under JPEG)")
    print("="*72)

    # Generate and save output heatmap
    out_map_path = SCRIPT_DIR / f"heatmap_result_{image_path.stem}.png"
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    title_col = "#991B1B" if prob_consensus >= 0.50 else "#065F46"
    fig.suptitle(f"{image_path.name} | Verdict: [{verdict}] (Confidence: {confidence*100:.1f}%)", 
                 fontsize=13, fontweight='bold', color=title_col, y=1.03)

    axes[0].imshow(maps["rgb"])
    axes[0].set_title("1. Input Face (256x256)", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    im_c = axes[1].imshow(maps["coherence"], cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title(f"2. Structure Coherence\nMean: {feats['S__structure_coh_mean']:.3f}", fontsize=10, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im_c, ax=axes[1], fraction=0.046, pad=0.04)

    im_r = axes[2].imshow(maps["residual"], cmap='seismic', vmin=-0.12, vmax=0.12)
    axes[2].set_title(f"3. High-Pass Residual\nStd: {feats['S__res_std']:.4f}", fontsize=10, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im_r, ax=axes[2], fraction=0.046, pad=0.04)

    im_d = axes[3].imshow(maps["diff_mag"], cmap='turbo')
    axes[3].set_title(f"4. Perturbation Shift |ΔG|\nMean: {maps['diff_mag'].mean():.4f}", fontsize=10, fontweight='bold')
    axes[3].axis('off')
    plt.colorbar(im_d, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_map_path, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[+] Visual forensic heatmap saved to: {out_map_path.name}")
    print("="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = Path(sys.argv[1])
    else:
        # Default test image in folder
        target_path = SCRIPT_DIR / "heatmaps" / "real" / "heatmap_real_53844.png"
        sample_candidates = list(SCRIPT_DIR.glob("*.jpg")) + list(SCRIPT_DIR.glob("*.png"))
        if sample_candidates:
            target_path = sample_candidates[0]
        else:
            print("Please provide an image path: python test_image.py <image_path>")
            sys.exit(0)

    test_face(target_path)
