import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import time
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.preprocess import load_and_preprocess
from src.features import (
    compute_block_a_edge_features,
    compute_block_b_texture_features,
    compute_block_c_frequency_features,
    compute_block_d_facial_zones,
    compute_block_e_perturbation_features,
    extract_all_features,
    extract_static_features,
    apply_perturbation
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pilot")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def get_feature_family(feat_name):
    if feat_name.startswith("delta_"):
        return "Perturbation (Block E)"
    elif feat_name.startswith("zone_"):
        return "Facial Zones (Block D)"
    elif any(feat_name.startswith(k) for k in ["edge_", "mean_grad_", "std_grad_", "grad_", "scharr_", "otsu_"]):
        return "Edge/Gradient (Block A)"
    elif any(feat_name.startswith(k) for k in ["local_tex_", "residual_", "high_pass_", "laplacian_", "lbp_"]):
        return "Texture/Residual (Block B)"
    elif any(feat_name.startswith(k) for k in ["low_freq_", "mid_freq_", "high_freq_", "spectral_"]):
        return "Frequency (Block C)"
    return "Other"

def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("Edge-Behavior Forensics for AI-Generated Face Detection")
    print("DynForensic-GBT 1-Hour Pilot Experiment Pipeline (~100-D Representation)")
    print("=" * 65)

    real_dir = os.path.join(DATA_DIR, "real")
    fake_dir = os.path.join(DATA_DIR, "fake")

    real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:200]
    fake_files = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:200]

    print(f"Loaded file paths: {len(real_files)} real, {len(fake_files)} fake.")

    records = []
    detection_counts = {"primary": 0, "fallback_profile": 0, "fallback_center": 0}
    preprocessed_grays = []
    labels = []
    classes = []

    print("Extracting features from 400 images...")
    t0 = time.time()
    all_items = [(f, 0, "real") for f in real_files] + [(f, 1, "fake") for f in fake_files]

    for idx, (path, label, cls_name) in enumerate(all_items):
        rgb_256, gray_256, det_status = load_and_preprocess(path)
        detection_counts[det_status] = detection_counts.get(det_status, 0) + 1

        feats = extract_all_features(gray_256)
        feats["label"] = label
        feats["class"] = cls_name
        feats["file"] = os.path.basename(path)
        feats["detection_status"] = det_status
        records.append(feats)

        preprocessed_grays.append(gray_256)
        labels.append(label)
        classes.append(cls_name)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/400 images...")

    extract_total_time = time.time() - t0
    avg_feat_time = (extract_total_time / 400.0) * 1000
    print(f"Feature extraction completed in {extract_total_time:.2f}s ({avg_feat_time:.2f} ms/image).")
    print(f"Face detection statistics: {detection_counts}")

    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c not in ["label", "class", "file", "detection_status"]]

    edge_cols = [c for c in feature_cols if get_feature_family(c) == "Edge/Gradient (Block A)"]
    tex_cols = [c for c in feature_cols if get_feature_family(c) == "Texture/Residual (Block B)"]
    freq_cols = [c for c in feature_cols if get_feature_family(c) == "Frequency (Block C)"]
    zone_cols = [c for c in feature_cols if get_feature_family(c) == "Facial Zones (Block D)"]
    pert_cols = [c for c in feature_cols if get_feature_family(c) == "Perturbation (Block E)"]
    static_cols = edge_cols + tex_cols + freq_cols + zone_cols

    print(f"Feature Breakdown: Edge={len(edge_cols)}, Texture={len(tex_cols)}, Frequency={len(freq_cols)}, Zones={len(zone_cols)}, Perturbation={len(pert_cols)}, Total={len(feature_cols)}")

    # Train / Test Split (70% train, 30% test, stratified, seed 42)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=df["label"], random_state=42
    )

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    y_train = df_train["label"].values
    y_test = df_test["label"].values

    print(f"Data Split: Train={len(df_train)} ({sum(y_train==0)} real, {sum(y_train==1)} fake), Test={len(df_test)} ({sum(y_test==0)} real, {sum(y_test==1)} fake)")

    # Ablation Study
    ablation_sets = {
        "1. Edge only": edge_cols,
        "2. Texture/residual only": tex_cols,
        "3. Frequency only": freq_cols,
        "4. Face zones only": zone_cols,
        "5. Edge + Texture": edge_cols + tex_cols,
        "6. Edge + Texture + Frequency": edge_cols + tex_cols + freq_cols,
        "7. Static (A+B+C+D)": static_cols,
        "8. Full (Static + Perturbation)": feature_cols
    }

    ablation_results = []
    trained_models = {}

    for name, cols in ablation_sets.items():
        X_tr = df_train[cols].values
        X_te = df_test[cols].values

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)

        # Logistic Regression
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_tr_scaled, y_train)
        lr_pred = lr.predict(X_te_scaled)
        lr_prob = lr.predict_proba(X_te_scaled)[:, 1]

        ablation_results.append({
            "feature_set": name,
            "num_features": len(cols),
            "model": "Logistic Regression",
            "accuracy": float(accuracy_score(y_test, lr_pred)),
            "precision": float(precision_score(y_test, lr_pred, zero_division=0)),
            "recall": float(recall_score(y_test, lr_pred, zero_division=0)),
            "f1": float(f1_score(y_test, lr_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, lr_prob))
        })

        # Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_tr, y_train)
        rf_pred = rf.predict(X_te)
        rf_prob = rf.predict_proba(X_te)[:, 1]

        ablation_results.append({
            "feature_set": name,
            "num_features": len(cols),
            "model": "Random Forest",
            "accuracy": float(accuracy_score(y_test, rf_pred)),
            "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
            "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
            "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, rf_prob))
        })

        if name == "8. Full (Static + Perturbation)":
            trained_models["lr"] = lr
            trained_models["scaler"] = scaler
            trained_models["rf"] = rf
            trained_models["lr_pred"] = lr_pred
            trained_models["lr_prob"] = lr_prob
            trained_models["rf_pred"] = rf_pred
            trained_models["rf_prob"] = rf_prob

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv(os.path.join(RESULTS_DIR, "ablation_results.csv"), index=False)
    print("\n=== Ablation Results ===")
    print(df_ablation.to_string(index=False))

    # Feature Importance
    full_rf = trained_models["rf"]
    full_lr = trained_models["lr"]

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "family": [get_feature_family(f) for f in feature_cols],
        "rf_importance": full_rf.feature_importances_,
        "lr_coef": full_lr.coef_[0],
        "lr_abs_coef": np.abs(full_lr.coef_[0])
    }).sort_values(by="rf_importance", ascending=False)

    fi_df.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    print("\n=== Top 15 Features by Random Forest Importance ===")
    print(fi_df.head(15)[["feature", "family", "rf_importance", "lr_coef"]].to_string(index=False))

    # Perturbation Robustness Analysis
    test_grays = [preprocessed_grays[i] for i in test_idx]
    pert_results = []

    clean_rf_acc = float(accuracy_score(y_test, trained_models["rf_pred"]))
    clean_lr_acc = float(accuracy_score(y_test, trained_models["lr_pred"]))

    perturbations = ["jpeg70", "resize50", "blur1", "crop80"]
    for p in perturbations:
        p_records = []
        for gray_img in test_grays:
            p_img = apply_perturbation(gray_img, p)
            p_feats = extract_all_features(p_img)
            p_records.append(p_feats)
        df_p = pd.DataFrame(p_records)[feature_cols]

        p_rf_pred = full_rf.predict(df_p.values)
        p_rf_prob = full_rf.predict_proba(df_p.values)[:, 1]
        rf_acc = float(accuracy_score(y_test, p_rf_pred))
        rf_f1 = float(f1_score(y_test, p_rf_pred, zero_division=0))
        rf_auc = float(roc_auc_score(y_test, p_rf_prob))

        X_p_scaled = trained_models["scaler"].transform(df_p.values)
        p_lr_pred = full_lr.predict(X_p_scaled)
        p_lr_prob = full_lr.predict_proba(X_p_scaled)[:, 1]
        lr_acc = float(accuracy_score(y_test, p_lr_pred))
        lr_f1 = float(f1_score(y_test, p_lr_pred, zero_division=0))
        lr_auc = float(roc_auc_score(y_test, p_lr_prob))

        pert_results.append({
            "perturbation": p,
            "model": "Random Forest",
            "clean_accuracy": clean_rf_acc,
            "perturbed_accuracy": rf_acc,
            "accuracy_delta": rf_acc - clean_rf_acc,
            "f1": rf_f1,
            "roc_auc": rf_auc
        })
        pert_results.append({
            "perturbation": p,
            "model": "Logistic Regression",
            "clean_accuracy": clean_lr_acc,
            "perturbed_accuracy": lr_acc,
            "accuracy_delta": lr_acc - clean_lr_acc,
            "f1": lr_f1,
            "roc_auc": lr_auc
        })

    df_pert = pd.DataFrame(pert_results)
    df_pert.to_csv(os.path.join(RESULTS_DIR, "perturbation_results.csv"), index=False)
    print("\n=== Perturbation Robustness Results ===")
    print(df_pert.to_string(index=False))

    # CPU Latency & Efficiency Benchmarking
    sample_gray = preprocessed_grays[0]

    for _ in range(5):
        _ = extract_all_features(sample_gray)

    n_runs = 50
    t_start = time.time()
    for _ in range(n_runs):
        _ = extract_static_features(sample_gray)
    static_time_per_img = ((time.time() - t_start) / n_runs) * 1000

    t_start = time.time()
    for _ in range(n_runs):
        _ = compute_block_e_perturbation_features(sample_gray)
    pert_time_per_img = ((time.time() - t_start) / n_runs) * 1000

    total_feat_time_per_img = static_time_per_img + pert_time_per_img

    sample_feat_vec = df[feature_cols].iloc[0].values.reshape(1, -1)
    sample_feat_scaled = trained_models["scaler"].transform(sample_feat_vec)

    for _ in range(20):
        _ = full_rf.predict(sample_feat_vec)
        _ = full_lr.predict(sample_feat_scaled)

    n_inf = 500
    t_start = time.time()
    for _ in range(n_inf):
        _ = full_rf.predict(sample_feat_vec)
    rf_inf_time = ((time.time() - t_start) / n_inf) * 1000

    t_start = time.time()
    for _ in range(n_inf):
        _ = full_lr.predict(sample_feat_scaled)
    lr_inf_time = ((time.time() - t_start) / n_inf) * 1000

    rf_bytes = len(pickle.dumps(full_rf))
    lr_bytes = len(pickle.dumps(full_lr))

    total_proc_time_rf = total_feat_time_per_img + rf_inf_time
    total_proc_time_lr = total_feat_time_per_img + lr_inf_time

    benchmarks = {
        "static_feature_time_ms": static_time_per_img,
        "perturbation_feature_time_ms": pert_time_per_img,
        "total_feature_extraction_time_ms": total_feat_time_per_img,
        "rf_inference_time_ms": rf_inf_time,
        "lr_inference_time_ms": lr_inf_time,
        "total_processing_time_rf_ms": total_proc_time_rf,
        "total_processing_time_lr_ms": total_proc_time_lr,
        "num_scalar_features": len(feature_cols),
        "rf_model_size_kb": rf_bytes / 1024.0,
        "lr_model_size_kb": lr_bytes / 1024.0
    }

    # Generate Plots
    # 1. Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    cm_lr = confusion_matrix(y_test, trained_models["lr_pred"])
    cm_rf = confusion_matrix(y_test, trained_models["rf_pred"])

    for ax, cm, title in zip(axes, [cm_lr, cm_rf], ["Logistic Regression", "Random Forest"]):
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix: {title}")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Real", "Fake"])
        ax.set_yticklabels(["Real", "Fake"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="red" if cm[i, j] > cm.max()/2 else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()

    # 2. ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, trained_models["lr_prob"])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, trained_models["rf_prob"])

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, trained_models['lr_prob']):.3f})", lw=2)
    plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, trained_models['rf_prob']):.3f})", lw=2)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Full 102-D Representation (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=200)
    plt.close()

    # 3. Feature Importance (Top 15)
    top15 = fi_df.head(15).iloc[::-1]
    plt.figure(figsize=(10, 6))
    colors = {
        "Edge/Gradient (Block A)": "#2b5c8f",
        "Texture/Residual (Block B)": "#2ca02c",
        "Frequency (Block C)": "#d62728",
        "Facial Zones (Block D)": "#ff7f0e",
        "Perturbation (Block E)": "#9467bd"
    }
    bar_colors = [colors.get(fam, "#7f7f7f") for fam in top15["family"]]
    plt.barh(top15["feature"], top15["rf_importance"], color=bar_colors)
    plt.xlabel("Random Forest Feature Importance (MDI)")
    plt.title("Top 15 Informative Features in Pilot Dataset")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in colors.values()]
    plt.legend(handles, colors.keys(), loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=200)
    plt.close()

    # 4. Ablation Plot
    plt.figure(figsize=(12, 5.5))
    df_ab_rf = df_ablation[df_ablation["model"] == "Random Forest"]
    df_ab_lr = df_ablation[df_ablation["model"] == "Logistic Regression"]
    x = np.arange(len(df_ab_rf))
    width = 0.35
    plt.bar(x - width/2, df_ab_lr["accuracy"], width, label="Logistic Regression", color="#4c72b0")
    plt.bar(x + width/2, df_ab_rf["accuracy"], width, label="Random Forest", color="#55a868")
    plt.xticks(x, [s.replace(". ", "\n") for s in df_ab_rf["feature_set"]], rotation=20, ha="right", fontsize=9)
    plt.ylabel("Accuracy")
    plt.ylim([0.45, 1.05])
    plt.title("Feature Ablation Performance Comparison (DynForensic Representation)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "ablation_plot.png"), dpi=200)
    plt.close()

    # 5. Perturbation Plot
    plt.figure(figsize=(9, 5))
    df_p_rf = df_pert[df_pert["model"] == "Random Forest"]
    df_p_lr = df_pert[df_pert["model"] == "Logistic Regression"]
    xp = np.arange(len(df_p_rf))
    plt.bar(xp - width/2, df_p_lr["perturbed_accuracy"], width, label="Logistic Regression", color="#e74c3c")
    plt.bar(xp + width/2, df_p_rf["perturbed_accuracy"], width, label="Random Forest", color="#3498db")
    plt.axhline(clean_rf_acc, color="#3498db", linestyle="--", label=f"RF Clean ({clean_rf_acc:.2f})")
    plt.axhline(clean_lr_acc, color="#e74c3c", linestyle=":", label=f"LR Clean ({clean_lr_acc:.2f})")
    plt.xticks(xp, df_p_rf["perturbation"])
    plt.xlabel("Perturbation Type")
    plt.ylabel("Test Accuracy under Degradation")
    plt.title("Performance Under Controlled Perturbations")
    plt.ylim([0.4, 1.05])
    plt.legend(loc="lower left")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "perturbation_plot.png"), dpi=200)
    plt.close()

    # Save CSV Results & JSON Summary
    full_results_list = []
    for model_name, pred, prob in [("Logistic Regression", trained_models["lr_pred"], trained_models["lr_prob"]),
                                   ("Random Forest", trained_models["rf_pred"], trained_models["rf_prob"])]:
        cm = confusion_matrix(y_test, pred)
        full_results_list.append({
            "model": model_name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1])
        })
    df_pilot_results = pd.DataFrame(full_results_list)
    df_pilot_results.to_csv(os.path.join(RESULTS_DIR, "pilot_results.csv"), index=False)

    summary_json = {
        "dataset": {
            "name": "TheKernel01/140k-Real-and-Fake-Faces",
            "split": "train (streamed)",
            "real_images": 200,
            "fake_images": 200,
            "total_images": 400,
            "face_detection": detection_counts
        },
        "features": {
            "edge_features_count": len(edge_cols),
            "texture_features_count": len(tex_cols),
            "frequency_features_count": len(freq_cols),
            "landmark_face_zones_count": len(zone_cols),
            "landmark_status": "geometric_aligned_face_zones",
            "perturbation_features_count": len(pert_cols),
            "total_features_count": len(feature_cols)
        },
        "models": full_results_list,
        "efficiency": benchmarks
    }

    with open(os.path.join(RESULTS_DIR, "pilot_summary.json"), "w") as f:
        json.dump(summary_json, f, indent=2)

    print("\nExperiment execution completed successfully. All artifacts written to results/")

if __name__ == "__main__":
    run_experiment()
