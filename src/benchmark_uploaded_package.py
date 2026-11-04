import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PKG_DIR = os.path.join(PROJECT_ROOT, "DynForensic_Testing_Package")
sys.path.insert(0, PKG_DIR)
import test_image

import matplotlib
matplotlib.use("Agg")
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pilot")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def run_package_benchmark():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 65)
    print("DynForensic Testing Package Evaluation on Pilot Dataset (400 Images)")
    print("=" * 65)

    real_dir = os.path.join(DATA_DIR, "real")
    fake_dir = os.path.join(DATA_DIR, "fake")

    real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:200]
    fake_files = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:200]

    all_items = [(f, 0, "real") for f in real_files] + [(f, 1, "fake") for f in fake_files]

    # Load expected feature columns from package dataset CSV
    df_sample = pd.read_csv(os.path.join(PKG_DIR, "dynforensic_ffhq_features_4000.csv"), nrows=1)
    df_cols = [c for c in df_sample.columns if c not in {"path", "filename", "label", "latency_ms"}]
    print(f"Expected 61-D feature columns from package: {len(df_cols)}")

    # 1. Feature extraction using package's extract_features
    print(f"\nExtracting 61-D features for all 400 images using DynForensic package...")
    records = []
    t_start = time.time()
    for idx, (path, label, cls_name) in enumerate(all_items):
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feats, _ = test_image.extract_features(rgb)
        feats["label"] = label
        feats["class"] = cls_name
        feats["filename"] = os.path.basename(path)
        records.append(feats)
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/400 images...")
    
    total_time = time.time() - t_start
    avg_feat_time_ms = (total_time / 400.0) * 1000.0
    print(f"Feature extraction done in {total_time:.2f}s ({avg_feat_time_ms:.2f} ms/image).")

    df_feats = pd.DataFrame(records)
    X = np.array([[row.get(c, 0.0) for c in df_cols] for _, row in df_feats.iterrows()])
    y = df_feats["label"].values

    # Train / Test Split identical to our previous experiment (70% train, 30% test, seed 42)
    indices = np.arange(len(df_feats))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.30, stratify=y, random_state=42
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # 2. Evaluate PRE-TRAINED models from package
    print("\n--- Evaluating Pre-trained Package Models (Joblib Weights) ---")
    pretrained_models = {
        "Logistic Regression (Pre-trained)": joblib.load(os.path.join(PKG_DIR, "dynforensic_ffhq_logistic_regression.joblib")),
        "Linear SVM (Pre-trained)": joblib.load(os.path.join(PKG_DIR, "dynforensic_ffhq_linear_svm.joblib")),
        "HistGradientBoosting (Pre-trained)": joblib.load(os.path.join(PKG_DIR, "dynforensic_ffhq_histgradientboosting.joblib")),
        "Random Forest (Pre-trained)": joblib.load(os.path.join(PKG_DIR, "dynforensic_ffhq_random_forest.joblib"))
    }

    eval_results = []
    for mname, model in pretrained_models.items():
        # Evaluate on the 120 test images
        t0 = time.time()
        n_eval = len(X_test)
        preds = model.predict(X_test)
        lat_ms = ((time.time() - t0) / n_eval) * 1000.0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_test)
            auc = float(roc_auc_score(y_test, probs))
        else:
            auc = 0.5

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)

        eval_results.append({
            "model": mname,
            "condition": "Pre-trained (Zero-Shot on Pilot Test)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "inference_ms": lat_ms,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
        })

        # Also evaluate on FULL 400 images
        preds_all = model.predict(X)
        if hasattr(model, "predict_proba"):
            probs_all = model.predict_proba(X)[:, 1]
            auc_all = float(roc_auc_score(y, probs_all))
        else:
            auc_all = 0.5
        eval_results.append({
            "model": mname,
            "condition": "Pre-trained (All 400 Images)",
            "accuracy": float(accuracy_score(y, preds_all)),
            "precision": float(precision_score(y, preds_all, zero_division=0)),
            "recall": float(recall_score(y, preds_all, zero_division=0)),
            "f1": float(f1_score(y, preds_all, zero_division=0)),
            "roc_auc": auc_all,
            "inference_ms": lat_ms,
            "tn": int(confusion_matrix(y, preds_all)[0, 0]), "fp": int(confusion_matrix(y, preds_all)[0, 1]),
            "fn": int(confusion_matrix(y, preds_all)[1, 0]), "tp": int(confusion_matrix(y, preds_all)[1, 1])
        })

    # 3. Train models from scratch on the 280-image training split using the 61-D representation
    print("\n--- Training Models from Scratch on Pilot 280-train Split (61-D features) ---")
    retrained_models = {
        "Logistic Regression (Retrained 61-D)": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Linear SVM (Retrained 61-D)": make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(dual=False, random_state=42))),
        "HistGradientBoosting (Retrained 61-D)": HistGradientBoostingClassifier(random_state=42),
        "Random Forest (Retrained 61-D)": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for mname, model in retrained_models.items():
        model.fit(X_train, y_train)
        t0 = time.time()
        preds = model.predict(X_test)
        lat_ms = ((time.time() - t0) / len(X_test)) * 1000.0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, probs))
        else:
            auc = 0.5

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        cm = confusion_matrix(y_test, preds)

        eval_results.append({
            "model": mname,
            "condition": "Retrained on Pilot (120 Test)",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "inference_ms": lat_ms,
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1])
        })

    df_eval = pd.DataFrame(eval_results)
    df_eval.to_csv(os.path.join(RESULTS_DIR, "package_evaluation_results.csv"), index=False)
    print("\nPackage Evaluation Results on Pilot Dataset:")
    print(df_eval[["model", "condition", "accuracy", "precision", "recall", "f1", "roc_auc", "inference_ms"]].to_string(index=False))

    # 4. Direct Comparison with our earlier 102-D Pilot Results
    our_pilot_csv = os.path.join(RESULTS_DIR, "pilot_results.csv")
    df_our_pilot = pd.read_csv(our_pilot_csv)

    comparison_records = []
    # Our pilot models (102-D)
    for _, row in df_our_pilot.iterrows():
        comparison_records.append({
            "Framework": "Our 102-D Pilot Pipeline",
            "Model": row["model"],
            "Training Split": "280 Train / 120 Test",
            "Feature Dimensions": 102,
            "Feature Extraction Latency (ms)": 224.78,
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "ROC-AUC": row["roc_auc"]
        })

    # Our static 54-D best model
    comparison_records.append({
        "Framework": "Our Static (A+B+C+D) Pipeline",
        "Model": "Random Forest",
        "Training Split": "280 Train / 120 Test",
        "Feature Dimensions": 54,
        "Feature Extraction Latency (ms)": 37.95,
        "Accuracy": 0.6583,
        "Precision": 0.6557,
        "Recall": 0.6667,
        "F1": 0.6612,
        "ROC-AUC": 0.7261
    })

    # Uploaded package pre-trained models on 120 Test
    df_pkg_test = df_eval[df_eval["condition"] == "Pre-trained (Zero-Shot on Pilot Test)"]
    for _, row in df_pkg_test.iterrows():
        clean_model_name = row["model"].replace(" (Pre-trained)", "")
        comparison_records.append({
            "Framework": "Uploaded Package (Pre-trained on 4000)",
            "Model": clean_model_name,
            "Training Split": "4,000 External -> 120 Zero-Shot Test",
            "Feature Dimensions": 61,
            "Feature Extraction Latency (ms)": avg_feat_time_ms,
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "ROC-AUC": row["roc_auc"]
        })

    # Uploaded package retrained models on 120 Test
    df_pkg_retrained = df_eval[df_eval["condition"] == "Retrained on Pilot (120 Test)"]
    for _, row in df_pkg_retrained.iterrows():
        clean_model_name = row["model"].replace(" (Retrained 61-D)", "")
        comparison_records.append({
            "Framework": "Uploaded Package 61-D (Retrained)",
            "Model": clean_model_name,
            "Training Split": "280 Train / 120 Test",
            "Feature Dimensions": 61,
            "Feature Extraction Latency (ms)": avg_feat_time_ms,
            "Accuracy": row["accuracy"],
            "Precision": row["precision"],
            "Recall": row["recall"],
            "F1": row["f1"],
            "ROC-AUC": row["roc_auc"]
        })

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "comparison_pilot_vs_package.csv"), index=False)

    print("\n" + "=" * 65)
    print("FINAL HEAD-TO-HEAD COMPARISON")
    print("=" * 65)
    print(df_comp[["Framework", "Model", "Feature Dimensions", "Feature Extraction Latency (ms)", "Accuracy", "F1", "ROC-AUC"]].to_string(index=False))

    # 5. Generate Comparison Visual Plot
    plt.figure(figsize=(12, 6))
    
    # Filter for comparable models
    plot_df = df_comp[df_comp["Model"].isin(["Logistic Regression", "Random Forest", "HistGradientBoosting", "Linear SVM"])]
    labels = [f"{r['Model']}\n({r['Framework'][:18]}...)" for _, r in plot_df.iterrows()]
    x = np.arange(len(plot_df))
    
    colors = []
    for _, r in plot_df.iterrows():
        if "Our" in r["Framework"]:
            colors.append("#2b5c8f")
        elif "Pre-trained" in r["Framework"]:
            colors.append("#2ca02c")
        else:
            colors.append("#d62728")

    bars = plt.bar(x, plot_df["Accuracy"], color=colors, width=0.6, edgecolor="black", alpha=0.85)
    plt.axhline(0.50, color="gray", linestyle="--", label="Chance Baseline (50%)")
    plt.ylabel("Accuracy on Pilot Test Set")
    plt.ylim([0.4, 0.85])
    plt.title("Head-to-Head Accuracy Comparison on 120-Image Pilot Test Set")
    plt.xticks(x, labels, rotation=35, ha="right", fontsize=8.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color="#2b5c8f", label="Our Pilot Pipeline"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Package Pre-trained (4k train)"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Package 61-D Retrained on Pilot")
    ]
    plt.legend(handles=legend_patches, loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "comparison_plot.png"), dpi=200)
    plt.close()

    # 6. Save JSON Summary
    summary_data = {
        "dataset_evaluated": {
            "name": "TheKernel01/140k-Real-and-Fake-Faces (Pilot)",
            "images_tested": 400,
            "real": 200,
            "fake": 200,
            "test_split_size": 120
        },
        "feature_extraction_package": {
            "num_features": 61,
            "avg_latency_ms": avg_feat_time_ms
        },
        "comparison_table": comparison_records
    }
    with open(os.path.join(RESULTS_DIR, "package_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    print("\nBenchmark and Comparison completed successfully!")
    print("Files created:")
    print("  - results/package_evaluation_results.csv")
    print("  - results/comparison_pilot_vs_package.csv")
    print("  - results/comparison_plot.png")
    print("  - results/package_summary.json")

if __name__ == "__main__":
    run_package_benchmark()
