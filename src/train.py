"""
SuSpy MVP — Mule Account Classifier
Phase 1 hackathon proof-of-concept.

Usage:
    python src/train.py --data data/your_dataset.csv

This script:
1. Loads the transaction dataset (target = F3924)
2. Cleans data and engineers a handful of behavioral features
3. Trains an XGBoost classifier with SHAP explainability
4. Saves model, metrics, feature importance plot, and SHAP plot to outputs/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, RocCurveDisplay
)

TARGET_COL = "F3924"

# Features highlighted by the bank as commonly used for fraud detection.
HINT_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
    "F3889", "F3891", "F3894",
]

OUT_DIR = "outputs"


def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column {TARGET_COL} not found in dataset.")
    return df


def basic_clean(df: pd.DataFrame, medians: pd.Series = None) -> tuple[pd.DataFrame, pd.Series]:
    """Drop columns that are entirely empty, drop index column, fill remaining NAs with median."""
    df = df.copy()

    # Drop index column if present (leakage fix)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if medians is None:
        # Drop fully empty columns during training
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            print(f"Dropping {len(empty_cols)} fully-empty columns")
            df = df.drop(columns=empty_cols)
    else:
        # Align columns to what medians has
        expected_cols = list(medians.index)
        if TARGET_COL in df.columns:
            expected_cols = expected_cols + [TARGET_COL]
        df = df.reindex(columns=expected_cols)

    # Coerce everything except target to numeric where possible
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill remaining NAs with column median
    if medians is None:
        medians = df[feature_cols].median()

    df[feature_cols] = df[feature_cols].fillna(medians)

    return df, medians


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small set of behavioral features on top of the hinted columns.
    These are intentionally generic so they work even if exact column
    semantics differ — adjust based on your actual data dictionary.
    """
    df = df.copy()
    available_hints = [c for c in HINT_FEATURES if c in df.columns]
    print(f"Found {len(available_hints)}/{len(HINT_FEATURES)} hint features in dataset")

    if len(available_hints) >= 2:
        # Ratio-style feature between the first two hinted features
        # (placeholder logic — replace with domain-meaningful pairs once
        # you know what each F-column actually represents)
        a, b = available_hints[0], available_hints[1]
        df[f"ratio_{a}_{b}"] = df[a] / (df[b].abs() + 1e-6)

    if len(available_hints) >= 1:
        col = available_hints[0]
        # Z-score style outlier signal
        df[f"zscore_{col}"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    # Generic aggregate signals across all hinted features
    if available_hints:
        df["hint_features_sum"] = df[available_hints].sum(axis=1)
        df["hint_features_mean"] = df[available_hints].mean(axis=1)
        df["hint_features_std"] = df[available_hints].std(axis=1)

    return df


def train_model(X_train, y_train, X_val, y_val):
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    print(f"Class balance -> positives: {pos}, negatives: {neg}, "
          f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def evaluate(model, X_val, y_val):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, proba)
    ap = average_precision_score(y_val, proba)
    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds).tolist()

    metrics = {
        "roc_auc": auc,
        "average_precision": ap,
        "classification_report": report,
        "confusion_matrix": cm,
    }
    print(f"ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")
    print(classification_report(y_val, preds))
    return metrics, proba


def plot_roc(y_val, proba, out_path):
    RocCurveDisplay.from_predictions(y_val, proba)
    plt.title("ROC Curve — Mule Account Classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pr(y_val, proba, out_path):
    precision, recall, _ = precision_recall_curve(y_val, proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Mule Account Classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names, out_path, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in idx][::-1], importances[idx][::-1])
    plt.title(f"Top {top_n} Feature Importances (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_shap_summary(model, X_sample, out_path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main(data_path: str, sample_for_shap: int = 500):
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(data_path)
    df, medians = basic_clean(df)
    df = engineer_features(df)

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Save medians and feature schema for scoring
    medians.to_json(os.path.join(OUT_DIR, "medians.json"))
    feature_names = list(X.columns)
    with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_model(X_train, y_train, X_val, y_val)

    metrics, proba = evaluate(model, X_val, y_val)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_roc(y_val, proba, os.path.join(OUT_DIR, "roc_curve.png"))
    plot_pr(y_val, proba, os.path.join(OUT_DIR, "pr_curve.png"))
    plot_feature_importance(model, list(X.columns), os.path.join(OUT_DIR, "feature_importance.png"))

    # SHAP on a sample for speed
    sample_n = min(sample_for_shap, len(X_val))
    plot_shap_summary(model, X_val.sample(sample_n, random_state=42),
                       os.path.join(OUT_DIR, "shap_summary.png"))

    model.save_model(os.path.join(OUT_DIR, "mule_classifier.json"))

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("  - metrics.json")
    print("  - roc_curve.png")
    print("  - pr_curve.png")
    print("  - feature_importance.png")
    print("  - shap_summary.png")
    print("  - mule_classifier.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuSpy MVP classifier")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--shap-sample", type=int, default=500,
                         help="Number of validation rows to use for SHAP plot")
    args = parser.parse_args()

    main(args.data, args.shap_sample)
