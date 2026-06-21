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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix, RocCurveDisplay
)
from diagnose_leakage import leakage_report

TARGET_COL = "F3924"

# Allowlist for features that are strong but verified as non-leakage
SAFE_FEATURES = []

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


def basic_clean(df: pd.DataFrame, medians: pd.Series = None, drop_index: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Drop columns that are entirely empty, drop index column if requested, fill remaining NAs with median."""
    df = df.copy()

    if drop_index and "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if medians is None:
        empty_cols = df.columns[df.isna().all()]
        if len(empty_cols) > 0:
            df = df.drop(columns=empty_cols)
    else:
        expected_cols = list(medians.index)
        if TARGET_COL in df.columns:
            expected_cols = expected_cols + [TARGET_COL]
        df = df.reindex(columns=expected_cols)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if medians is None:
        medians = df[feature_cols].median()

    df[feature_cols] = df[feature_cols].fillna(medians)
    return df, medians


def get_log_transform_cols(X_train: pd.DataFrame, skew_thresh: float = 5.0) -> dict:
    """Identify highly skewed columns on the training set only (no leakage)."""
    transforms = {}
    for col in X_train.columns:
        if col == TARGET_COL:
            continue
        try:
            x = pd.to_numeric(X_train[col], errors="coerce")
            if x.isna().all() or x.nunique() <= 5:
                continue
            x_filled = x.fillna(x.median())
            skew_val = x_filled.skew()
            if abs(skew_val) > skew_thresh:
                has_negative = bool((x_filled < 0).any())
                transforms[col] = {"skew_before": float(skew_val), "has_negative": has_negative}
        except Exception:
            pass
    return transforms


def apply_log_transforms(df: pd.DataFrame, transforms: dict) -> pd.DataFrame:
    """Apply log1p or signed log transform consistently (same list as training)."""
    df = df.copy()
    for col, info in transforms.items():
        if col not in df.columns:
            continue
        try:
            x = pd.to_numeric(df[col], errors="coerce").fillna(0)
            if info["has_negative"]:
                df[col] = np.sign(x) * np.log1p(np.abs(x))
            else:
                df[col] = np.log1p(np.maximum(x, 0))
        except Exception:
            pass
    return df


def engineer_features(df: pd.DataFrame, train_stats: dict = None) -> pd.DataFrame:
    """
    Add a small set of behavioral features on top of the hinted columns.
    Uses train_stats for z-score normalization and applies log-transforms if stored.
    """
    df = df.copy()

    if train_stats is None:
        stats_path = os.path.join(OUT_DIR, "train_stats.json")
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "r") as f:
                    train_stats = json.load(f)
            except Exception:
                pass

    # Apply log-transforms stored from training (for inference/scoring compatibility)
    if train_stats is not None and "log_transforms" in train_stats:
        df = apply_log_transforms(df, train_stats["log_transforms"])

    available_hints = [c for c in HINT_FEATURES if c in df.columns]

    if len(available_hints) >= 2:
        a, b = available_hints[0], available_hints[1]
        df[f"ratio_{a}_{b}"] = df[a] / (df[b].abs() + 1e-6)

    if len(available_hints) >= 1:
        col = available_hints[0]
        mean_val = train_stats.get(f"{col}_mean") if train_stats else None
        std_val = train_stats.get(f"{col}_std") if train_stats else None
        if mean_val is None or std_val is None:
            mean_val = df[col].mean()
            std_val = df[col].std()
        df[f"zscore_{col}"] = (df[col] - mean_val) / (std_val + 1e-6)

    if available_hints:
        df["hint_features_sum"] = df[available_hints].sum(axis=1)
        df["hint_features_mean"] = df[available_hints].mean(axis=1)
        df["hint_features_std"] = df[available_hints].std(axis=1)

    return df


def train_model(X_train, y_train, X_val, y_val):
    pos = int(y_train.sum())
    neg = len(y_train) - pos
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        # Reduced from 0.8 → 0.4: halves per-tree column sampling,
        # cutting fit time ~2× on 3800-wide sparse data
        colsample_bytree=0.4,
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate(model, X_val, y_val, decision_threshold: float = 0.5):
    proba = model.predict_proba(X_val)[:, 1]
    preds = (proba >= decision_threshold).astype(int)

    auc = roc_auc_score(y_val, proba)
    ap = average_precision_score(y_val, proba)
    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds).tolist()

    return {
        "roc_auc": auc,
        "average_precision": ap,
        "classification_report": report,
        "confusion_matrix": cm,
    }, proba


def plot_roc(y_val, proba, out_path):
    RocCurveDisplay.from_predictions(y_val, proba)
    plt.title("ROC Curve — SuSpy Account Classifier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pr(y_val, proba, out_path):
    precision, recall, _ = precision_recall_curve(y_val, proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — SuSpy Account Classifier")
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


def _prep_fold(X_train_raw, X_val_raw, y_train, apply_log: bool = True):
    """Clean, optionally log-transform, and engineer features for a single fold."""
    X_train_c, medians = basic_clean(X_train_raw, drop_index=False)
    X_val_c, _ = basic_clean(X_val_raw, medians=medians, drop_index=False)

    transforms = {}
    if apply_log:
        transforms = get_log_transform_cols(X_train_c, skew_thresh=5.0)
        X_train_c = apply_log_transforms(X_train_c, transforms)
        X_val_c = apply_log_transforms(X_val_c, transforms)

    available_hints = [c for c in HINT_FEATURES if c in X_train_c.columns]
    train_stats = {}
    if available_hints:
        col = available_hints[0]
        train_stats[f"{col}_mean"] = float(X_train_c[col].mean())
        train_stats[f"{col}_std"] = float(X_train_c[col].std())

    X_tr = engineer_features(X_train_c, train_stats)
    X_vl = engineer_features(X_val_c, train_stats)
    return X_tr, X_vl, transforms


def run_cv(df: pd.DataFrame, exclude_features: list, n_splits: int = 5,
           apply_log: bool = True) -> dict:
    """
    Stratified k-fold CV.
    Threshold sweep is done POST-HOC from raw probabilities — trains each fold
    only ONCE, then evaluates every threshold without re-fitting.
    """
    df_clean = df.drop(columns=[c for c in exclude_features if c in df.columns], errors="ignore")
    y = df_clean[TARGET_COL].astype(int).values
    X = df_clean.drop(columns=[TARGET_COL])

    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Collect per-fold raw probabilities and labels (one model fit per fold)
    fold_probas = []
    fold_labels = []

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold+1}/{n_splits} ...", flush=True)
        X_tr_raw, y_tr = X.iloc[tr_idx].copy(), y[tr_idx]
        X_vl_raw, y_vl = X.iloc[vl_idx].copy(), y[vl_idx]

        X_tr, X_vl, _ = _prep_fold(X_tr_raw, X_vl_raw, y_tr, apply_log=apply_log)
        model = train_model(X_tr, y_tr, X_vl, y_vl)
        proba = model.predict_proba(X_vl)[:, 1]
        fold_probas.append(proba)
        fold_labels.append(y_vl)

    # Evaluate all thresholds post-hoc (no extra training)
    sweep_results = {}
    for t in thresholds:
        t_metrics = []
        for proba, y_vl in zip(fold_probas, fold_labels):
            preds = (proba >= t).astype(int)
            rep = classification_report(y_vl, preds, output_dict=True, zero_division=0)
            cm = confusion_matrix(y_vl, preds).tolist()
            t_metrics.append({
                "roc_auc": float(roc_auc_score(y_vl, proba)),
                "average_precision": float(average_precision_score(y_vl, proba)),
                "precision": rep["1"]["precision"],
                "recall": rep["1"]["recall"],
                "f1_score": rep["1"]["f1-score"],
                "confusion_matrix": cm,
            })
        sweep_results[str(t)] = t_metrics

    # Build summary for each threshold
    sweep_summary = []
    for t in thresholds:
        ms = sweep_results[str(t)]
        sweep_summary.append({
            "threshold": t,
            "mean_roc_auc": float(np.mean([m["roc_auc"] for m in ms])),
            "mean_precision": float(np.mean([m["precision"] for m in ms])),
            "std_precision": float(np.std([m["precision"] for m in ms])),
            "mean_recall": float(np.mean([m["recall"] for m in ms])),
            "std_recall": float(np.std([m["recall"] for m in ms])),
            "mean_f1_score": float(np.mean([m["f1_score"] for m in ms])),
            "std_f1_score": float(np.std([m["f1_score"] for m in ms])),
        })

    # Main CV summary at threshold=0.5 for backward compat
    def _summary(t):
        ms = sweep_results[str(t)]
        fold_details = []
        for i, (m, y_vl) in enumerate(zip(ms, fold_labels)):
            fold_details.append({
                "fold": i + 1,
                "roc_auc": m["roc_auc"],
                "average_precision": m["average_precision"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "confusion_matrix": m["confusion_matrix"],
            })
        aucs = [m["roc_auc"] for m in ms]
        aps = [m["average_precision"] for m in ms]
        precs = [m["precision"] for m in ms]
        recs = [m["recall"] for m in ms]
        f1s = [m["f1_score"] for m in ms]
        return {
            "threshold": t,
            "mean_roc_auc": float(np.mean(aucs)), "std_roc_auc": float(np.std(aucs)),
            "mean_average_precision": float(np.mean(aps)), "std_average_precision": float(np.std(aps)),
            "mean_precision": float(np.mean(precs)), "std_precision": float(np.std(precs)),
            "mean_recall": float(np.mean(recs)), "std_recall": float(np.std(recs)),
            "mean_f1_score": float(np.mean(f1s)), "std_f1_score": float(np.std(f1s)),
            "folds": fold_details,
        }

    return {
        "cv_default_threshold": _summary(0.50),
        "sweep_summary": sweep_summary,
    }


def run_pipeline(df: pd.DataFrame, exclude_features: list,
                 save_artifacts: bool = False, decision_threshold: float = 0.5):
    """Single 80/20 split pipeline for final model training and artifact generation."""
    df_clean, medians = basic_clean(df, drop_index=False)
    df_clean = df_clean.drop(columns=[c for c in exclude_features if c in df_clean.columns], errors="ignore")

    y = df_clean[TARGET_COL].astype(int)
    X = df_clean.drop(columns=[TARGET_COL])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    transforms = get_log_transform_cols(X_train, skew_thresh=5.0)

    # F1706 diagnostic print (only when saving final artifacts)
    if "F1706" in transforms and save_artifacts:
        print("\n=== F1706 Log Transform Check ===")
        print(f"  Skew BEFORE: {X_train['F1706'].skew():.3f}")
        tmp = X_train.copy(); tmp[TARGET_COL] = y_train
        print(tmp.groupby(TARGET_COL)["F1706"].describe())

    X_train_t = apply_log_transforms(X_train, transforms)
    X_val_t = apply_log_transforms(X_val, transforms)

    if "F1706" in transforms and save_artifacts:
        print(f"  Skew AFTER:  {X_train_t['F1706'].skew():.3f}")
        tmp2 = X_train_t.copy(); tmp2[TARGET_COL] = y_train
        print(tmp2.groupby(TARGET_COL)["F1706"].describe())
        print("=================================\n")

    if save_artifacts:
        transforms_record = {
            col: {"skew_before": info["skew_before"],
                  "skew_after": float(X_train_t[col].skew()),
                  "has_negative": info["has_negative"]}
            for col, info in transforms.items()
        }
        with open(os.path.join(OUT_DIR, "log_transformed_features.json"), "w") as f:
            json.dump(transforms_record, f, indent=2)

    available_hints = [c for c in HINT_FEATURES if c in X_train_t.columns]
    train_stats: dict = {}
    if available_hints:
        col = available_hints[0]
        train_stats[f"{col}_mean"] = float(X_train_t[col].mean())
        train_stats[f"{col}_std"] = float(X_train_t[col].std())

    if save_artifacts:
        train_stats["log_transforms"] = transforms
        with open(os.path.join(OUT_DIR, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=2)

    X_tr_feat = engineer_features(X_train_t, train_stats)
    X_vl_feat = engineer_features(X_val_t, train_stats)

    if save_artifacts:
        medians.to_json(os.path.join(OUT_DIR, "medians.json"))
        with open(os.path.join(OUT_DIR, "feature_names.json"), "w") as f:
            json.dump(list(X_tr_feat.columns), f, indent=2)

    model = train_model(X_tr_feat, y_train, X_vl_feat, y_val)
    metrics, proba = evaluate(model, X_vl_feat, y_val, decision_threshold)

    if save_artifacts:
        plot_roc(y_val, proba, os.path.join(OUT_DIR, "roc_curve.png"))
        plot_pr(y_val, proba, os.path.join(OUT_DIR, "pr_curve.png"))
        plot_feature_importance(model, list(X_tr_feat.columns), os.path.join(OUT_DIR, "feature_importance.png"))
        sample_n = min(500, len(X_vl_feat))
        plot_shap_summary(model, X_vl_feat.sample(sample_n, random_state=42),
                          os.path.join(OUT_DIR, "shap_summary.png"))
        model.save_model(os.path.join(OUT_DIR, "suspy_classifier.json"))

    return metrics


def main(data_path: str, sample_for_shap: int = 500):
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data(data_path)

    # ── 1. Leakage diagnosis ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("DIAGNOSING TARGET LEAKAGE")
    print("="*55)
    report = leakage_report(df, target_col=TARGET_COL)
    suspects = report[report["suspect"]]["feature"].tolist()

    exclude_features, dropped_reasons = [], {}
    for s in suspects:
        if s in SAFE_FEATURES:
            print(f"  [KEEP] {s} is in SAFE_FEATURES")
        else:
            exclude_features.append(s)
            dropped_reasons[s] = {
                "abs_corr": float(report.loc[report["feature"] == s, "abs_corr"].values[0]),
                "single_feature_auc": float(report.loc[report["feature"] == s, "single_feature_auc"].values[0]),
                "reason": "Flagged as target leakage suspect (|corr| > 0.85 or AUC > 0.97)"
            }
            print(f"  [DROP] {s}")

    with open(os.path.join(OUT_DIR, "dropped_features.json"), "w") as f:
        json.dump(dropped_reasons, f, indent=2)

    # Note: We skip a baseline-leakage CV run because leakage was already
    # confirmed during diagnosis (ROC-AUC = 1.0 with Unnamed:0 + F3912).
    # Rerunning it with leakage-in is O(5 × 3923-col XGBoost) — too slow
    # and not informative for the submission.
    cv_metrics_before_note = {
        "note": "Baseline (leakage-in) CV skipped — leakage already confirmed by diagnose_leakage.py",
        "known_leaked_features": exclude_features,
        "expected_roc_auc": "~1.0 (confirmed in prior session)",
    }
    with open(os.path.join(OUT_DIR, "cv_metrics_before.json"), "w") as f:
        json.dump(cv_metrics_before_note, f, indent=2)

    # ── 2. Fixed CV — leakage removed, log-transform ON, all thresholds ───────
    print("\n" + "="*55)
    print("FIXED 5-FOLD CV — leakage removed + log-transform (all thresholds)")
    print("="*55)
    cv_fixed = run_cv(df, exclude_features=exclude_features, n_splits=5, apply_log=True)
    f_def = cv_fixed["cv_default_threshold"]
    print(f"  ROC-AUC:  {f_def['mean_roc_auc']:.4f} ± {f_def['std_roc_auc']:.4f}")
    print(f"  Recall:   {f_def['mean_recall']:.4f} ± {f_def['std_recall']:.4f}")

    print("\n  Threshold sweep (post-hoc, no extra training):")
    print("  Threshold | Precision | Recall | F1")
    print("  ----------|-----------|--------|----")
    for s in cv_fixed["sweep_summary"]:
        print(f"    {s['threshold']:.2f}    |  {s['mean_precision']:.4f}   | {s['mean_recall']:.4f} | {s['mean_f1_score']:.4f}"
              f"  (±prec {s['std_precision']:.3f} / ±rec {s['std_recall']:.3f})")

    with open(os.path.join(OUT_DIR, "threshold_sweep.json"), "w") as f:
        json.dump(cv_fixed["sweep_summary"], f, indent=2)

    # Pick optimal threshold (max F1)
    best = max(cv_fixed["sweep_summary"], key=lambda x: x["mean_f1_score"])
    opt_threshold = best["threshold"]
    print(f"\n  Optimal threshold (max F1): {opt_threshold:.2f}")

    # Save per-fold breakdown (from the fixed CV run, at default threshold 0.5)
    opt_fold_details = cv_fixed["cv_default_threshold"]["folds"]
    with open(os.path.join(OUT_DIR, "cv_fold_breakdown.json"), "w") as f:
        json.dump(opt_fold_details, f, indent=2)

    # Save main cv_metrics as the fixed-CV at default threshold
    with open(os.path.join(OUT_DIR, "cv_metrics.json"), "w") as f:
        json.dump(f_def, f, indent=2)

    # ── 3. Final model training ───────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"TRAINING FINAL MODEL (threshold={opt_threshold:.2f})")
    print("="*55)
    metrics_after = run_pipeline(df, exclude_features=exclude_features,
                                 save_artifacts=True, decision_threshold=opt_threshold)

    for path, data in [
        ("metrics_after_leakage_fix.json", metrics_after),
        ("metrics.json", metrics_after),
    ]:
        with open(os.path.join(OUT_DIR, path), "w") as f:
            json.dump(data, f, indent=2)

    # ── 4. Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FINAL SUMMARY")
    print("="*55)
    cv_f = f_def
    opt_sw = best

    print(f"{'Metric':<28} | {'Fixed CV (thr=0.5)':>20} | {'Optimal (thr={:.2f})'.format(opt_threshold):>20}")
    print("-" * 75)
    print(f"{'CV ROC-AUC':<28} | {cv_f['mean_roc_auc']:>8.4f} ± {cv_f['std_roc_auc']:.4f}        | {opt_sw['mean_roc_auc']:>8.4f}")
    print(f"{'CV Precision':<28} | {cv_f['mean_precision']:>8.4f} ± {cv_f['std_precision']:.4f}        | {opt_sw['mean_precision']:>8.4f} ± {opt_sw['std_precision']:.4f}")
    print(f"{'CV Recall':<28} | {cv_f['mean_recall']:>8.4f} ± {cv_f['std_recall']:.4f}        | {opt_sw['mean_recall']:>8.4f} ± {opt_sw['std_recall']:.4f}")
    print(f"{'CV F1-Score':<28} | {cv_f['mean_f1_score']:>8.4f} ± {cv_f['std_f1_score']:.4f}        | {opt_sw['mean_f1_score']:>8.4f} ± {opt_sw['std_f1_score']:.4f}")
    try:
        with open(os.path.join(OUT_DIR, "log_transformed_features.json"), "r") as f:
            log_cols_count = len(json.load(f))
    except Exception:
        log_cols_count = 0
    print(f"\nLeakage dropped: {exclude_features}")
    print(f"Log-transformed features: {log_cols_count} columns with skew > 5")
    print(f"\nAll outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SuSpy MVP classifier")
    parser.add_argument("--data", required=True, help="Path to dataset CSV")
    parser.add_argument("--shap-sample", type=int, default=500,
                        help="Number of validation rows to use for SHAP plot")
    args = parser.parse_args()
    main(args.data, args.shap_sample)
