import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def leakage_report(df, target_col, exclude_cols=None, corr_thresh=0.85, auc_thresh=0.97):
    exclude_cols = exclude_cols or []
    results = []
    y = df[target_col].astype(int)
    for col in df.columns:
        if col == target_col or col in exclude_cols:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.isna().all() or x.nunique() <= 1:
            continue
        x_filled = x.fillna(x.median())
        corr = x_filled.corr(y)
        try:
            auc = roc_auc_score(y, x_filled)
            auc = max(auc, 1 - auc)  # direction-agnostic
        except ValueError:
            auc = np.nan
        results.append({
            "feature": col,
            "abs_corr": abs(corr),
            "single_feature_auc": auc,
            "suspect": (abs(corr) > corr_thresh) or (auc > auc_thresh),
        })
    report = pd.DataFrame(results).sort_values("single_feature_auc", ascending=False)
    return report

if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    report = leakage_report(df, target_col="F3924")
    print(report.head(20).to_string(index=False))
    suspects = report[report["suspect"]]
    print(f"\n{len(suspects)} suspect feature(s) flagged for leakage review:")
    print(suspects["feature"].tolist())
    import os
    os.makedirs("outputs", exist_ok=True)
    report.to_csv("outputs/leakage_report.csv", index=False)
