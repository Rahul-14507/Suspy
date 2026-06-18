import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.dirname(__file__))
from train import basic_clean, engineer_features

TARGET_COL = "F3924"


def risk_tier(score: float) -> str:
    if score <= 30:
        return "Low"
    elif score <= 60:
        return "Medium"
    elif score <= 80:
        return "High"
    else:
        return "Critical"


def main(data_path: str, model_path: str, out_path: str):
    df = pd.read_csv(data_path)

    # Load medians and feature names from the model's directory
    model_dir = os.path.dirname(model_path) or "outputs"
    medians_path = os.path.join(model_dir, "medians.json")
    feature_names_path = os.path.join(model_dir, "feature_names.json")

    if os.path.exists(medians_path):
        medians = pd.read_json(medians_path, typ="series")
    else:
        medians = None

    df_clean, _ = basic_clean(df, medians=medians)
    df_feat = engineer_features(df_clean)

    if os.path.exists(feature_names_path):
        with open(feature_names_path, "r") as f:
            feature_names = json.load(f)
        X = df_feat.reindex(columns=feature_names)
    else:
        X = df_feat.drop(columns=[TARGET_COL], errors="ignore")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    proba = model.predict_proba(X)[:, 1]
    dri = (proba * 100).round(2)

    acc_idx = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else df.index
    results = pd.DataFrame({
        "account_index": acc_idx,
        "risk_probability": proba,
        "DRI": dri,
        "risk_tier": [risk_tier(s) for s in dri],
    })

    results = results.sort_values("DRI", ascending=False)
    results.to_csv(out_path, index=False)

    print(f"Scored {len(results)} accounts")
    print(results["risk_tier"].value_counts())
    print(f"\nSaved results to {out_path}")
    print("\nTop 10 highest-risk accounts:")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score accounts with trained MuleNet model")
    parser.add_argument("--data", required=True, help="Path to dataset CSV (with or without target)")
    parser.add_argument("--model", default="outputs/mule_classifier.json", help="Path to trained model")
    parser.add_argument("--out", default="outputs/risk_scores.csv", help="Output CSV path")
    args = parser.parse_args()

    main(args.data, args.model, args.out)
