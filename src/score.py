"""
MuleNet MVP — Risk scoring / inference

Loads a trained XGBoost model and scores new accounts, producing
a 0-100 Dynamic Risk Index (DRI) and risk tier per account.

Usage:
    python src/score.py --data data/new_accounts.csv --model outputs/mule_classifier.json
"""

import argparse
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
    df_clean = basic_clean(df)
    df_feat = engineer_features(df_clean)

    X = df_feat.drop(columns=[TARGET_COL], errors="ignore")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    proba = model.predict_proba(X)[:, 1]
    dri = (proba * 100).round(2)

    results = pd.DataFrame({
        "account_index": df.index,
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
