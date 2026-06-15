# MuleNet — Mule Account Classification (Phase 1 MVP)

A proof-of-concept for **Problem Statement 2: AI/ML-Based Classification of Suspicious Mule Accounts**.

This MVP implements **Layer 1 and Layer 2a** of the full MuleNet architecture: feature
engineering + an XGBoost classifier with SHAP-based explainability, trained on the
provided transaction dataset (target variable `F3924`).

> The full MuleNet vision additionally includes a Graph Neural Network (GraphSAGE)
> layer for network-level mule ring detection and a dynamic, time-decaying risk score.
> This MVP focuses on the foundational classifier to demonstrate feasibility within
> the hackathon timeline — the GNN layer is the proposed next phase (see `docs/roadmap.md`).

## What this MVP does

1. **Loads** the provided transaction dataset (3924 features, target = `F3924`)
2. **Cleans** the data (handles missing values, coerces types)
3. **Engineers features** on top of the bank-highlighted columns
   (`F115, F321, F527, F531, F670, F1692, F2082, F2122, F2582, F2678, F2737,
   F2956, F3043, F3836, F3887, F3889, F3891, F3894`)
4. **Trains** an XGBoost classifier with class-imbalance handling
5. **Evaluates** with ROC-AUC, average precision, confusion matrix
6. **Explains** predictions with SHAP summary plots
7. **Scores** new accounts and assigns a 0–100 Dynamic Risk Index (DRI) with
   risk tiers (Low / Medium / High / Critical)

## Project structure

```
mulenet-mvp/
├── data/                 # place your dataset CSV here (not committed)
├── notebooks/
│   └── exploration.ipynb # EDA + walkthrough notebook
├── src/
│   ├── train.py          # training pipeline
│   └── score.py          # inference / risk scoring
├── outputs/               # generated metrics, plots, model artifact
├── docs/
│   └── roadmap.md        # GNN layer & future phases
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your dataset at data/dataset.csv (must contain column F3924)

# 3. Train the model
python src/train.py --data data/dataset.csv

# 4. Score new/unseen accounts
python src/score.py --data data/dataset.csv --model outputs/mule_classifier.json
```

## Outputs

After training, `outputs/` contains:

- `metrics.json` — ROC-AUC, average precision, classification report, confusion matrix
- `roc_curve.png` / `pr_curve.png` — performance curves
- `feature_importance.png` — top 20 most important features
- `shap_summary.png` — SHAP explainability summary across the validation set
- `mule_classifier.json` — trained XGBoost model
- `risk_scores.csv` (after running `score.py`) — per-account DRI and risk tier

## Risk tiers

| DRI range | Tier     | Recommended action                          |
|-----------|----------|----------------------------------------------|
| 0–30      | Low      | Normal monitoring                            |
| 31–60     | Medium   | Enhanced transaction monitoring              |
| 61–80     | High     | Investigator alert with SHAP explanation     |
| 81–100    | Critical | Immediate alert + fraud network review       |

## Next phases (post-hackathon)

See [`docs/roadmap.md`](docs/roadmap.md) for the full MuleNet vision, including
the GraphSAGE-based network intelligence layer and the dynamic, time-decaying
risk scoring engine.
