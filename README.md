# SuSpy — AI/ML Mule Account Classifier

> **Hackathon Submission** · Problem Statement 2: AI/ML-Based Classification of Suspicious Mule Accounts

---

![Stats Banner](docs/stats_banner.png)

---

## Overview

Financial mule accounts are the conduit layer of money-laundering operations — controlled by criminals to receive and forward illicit funds while appearing legitimate on the surface. Detecting them requires separating a handful of bad actors from tens of thousands of normal accounts using only anonymised behavioural features.

**SuSpy** is a layered machine-learning system designed to do exactly that. This submission delivers **Layers 1 and 2a** of the full vision: a production-ready feature engineering and XGBoost classification pipeline with SHAP-based explainability and a 0–100 **Dynamic Risk Index (DRI)** output. The full roadmap through a Graph Neural Network layer and an investigator dashboard is documented in [`docs/roadmap.md`](docs/roadmap.md).

---

## Architecture

![Architecture](docs/architecture.png)

| Layer | Component | Status |
|---|---|---|
| **Layer 1** | Feature Engineering (3,924 features + engineered behavioural signals) | ✅ MVP |
| **Layer 2a** | XGBoost Classifier with SHAP explainability | ✅ MVP |
| **Layer 2b** | GraphSAGE GNN for network-level mule ring detection | 🗺️ Roadmap |
| **Layer 3** | DRI Fusion Meta-Learner with temporal decay | 🗺️ Roadmap |
| **Future** | Investigator Dashboard (SHAP waterfall + fraud subgraph) | 🗺️ Roadmap |

---

## Dataset & Class Imbalance

![Class Imbalance](docs/class_imbalance.png)

The dataset presents a **111:1 class imbalance** — a realistic and severe challenge. SuSpy handles this via XGBoost's `scale_pos_weight` parameter (set to **110.8×**), which ensures the model is appropriately penalised for missing the rare positive class.

| Metric | Value |
|---|---|
| Total accounts | 9,082 |
| Clean accounts (F3924 = 0) | 9,001 (99.11%) |
| Mule accounts (F3924 = 1) | 81 (0.89%) |
| Features (raw) | 3,924 |
| Features (after cleaning) | 3,861 (63 fully-empty cols dropped) |

---

## Model Performance

![ROC Curve](docs/roc_curve.png)

| Metric | Value |
|---|---|
| **ROC-AUC** | **1.0000** |
| **Average Precision** | **1.0000** |
| **Recall (mule class)** | **1.00** (0 missed mule accounts) |
| **Precision (mule class)** | **1.00** (0 false alarms) |
| **Validation set size** | 1,817 accounts (20% stratified split) |

The confusion matrix on the held-out validation set:

```
Predicted →       Clean   Mule
Actual Clean   [ 1801      0  ]
Actual Mule    [    0     16  ]
```

**No false negatives.** Every mule account in the validation set was correctly flagged.

### Precision-Recall Curve

![PR Curve](docs/pr_curve.png)

---

## Explainability — SHAP

A core requirement for compliance teams is **explainability**: investigators need to understand *why* an account was flagged. SuSpy integrates SHAP (SHapley Additive exPlanations) directly into the training pipeline.

![SHAP Summary](docs/shap_summary.png)

Each flagged account can be accompanied by a SHAP waterfall plot showing the exact feature contributions that drove its risk score — ready for an investigator's case report.

---

## Feature Importance

![Top Features](docs/top_features.png)

The top 15 features by information gain are shown above. The model strongly leverages features in the `F3900`-range alongside several mid-range features (`F1382`, `F1603`, `F234`). All 18 bank-highlighted hint features were present in the dataset and were used by the model.

---

## Dynamic Risk Index (DRI)

Every account receives a **DRI score from 0 to 100** derived from the classifier's probability output, enabling risk-ranked prioritisation of investigative resources.

![DRI Distribution](docs/dri_distribution.png)

| DRI Range | Tier | Count | Recommended Action |
|---|---|---|---|
| 0 – 30 | 🟢 **Low** | 9,001 | Normal monitoring |
| 31 – 60 | 🔵 **Medium** | 0 | Enhanced transaction monitoring |
| 61 – 80 | 🟡 **High** | 0 | Investigator alert + SHAP explanation |
| 81 – 100 | 🔴 **Critical** | 81 | Immediate escalation + fraud network review |

The bimodal distribution (9,001 accounts at DRI ≈ 0, 81 accounts at DRI = 100) reflects the model's high confidence — consistent with the clean decision boundary observed in the ROC curve.

---

## Project Structure

```
SuSpy/
├── data/                          # Dataset directory (CSV not committed)
├── docs/                          # Report assets, roadmap, charts
│   ├── architecture.png
│   ├── stats_banner.png
│   ├── dri_distribution.png
│   ├── class_imbalance.png
│   ├── top_features.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── shap_summary.png
│   └── roadmap.md
├── notebooks/
│   └── exploration.ipynb          # EDA walkthrough
├── outputs/                       # Generated model artifacts
│   ├── mule_classifier.json       # Trained XGBoost model
│   ├── medians.json               # Training imputation medians (for inference)
│   ├── feature_names.json         # Training feature schema (for inference alignment)
│   ├── metrics.json               # Evaluation metrics
│   ├── risk_scores.csv            # Per-account DRI scores
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   └── shap_summary.png
├── src/
│   ├── train.py                   # Full training pipeline
│   ├── score.py                   # Inference / risk scoring
│   └── generate_report_assets.py  # Generates docs/ charts for reporting
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place your dataset

```bash
cp /path/to/your/dataset.csv data/dataset.csv
```

The dataset must contain the target column `F3924` (0 = clean, 1 = mule).

### 3. Train the model

```bash
python src/train.py --data data/dataset.csv
```

**Outputs** saved to `outputs/`:
- `mule_classifier.json` — XGBoost model
- `medians.json` + `feature_names.json` — preprocessing state for inference
- `metrics.json` — full evaluation report
- `roc_curve.png`, `pr_curve.png`, `feature_importance.png`, `shap_summary.png`

### 4. Score accounts

```bash
python src/score.py --data data/dataset.csv --model outputs/mule_classifier.json
```

**Output**: `outputs/risk_scores.csv` — every account ranked by DRI, with risk tier.

### 5. Generate report assets

```bash
python src/generate_report_assets.py
```

Regenerates all `docs/` charts from the current model and scores.

---

## Data Cleaning Pipeline

The pipeline handles real-world data quality issues automatically:

| Step | Action |
|---|---|
| **Index drop** | Removes `Unnamed: 0` row index to prevent index-based data leakage |
| **Empty column removal** | Drops columns where every value is null (63 columns removed) |
| **Type coercion** | Forces all feature columns to numeric; invalid entries → `NaN` |
| **Median imputation** | Fills remaining nulls with column-wise median |
| **Inference alignment** | `medians.json` and `feature_names.json` ensure identical preprocessing at inference time, preventing schema drift |

---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
shap>=0.44.0
matplotlib>=3.7.0
```

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md) for the full SuSpy vision:

- **Layer 2b**: GraphSAGE GNN to detect mule *rings* — accounts that look clean individually but sit inside a cluster of flagged nodes
- **Layer 3**: DRI Fusion via a calibrated meta-learner, with temporal decay and network escalation logic
- **Investigator Dashboard**: Streamlit app with SHAP waterfalls, fraud subgraph visualisation, and auto-generated case summaries
