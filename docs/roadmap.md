# Roadmap — SuSpy Full Vision

This MVP covers Layer 1 (feature engineering) and Layer 2a (XGBoost classifier
with SHAP explainability). The full proposed solution adds two further layers:

## Layer 2b — Graph Neural Network (GraphSAGE)

- Construct a transaction graph: accounts as nodes, transactions as directed,
  weighted edges (weight = amount + frequency)
- Train a GraphSAGE model (PyTorch Geometric) on top of the engineered node
  features to produce neighborhood-aware embeddings
- Inductive design — scores new accounts at inference time without retraining
- Goal: detect accounts that look individually clean but sit inside a
  cluster of flagged mule accounts

## Layer 3 — Dynamic Risk Index (DRI)

- Fuse the XGBoost probability and GraphSAGE neighborhood score via a
  calibrated meta-learner
- Add temporal decay: DRI relaxes toward baseline if no new suspicious
  activity is observed
- Add network escalation: DRI is re-evaluated when an account gains new
  high-risk neighbors

## Investigator dashboard

- SHAP waterfall per flagged account
- Fraud subgraph visualization (network map of the account's neighborhood)
- Auto-generated case summary and recommended action

## Suggested implementation order for next phase

1. Build transaction graph from the dataset (NetworkX prototype)
2. Train baseline GraphSAGE model (PyTorch Geometric) on graph + Layer 1 features
3. Calibrate fusion model combining XGBoost + GraphSAGE outputs
4. Add temporal decay / escalation logic
5. Build Streamlit dashboard for investigator view with SHAP + subgraph plots
