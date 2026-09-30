# PharmaLens — Experiment Plan

## Overview

This document defines the experimental protocol for PharmaLens. Every experiment follows reproducible procedures with tracked configurations, seeds, and outputs.

---

## Research Questions

| ID | Question |
|----|----------|
| RQ1 | Can ML models predict drug–protein interaction scores? |
| RQ2 | Does molecular graph representation improve prediction vs. simpler representations? |
| RQ3 | Which model performs best on the KIBA benchmark? |
| RQ4 | How does performance change when evaluating unseen drugs? |
| RQ5 | How does performance change when evaluating unseen protein targets? |
| RQ6 | Can XAI identify influential features in model predictions? |
| RQ7 | Can the selected model rank promising drug–protein combinations? |
| RQ8 | Does the final model provide a useful balance of performance, generalization, interpretability, and cost? |

---

## Experiment Registry

Every experiment produces a record with:

```json
{
  "experiment_id": "EXP_001",
  "timestamp": "2026-09-03T14:00:00",
  "dataset": "KIBA",
  "model": "RandomForest",
  "representation": "descriptors+fingerprints+aa_composition",
  "split": "random",
  "seed": 42,
  "hyperparameters": {"n_estimators": 500, "max_depth": null},
  "metrics": {
    "rmse": null,
    "mae": null,
    "pearson": null,
    "spearman": null,
    "r2": null,
    "ci": null
  },
  "training_time_sec": null,
  "inference_time_sec": null,
  "num_parameters": null,
  "notes": ""
}
```

Stored in: `experiments/results/experiment_log.csv` and `experiment_log.json`

---

## Models to Evaluate

### Tier 1: Baselines (Notebook 05)

| Model | Input Representation | Purpose |
|-------|---------------------|---------|
| MeanPredictor | None | Lower bound |
| Ridge Regression | Drug descriptors + protein AA composition | Linear baseline |
| Random Forest | Drug descriptors + Morgan FP + protein AA composition | Nonlinear baseline |

### Tier 2: Strong ML (Notebook 06)

| Model | Input Representation | Purpose |
|-------|---------------------|---------|
| XGBoost | Drug descriptors + Morgan FP + protein AA composition | Strong tree-based model |

### Tier 3: Deep Learning (Notebooks 07–08)

| Model | Drug Input | Protein Input | Purpose |
|-------|-----------|---------------|---------|
| DeepDTA | SMILES (char-level) | Sequence (char-level) | CNN baseline (base paper) |
| GNN-GCN | Molecular graph | Sequence (char-level) | Graph-based representation |
| GNN-GAT | Molecular graph | Sequence (char-level) | Attention-based graph |

### Tier 4: Optional Advanced (if resources allow)

| Model | Enhancement | Purpose |
|-------|-------------|---------|
| DeepDTA + pretrained drug encoder | e.g., MolBERT / ChemBERTa embeddings | Learned drug representation |

---

## Representations

### Drug Representations

| Name | Dimensions | Used By |
|------|-----------|---------|
| Molecular descriptors | 8 features (MW, HBA, HBD, LogP, TPSA, RotBonds, Rings, HeavyAtoms) | RF, XGBoost |
| Morgan fingerprint (r=2, 1024 bits) | 1024 | RF, XGBoost |
| SMILES encoding (char-level, max_len=100) | 100 integers | DeepDTA |
| Molecular graph (atoms=nodes, bonds=edges) | Variable | GNN |

### Protein Representations

| Name | Dimensions | Used By |
|------|-----------|---------|
| AA composition | 20 | RF, XGBoost |
| Dipeptide composition | 400 | RF, XGBoost |
| Sequence encoding (char-level, max_len=1000) | 1000 integers | DeepDTA, GNN |

---

## Evaluation Protocol

### Metrics (Mandatory)

| Metric | Formula/Library | Interpretation |
|--------|----------------|---------------|
| RMSE | `sqrt(mean((y-ŷ)²))` | Lower = better |
| MAE | `mean(abs(y-ŷ))` | Lower = better |
| Pearson r | `scipy.stats.pearsonr` | Higher = better (linear correlation) |
| Spearman ρ | `scipy.stats.spearmanr` | Higher = better (rank correlation) |
| R² | `sklearn.metrics.r2_score` | Higher = better (variance explained) |
| CI | Custom implementation | Higher = better (concordance) |

### Ranking Metrics (for batch screening)

| Metric | K values |
|--------|----------|
| Precision@K | 5, 10, 20 |
| Recall@K | 5, 10, 20 |
| NDCG@K | 5, 10, 20 |
| Hit Rate@K | 5, 10, 20 |
| Enrichment Factor@K | 5, 10, 20 |

---

## Split Experiments

### Experiment Set A: Random Split (All Models)
- Split: 80/10/10 random
- Seeds: [42, 123, 456] (minimum 3 runs)
- Report: mean ± std for all metrics

### Experiment Set B: Cold-Drug (Best 2–3 Models)
- Split: Cold-drug (test drugs unseen)
- Seeds: [42, 123, 456]
- Answers: RQ4

### Experiment Set C: Cold-Target (Best 2–3 Models)
- Split: Cold-target (test proteins unseen)
- Seeds: [42, 123, 456]
- Answers: RQ5

### Experiment Set D: Cold-Both (Optional, Best Model Only)
- Split: Cold-both
- Seeds: [42]
- Only if computationally feasible

---

## Hyperparameter Search Spaces

### XGBoost (Optuna)
```python
{
    "n_estimators": [100, 300, 500, 800, 1000],
    "max_depth": [3, 5, 7, 9, 11],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha": [0, 0.01, 0.1, 1.0],
    "reg_lambda": [0, 0.01, 0.1, 1.0]
}
```

### DeepDTA (Optuna)
```python
{
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "num_filters": [32, 64, 128],
    "filter_sizes": [[4,6,8], [4,8,12]],
    "fc_dims": [[1024, 512], [512, 256]],
    "dropout": [0.1, 0.2, 0.3],
    "batch_size": [128, 256, 512]
}
```

### GNN (Optuna)
```python
{
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "hidden_dim": [64, 128, 256],
    "num_gnn_layers": [2, 3, 4],
    "dropout": [0.1, 0.2, 0.3],
    "pooling": ["mean", "max", "add"],
    "batch_size": [128, 256, 512]
}
```

---

## Reproducibility Protocol

### Seeds
```python
SEEDS = [42, 123, 456]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Environment
- Python version: recorded
- Package versions: recorded in `requirements.txt`
- Hardware: recorded (CPU/GPU model, RAM)
- OS: recorded

---

## XAI Experiments

### For Tree Models (XGBoost)
- SHAP TreeExplainer
- Global: summary plot, bar plot, dependence plots
- Local: force plot, waterfall plot for N=10 example predictions

### For DeepDTA
- Integrated Gradients (Captum)
- Identify top-contributing SMILES characters / protein residue positions
- Attention visualization if attention layers are added

### For GNN
- GNNExplainer: identify important atoms/bonds
- Attention weights (GAT): which atoms attend to which
- Visualize on 2D molecular structure using RDKit

### XAI Disclaimer
All explanations must include:
> "These features influenced the model's prediction. They do not automatically imply biological causality or experimental validity."

---

## Expected Outputs per Experiment

```
experiments/
├── configs/
│   ├── exp_baseline_rf.json
│   ├── exp_xgboost.json
│   ├── exp_deeptda.json
│   └── exp_gnn_gcn.json
├── results/
│   ├── experiment_log.csv
│   ├── experiment_log.json
│   ├── baseline_results.json
│   ├── xgboost_results.json
│   ├── deeptda_results.json
│   ├── gnn_results.json
│   ├── cold_start_results.json
│   └── tuning_results.json
└── logs/
    ├── deeptda_training_history.csv
    └── gnn_training_history.csv
```

---

## Final Model Selection Criteria

The best model is selected based on a weighted assessment of:

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| RMSE (random split) | High | Primary performance metric |
| CI (random split) | High | Ranking ability |
| Cold-drug performance | Medium | Generalization to new drugs |
| Cold-target performance | Medium | Generalization to new targets |
| Explainability | Medium | Required for PharmaLens |
| Training time | Low | Practical consideration |
| Inference time | Medium | Must be fast for batch screening |
| Stability (std across seeds) | Medium | Reliability |

The selection is **data-driven** — the winner is determined by experiments, not predetermined.

---

## Timeline Mapping

| Week | Experiments |
|------|------------|
| 1 | Dataset acquisition, exploration, preprocessing |
| 2 | Drug/protein features, baseline models |
| 3 | XGBoost training + SHAP |
| 4 | DeepDTA implementation + training |
| 5 | GNN implementation + training |
| 6 | Model comparison, hyperparameter tuning |
| 7 | Cold-start evaluation, XAI analysis |
| 8 | Batch screening, final evaluation |
| 9–10 | FastAPI backend + React frontend |
| 11 | Integration, testing, documentation |

> This is a suggested timeline. Actual pacing depends on hardware and available time.
