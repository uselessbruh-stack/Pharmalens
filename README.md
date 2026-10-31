# PharmaLens

**An Explainable AI Framework for Drug–Target Interaction Prediction and Candidate Prioritization**

MCA Final-Year Research Project

## Current implementation phases

The expanded research benchmark is complete: 57 trained runs across corrected KIBA, broad non-kinase Ki/Kd and 11 supported family cohorts, followed by validation-locked held-out evaluation. The non-kinase data contains 309,685 Ki and 21,219 Kd observations across 1,638 distinct proteins. Family views overlap their broad datasets; they are not independent external sources.

On corrected KIBA, held-out RMSE is **0.3689 for hybrid GINE + ESM-2**, **0.3730 for hybrid CNN + ESM-2**, and **0.4189 for XGBoost + ESM-2**. Broad non-kinase Ki GNN RMSE is **0.7519**, versus **0.7633** for boosting. Gains vary by task; the complete [benchmark table](results/reports/final_benchmark_v1.md) includes weaker neural results too. These are identity-grouped random-split results, not proof of cold-target or external validity.

The Next.js dashboard selects a compatible validation default, exposes all registered alternatives, and displays separate validation/test results. Saved CSV batch jobs support progress, cancellation, bounded retries, history and result recovery after refresh. Start with [dashboard setup](docs/DASHBOARD.md), [pipeline commands](docs/DATA_PIPELINE.md), and [research details](docs/RESEARCH_EXPANSION.md). Artifacts, datasets and the local job database are ignored by Git and must be archived separately.

This is a verified local research implementation. Deployment authentication, capacity testing and additional generalization protocols remain release work.

See the [project summary](docs/PROJECT_SUMMARY.md) or [Word report](docs/PharmaLens_Project_Summary.docx) for the original system, completed changes, dataset counts, methods, results and the recommendation on endpoint-specific models.

---

## Overview

PharmaLens is a computational screening and prioritization system for drug–target interaction (DTI) prediction. It combines multiple machine learning approaches with explainable AI to predict, explain, and rank potential drug–protein interactions.

**PharmaLens is a research tool — not a clinical decision-support system.**

### What PharmaLens Does

```
Drug (SMILES) + Protein (Sequence)
        ↓
Feature Extraction / Learned Representation
        ↓
ML Prediction (Multiple Models Compared)
        ↓
Interaction Score
        ↓
Explainable AI Analysis
        ↓
Rank & Prioritize Candidates
```

### Key Features

- **Multiple ML Models** — Random Forest, XGBoost, DeepDTA (CNN), Graph Neural Networks
- **Fair Comparison** — All models evaluated on the same splits with the same metrics
- **Cold-Start Evaluation** — Tests generalization to unseen drugs and unseen targets
- **Explainable AI** — SHAP, Integrated Gradients, GNNExplainer
- **Batch Screening** — Screen many drug–protein combinations and rank results
- **Research Dashboard** — Next.js/TypeScript + FastAPI workspace; legacy HTML retained

---

## Project Structure

```
pharmalens/
├── notebooks/          # Jupyter research notebooks (01–14)
├── src/                # Reusable Python modules
│   ├── data/           # Dataset download and loading
│   ├── preprocessing/  # Data cleaning and validation
│   ├── features/       # Feature extraction (descriptors, fingerprints, graphs)
│   ├── models/         # Model implementations
│   ├── training/       # Training loops and utilities
│   ├── evaluation/     # Metrics, experiment tracking, split strategies
│   ├── explainability/ # XAI (SHAP, Captum, GNNExplainer)
│   ├── screening/      # Batch screening and ranking
│   ├── inference/      # Production inference pipeline
│   └── utils/          # Configuration, logging, plotting
├── models/             # Saved model checkpoints
├── experiments/        # Experiment configs, results, logs
├── results/            # Figures, tables, reports
├── backend/            # FastAPI REST API
├── frontend/           # React application
├── tests/              # Unit and integration tests
└── docs/               # Documentation
```

---

## Setup

### Requirements

- Python ≥ 3.10
- 32 GB RAM recommended (for large feature matrices)
- CPU training supported (GPU optional)

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-foundation.lock
# Optional notebook dependencies: requirements.txt (not the tested runtime lock)

# For Intel Arc GPU acceleration (optional):
# pip install intel-extension-for-pytorch
```

### Quick Start

```bash
# 1. Inspect/download the new broad and non-kinase sources
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py

# 2. Follow docs/DATA_PIPELINE.md to prepare, curate and train
python scripts/train_model.py --config experiments/configs/nonkinase_ki_xgboost.json --output experiments/runs/my_nonkinase_xgb

# 3. Start the backend API (after model training)
uvicorn backend.main:app --reload

# 4. Start the new Next.js workspace (Node 24 / pnpm 11.25.0)
cd dashboard
pnpm install --frozen-lockfile
pnpm build
pnpm start
# Open http://localhost:3000; see docs/DASHBOARD.md.
# Legacy frontend remains at http://localhost:8000/.
```

---

## Research Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Dataset Exploration | Load, inspect, and visualize the KIBA dataset |
| 02 | Data Preprocessing | Validate SMILES, sequences; create splits |
| 03 | Drug Features | Molecular descriptors, fingerprints, graphs |
| 04 | Protein Features | Amino acid composition, sequence encoding |
| 05 | Baseline Models | Mean predictor, Ridge, Random Forest |
| 06 | XGBoost Model | Gradient boosting + SHAP explainability |
| 07 | DeepDTA Model | CNN-based DTI prediction (base paper) |
| 08 | GNN Model | Graph neural network for molecular graphs |
| 09 | Model Comparison | Compare all models fairly |
| 10 | Hyperparameter Tuning | Optuna-based optimization |
| 11 | Cold-Start Evaluation | Unseen drug/target generalization |
| 12 | XAI Analysis | Explainability for all model types |
| 13 | Batch Screening | Drug–target combination screening |
| 14 | Final Evaluation | Consolidated results and reporting |

---

## Datasets

- **Primary:** KIBA (Kinase Inhibitor BioActivity)
- **Secondary:** Davis (optional)
- **New acquisition:** BindingDB article/PDSP snapshots (Ki/Kd preparation), optional full BindingDB
- **Non-kinase starter panel:** ChEMBL GPCR, protease, nuclear-receptor, channel, transporter and enzyme observations

---

## Research Papers

1. **DeepDTA** — Öztürk et al., 2018 (base paper)
2. **DeepPurpose** — Huang et al., 2020
3. **GraphDTA** — Nguyen et al., 2021
4. **iNGNN-DTI** — Bai et al., 2023
5. XAI for DTI (various)

---

## Limitations

1. Predictions are computational — not experimentally confirmed
2. KIBA is focused on kinase-related interactions
3. High predicted score ≠ biological effectiveness
4. XAI explains model behavior, not biological causality
5. PharmaLens is NOT a clinical tool

---

## License

MIT License — Academic use
