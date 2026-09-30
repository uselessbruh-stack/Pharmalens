# PharmaLens

**An Explainable AI Framework for Drug–Target Interaction Prediction and Candidate Prioritization**

MCA Final-Year Research Project

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
- **Research Dashboard** — Interactive React + FastAPI application

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
pip install -r requirements.txt

# For Intel Arc GPU acceleration (optional):
# pip install intel-extension-for-pytorch
```

### Quick Start

```bash
# 1. Download the KIBA dataset
python -m src.data.download

# 2. Run Jupyter notebooks in order
jupyter lab notebooks/

# 3. Start the backend API (after model training)
uvicorn backend.main:app --reload

# 4. Start the frontend (after backend setup)
cd frontend && npm install && npm run dev
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
