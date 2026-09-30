# PharmaLens — Project Plan

## Project Title
**PharmaLens — An Explainable AI Framework for Drug–Target Interaction Prediction and Candidate Prioritization**

**Type**: MCA Final-Year Research Project

---

## Problem Statement

Drug–Target Interaction (DTI) prediction aims to predict whether (and how strongly) a drug molecule may interact with a protein target. Experimental testing of all possible drug–protein combinations is prohibitively expensive and time-consuming.

PharmaLens is a **computational screening and prioritization system** that:
1. Takes a drug (SMILES) and protein (sequence) as input
2. Extracts/learns representations
3. Predicts an interaction score via ML
4. Explains the prediction using XAI
5. Ranks candidates for further investigation

**PharmaLens does NOT**:
- Claim predictions are experimentally confirmed
- Replace laboratory experiments
- Provide clinical recommendations
- Serve as a clinical decision-support system

---

## Research Gap

Existing research provides strong individual approaches for DTI prediction. PharmaLens integrates:

1. Multiple representation strategies (descriptors, fingerprints, graphs, sequences)
2. Multiple ML approaches (baselines, XGBoost, CNN, GNN)
3. Fair experimental comparison across all models
4. Best-model selection based on evidence
5. Explainable prediction (SHAP, Integrated Gradients, GNNExplainer)
6. Unseen-drug evaluation (cold-drug split)
7. Unseen-target evaluation (cold-target split)
8. Batch screening and candidate ranking
9. A usable researcher-oriented application/dashboard

The contribution is the **integration, experimentation, evaluation, and practical prioritization workflow** — not any single model innovation.

---

## Research Foundation

| Paper | Role in PharmaLens |
|-------|-------------------|
| **DeepDTA** (Öztürk et al., 2018) | Base paper. Foundational CNN architecture for SMILES + sequence → affinity |
| **DeepPurpose** (Huang et al., 2020) | Inspiration for multiple representations, architectures, and virtual screening |
| **GraphDTA** (Nguyen et al., 2021) | Inspiration for molecular graph + GNN approach |
| **iNGNN-DTI** (Bai et al., 2023) | Inspiration for interpretability, cold-start evaluation, attention visualization |
| **XAI for DTI** (various) | Inspiration for explaining predictions and evaluating explanation quality |

---

## Development Phases

| Phase | Description | Deliverables |
|-------|-------------|-------------|
| 0 | Project scaffolding & environment | Directory structure, requirements.txt, .gitignore |
| 1 | Dataset acquisition | Downloaded KIBA data, dataset_summary.json |
| 2 | Dataset exploration (NB 01) | Statistics, plots, observations |
| 3 | Data preprocessing (NB 02) | Cleaned dataset, cleaning_report.json |
| 4 | Drug features (NB 03) | Descriptors, fingerprints, molecular graphs |
| 5 | Protein features (NB 04) | AA composition, sequence encoding |
| 6 | Baseline models (NB 05) | Mean, Ridge, RF trained + evaluated |
| 7 | XGBoost (NB 06) | XGBoost trained + SHAP analysis |
| 8 | DeepDTA (NB 07) | CNN model trained + evaluated |
| 9 | GNN (NB 08) | GCN/GAT trained + evaluated |
| 10 | Model comparison (NB 09) | Comparison table + visualizations |
| 11 | Hyperparameter tuning (NB 10) | Optimized models, before/after |
| 12 | Cold-start evaluation (NB 11) | Generalization results |
| 13 | XAI analysis (NB 12) | Explanations for all model types |
| 14 | Batch screening (NB 13) | Screening + ranking demonstration |
| 15 | Final evaluation (NB 14) | FINAL_MODEL_REPORT, model_selection_report |
| 16 | Inference pipeline | Reusable Python pipeline |
| 17 | FastAPI backend | REST API for predictions |
| 18 | React frontend | 10-page application |
| 19 | Integration & testing | End-to-end testing |
| 20 | Documentation | Methodology, limitations, guides |

---

## Technology Stack

### Research & ML
| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Notebooks | Jupyter / JupyterLab |
| Data | pandas, NumPy |
| Chemistry | RDKit |
| ML | scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| GNN | PyTorch Geometric |
| XAI | SHAP, Captum |
| Tuning | Optuna |
| Visualization | matplotlib, seaborn, plotly |

### Application
| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | React (Vite) |
| Database | SQLite (or JSON/CSV) |
| Deployment | Local development |

---

## Success Criteria

- [ ] Dataset downloaded and documented
- [ ] 14 Jupyter notebooks with executable experiments
- [ ] At least 4 model types trained and compared
- [ ] Hyperparameter tuning performed
- [ ] Cold-drug and cold-target evaluation completed
- [ ] XAI implemented for multiple model types
- [ ] Batch screening with candidate ranking
- [ ] Best model selected with evidence-based justification
- [ ] FastAPI backend serving predictions
- [ ] React frontend with 10 pages
- [ ] No fabricated results
- [ ] Reproducible experiments (seeds, configs saved)
- [ ] Documentation complete

---

## Limitations (to be documented)

1. Predictions are computational — not experimentally confirmed
2. KIBA dataset is focused on kinase-related interactions
3. Dataset distribution may not represent all biological systems
4. High predicted score ≠ biological effectiveness
5. XAI explains model behavior, not biological causality
6. Cold-start performance expected to degrade
7. Out-of-distribution predictions may be unreliable
8. PharmaLens is NOT a clinical tool

---

## Teacher-Friendly Glossary

| Term | Simple Explanation |
|------|-------------------|
| Drug | A chemical molecule being studied |
| Protein target | A protein in the body that the drug may interact with |
| Drug–target interaction | The relationship between a drug and protein, including how strongly they may interact |
| SMILES | A text-based representation of a chemical molecule |
| Molecular fingerprint | A numerical representation of structural patterns in a molecule |
| Molecular graph | Atoms as points, chemical bonds as connections |
| GNN | A neural network designed to learn from connected structures like molecular graphs |
| Interaction score | A numerical value predicted by the model, used for comparison and ranking |
| XAI | Methods that help identify what information influenced the model's prediction |
| Batch screening | Testing many drug–protein combinations computationally and ranking results |
| Cold-start | Testing the model on drugs or proteins it has never seen during training |
