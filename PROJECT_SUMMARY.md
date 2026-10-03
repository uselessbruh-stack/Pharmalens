# PharmaLens — Project Summary

**An Explainable AI Framework for Drug–Target Interaction Prediction and Candidate Prioritization**

---

## What Is This Project?

PharmaLens predicts how strongly a drug molecule will bind to a protein target. In drug discovery, testing every drug against every protein in a lab is expensive and slow. PharmaLens uses machine learning to predict these interactions computationally, so researchers can narrow down which drug-target pairs are worth testing in the lab.

We built and compared 7 different models — from simple baselines to deep learning and graph neural networks — and used Explainable AI (SHAP) to understand why the model makes a particular prediction.

---

## Dataset

**KIBA Dataset** (Kinase Inhibitor BioActivity)

- **118,254** drug-target interaction records
- **2,111** unique drug molecules (represented as SMILES strings)
- **229** unique protein targets (represented as amino acid sequences)
- Interaction scores range from **0.0 to 17.2** (lower score = stronger binding)
- This is a well-established benchmark dataset used in DTI prediction research

The dataset was split into:
- **Training set:** 80% of interactions (for learning)
- **Validation set:** 10% (for tuning during training)
- **Test set:** 10% (for final evaluation — the model never sees this during training)

---

## Step-by-Step: What Was Done and What Came Out

### Step 1 — Dataset Exploration

We looked at the raw KIBA data to understand what we are working with. The score distribution showed that most interactions cluster between 10–15 (weak binding), with fewer strong binders below 3. Some drugs interact with many targets (promiscuous drugs), while some targets are hit by many drugs. This gives us a sense of the data imbalance.

### Step 2 — Data Preprocessing

Raw data is messy. We cleaned it:
- Validated all 2,111 drug SMILES using RDKit (removed invalid molecules)
- Normalized protein sequences (uppercase, removed non-standard amino acids)
- Removed duplicate entries
- Created 4 types of data splits:
  - **Random split** (standard evaluation)
  - **Cold-drug split** (test drugs are completely unseen during training)
  - **Cold-target split** (test proteins are completely unseen)

After cleaning: **118,254** valid interaction records remained.

### Step 3 — Drug Feature Extraction

Each drug molecule was converted into a numerical vector using two approaches:
- **10 molecular descriptors** — molecular weight, hydrogen bond donors/acceptors, LogP (lipophilicity), topological polar surface area, aromatic ring count, etc.
- **1,024-bit Morgan fingerprint** — encodes the substructure patterns present in the molecule (similar to a molecular barcode)

So each drug becomes a vector of **1,034 numbers**.

### Step 4 — Protein Feature Extraction

Each protein sequence was converted into numerical features:
- **Amino Acid Composition (AAC)** — frequency of each of the 20 amino acids (20 features)
- **Dipeptide Composition (DPC)** — frequency of each pair of consecutive amino acids (400 features)

So each protein becomes a vector of **420 numbers**.

For each drug-target pair, we concatenate: drug vector (1,034) + protein vector (420) = **1,454-dimensional input** for the ML models.

### Step 5 — Baseline Models

We started with simple models to set a performance floor:

| Model | RMSE | Pearson r | What It Does |
|-------|------|-----------|--------------|
| **Mean Predictor** | 0.841 | — | Always predicts the average score. The dumbest possible model. |
| **Ridge Regression** | 0.655 | 0.628 | Simple linear model with regularization. |
| **Random Forest** | 0.487 | 0.820 | Ensemble of 500 decision trees. Already decent. |

The Mean Predictor gives us a lower bound — any useful model must beat 0.841 RMSE. Random Forest showed that the features we engineered are meaningful — it got Pearson correlation of 0.82, meaning it captures real patterns.

### Step 6 — XGBoost

XGBoost (Extreme Gradient Boosting) is a powerful tree-based model. It builds trees sequentially, where each new tree corrects the mistakes of the previous ones.

| Metric | Value |
|--------|-------|
| **RMSE** | 0.435 |
| **Pearson r** | 0.857 |
| **R²** | 0.733 |
| **CI** | 0.850 |
| **Training time** | ~71 seconds |

XGBoost beat Random Forest by a clear margin — RMSE dropped from 0.487 to 0.435 (a 10.7% improvement). It also trains in just 71 seconds, making it very practical.

We also ran SHAP (Shapley Additive Explanations) on XGBoost. SHAP showed which features matter most — molecular weight, LogP, certain fingerprint bits, and specific dipeptide compositions were the top contributors.

### Step 7 — DeepDTA (Deep Learning)

DeepDTA uses two CNN (Convolutional Neural Network) encoders — one for the drug SMILES string and one for the protein sequence — and combines their representations to predict binding.

Unlike the previous models, DeepDTA works directly on raw SMILES and sequences — it learns its own representations instead of relying on hand-crafted features.

| Metric | Value |
|--------|-------|
| **RMSE** | 0.425 |
| **Pearson r** | 0.864 |
| **R²** | 0.745 |
| **CI** | 0.866 |
| **Training time** | ~8.2 hours |

DeepDTA improved slightly over XGBoost (RMSE 0.425 vs 0.435), but at the cost of much longer training time.

### Step 8 — GraphDTA (Graph Neural Network)

GraphDTA represents each drug molecule as a molecular graph — atoms are nodes, chemical bonds are edges. A Graph Convolutional Network (GCN) learns from this structure.

This is the most chemically meaningful representation because molecular graphs directly capture how atoms are connected and bonded.

| Metric | Value |
|--------|-------|
| **RMSE** | 0.430 |
| **Pearson r** | 0.860 |
| **R²** | 0.738 |
| **CI** | 0.856 |
| **Training time** | ~11.1 hours |

GraphDTA performed comparably to DeepDTA but didn't surpass it on this dataset. The likely reason: KIBA is a medium-sized dataset, and GNNs tend to shine more with larger datasets where they can learn richer structural patterns.

### Step 9 — Model Comparison

All 7 models on the same test set (random split):

| Rank | Model | RMSE ↓ | Pearson r ↑ | R² ↑ | CI ↑ | Train Time |
|------|-------|--------|------------|------|------|------------|
| 1 | **XGBoost (Tuned)** | **0.421** | **0.867** | **0.750** | **0.860** | 5.4 min |
| 2 | DeepDTA | 0.425 | 0.864 | 0.745 | 0.866 | 8.2 hrs |
| 3 | GraphDTA (GCN) | 0.430 | 0.860 | 0.738 | 0.856 | 11.1 hrs |
| 4 | XGBoost (Default) | 0.435 | 0.857 | 0.733 | 0.850 | 1.2 min |
| 5 | Random Forest | 0.487 | 0.820 | 0.665 | 0.845 | 33 min |
| 6 | Ridge | 0.655 | 0.628 | 0.394 | 0.739 | 1.3 sec |
| 7 | Mean Predictor | 0.841 | — | −0.0 | 0.500 | instant |

The top 4 models are all closely competitive (RMSE 0.42–0.44). The tuned XGBoost is the overall winner — it achieves the best RMSE while training in just 5 minutes, compared to 8–11 hours for deep learning models.

### Step 10 — Hyperparameter Tuning

We used Optuna (Bayesian optimization) to find the best hyperparameters for XGBoost. Optuna ran 50 trials, each trying different combinations of learning rate, max depth, number of trees, regularization, etc.

**Before tuning → After tuning:**

| Metric | Default XGBoost | Tuned XGBoost | Improvement |
|--------|----------------|---------------|-------------|
| RMSE | 0.435 | **0.421** | 3.2% better |
| Pearson r | 0.857 | **0.867** | +0.010 |
| R² | 0.733 | **0.750** | +0.017 |

Tuning gave a meaningful but modest improvement. The key finding from Optuna: learning rate, max_depth, and the number of estimators had the highest impact on performance.

### Step 11 — Cold-Start Evaluation

This is the most practically relevant test. In real drug discovery, you want to predict interactions for **new drugs** or **new proteins** that the model has never seen.

**Cold-Drug** (test set has completely new drugs):

| Model | Random RMSE | Cold-Drug RMSE | Drop |
|-------|-------------|---------------|------|
| XGBoost | 0.435 | 0.574 | +32% worse |
| Random Forest | 0.487 | 0.581 | +19% worse |

**Cold-Target** (test set has completely new proteins):

| Model | Random RMSE | Cold-Target RMSE | Drop |
|-------|-------------|-----------------|------|
| XGBoost | 0.435 | 0.593 | +36% worse |
| Random Forest | 0.487 | 0.684 | +40% worse |

Performance drops significantly in cold-start — this is expected and normal. The model struggles more with unseen proteins (cold-target) than unseen drugs (cold-drug), meaning the drug features generalize better than protein features.

XGBoost handles cold-start better than Random Forest in both cases, showing it learned more generalizable patterns.

### Step 12 — Explainability (SHAP Analysis)

SHAP values tell us **why** the model made a specific prediction. For each prediction, SHAP decomposes the output into contributions from each input feature.

Key findings:
- **Morgan fingerprint bits** contribute the most overall — specific substructure patterns in the drug molecule are the strongest predictors of binding
- **Molecular descriptors** like molecular weight, LogP, and TPSA also matter — heavier, more lipophilic molecules tend to bind more strongly
- **Protein features** (dipeptide composition) contribute less than drug features — the model relies more on drug-side information
- The drug-side features account for roughly **65–70%** of the total SHAP importance, protein-side features account for **30–35%**

This makes biological sense — the structural properties of the drug molecule largely determine whether it can fit into and bind a protein's active site.

### Step 13 — Batch Screening

We demonstrated the practical use case: screening 20 drugs against 10 targets (200 combinations), predicting and ranking all interactions. The heatmap visualization shows which drug-target pairs are predicted as strong binders.

This is how PharmaLens would be used in practice — a researcher provides candidate drugs and target proteins, and the system ranks the most promising combinations.

### Step 14 — Final Results

All results, figures, and tables were consolidated. The project produced:
- **37 publication-ready figures** (distribution plots, training curves, SHAP plots, comparison charts, heatmaps)
- **6 result tables** (model comparison, cross-split analysis, screening results)
- **2 trained XGBoost models** (default + tuned)
- **Trained DeepDTA and GraphDTA models**

---

## Final Conclusion

1. **Best model: Tuned XGBoost** — RMSE 0.421, Pearson 0.867, trains in 5 minutes. It beats deep learning models while being 100× faster to train.

2. **Deep learning (DeepDTA, GraphDTA) performs comparably** but requires hours of training. On larger datasets, they would likely overtake XGBoost.

3. **Feature engineering matters** — the hand-crafted molecular descriptors and fingerprints capture enough signal for tree-based models to compete with deep learning.

4. **SHAP explainability** reveals that drug structural features (fingerprints, LogP, molecular weight) are the primary drivers of predictions, which aligns with biochemical intuition.

5. **Cold-start is the hard problem** — predicting for completely novel drugs/proteins is significantly harder (32–40% performance drop). This is the key challenge for real-world deployment.

---

## Limitations

- **Dataset is kinase-focused** — KIBA only contains kinase inhibitors. Performance on other protein families (GPCRs, proteases, ion channels) is untested.
- **No 3D structure information** — we use sequence and fingerprint-based features, not the 3D shape of the protein binding site. Adding structural data (from AlphaFold) could improve predictions.
- **Cold-start performance drops significantly** — for truly novel drugs or proteins with no similar examples in training, the model will struggle.
- **All predictions are computational** — no prediction replaces actual lab experiments. The model prioritizes candidates, it doesn't confirm binding.
- **Limited to binding affinity** — the model predicts how strongly a drug binds, not whether it's safe, effective, or has side effects.

---

## Where This Model Works Well

- Predicting binding for drugs **similar** to ones in the training set (the KIBA dataset covers kinase inhibitors well)
- **Ranking** a set of candidates — even if individual scores aren't perfect, the relative ranking is useful
- **Fast screening** — predicting thousands of drug-target pairs in seconds
- When you need **interpretable predictions** — SHAP explains each prediction at the feature level

## Where This Model Will Struggle

- Completely new drug families that look nothing like kinase inhibitors
- Proteins from non-human organisms (training data is human-focused)
- Predicting exact binding affinity values (ranking is more reliable than absolute numbers)
- Very large, flexible proteins where sequence-based features lose important conformational information

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.12 |
| ML Models | scikit-learn, XGBoost, PyTorch, PyTorch Geometric |
| Chemistry | RDKit (molecular processing) |
| Explainability | SHAP |
| Tuning | Optuna |
| Backend | FastAPI |
| Frontend | HTML/CSS/JS (dark-theme dashboard) |
| Dataset | KIBA (from DeepDTA benchmark) |

---

## Project Structure

```
PharmaLens/
├── notebooks/          ← 14 research notebooks (the full ML pipeline)
├── src/
│   ├── preprocessing/  ← Drug, protein, interaction cleaning
│   ├── features/       ← Feature extraction (descriptors, fingerprints, graphs)
│   ├── models/         ← All model architectures
│   ├── training/       ← Training loop with early stopping
│   ├── evaluation/     ← Metrics, experiment tracking, splits
│   ├── explainability/ ← SHAP analysis
│   ├── inference/      ← Production prediction pipeline
│   └── screening/      ← Batch drug screening
├── backend/            ← FastAPI REST API
├── frontend/           ← Web dashboard
├── models/             ← Saved trained models
├── results/            ← Figures, tables, reports
└── data/               ← Raw and processed datasets
```
