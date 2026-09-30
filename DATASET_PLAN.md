# PharmaLens — Dataset Plan

## Primary Dataset: KIBA

### Overview
KIBA (Kinase Inhibitor BioActivity) integrates bioactivity data from multiple sources (Ki, Kd, IC50) into a single unified score called the **KIBA score**. Higher KIBA scores indicate weaker binding (less interaction).

### Source
- **Origin**: He et al., 2017 — "SimBoost: a read-across approach for predicting drug–target binding affinities using gradient boosting machines"
- **Common download**: [DeepDTA GitHub](https://github.com/hkmztrk/DeepDTA/tree/master/data) or [Therapeutics Data Commons (TDC)](https://tdcommons.ai/multi_pred_tasks/dti/)
- **License**: Academic/research use

### Expected Format
```
data/raw/kiba/
├── ligands_can.txt       # Drug SMILES (dict: drug_id → SMILES)
├── proteins.txt          # Protein sequences (dict: target_id → sequence)
└── Y                     # Interaction matrix (drugs × targets), NaN = no measurement
```

### Expected Statistics (approximate)
| Metric | Value |
|--------|-------|
| Total interactions | ~118,254 (non-NaN) |
| Unique drugs | ~2,111 |
| Unique proteins | ~229 |
| Matrix size | 2,111 × 229 |
| Sparsity | ~75% NaN |
| Score range | 0 – ~17.2 |
| Median score | ~12.1 |

> **IMPORTANT**: These numbers are approximate. The actual values must be computed from the downloaded data and reported in Notebook 01.

### KIBA Score Interpretation
- **Low KIBA score** → Strong binding / high affinity
- **High KIBA score** → Weak binding / low affinity
- Common threshold for "active": KIBA < 12.1 (median), but this is dataset-dependent

---

## Secondary Dataset: Davis

### Overview
Davis kinase dataset contains Kd (dissociation constant) values for kinase inhibitors against kinases.

### Source
- **Origin**: Davis et al., 2011
- **Download**: Same sources as KIBA (DeepDTA GitHub or TDC)

### Expected Statistics (approximate)
| Metric | Value |
|--------|-------|
| Total interactions | ~30,056 |
| Unique drugs | ~68 |
| Unique proteins | ~442 |
| Score range | Kd values (nM) |

### Davis Score Transformation
Following DeepDTA convention:
```
binding_score = -log10(Kd / 1e9)
```

> **NOTE**: Davis is smaller and has fewer drugs. It serves as a secondary validation dataset. If time is limited, focus on KIBA only.

---

## Data Quality Checks

### Drug (SMILES) Validation
1. Parse with RDKit `Chem.MolFromSmiles()`
2. Reject if returns `None`
3. Canonicalize: `Chem.MolToSmiles(mol, canonical=True)`
4. Check for valid atom types
5. Remove salts/fragments (keep largest fragment)

### Protein Sequence Validation
1. Verify all characters are in standard amino acid alphabet: `ACDEFGHIKLMNPQRSTVWY`
2. Flag/remove sequences with non-standard characters (B, J, O, U, X, Z)
3. Filter by length: keep sequences with 50 ≤ length ≤ 5000
4. Remove exact duplicate sequences

### Interaction Data Validation
1. Remove rows with NaN/missing scores
2. Remove duplicate (drug_id, target_id) pairs (keep first or average)
3. Check for score outliers (beyond 3σ from mean)
4. Log-transform if distribution is highly skewed

---

## Data Splitting Strategies

### Strategy 1: Random Split
```
All (drug, target) pairs → random 80/10/10 split
```
- Standard evaluation
- Tests interpolation ability

### Strategy 2: Cold-Drug Split
```
Drugs partitioned into train/val/test sets
All interactions of test drugs are in test set
```
- Tests generalization to **unseen drugs**
- Expected: performance degradation vs random split

### Strategy 3: Cold-Target Split
```
Targets partitioned into train/val/test sets
All interactions of test targets are in test set
```
- Tests generalization to **unseen protein targets**
- Expected: performance degradation vs random split

### Strategy 4: Cold-Both Split (Optional)
```
Both drugs AND targets in test set are unseen during training
```
- Most challenging setting
- May show significant performance drop

### Split Ratios
| Split | Train | Validation | Test |
|-------|-------|------------|------|
| Random | 80% | 10% | 10% |
| Cold-Drug | ~80% drugs | ~10% drugs | ~10% drugs |
| Cold-Target | ~80% targets | ~10% targets | ~10% targets |

---

## Data Pipeline

```
Download raw files
       ↓
Inspect and parse format
       ↓
Validate drugs (SMILES → RDKit)
       ↓
Validate proteins (sequence check)
       ↓
Clean interaction data
       ↓
Generate dataset_summary.json
       ↓
Save to data/processed/
       ↓
Create train/val/test splits
       ↓
Save splits to data/processed/splits/
```

### Output Files
```
data/
├── raw/
│   └── kiba/
│       ├── ligands_can.txt
│       ├── proteins.txt
│       └── Y
├── processed/
│   ├── kiba_clean.csv          # drug_id, target_id, smiles, sequence, score
│   ├── drug_features.pkl       # precomputed drug features
│   ├── protein_features.pkl    # precomputed protein features
│   └── splits/
│       ├── random_train.csv
│       ├── random_val.csv
│       ├── random_test.csv
│       ├── cold_drug_train.csv
│       ├── cold_drug_val.csv
│       ├── cold_drug_test.csv
│       ├── cold_target_train.csv
│       ├── cold_target_val.csv
│       └── cold_target_test.csv
└── metadata/
    ├── dataset_summary.json
    └── cleaning_report.json
```
