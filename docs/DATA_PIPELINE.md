# Dataset and model foundation commands

Run from the repository root. The downloader uses the Python standard library. Preparation and integrity/model tests need the project virtual environment.

## Environment

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install -r requirements-foundation.lock
```

The lock describes the tested Windows/Python 3.12 foundation, not every research notebook dependency. Install extra notebook/tuning packages deliberately as needed. For GPU training choose and validate the appropriate runtime rather than reusing the CPU pin.

## Download

```powershell
python scripts/download_datasets.py --dry-run
python scripts/download_datasets.py
```

The default downloads BindingDB article-curated and PDSP-derived TSV ZIPs for release `202610`, plus all matching Ki/Kd binding observations for the 18 configured human ChEMBL targets. ChEMBL must report `ChEMBL_37`; the script fails if the release differs. Full-panel API acquisition can take time; raw response caching makes reruns reusable. Downloads do not imply records are training-ready.

Select individual sources/families or run a bounded integration sample:

```powershell
python scripts/download_datasets.py --datasets bindingdb_articles bindingdb_pdsp
python scripts/download_datasets.py --datasets chembl --families gpcr protease nuclear_receptor
python scripts/download_datasets.py --datasets chembl --families all --max-targets-per-family 1 --max-activities-per-target 25
python scripts/download_datasets.py --datasets bindingdb_all
```

The optional full BindingDB archive is substantially larger (roughly 567 MB compressed for the pinned release). Downloads stream to a temporary file; they are never automatically extracted. Archive CRC and publisher MD5, where pinned, are checked; every final artifact has a local SHA-256 manifest. An existing unverified/corrupt file is rejected rather than silently trusted or overwritten. Transient HTTP connection failures are retried; a broken transfer can be rerun from scratch. This implementation does not claim byte-range transfer resume.

ChEMBL uses a release directory and a run-signature directory printed on success. Each contains raw `observations.jsonl`, a manifest, and checksummed cached API responses. Limits mark a sample explicitly; samples must not be reported as complete benchmarks. Assay confidence, molecule type, source validity and duplicates are checked during preparation, so a bounded raw sample may retain few or zero usable records for some families.

## Prepare

```powershell
.venv/Scripts/python.exe scripts/prepare_datasets.py --source bindingdb --input data/raw/bindingdb/202610/BindingDB_PDSPKi_202610_tsv.zip --output data/processed/prepared/bindingdb_pdsp_202610_v1
.venv/Scripts/python.exe scripts/prepare_datasets.py --source bindingdb --input data/raw/bindingdb/202610/BindingDB_BindingDB_Articles_202610_tsv.zip --output data/processed/prepared/bindingdb_articles_202610_v1
```

For ChEMBL replace `<run-signature>` with the directory printed by the downloader:

```powershell
.venv/Scripts/python.exe scripts/prepare_datasets.py --source chembl --input data/raw/chembl/ChEMBL_37/<run-signature>/observations.jsonl --output data/processed/prepared/chembl_panel_v1
```

Preparation verifies the raw file's manifest hash. It writes `Ki.csv`, `Kd.csv`, `excluded.jsonl` and a complete manifest into a new version directory. Existing versions are never overwritten. An incomplete preparation retains an explicitly named `.part` directory for diagnostics; it is not a published dataset.

Exact positive concentrations are converted using explicit units; censored boundaries are excluded and counted. The standardized policy preserves charge/stereochemistry, selects the largest organic fragment, and rejects unresolved amino acids rather than inventing residues. Ki and Kd remain separate endpoints. Source replicates are retained, with stable observation IDs; chemically identical pairs must be grouped together in splits. Family labels are only set when verified. BindingDB source data still needs curated family/assay/variant resolution and cross-source overlap review before final research claims.

## Rebuild splits

```powershell
.venv/Scripts/python.exe scripts/rebuild_splits.py
.venv/Scripts/python.exe scripts/rebuild_splits.py --input data/processed/prepared/bindingdb_pdsp_202610_v1/Ki.csv --output data/processed/datasets/pdsp_ki_v1/splits
```

The default creates corrected KIBA splits under `data/processed/datasets/kiba_identity_v2/splits`. Historical CSVs are untouched. Random splits hold repeated pair identities together; cold splits group cleaned SMILES/exact sequences rather than source aliases. Strict cold-both drops cross-block observations and records the count. Group fractions do not imply the same row fractions, and sparse panels may not support nonempty cold-both evaluation. Chemical standardization is a preparation prerequisite; scaffold/homology holdouts are a later protocol, not an implicit guarantee of these splits.

## Register and select models

```powershell
.venv/Scripts/python.exe scripts/register_model.py --name broad-ki-xgb-v1 --type xgboost --artifact models/xgboost/broad_ki_v1.json --dataset broad --dataset-version v1 --endpoint Ki --preprocessing-version standardized_v1 --validation-metrics experiments/results/broad_ki_validation.json --default
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The registration example requires a real trained artifact and validation-metrics JSON, e.g. `{"rmse": 0.7}` from the actual validation run. Do not invent values or substitute test metrics. Local validation artifacts are available after running the experiment configs; they are provisional research candidates rather than final champions. Old checkpoints can use `legacy_v1` only if their feature/tokenizer contract matches the old pipeline; standardized new datasets use `standardized_v1`. IDs are immutable; register new versions under new IDs. A finite validation RMSE is required to mark a default. Defaults are filtered by dataset/endpoint/release, never globally selected across label scales.

`/predict`, `/predict/batch` and `/screen` accept `model_id`, `dataset`, `endpoint` and `dataset_version`; omitting `model_id` resolves a compatible validation-selected default. Example:

```json
{
  "dataset": "broad",
  "dataset_version": "v1",
  "endpoint": "Ki",
  "model_id": "broad-ki-xgb-v1",
  "smiles": "CCO",
  "sequence": "ACDEFGHIKLMNPQRSTVWY"
}
```

Responses include the actual model ID, dataset release and label direction. Synchronous batch requests accept at most 1,000 rows; Cartesian screening accepts at most 10,000 pairs. These limits bound the current API and do not replace a durable job system or concurrency/load testing. Missing/invalid artifacts give an explicit unavailable response. Configure `PHARMALENS_REGISTRY_PATH` and comma-separated `PHARMALENS_CORS_ORIGINS` for the deployment.

## Verify

```powershell
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m pip check
```

Offline tests cover duplicate-identity leakage, strict cold-both exclusion, units/censoring, invalid metrics, atomic download/cache integrity, source preparation, short batches, real GNN batches, XGBoost early stopping, CNN/GNN checkpoint parity, scoped registry selection and API partial failures. Tests use small synthetic models; their predictions are not research performance results. Source downloads are integration checks performed separately from offline CI.

## Curate the six-family Ki task

The completed configured ChEMBL source is under `data/raw/chembl/ChEMBL_37/1f923a11148d/`. This is an 18-target starter panel, not all ChEMBL. Preparation excludes unresolved assay variants and retains exact Ki/Kd separately. Q13547 did not satisfy the enzyme ancestry filter; the verified enzyme panel uses PDE5A O76074 instead.

```powershell
python scripts/curate_datasets.py --inputs data/processed/prepared/bindingdb_pdsp_202610_v1/Ki.csv data/processed/prepared/bindingdb_articles_202610_v1/Ki.csv data/processed/prepared/chembl_panel37_v1/Ki.csv --endpoint Ki --family-reference data/raw/chembl/ChEMBL_37/1f923a11148d/observations.jsonl --non-kinase-only --output data/processed/prepared/nonkinase_ki_202610_v1
```

Version directories must be new. The published local output has 27,870 observations / 22,305 unique pairs. Family counts are GPCR 9,744, other enzyme 5,318, transporter 5,151, protease 4,656, nuclear receptor 1,854 and ion channel 1,147. Classification uses exact verified reference sequences; 39,153 unresolved-family observations were excluded. The overlap ledger prevents treating overlapping databases as independent external validation. Assay/publication reuse and heterogeneous replicate disagreement still require further analysis.

## Train and summarize

```powershell
python scripts/train_model.py --config experiments/configs/nonkinase_ki_xgboost.json --output experiments/runs/nonkinase_xgb_new_seed42
python scripts/train_model.py --config experiments/configs/nonkinase_ki_xgboost.json --seed 123 --output experiments/runs/nonkinase_xgb_new_seed123
python scripts/train_model.py --config experiments/configs/nonkinase_ki_cnn_pilot.json --output experiments/runs/nonkinase_cnn_new_seed42
python scripts/train_model.py --config experiments/configs/nonkinase_ki_gine_pilot.json --output experiments/runs/nonkinase_gine_new_seed42
python scripts/summarize_experiments.py
```

Outputs are immutable. The CLI records data/split membership/source/package hashes, a source snapshot, actual validation predictions and metrics, training-only shortcut baselines, truncation counts and serving reload parity. Keep source unchanged during training. Neural checkpoints include training-only target normalization; reported validation metrics are restored to the original endpoint scale. Concordance uses an exact O(n log n) algorithm rather than sampling pairs. Smoke configurations are excluded from ranking. `split_seed` is independent from model initialization `seed`.

Test data is not evaluated by default. Reserve `--evaluate-test` for the locked final evaluation after model/configuration selection. Repeated tuning against validation is not an unbiased final result. The initial CNN/GINE pilots each have one seed and a finite 40-epoch budget; comparable hyperparameter searches and further seeds remain necessary.

```powershell
python scripts/register_experiment.py --run experiments/runs/nonkinase_ki_xgb_final_seed42 --name nonkinase-ki-xgb-validation-v1 --default
python scripts/register_experiment.py --run experiments/runs/nonkinase_ki_cnn_pilot_seed42 --name nonkinase-ki-cnn-pilot-v1
python scripts/register_experiment.py --run experiments/runs/nonkinase_ki_gine_pilot_seed42 --name nonkinase-ki-gine-pilot-v1
```

These commands were already run locally; do not repeat immutable IDs. Use new names for new versions. Registration rejects smoke/incomplete runs, failed serving parity and artifact hash mismatches. Local defaults are provisional validation candidates, not final research champions. Model/data artifacts are ignored by Git and must be archived separately. See `DASHBOARD.md` for the Next.js workspace.
