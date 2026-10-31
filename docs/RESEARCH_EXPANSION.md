# Broader affinity experiments

The objective is to improve neural affinity prediction with stronger representations, sufficient data and reproducible evaluation. A neural winner must be measured, not manufactured by weakening the baseline or choosing only favorable datasets.

## Data acquired and prepared

The complete pinned BindingDB October 2026 archive is 594,284,548 bytes. Its publisher MD5 is `ab7ca3dc152d395b773ff987d8e09f64`; SHA-256 is `e968f90e900c7833d5e6eceae282d0ca3b75201448bccaf4df6b09cc410f7d76`. The archive passed CRC/checksum checks before preparation.

ChEMBL 37 supplies 2,999 human protein-component reference sequences classified into the six supported non-kinase families. Family assignment uses exact sequences and verified classification ancestry; unknown families are excluded. The reference catalog is an annotation source, not 2,999 experimentally supported training targets.

| Published task | Observations | Molecules | Proteins | Unique pairs |
|---|---:|---:|---:|---:|
| Broad non-kinase Ki, v3 | 309,685 | 143,035 | 995 | 244,344 |
| Broad non-kinase Kd, v3 | 21,219 | 10,007 | 1,134 | 15,952 |
| Combined starter-source Ki, v2 | 45,517 | 23,093 | 355 | 35,561 |

Ki and Kd stay separate regression tasks. The v3 tasks use the full BindingDB source alone rather than pooling overlapping ChEMBL and BindingDB copies again. This does not resolve publication/assay reuse within BindingDB. Family views overlap the broad tasks and must not be called independent external datasets.

The v3 family/endpoint views support all six Ki families and five Kd families under fixed coverage criteria: at least 500 observations, 100 molecules and three proteins. Transporter Kd has 335 observations and is explicitly unsupported under this policy. The cohort manifest records every panel, including those below the criteria.

`results/reports/expanded_affinity_quality.json` reports replicate disagreement, missing publication metadata and sequence/SMILES truncation exposure. It does not remove difficult observations or average discordant assays. These are descriptive all-data audits, never test-derived training transformations.

## Model improvements and controls

Hybrid CNN/GINE networks retain their learned sequence/graph branches and add molecular fingerprints, physicochemical descriptors and protein composition features already available to XGBoost. Their input normalization is fitted on training data and saved inside the checkpoint. Old non-hybrid checkpoints remain loadable.

The extended v1 comparison completed with early stopping:

| Model/configuration | Validation RMSE, pKi |
|---|---:|
| Initial CNN pilot | 0.8534 |
| Hybrid CNN | 0.7945 |
| Initial GINE pilot | 0.8616 |
| Hybrid GINE | 0.7957 |
| Tuned XGBoost, depth 9 / 1,800-tree budget | 0.7604 |

These results use the original fixed identity-grouped split, not the larger v3 data. The neural models improved, while the stronger boosting baseline remains ahead. Target-group paired bootstrap reports are conditional on the validation set and do not correct tuning bias or substitute for final test evaluation.

An isolated `torch==2.15.0+xpu` environment passed actual CNN/GINE CPU-versus-Intel-Arc parity and finite-backward checks. `requirements-xpu-windows.lock` records its installed packages. The CPU runtime remains separately pinned and working.

Frozen ESM-2 `esm2_t6_8M_UR50D` features have been cached for 1,638 distinct proteins. They use layer 6, 320 dimensions, residue-mean pooling and nonoverlapping 1,022-residue chunks weighted by residue count. Padding and special tokens are excluded. The encoder uses no affinity labels and is not fine-tuned. Long proteins lose cross-chunk context; public pretraining may include related sequences. CPU local encoding was checked against GPU-cached representations and also tested on an unseen sequence.

The completed matched representation comparison gives the same frozen protein features to XGBoost, CNN and GNN. Cached representations alone are not a new trained affinity model. Model/feature artifacts and execution state distinguish these stages explicitly.

## Reproduce data expansion

Run from the repository root, using the tested environment:

```powershell
python scripts/discover_protein_catalog.py --output data/raw/chembl/human_nonkinase_catalog37_v1
python scripts/download_datasets.py --datasets bindingdb_all
python scripts/prepare_datasets.py --source bindingdb --input data/raw/bindingdb/202610/BindingDB_All_202610_tsv.zip --family-reference data/raw/chembl/human_nonkinase_catalog37_v1/observations.jsonl --non-kinase-only --output data/processed/prepared/bindingdb_all_nonkinase_202610_v1
python scripts/curate_datasets.py --inputs data/processed/prepared/bindingdb_all_nonkinase_202610_v1/Ki.csv --endpoint Ki --output data/processed/prepared/nonkinase_ki_202610_v3
python scripts/curate_datasets.py --inputs data/processed/prepared/bindingdb_all_nonkinase_202610_v1/Kd.csv --endpoint Kd --output data/processed/prepared/nonkinase_kd_202610_v3
python scripts/build_research_cohorts.py --input data/processed/prepared/bindingdb_all_nonkinase_202610_v1 --output data/processed/prepared/family_affinity_202610_v3
```

Use new version directories when outputs already exist. The full acquisition/preparation and annotation manifests must accompany any archived dataset. Family selection is independent of model performance. The acquisition CLI also accepts `--panel-json` from the discovery output for explicitly expanded ChEMBL accession panels; this does not imply those additional activity records have been downloaded.

## Execution and model selection

`plan_research_matrix.py` creates immutable per-task configurations and a code/config-hashed plan. `run_experiment_matrix.py` bounds worker processes, records logs/status, verifies completed outputs before resuming, and rejects changed configurations. Plans never request test evaluation. Smoke/incomplete runs are excluded from ranking and registration.

For a fresh local reproduction of the 39 broad/family comparisons, first create the label-free protein cache, then generate a new plan. These commands use CPU; the tested Intel environment can use `--device xpu` with `.venv-xpu` instead. Choose new output names if they already exist.

```powershell
python scripts/cache_protein_embeddings.py --inputs data/processed/prepared/nonkinase_ki_202610_v3/Ki.csv data/processed/prepared/nonkinase_kd_202610_v3/Kd.csv --output data/processed/features/esm2_local_reproduction_v1 --device cpu
python scripts/plan_research_matrix.py --output experiments/configs/local_reproduction_v1 --scope all --protein-cache data/processed/features/esm2_local_reproduction_v1 --device cpu
python scripts/run_experiment_matrix.py --plan experiments/configs/local_reproduction_v1/plan.json --status experiments/runs/local_reproduction_v1_status.json --workers 1 --register
```

Cache generation may download the pinned ESM weights on its first run. Archived original plans contain absolute feature-cache paths and source hashes: retain them as evidence, and generate fresh plans on another machine rather than editing frozen records. The three KIBA configurations and nuclear-repeat plans are archived separately. After all validation runs finish, freeze the complete candidate set before evaluating test data:

```powershell
python scripts/evaluate_frozen_matrix.py --plans experiments/configs/local_reproduction_v1/plan.json --selection results/reports/local_reproduction_selection_v1.json
python scripts/evaluate_frozen_matrix.py --selection results/reports/local_reproduction_selection_v1.json --output experiments/runs/local_reproduction_heldout_v1
```

Keep test evaluation separate from further hyperparameter choices. Large datasets, feature caches, trained weights, job databases and run directories are ignored by Git and require a separate archive with their manifests and hashes. Install CPU PyTorch separately before `requirements-foundation.lock`; for Intel, install the exact XPU PyTorch build from its official wheel index before the Windows XPU lock file. Do not mix the CPU and XPU environments.

`compare_validation_runs.py` requires identical source data, validation membership, observation labels and target identities, then computes paired target-group bootstrap differences. Compare within an endpoint, release and split; keep random, cold-target, scaffold and family holdouts distinct. Scaffold grouping keeps the entire empty acyclic scaffold together rather than dispersing it for convenient row fractions.

The default for each dashboard task should follow its compatible, loadable validation evidence and remain marked provisional until the final protocol is locked. Publish every attempted panel/model result, initialization seed and training budget. Do not claim that broader source coverage proves external validity, or that a single favorable result establishes general architecture superiority.

## Remaining research and operational work

The 42-run expanded matrix, three corrected-KIBA runs and 12 repeat-seed runs are complete. All 57 artifacts have separate held-out evaluations after selection was locked. Most tasks still need repeated initialization seeds, harder holdouts and uncertainty review. Publication/assay holdouts and sequence-homology clustering remain additional controls. The released test sets must not become tuning sets for subsequent experiments. Persistent user batch jobs are implemented and locally verified; worker deployment, authentication and sustained capacity testing remain deployment work. A successful local research experiment does not certify production readiness.

Source documentation: [BindingDB downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp), [ChEMBL API](https://www.ebi.ac.uk/chembl/api/data/docs), [official ESM implementation](https://github.com/facebookresearch/esm).

## Completed held-out benchmark (31 October 2026)

Selection was locked in `results/reports/final_validation_selection_v1.json` before accessing test scores. Nuclear-receptor architecture selection used mean validation RMSE across three seeds; the fixed seed-42 artifact represents the selected architecture in serving. Test data did not change the selected defaults.

| Task | XGBoost test RMSE | CNN test RMSE | GINE test RMSE |
|---|---:|---:|---:|
| Corrected KIBA | 0.4189 | 0.3730 | 0.3689 |
| Broad non-kinase Ki | 0.7633 | 0.7645 | 0.7519 |
| Broad non-kinase Kd | 1.0388 | 1.0731 | 1.0677 |

The KIBA GNN gain is about 11.9% relative RMSE reduction versus matched boosting. The broad Ki gain is about 1.5%. These are hybrid learned molecular encoders with frozen ESM-2 and shared descriptor/fingerprint features, not evidence isolating the contribution of graph convolutions alone. Most tasks have one initialization.

Nuclear-receptor validation improvements persisted across seeds but did not transfer to the held-out test as a clear advantage. Mean test RMSE over three seeds is Ki: boosting 0.7223, CNN 0.7649, GINE 0.7867; Kd: boosting 0.8453, CNN 0.8484, GINE 0.8582. The full table reports every task. Do not present the favorable validation results as final test superiority or retune against this test set.

A Windows sharing-lock error prevented one completed protease CNN run from being published initially. Its artifact, snapshot, data and membership hashes and serving parity were checked before recovering the directory without retraining. Original failure and recovery evidence remain beside the artifact. Publication now retries transient permission locks while refusing to overwrite an existing published directory.
