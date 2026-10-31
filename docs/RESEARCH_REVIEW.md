# PharmaLens research and product review

Review date: 30 October 2026. Scope: Python source, all 14 notebook code paths and saved error outputs, committed datasets/splits, experiment logs, result tables, inference, screening, and existing frontend. This is a review and implementation specification; training, model promotion, dataset ingestion, and the Next.js migration have not been performed.

Implementation update: the review below records the original findings. Dataset acquisition/preparation and the first reliability fixes have since been implemented; see [phase tracker](IMPLEMENTATION_PHASES.md), [commands](DATA_PIPELINE.md) and `results/reports/implementation_validation.json` for actual completion and remaining work. Historical performance claims have not been replaced with new training results.

## Recommendation

First repair data identity leakage, affinity interpretation, and evaluation reliability. Then add BindingDB Ki and Kd as separate broad-target regression benchmarks, improve and fairly tune neural models, and build the Next.js research dashboard around a versioned model registry. Preserve KIBA as a historical benchmark. Choose a default per dataset and endpoint using validation evidence and artifact checks.

The defensible research question is: **When do learned molecular and protein representations improve affinity prediction and generalization beyond engineered-feature boosting?** Larger datasets provide an opportunity to investigate this; they cannot guarantee a neural-network victory. Report failures and uncertainty as well as improvements. An existing-model dashboard plus extra datasets alone does not establish methodological novelty.

## What is already implemented

- Data: downloader and basic loaders for KIBA and Davis; only KIBA processed data is present.
- Representations: ten descriptors plus 1,024-bit Morgan fingerprints; 20 AAC plus 400 DPC protein features; character SMILES/protein encoding; atom and bond graph construction.
- Models: mean, Ridge, Random Forest, XGBoost, DeepDTA-style CNN, and GraphDTA-style GCN/GAT implementation. The recorded GNN experiment uses GCN.
- Training: shared PyTorch trainer, gradient clipping, validation, ReduceLROnPlateau, early stopping, best-weight restoration, checkpoints and histories. These mechanisms already exist; they need corrections and better verification.
- Experiments: saved random results for seven model configurations; cold-drug and cold-target results for default XGBoost and Random Forest only. Fifty-trial Optuna tuning is implemented for XGBoost only.
- Explanations: XGBoost SHAP code and a faulty/incomplete CNN Integrated Gradients helper. No implemented GNNExplainer was found in source despite README claims.
- Serving: FastAPI single/batch prediction, Cartesian screening, dataset statistics, experiment comparison, and registry listing.
- UI: plain HTML/CSS/JavaScript served by FastAPI, with prediction, screening, SHAP bars, dataset stats and comparison code. It is neither React nor Next.js; README and pending-requirement descriptions are partly stale.
- Availability: this checkout has no trained model files, registry JSON, raw source files, or cached feature pickle/NumPy files. Model directories contain placeholders. The active Python environment lacks the scientific/API dependencies. Saved outputs document earlier runs; they do not prove those models can currently be loaded.

## What the saved performance actually says

Source: `experiments/results/experiment_log.json`, random KIBA split, seed 42, 11,826 test examples. These are recorded values, not independently recomputed predictions.

| Model | RMSE | MAE | Concordance index | Training time |
|---|---:|---:|---:|---:|
| Tuned XGBoost | 0.4206 | 0.2624 | 0.8595 | 326 s |
| DeepDTA | 0.4249 | **0.2470** | **0.8657** | 29,558 s |
| GraphDTA GCN | 0.4303 | 0.2648 | 0.8561 | 39,955 s |
| Default XGBoost | 0.4346 | 0.2804 | 0.8497 | 71 s |
| Random Forest | 0.4870 | 0.2933 | 0.8445 | 1,984 s |
| Ridge | 0.6547 | 0.4475 | 0.7386 | 1.3 s |
| Mean predictor | 0.8410 | 0.5945 | 0.5000 | — |

DeepDTA and GCN already beat default XGBoost on RMSE. Tuned XGBoost beats DeepDTA by 0.0043 RMSE, about 1%; one seed provides no evidence that this gap is statistically reliable. DeepDTA wins MAE and CI. The CI implementation samples pairs for this test size, so these CI values are estimates. The historical RMSE leader is tuned XGBoost, but the leaked split prevents treating this as a validated generalization champion.

An error of 0.5 is not 50% error: RMSE has the target label's units. KIBA's saved standard deviation is about 0.84, and the test mean baseline is 0.841 RMSE. Tuned XGBoost's 0.4206 is about half that baseline RMSE. Its MSE is 0.1769; comparing RMSE 0.4206 with a paper's MSE around 0.18 would be misleading. Compare identical metrics, label transformations, dataset releases and split protocols. KIBA error and pKi/pKd error are not directly comparable.

Default XGBoost's recorded RMSE is 0.5744 on cold drugs and 0.5925 on cold targets. There are no equivalent neural cold-start results in the log. Generalization superiority for any neural model remains untested here.

## Findings that must be fixed before new claims

### 1. Actual identity leakage in saved data — critical

The standard-library audit of the committed cleaned CSV found 118,254 records, 2,111 drug IDs, **2,065 distinct cleaned SMILES**, 229 distinct protein sequences, and **648 duplicate molecule–sequence interaction rows** beyond the first occurrence. Drug-ID deduplication does not resolve molecules that become identical after cleaning.

| Saved split | Train/test ID-pair overlap | Train/test identical SMILES–sequence overlap | Additional issue |
|---|---:|---:|---|
| Random | 0 | **90** | Same chemical/target pair can cross splits under different IDs |
| Cold drug | 0 | **106** | **8 identical molecules** appear on both sides under different drug IDs |
| Cold target | 0 | 0 | Exact sequences are disjoint; homologous proteins remain untested |

Overlap counts are distinct identities, not test-row counts. Random splits normally share individual drugs and proteins; the problem is sharing the *same interaction identity*. Cold-drug evaluation additionally requires molecule identity disjointness.

Fix: assign standardized molecule identity and sequence hashes before splitting; preserve original IDs as provenance. Group repeated interaction identities together. Aggregate replicates only within compatible endpoint/assay contexts, or keep assay-specific observations grouped. Freeze new split manifests and hashes, then rerun every model. Check train/validation and validation/test too. Saved metrics must remain labeled as historical until rerun. Exact SMILES equality is a lower-bound identity check; a documented stereochemistry, salt, charge and tautomer policy is also needed.

### 2. `cold_both_split` does not enforce cold-both — critical

In `src/evaluation/split_strategies.py`, test and validation select within their respective entity blocks, but training uses the complement of those two blocks. A test drug paired with a training target therefore enters training; so does a test target paired with a training drug. The supposed held-out entities become visible.

Fix: training must use training-drug AND training-target membership; validation and test use their own disjoint blocks. Explicitly discard or separately report cross-block interactions and resulting sample counts. Fractions of entities do not imply the same fractions of interaction rows. Assert entity and pair disjointness across all partitions. This function is not used by the three committed split sets, so its bug should not be confused with the measured identity leakage above.

### 3. Binding interpretation and screening rank are reversed — critical

The configured download comes from DeepDTA, whose benchmark KIBA transformation reverses the original raw-score orientation. In that transformed convention, larger scores correspond to stronger binding. The frontend's `score < 3`/`score < 10` classes, both screening implementations' ascending sort, and local dataset documentation apply a lower-is-stronger interpretation. Ranking metrics meanwhile assume larger is better, creating an internal contradiction.

Sources: [DeepDTA data documentation](https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md), [SimBoost original preprocessing description](https://pmc.ncbi.nlm.nih.gov/articles/5395521/), [DeepDTA paper](https://pmc.ncbi.nlm.nih.gov/articles/6129291/). The raw source matrix is absent locally, so independently verify its release/checksum and transformation against the committed CSV before migration. Do not assume every file labeled KIBA uses the same convention.

Fix: dataset manifests must declare `label_name`, `unit`, `transform`, and `higher_is_stronger`. Use that contract for labels, ranking, thresholding, exports and explanations. Remove unsupported strong/moderate/weak categories until thresholds are specified and justified. KIBA is not a physical concentration and must not be converted to nM. This issue affects interpretation/ranking; merely reversing presentation does not lower regression RMSE.

### 4. Comparison is incomplete and cannot isolate architecture effects — high

Only XGBoost receives a tuning study; the recorded results use one seed. There are no neural cold-start runs, confidence intervals, scaffold/sequence-cluster splits, family-level evaluations, or independent broad-target validation. Feature builders/datasets can skip missing entries independently, so equal sample counts alone do not guarantee identical evaluation rows.

Fix: identical retained interaction IDs and label semantics across models; common splits; comparable documented search budgets; seeded Optuna sampler; at least three repeated runs, preferably five for finalists. Distinguish model-initialization seeds from split seeds. Choose hyperparameters on validation only and lock finalists before one final test evaluation. Report uncertainty across runs and paired differences with bootstrap resampling at the held-out drug/target group level where appropriate. Do not infer significance from overlapping or non-overlapping rounded metrics alone.

### 5. Training and inference reliability — high

- `Trainer` detects tuples using `len(batch) == 3` before checking PyG attributes. Use explicit batch-type handling. `batch.y.squeeze()` becomes scalar for a one-sample batch, causing inconsistent shapes and possible concatenation failure. Flatten both predictions and labels to one dimension and enforce equal shapes.
- Trainer loss averages batches equally; the last short batch receives disproportionate influence. Weight by sample count for checkpoint/early-stop metrics. Reject empty loaders and non-finite losses/predictions explicitly.
- XGBoost accepts `early_stopping_rounds` but never forwards it to the regressor. Wire it to the supported installed API when a validation set exists, and verify best-iteration behavior with a small real training check.
- Neural inference hardcodes SMILES length 100 and protein length 1,000 instead of the artifact's preprocessing configuration. GraphDTA configuration omits some constructor settings, including protein vocabulary and maximum length. Persist a complete feature/tokenizer/length/normalization contract and load it consistently.
- **44 of 229 proteins** exceed 1,000 residues; **24,103 interactions (20.38%)** lose their protein tail. Forty-four drug IDs exceed 100 SMILES characters, affecting 516 interactions. Measure truncation by split and family; test longer inputs, masked pooling or chunking with validation. Do not assume every missing tail contains the binding region.
- CNN max pooling includes padding-related activations; unknown SMILES characters map to padding. Separate unknown and padding tokens and mask padded positions throughout a compatible encoder implementation. Retrain after changes; old weights are not automatically compatible.
- GNN datasets rebuild the same molecule for each interaction. Cache immutable graphs once per molecule and sequence representations once per target. The GNN receives connectivity and atom features, but its forward path ignores the available bond attributes.
- Input cleaning changes ambiguous amino acids and strips nonletters silently; training and inference also enforce different minimum lengths. Specify FASTA parsing, ambiguity policy, maximum lengths and explicit user-visible validation outcomes.
- Metric code silently removes NaN pairs, does not comprehensively reject infinity, and has degenerate constant/small-array cases. Persist evaluated/excluded row IDs; fail final evaluation on unexpected invalid predictions. Keep full precision in records and round only in displays. Use a validated exact efficient CI implementation or label sampling and its uncertainty.

### 6. Registry, API and screening do not yet support the requested product — high

- Backend startup tries two XGBoost paths directly; prediction requests accept no model ID. `/models/list` cannot enable selection by itself. No training notebooks register artifacts.
- Registry fallback compares unfiltered RMSE across datasets and splits. Once multiple endpoints exist, that can choose a model on an incomparable scale. Missing metrics also sort incorrectly for some maximized metrics in `compare`.
- CPU inference, SHAP, CSV reading and large screening loops execute directly inside async routes. These operations can block request handling. Use bounded workers/thread execution for small requests and persistent job workers for larger batches.
- Batch pairs are untyped dictionaries; ID lists can truncate silently through `zip`; empty screening inputs can fail during sorting; there are no workload limits or persistent jobs. Failed rows receive ranks, and mixed missing fields/NaN can make JSON response serialization fail.
- `/models/comparison` serializes records containing undefined correlations and missing values. The saved log already contains nonstandard JSON NaN. Normalize to JSON null and use explicit response schemas. UI comparison mixes split types in one ranking.
- The frontend inserts user-supplied screening IDs into `innerHTML`, which can execute markup. Use escaped text/React rendering. Configure CORS for actual frontend origins and environment-based API URLs.
- `tests/` contains only an initializer. There is no regression test coverage for these failure modes. Broad lower-bound dependencies and omitted default deep-learning installs also weaken reproducibility.

### 7. Explainability needs a correctness pass — medium/high

The Integrated Gradients helper interpolates integer token IDs instead of embedding vectors, then reads `.grad` from a non-leaf embedding output without retaining it. It returns drug attribution only despite advertising both branches. Replace with tested embedding-layer attribution, choose meaningful baselines, handle device placement and verify completeness/convergence. GNN attribution requires an actual compatible explainer and masking validation. SHAP feature importance describes model behavior; it does not establish binding mechanism or biological causality.

## Dataset expansion

The expanded [non-kinase dataset specification](NON_KINASE_DATASETS.md) defines six explicit family panels (GPCRs, proteases, nuclear receptors, ion channels, transporters and other enzymes), additional curated/screening/structure sources, and problem-by-problem controls. The planned machine-readable catalog is `experiments/configs/dataset_catalog.json`. These panels have not been ingested or trained.

| Dataset | Role | Preparation and caveats |
|---|---|---|
| **BindingDB Ki** | First broad-target benchmark; primary expansion recommendation | TDC documents 375,032 pairs, 174,662 drugs and 3,070 proteins. Treat these as source-release counts, not promised cleaned counts. Use endpoint-aware pKi and quantify family coverage. |
| **BindingDB Kd** | Independent endpoint experiment and smaller integration smoke test | TDC documents 52,284 pairs, 10,665 drugs and 1,413 proteins. Evaluate as a separate task; overlap with Ki data prevents claiming independence without identity/source checks. |
| **BindingDB IC50** | Later scale experiment | TDC documents 991,486 pairs. Assay-dependent inhibition is not interchangeable with equilibrium affinity; use its own endpoint or a clearly defined multi-task head. |
| **Curated ChEMBL binding assays** | Optional family/temporal external evaluation | Select explicit single-protein targets, assay confidence 9, binding assays, supported positive numeric units and exact relations initially. Preserve publication/assay metadata and remove source overlap with BindingDB. |
| KIBA | Historical benchmark | Freeze the existing provenance and add corrected identity-grouped splits. Keep historical and corrected results separate. |
| Davis | Optional kinase reproducibility benchmark | It does not address non-kinase coverage; versions differ across providers. Do not silently replace one release with another. |

References: [TDC DTI benchmark descriptions and counts](https://tdcommons.ai/multi_pred_tasks/dti/), [BindingDB timestamped downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp), [BindingDB TSV field specification](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/BindingDB-TSV-Format.pdf), [ChEMBL assay and activity documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions).

Recommended ingestion contract:

1. Store immutable raw files, source URL, retrieval time, source/package release, checksum and dataset-specific terms/citation. Pin the loader and data version; TDC KIBA/Davis counts differ from the DeepDTA release already used here.
2. Normalize into `dataset_id, source_record_id, molecule_id, target_id, canonical_smiles, sequence_hash, sequence, endpoint, value, unit, relation, assay_id, publication_id, organism, target_family` plus raw values and provenance. Record missing metadata explicitly.
3. Convert supported positive concentrations with explicit units: for nM, `pX = 9 - log10(X_nM)`; for molar, `pX = -log10(X_M)`. Retain endpoint type. Do not use the present `max > 100` heuristic or clip invalid zero values into artificial high affinity.
4. For the first regression benchmark accept exact `=` measurements; count and retain censored `<`/`>` rows separately for later interval/censoring-aware work. Never treat an inequality boundary as an exact observation. Keep unmeasured interactions unknown.
5. Resolve chemical and sequence identity before deduplication and splitting. Assess repeated-assay disagreement and define aggregation before results are inspected. Hold publication/assay groups together where the evaluation question requires it.
6. Stratify reports by GPCR, protease, ion-channel, nuclear-receptor and kinase families where sufficient verified data exists. Do not assume a broad source is balanced across families. Include coverage and sample sizes.
7. For cross-source validation remove overlapping molecule–protein–endpoint identities and reused assay/publication records. For cold evaluation also exclude the relevant molecule/scaffold or protein/homology groups. BindingDB imports some ChEMBL data, so different source names do not guarantee independent test sets.

Avoid pooling KIBA with pKi/pKd/pIC50 under one undifferentiated regression label. Start with separate trained models. Later test shared encoders with endpoint-specific heads and training-only normalization, preserving endpoint-specific output and evaluation.

## Model improvement experiments

Start by reproducing all baselines on corrected data. Compare **tuned XGBoost, tuned CNN and tuned GNN**; never weaken XGBoost to make neural results appear stronger.

| Experiment | Change | What it tests |
|---|---|---|
| A | Corrected existing features/models | Reliable new baseline after identity and serving fixes |
| B | CNN width/kernel/dropout/lr/batch/weight-decay tuning; length and masked-pooling ablations | Whether the current CNN is under-tuned or loses useful sequence context |
| C | Bond-aware GINE or message-passing encoder; residual connections and validated pooling | Whether explicit bond information improves on the current GCN |
| D | Cached pretrained protein embeddings with a small prediction head | Whether protein representation limits cold-target performance |
| E | The same pretrained protein embeddings supplied to XGBoost and neural heads | Distinguishes representation benefits from architecture benefits |
| F | Molecular GNN plus pretrained protein representation and interaction fusion | Candidate primary research model if supported by validation |
| G | Validation-fitted ensemble of eligible models | Whether complementary errors improve prediction enough to justify complexity |

[ESM's official repository](https://github.com/facebookresearch/esm) provides pretrained protein models and embedding extraction. Start with a smaller frozen encoder subject to available hardware and record pooling/truncation. This is a proposed experiment, not a measured improvement. Cross-attention and fine-tuning are later ablations if data, memory and compute support them. The existing GraphDTA is a local variant, not a demonstrated exact reproduction of the published model; the [GraphDTA paper](https://academic.oup.com/bioinformatics/article/37/8/1140/5942970) compares several graph variants.

Report RMSE, MAE, R², Pearson, Spearman and exact/clearly identified approximate CI; macro per-target and family metrics; top-k/enrichment with prespecified relevance and per-target grouping; seed variation; latency, memory and compute. Add scaffold and protein-sequence-cluster holdouts after basic identity splits pass. Genuine generalization tests can produce higher errors even when the methodology improves.

For uncertainty, test small seed ensembles and validation-calibrated prediction intervals, then report observed coverage and width on each split. Prediction variance is not automatically calibrated confidence; applicability checks should reflect similarity to training compounds/proteins and the model's actual training domain.

## Primary-model selection and serving

**Current decision:** tuned XGBoost is the provisional historical KIBA RMSE leader; DeepDTA is the historical MAE/CI leader. There is no deployable primary artifact in this checkout. Selecting or promoting a new winner requires restored/retrained artifacts and corrected evaluation.

The model registry should store model ID, architecture, dataset release, endpoint and scale, preprocessing/tokenizer version, input limits, artifact checksum, split manifest, seed, code/dependency versions, validation metrics, locked test metrics, supported explanation method, training-domain coverage, latency and load status.

Define one default for each `(dataset_version, endpoint, intended_use)` group. Prespecify the primary validation criterion: e.g. RMSE for affinity estimation, with cold-target performance used when new targets are the intended use. Select among loadable models with matching contracts; use documented tie-breaks and uncertainty rather than differences in rounded numbers. Report the final test performance only after selection is locked. Keep an explicit promotion event and rollback reference.

All prediction/batch/screen requests should accept a model ID; omission resolves to that group's default. Return the resolved model ID/version and dataset/endpoint/scale every time. Model selection must be request-scoped so two simultaneous users cannot change one global pipeline. Use a bounded cache for loaded artifacts and report unavailable models as unavailable. Expose compatible trained alternatives in the dashboard; experiment log rows without artifacts are historical results, not runnable models.

## Next.js dashboard specification

Use Next.js App Router with TypeScript and accessible reusable UI components; keep FastAPI/PyTorch/RDKit inference as a Python service. Follow [official App Router guidance](https://nextjs.org/docs/app). Prefer a same-origin API proxy and explicit server configuration. Dataset preprocessing and model training run as worker/CLI jobs, never inside a page render or request handler.

Visual direction: professional laboratory workspace, white/off-white background, slate text, one restrained blue/teal accent, fine borders, compact tables, clear typography, and an optional accessible dark theme. Use a persistent sidebar, functional toolbar and sensible page titles. Remove the current promotional hero/glow/emoji styling in the migration. Focus on usable research workflows and explicit data provenance.

| Page | Required capabilities |
|---|---|
| Overview | Dataset coverage, actual runnable models, selected default, recent jobs, service readiness and concise validation summary |
| Predict | Dataset/endpoint then compatible model selector; SMILES and FASTA/sequence validation; molecule drawing; numeric score and scale; applicability warnings; supported explanations |
| Compare predictions | Same inputs across compatible models; per-model scores and timings; no ranking across incompatible output scales |
| Batch jobs | CSV template/upload with preview; paired rows versus drugs×targets mode; validation; workload estimate; progress/cancel/retry; failed-row report; downloadable results |
| Screening results | Filter/sort/paginate; per-target ranking; heatmap for bounded selections; row drill-down, optional top-hit explanation and export |
| Models | Separate experiment results from loadable artifacts; filter by dataset, endpoint, split and metric; uncertainty; model detail and default badge |
| Datasets | Release/license/provenance, endpoint distribution, family coverage, score distribution, duplicate/censoring exclusions and split manifests |
| Experiments | Frozen configs, seeds, training curves, metrics, ablations and reproducible run/export details |

Proposed API contract: dataset catalog/details; model catalog/details/default resolution; validated single predictions with `model_id`; persistent batch/screen job creation; job status, cancellation, paginated rows and exports. Job records must retain source row IDs, model/preprocessor versions, successful/failed counts, timestamps and input hashes. Return null rather than NaN and structured error codes; failed rows remain unranked. Constrain file size, sequence length and Cartesian pair count. Stream/chunk results and cache unique inputs. Use SQLite for an initial single-machine persistent job store, or Postgres with worker coordination when deployed; select the queue based on the actual deployment environment.

## Implementation order and completion criteria

1. **Repair research foundation.** Correct identity grouping, cold-both, score metadata, finite metrics, early stopping, tensor shapes and complete checkpoint contracts. Add meaningful synthetic leakage tests, one-sample graph batch checks, a save/load prediction parity test for each supported model, unit-conversion tests and strict-JSON API tests. Rebuild splits and record old/new provenance.
2. **Integrate broad data.** Run a small BindingDB Kd pipeline smoke test, then full Ki and Kd ingestion. Deliver immutable manifests, quality reports, family coverage, deterministic splits and reproducible CLI/config training. Preserve KIBA compatibility.
3. **Run fair experiments.** Reproduce baselines, tune CNN/GNN, test representations and finalists across fixed seeds and intended-use splits. Deliver per-row held-out predictions, confidence analysis and comparisons that can be recomputed. Set performance targets only after this baseline; no guaranteed RMSE or model ordering.
4. **Wire serving and jobs.** Restore/register valid artifacts, implement scoped defaults/model selection, vectorized/chunked inference, persistent jobs and proper error contracts. Verify that every advertised model loads and predicts and that missing artifacts produce truthful readiness status.
5. **Build Next.js workspace.** Deliver the pages above using actual API data, with loading/empty/error states, responsive keyboard-accessible interactions and reproducible exports. Test single prediction, alternative-model comparison, invalid rows, partly failing batch, cancellation and download end to end.
6. **Finalize research output.** Write methods, dataset versions, split policy, compute budgets, ablations, statistical limits and reproducibility instructions. Revise README, summary, slides and old requirement lists to match measured capabilities. A negative neural result remains valid research evidence.

## Verification performed for this review

`python scripts/audit_research.py` scans actual committed CSV identities/splits and saved experiment/notebook records using only the standard library. Its machine-readable output is `results/reports/research_audit.json`; it includes the cleaned-data SHA-256, all partition-pair overlap counts, truncation counts, recorded model metrics/seeds, artifact inventory and dependency availability. It completed successfully.

All 14 notebooks were parsed and their code paths reviewed; none contains a saved error output. That does not mean they currently run successfully. No training or API/browser execution was performed: trained artifacts and scientific/API dependencies are absent. No dataset was downloaded, model metric improved, default promoted, or dashboard migrated during this review. Only this review, its audit script and audit results were added.
