# PharmaLens Project Summary

Research implementation and benchmark review

Prepared for the project owner on 31 October 2026. This report describes the inspected original system, the changes completed in this workspace, measured results and the remaining research and deployment work. Counts and scores refer to the pinned local releases and reports, not future downloads.

PharmaLens now supports broad non-kinase affinity research alongside KIBA. The expansion produced 330,904 curated non-kinase observations, 57 completed training runs with separate held-out evaluation, and a Next.js dashboard with persistent batch processing. Hybrid GNN models improved test RMSE on corrected KIBA and broad non-kinase Ki; XGBoost remains stronger on several other tasks.

### Decision on one model for everything

Keep the existing models for now. Separate artifacts currently represent different datasets, endpoints and experimental alternatives. However, the broad Ki model already covers all six non-kinase families in one trained model; using it does not require choosing a family specialist. The broad Kd model likewise spans the available families, including the small transporter subset.

KIBA is a benchmark score, while Ki and Kd are distinct experimental endpoints represented here as negative logarithms of molar concentrations. They should not be merged into one unqualified numeric output. Our recommendation is one application with clear endpoint selection and compatible defaults. Family models remain optional research comparisons.

A future single checkpoint could share molecular and protein encoders and expose separate Ki, Kd and KIBA prediction heads. This is a proposed multitask experiment, not an implemented or validated universal model. It would need deduplication across sources, compatible task labels, loss balancing and evaluation against the current endpoint models before replacing them. No current defaults or artifacts were changed for this summary.

### Original system and completed changes

| Area | Inspected original state | Current implementation |
|---|---|---|
| Data | KIBA processed data; Davis loader existed | Full BindingDB non-kinase Ki/Kd plus ChEMBL reference annotations |
| Models | Engineered-feature baselines, CNN and GCN/GAT code | Matched XGBoost and hybrid CNN/GINE with frozen ESM-2 features |
| Evaluation | Historical splits contained identity leakage | Identity grouping, immutable split evidence and validation-locked test evaluation |
| Serving | FastAPI and a plain HTML dashboard; no loadable artifacts initially present | 69 locally registered artifacts and a Next.js workspace |
| Batch work | Synchronous prediction and screening | Saved jobs with progress, cancellation, retry, history and exports |

Historical values and claims are preserved as historical evidence. They are not treated as directly comparable to the corrected benchmark.

## Data expansion and preparation

The full BindingDB October 2026 archive was acquired and verified before preparation. Its size is 594,284,548 bytes and SHA-256 is e968f90e900c7833d5e6eceae282d0ca3b75201448bccaf4df6b09cc410f7d76. ChEMBL 37 supplied 2,999 classified human reference sequences for exact-sequence family annotation. These are reference entries, not 2,999 trained targets.

| Broad task | Observations | Unique molecules | Proteins | Unique pairs |
|---|---:|---:|---:|---:|
| Non-kinase Ki v3 | 309,685 | 143,035 | 995 | 244,344 |
| Non-kinase Kd v3 | 21,219 | 10,007 | 1,134 | 15,952 |

The union contains 1,638 distinct protein sequences. Molecule and protein counts overlap between endpoints and should not be added. The existing KIBA benchmark contains 118,254 rows and 229 protein sequences.

| Non-kinase family | Ki observations | Kd observations |
|---|---:|---:|
| GPCR | 165,822 | 3,223 |
| Other enzyme | 73,333 | 12,853 |
| Protease | 39,797 | 1,332 |
| Ion channel | 13,938 | 566 |
| Transporter | 9,204 | 335 |
| Nuclear receptor | 7,591 | 2,910 |

All six Ki and five Kd family cohorts met the fixed support rule of at least 500 observations, 100 molecules and three proteins. Transporter Kd did not meet that rule and was not trained as a separate family task. Family views are subsets of the broad datasets, not additional independent data sources.

The pipeline downloads with receipts and hashes, validates molecular structures and protein sequences, separates endpoints, converts accepted concentrations to pKi or pKd, assigns molecule and sequence identities, maps families, and publishes versioned datasets and manifests. Larger pKi/pKd values mean stronger binding. KIBA retains its own documented score convention and is not converted to nM.

From 755,232 source endpoint observations, preparation excluded unresolved families (355,821), censored measurements (54,214), multi-chain entries (12,816), invalid SMILES (1,328), ambiguous sequences (91), missing SMILES (40), and invalid concentrations (18). The retained counts sum to 330,904. Unknown family is an exclusion, not evidence of non-kinase identity.

An earlier bounded ChEMBL activity panel and starter BindingDB combinations were used during development. The final v3 broad tasks use BindingDB alone to avoid pooling overlapping source copies again. Repeated measurements remain grouped by interaction identity; discordant assays were not selectively removed to improve scores. The Ki audit found 7,889 repeated pairs with more than 0.5 log-unit spread and 4,687 with more than 1 log-unit spread. More reference proteins do not automatically mean more measured training pairs.

## Model and evaluation methods

The new comparisons use XGBoost, a hybrid molecular CNN and a hybrid molecular graph network using bond-aware GINE layers. The ESM variants combine learned molecular representations with engineered descriptors, fingerprints and frozen protein embeddings. They do not establish the effect of graph convolutions in isolation.

All three architectures receive the same frozen ESM-2 protein features. The model is esm2_t6_8M_UR50D, using layer 6 and 320 dimensions. Residue means exclude padding and special tokens. Proteins longer than 1,022 residues are encoded in nonoverlapping chunks and pooled by residue count; this loses context across chunks. No affinity labels were used to fit the protein encoder. The expanded cache covers 1,638 proteins and a separate KIBA cache covers 229.

Shared auxiliary features contain molecular descriptors and fingerprints plus protein composition; ESM extends the 1,454-dimensional branch to 1,774 dimensions. Neural normalization is fitted on training data and stored in checkpoints. Optional interaction fusion allows molecular and protein features to interact. The ESM variants use frozen protein representations rather than the earlier protein CNN branch.

Training corrections include masked CNN processing, graph and protein caches, per-batch protein deduplication, finite loss and gradient checks, early stopping, best-weight restoration and explicit CPU/XPU/CUDA selection. The model is moved to the device before optimizer creation. Separate pinned CPU and Intel XPU environments were checked; serving uses verified local feature and model contracts with checksum validation.

### Completed experiment protocol

The 57 runs comprise 42 expanded comparisons (three starter-panel runs and 39 broad/family runs), three corrected KIBA comparisons, and 12 additional nuclear-receptor runs. The latter extend both nuclear endpoints to three initialization seeds per architecture. Most other tasks use one initialization. The run count does not mean 57 independent datasets.

Repeated molecule–sequence interaction identities are kept within one partition. Random splits may share individual molecules and proteins across partitions; they hold out pair identities. Source data, split memberships, configuration, code snapshot and artifact hashes accompany each run. Incomplete or smoke runs do not become benchmark winners.

Selection uses validation RMSE only, with mean validation performance across available initialization seeds. Seed 42 is the fixed serving representative; the best seed is not cherry-picked. Selection was frozen before all 57 held-out evaluations. Test scores did not change the selections. Prediction files and artifact checksums link the displayed scores to the evaluated weights.

Reported metrics include RMSE, MAE, correlation, R-squared and exact concordance, with per-target/family diagnostics and conditional paired target-group bootstrap comparisons. Scaffold and family split implementations are tested, but full neural experiments under these harder protocols have not been completed. The released test sets must not become hyperparameter tuning sets.

## Measured results and interpretation

Lower RMSE is better. Each row below compares models within the same task and split. KIBA RMSE and pKi/pKd RMSE have different target meanings and should not be compared directly across rows.

| Task | XGBoost test RMSE | CNN test RMSE | GNN test RMSE |
|---|---:|---:|---:|
| Corrected KIBA | 0.4189 | 0.3730 | 0.3689 |
| Broad non-kinase Ki | 0.7633 | 0.7645 | 0.7519 |
| Broad non-kinase Kd | 1.0388 | 1.0731 | 1.0677 |

The hybrid GNN reduced KIBA test RMSE by about 11.9% relative to matched XGBoost, and broad Ki RMSE by about 1.5%. Conditional target-bootstrap intervals for GNN minus XGBoost RMSE were approximately minus 0.0608 to minus 0.0390 on KIBA and minus 0.0204 to minus 0.0014 on broad Ki. These intervals are conditional on the evaluated split and do not resolve tuning bias, multiple comparisons or variation across initialization seeds.

Current broad defaults are GNN for KIBA, GNN for non-kinase Ki and XGBoost for non-kinase Kd. They were chosen using validation evidence, not the test table. All alternatives remain available for comparison. A family-specific winner is not required to use the broad endpoint model.

### Repeated seeds and unsuccessful improvements

| Nuclear receptor task | XGBoost test RMSE | CNN test RMSE | GNN test RMSE |
|---|---:|---:|---:|
| Ki mean of three seeds | 0.7223 | 0.7649 | 0.7867 |
| Kd mean of three seeds | 0.8453 | 0.8484 | 0.8582 |

Nuclear CNN validation means were better than boosting (Ki 0.7754 versus 0.8052; Kd 0.7517 versus 0.7791), but that advantage did not carry over as clear test superiority. The validation-selected CNN remains the representative rather than switching after observing the test. Other tasks also retain stronger boosting results; the appendix reports every scope.

The original saved KIBA log recorded tuned XGBoost RMSE 0.4206, DeepDTA 0.4249 and GCN 0.4303. These were historical values with measured identity leakage in the saved partitions. They are not a clean before/after causal estimate. The new 0.4189 boosting result is the relevant matched comparator for the corrected neural results.

An RMSE around 0.5 does not mean 50 percent error. It measures error in the target score units. For concentration endpoints, one log unit represents a tenfold concentration ratio, but converting a dataset RMSE into a guaranteed per-pair fold error would be incorrect. Lower error requires improved evidence and generalization, not a cosmetic change in score scale.

## Application delivery and verification

The Next.js and TypeScript workspace provides overview, single prediction, batch processing, models, datasets and experiments pages. A neutral laboratory-style interface replaces the simple dashboard as the primary workspace. FastAPI remains the prediction service, and the legacy interface is retained.

Model selection is scoped by dataset release and endpoint. The picker selects an available compatible default, and alternative trained artifacts remain accessible. The experiments view separates training loss, validation RMSE and artifact-linked test RMSE. It refreshes saved training state while the page is open. The dataset catalog verifies manifests and avoids adding overlapping family views as independent data.

Saved CSV jobs accept up to 1,000 pairs and 2 MB. SQLite stores job history and results. Atomic claims and renewable leases recover abandoned work; cancellation is cooperative between rows. A lineage permits three attempts and the queue permits 50 pending or running jobs. Model identity and checksum are fixed when a job is submitted. Retries may repeat computation, so this is not an exactly-once execution guarantee.

Exports retain row identifiers, successes, failures, score units and model provenance, and protect against spreadsheet formula execution. Short invalid sequences and invalid chemistry produce row errors instead of invalid numeric predictions. Page refresh can restore the saved result.

| Verification | Recorded outcome |
|---|---|
| Python suite | 93 passed with zero failures; two dependency deprecation warnings |
| Dashboard checks | TypeScript check and Next.js production build passed |
| Serving integration | Four actual artifacts; 16 concurrent requests with prediction parity |
| Browser batch flow | Mixed valid/invalid rows, refresh restoration, cancellation and retry passed |
| Export workflow | 1,000 completed rows; identifiers and artifact checksum verified |
| Runtime checks | CPU and XPU dependency checks and critical static checks passed |
| Remote CI | Workflow configured; remote execution not recorded |

One completed protease CNN artifact initially failed directory publication because of a Windows sharing lock. Hashes and serving parity were checked before recovery; no retraining or score editing occurred. Publication now retries transient permission locks without overwriting an existing published run.

The result is a verified local research implementation. Remaining work includes cold-drug/cold-target/scaffold and sequence-homology evaluation, publication or assay holdouts, more seeds, calibration and out-of-domain assessment. Deployment still requires authentication, worker isolation, backup operations and sustained resource testing. Existing explanation code should be reviewed separately; this work does not establish complete validated explanations for every new model. Large data, weights, feature caches and job databases require separate archiving because Git excludes them.

## Complete benchmark and evidence

The table includes all 15 evaluated scopes. Values are mean held-out RMSE over available initialization seeds. Most rows use seed 42; nuclear Ki and Kd each use seeds 42, 123 and 456. CNN and GNN mean the hybrid ESM-2 variants. The starter panel is an earlier release, and family views overlap the broad tasks.

<!-- BENCHMARK_START -->
| Scope | Endpoint | XGBoost | CNN | GNN | Seeds |
|---|---|---:|---:|---:|---:|
| KIBA | KIBA | 0.4189 | 0.3730 | 0.3689 | 1 |
| Broad non-kinase | Kd | 1.0388 | 1.0731 | 1.0677 | 1 |
| Broad non-kinase | Ki | 0.7633 | 0.7645 | 0.7519 | 1 |
| Starter panel | Ki | 0.7488 | 0.7765 | 0.7600 | 1 |
| GPCR | Kd | 0.9425 | 0.8751 | 0.8702 | 1 |
| GPCR | Ki | 0.7243 | 0.7317 | 0.7380 | 1 |
| Ion Channel | Kd | 1.0506 | 1.1198 | 1.1476 | 1 |
| Ion Channel | Ki | 0.6026 | 0.6613 | 0.6657 | 1 |
| Nuclear Receptor | Kd | 0.8453 | 0.8484 | 0.8582 | 3 |
| Nuclear Receptor | Ki | 0.7223 | 0.7649 | 0.7867 | 3 |
| Other Enzyme | Kd | 1.1403 | 1.1625 | 1.1680 | 1 |
| Other Enzyme | Ki | 0.7408 | 0.7727 | 0.7762 | 1 |
| Protease | Kd | 0.8346 | 0.9191 | 0.9754 | 1 |
| Protease | Ki | 0.7424 | 0.7460 | 0.7520 | 1 |
| Transporter | Ki | 0.6694 | 0.6682 | 0.6655 | 1 |
<!-- BENCHMARK_END -->

### Reproduce and inspect

Use docs/RESEARCH_EXPANSION.md for download, preparation, curation, feature caching, planning, training and frozen evaluation commands. Use docs/DASHBOARD.md to start FastAPI and Next.js. Generate fresh plans on another machine instead of editing archived paths and hashes. Read docs/IMPLEMENTATION_PHASES.md for completed and pending phases.

The main evidence files are results/reports/final_benchmark_v1.json, final_benchmark_v1.md, final_validation_selection_v1.json, expanded_affinity_quality.json, expanded_validation.json and expanded_tests.xml. Configurations are under experiments/configs. Full run artifacts and the 69-model registry are local and require separate preservation. The document reflects this local snapshot, not a production deployment or a guarantee for unseen drugs and proteins.

### Primary source documentation

BindingDB downloads: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

ChEMBL assay and activity definitions: https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions

DeepDTA dataset conventions: https://github.com/hkmztrk/DeepDTA/blob/master/data/README.md

ESM reference implementation: https://github.com/facebookresearch/esm

The recommendation to retain endpoint-specific outputs is a project design judgment based on these label distinctions and the current evidence. A shared multitask checkpoint remains proposed work.
