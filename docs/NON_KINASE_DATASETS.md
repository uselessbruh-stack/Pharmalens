# Non-kinase datasets and research controls

Updated: 30 October 2026. This document specifies the research expansion. Acquisition and preparation are now implemented for BindingDB and an 18-target ChEMBL starter panel; see [actual phase status](IMPLEMENTATION_PHASES.md) and [commands](DATA_PIPELINE.md). Bounded download samples do not establish a complete family benchmark, and no new champion has been trained. Family names describe curated subsets; they are not claims that six independent databases exist.

## Recommended scope

Keep KIBA as the kinase reference and create six explicit non-kinase panels. Build the first release from a pinned ChEMBL release and a pinned BindingDB snapshot; retain database, assay, publication and record IDs. Extract a kinase-excluded broad benchmark as well as family slices. Classify targets using curated identifiers and family hierarchies, not target-name keyword matching.

| Panel | Illustrative targets to seek | Primary records | Special handling |
|---|---|---|---|
| **GPCRs** | Dopamine, serotonin, adrenergic and opioid receptors | ChEMBL/BindingDB Ki or Kd; GPCRdb for receptor annotation and record inspection | Separate binding from agonist/antagonist functional potency. Retain species, receptor subtype and construct identity. Restrict initial molecular scope to supported small molecules. |
| **Proteases** | Thrombin, factor Xa, cathepsins, HIV protease and viral main proteases | ChEMBL/BindingDB Ki, Kd and a separate IC50 task | Separate reversible affinity from covalent or time-dependent inhibition; preserve mutants and assay conditions. Do not merge kinetic constants such as kinact/KI into Ki. |
| **Nuclear receptors** | Estrogen, androgen, glucocorticoid and PPAR receptors | ChEMBL/BindingDB binding records; curated pharmacology for checks | Ligand binding and transcriptional activation are different tasks. Preserve full-length versus ligand-binding-domain constructs. |
| **Ion channels** | hERG/KCNH2 and appropriately identified voltage-/ligand-gated channels | ChEMBL binding assays where available; channel inhibition as a separate functional task | Many measurements are functional block rather than binding. Multisubunit or state-dependent measurements require explicit context; exclude unresolved complexes from the first single-protein benchmark. |
| **Transporters** | SERT/SLC6A4, DAT/SLC6A3 and NET/SLC6A2 | ChEMBL/BindingDB exact-target affinity records; uptake inhibition separately | Uptake/transport activity is not automatically affinity. Keep membrane system, species and endpoint annotations. |
| **Other non-kinase enzymes** | Carbonic anhydrases, acetylcholinesterase, phosphodiesterases, HDACs and selected CYPs | ChEMBL/BindingDB, endpoint-specific extraction | Subfamily and cofactor/substrate context matter. Distinguish inhibitor binding, enzyme inhibition, substrate status and induction. |

These are candidate target lists, not verified counts or guaranteed available exact-binding records. Measure retained targets, scaffolds, interactions and endpoint distribution after filtering. Sparse panels remain descriptive/evaluation subsets until there is enough data for defensible training and held-out evaluation. Do not fill a sparse binding panel with functional measurements just to increase row count.

ChEMBL documents a hierarchical target classification and separately labels binding and functional assays. Use these fields to implement the panel filters: [target classification](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-interface-questions), [assay/endpoint/confidence documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions). BindingDB provides measured affinity records and source-specific downloads: [downloads](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp), [field specification](https://www.bindingdb.org/rwd/bind/chemsearch/marvin/BindingDB-TSV-Format.pdf).

## Sources and their roles

1. **ChEMBL family panels — primary family coverage.** Select explicit small-molecule structures, supported target mappings, assay type B and exact positive Ki/Kd initially. For the first single-protein release require target type SINGLE PROTEIN and confidence score 9. Save endpoint-specific counts and excluded-complex counts. A separately named functional dataset can later use appropriate IC50/EC50 measurements.
2. **BindingDB Ki/Kd — primary broad affinity training.** Use these separately and annotate families through exact UniProt/sequence mappings. Publish both full-source results and a verified non-kinase-only slice. Do not assume every TDC record has enough assay metadata for strict publication-level overlap checks; use a full snapshot when necessary, or explicitly state the limitation. [TDC DTI documentation](https://tdcommons.ai/multi_pred_tasks/dti/).
3. **IUPHAR/BPS Guide to PHARMACOLOGY — curated checks and annotations.** Provides family-specific interaction files for GPCRs, ion channels, nuclear receptors, transporters and enzymes. Join interactions with ligand structures and exact target sequences; exclude peptide/antibody ligands from the initial small-molecule pipeline. Preserve quantitative affinity type and relation. Its API currently requires registration and an API key; use a permitted versioned export or authorized API access. The official download page states ODbL for the database and CC BY-SA 4.0 for contents; preserve the applicable attribution/redistribution terms. [Downloads](https://www.guidetopharmacology.org/download.jsp), [API access](https://www.guidetopharmacology.org/webServices.jsp).
4. **GPCRdb — specialist GPCR annotation and optional curated activity slice.** Use extended per-activity records rather than compact ligand-level min/mean/max summaries. The resource incorporates ChEMBL, Guide to PHARMACOLOGY and PDSP Ki records. It is not inherently independent validation. [Activity documentation](https://docs.gpcrdb.org/ligands.html).
5. **LIT-PCBA — optional screening benchmark.** Use eligible non-kinase target assays to evaluate active/inactive ranking with experimental inactive compounds. This is a separate classification/screening task, not continuous-affinity training data. Inspect each assay's biological context and target mapping; a cell-based response must not be relabeled as direct binding to a single protein. [Official dataset](https://lab.drugdesign.unistra.fr/datasets/lit-pcba/).
6. **PDBbind — optional later structure experiment.** Contains measured affinities paired with complex structures, enabling a distinct pocket/3D-model study. It is not required for the current SMILES+sequence pipeline, although a compatible subset could also be evaluated without structural inputs. Current PDBbind+ access is commercialized; inspect release-specific access and terms before choosing it. Prevent ligand, protein and complex-similarity leakage. [Official PDBbind+](https://pdbbind-plus.org.cn/).

Do not add drug-only ADMET datasets, molecule property benchmarks, drug–drug interactions or protein–protein interactions to the affinity target column. They can support separately defined auxiliary tasks, but their labels do not measure small-molecule–protein affinity. DUD-E-style generated decoys must not be treated as experimentally established non-binders.

## Initial datasets to produce

- `bindingdb_ki_broad` and `bindingdb_kd_broad`: independent endpoint definitions, with overlap reports rather than an independence claim.
- `non_kinase_ki` and `non_kinase_kd`: verified kinase-excluded slices; unresolved/multi-family targets remain excluded or separately labeled.
- `gpcr_binding`, `protease_binding`, `nuclear_receptor_binding`, `ion_channel_binding`, `transporter_binding`, `other_enzyme_binding`: family panels partitioned by endpoint. A panel is a view over retained records, not duplicated training rows.
- `gtopdb_curated_binding`: a candidate external check only after overlap and selection-bias analysis; otherwise a labeled in-domain curated check.
- Later: `functional_inhibition_ic50` and `lit_pcba_screening`, each with its own task, labels, metrics and compatible models.

Start with Ki and Kd rather than pooling every assay. For nM values, `pKi/pKd = 9 - log10(value_nM)`; higher is stronger. Preserve original concentration, units and inequality relation. Conversion to a common logarithmic scale does not make Ki, Kd, IC50 and EC50 interchangeable. Family and endpoint are separate axes: every family may have several endpoint-specific tasks.

## How to counter the problems properly

| Problem | Required control | Evidence required before accepting results |
|---|---|---|
| Duplicate molecules under different IDs | Standardize molecules, assign stable identity, hash exact sequences, group repeated identities before splitting | No identical molecule–sequence–endpoint observation crosses partitions; zero molecule overlap in cold-drug evaluation |
| Similar chemicals/proteins leaking across splits | Add scaffold-held-out and sequence-cluster-held-out evaluations; document clustering thresholds and alignment method | Group disjointness and train/test similarity summaries, including per-family sample sizes |
| Broken cold-both split | Restrict training to training molecules AND training proteins; report discarded cross-block pairs | Zero held-out molecule/protein identities in training; nonempty validation/test blocks |
| Database overlap | Match normalized identity, endpoint, assay/publication and source record IDs | An overlap ledger and explicitly quarantined duplicates across all sources |
| Mixed/incorrect labels | Explicit endpoint, units, transform and direction metadata; exact numeric observations first | Unit conversion checks, invalid-value counts and separate regression tasks |
| Inequalities and discordant replicates | Keep censored measurements separate; group assay-compatible replicates and report dispersion | No censored boundaries passed off as exact values; versioned aggregation policy |
| Kinases or popular targets dominate | Report non-kinase-only, per-family and macro-per-target scores alongside micro scores; consider training-only group weighting | All family counts disclosed; improvement cannot be explained only by the largest family |
| Source/assay shortcuts | Publication/assay holdouts and source-specific checks where metadata supports them | Residuals by source, assay and family; explicit unknown-metadata rate |
| Single-target shortcuts | Drug-only, protein-only, per-target mean and similarity baselines fitted using training data only | Pair models improve beyond shortcut baselines under the relevant split; defined fallback for unseen entities |
| Long sequences and unknown tokens | Measure truncation; consistent unknown/padding tokens; validated masking/chunking/longer-input ablations | Save/load preprocessing parity and errors stratified by length/truncation |
| Fragile training/inference | Correct one-sample tensor handling, weighted validation loss, finite checks and real XGBoost early stopping | Batch-size-one checks, finite outputs and checkpoint round-trip prediction parity |
| CNN/GNN under-tuned | Comparable validation-only search budgets; full feature-matched boosting baselines | Search logs, multiple initialization seeds and locked final-test protocol |
| Protein representation too weak | Compare CNN and cached pretrained protein embeddings; give the same embeddings to XGBoost too | Ablation distinguishes representation gains from architecture gains |
| GNN ignores bond chemistry | Test bond-aware message passing and cache immutable molecular graphs | Bond-attribute ablation and measured quality/latency/memory |
| Uncertain winner | Prespecify primary metric, repeat seeds and report paired group-bootstrap differences | Effect sizes, uncertainty and validation-based selection; test set remains untouched until selection |
| Good RMSE but poor screening | Evaluate per-target top-k/enrichment on compatible exact labels or appropriate screened actives/inactives | Relevance threshold chosen beforehand, class prevalence and ranking uncertainty |
| Overconfident novel-family predictions | Family holdouts, similarity applicability checks and validation-calibrated intervals | Coverage/width and error reported on in-domain and shifted groups separately |
| Dashboard selects incompatible model | Dataset/endpoint-scoped defaults and compatible-model filtering | Every response/export identifies model, version, endpoint, scale and direction |
| Batch failures and blocking requests | Typed rows, workload bounds, worker jobs, cached features, failed-row reports and strict JSON nulls | Partial failures preserve row IDs; failed rows unranked; concurrent model choices isolated |

Scaffold split and cold-drug split answer different questions; neither should be described as automatically solving every chemical-similarity leak. A whole-family holdout asks a substantially harder question than new targets within a familiar family. Record these settings separately. Greater biological diversity can increase measured error while making the experiment more useful.

## Experiment and selection policy

Use the existing KIBA results as historical evidence only. Rebuild identity-correct splits and establish new baselines before setting numeric improvement targets. The first neural candidate should be a bond-aware molecular GNN plus a compact pretrained protein representation; a tuned masked CNN remains an important comparator. Include a feature-matched XGBoost baseline. This is an experiment proposal, not a prediction that the GNN will win.

For each endpoint run random identity-grouped, cold-drug, cold-target and corrected cold-both splits where sample support permits. Add scaffold/sequence-cluster and family holdouts for finalists. Use at least three model seeds for finalists; distinguish repeated model initialization from changing the actual split. Tune and select with validation data only. Publish all attempted panel results, including weak or unsupported panels.

Compare RMSE, MAE, R², correlation, concordance and per-target/family aggregates. Compare errors within the same endpoint, release and split. For classification use PR-AUC, ROC-AUC and screening enrichment with prevalence stated; do not compare those scores directly with regression RMSE. Never convert model disagreement into an unsupported confidence percentage.

Choose the dashboard default within each endpoint/dataset/use-case group from valid loadable artifacts using the prespecified validation criterion. A cross-family model can become the default only when evidence supports its intended domain. Family-specific fine-tuning is optional and requires sufficient data and its own held-out evaluation. Show all compatible alternatives and their limitations.

## Delivery sequence

1. Correct the existing research defects and add split/label/inference regression checks.
2. Pin source releases, acquire permitted exports, build target-family mappings and inspect a small Ki/Kd ingestion sample.
3. Produce the broad and six-panel datasets with exclusions, provenance and family coverage; freeze split manifests.
4. Reproduce/tune baselines and neural models; run the planned controls and select finalists using validation evidence.
5. Register actual trained artifacts; integrate dataset/family/endpoint/model selectors and persistent batch workflows in the Next.js dashboard.

The catalog at `experiments/configs/dataset_catalog.json` makes the intended sources, panels, endpoints, task types and status explicit. The downloader, preparation/curation pipeline and initial Ki validation models are implemented; catalog status is distinct from artifact availability. See `RESEARCH_REVIEW.md` for the complete code-level repair and dashboard specification.
