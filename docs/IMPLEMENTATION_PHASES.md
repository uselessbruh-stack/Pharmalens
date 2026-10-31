# Implementation phases

Status: 31 October 2026. The expanded identity-grouped random benchmark and local dashboard/job workflows are implemented and verified. This is not production certification or a claim of external-domain validity.

## Phase 1 — Dataset acquisition and research integrity: implemented

- Dependency-free acquisition CLI for pinned BindingDB article/PDSP snapshots; full BindingDB snapshot is optional.
- Configured ChEMBL starter panel: 18 accessions across six verified non-kinase families. This is a deliberate starter selection, not an exhaustive or balanced family benchmark.
- HTTPS, request timeouts, transient-request retries, streamed downloads, archive validation, publisher checksum checks where supplied, SHA-256 receipts and atomic file publication.
- ChEMBL release guards, cached checksummed responses, classification ancestry checks, source assay/molecule metadata and explicit sample status.
- Exact endpoint/unit contracts and a preparation CLI with molecule/sequence identities, source provenance, endpoint-specific CSVs and per-observation exclusion reports.
- Identity-grouped random/cold-drug/cold-target splits and corrected strict cold-both evaluation. Corrected KIBA splits are separate from historical CSVs; old metrics do not apply to them.

## Phase 2 — Model-serving and training reliability: implemented foundation

- Real XGBoost early stopping; sample-weighted validation loss; explicit batch types; one-sample graph handling; non-finite loss/gradient/metric checks.
- Full GNN constructor configuration and configured neural input lengths on reload; CPU/device-safe inference; checkpoint round-trip regression checks.
- Model IDs and defaults scoped by dataset, release and endpoint; validation-based selection, immutable registration IDs and artifact checksums. Registration requires actual artifacts. Model catalog availability is an artifact-presence/provenance indicator; inference still checks the artifact on first load.
- Model-selectable single/batch/screen endpoints; typed bounded inputs; failed rows retain identities and remain unranked; strict JSON nulls; blocking inference runs through synchronous FastAPI worker handlers rather than the async event loop.
- Correct score direction and removal of unsupported binder categories in legacy frontend; escaped screening IDs; comparison restricted to historical KIBA random results.
- Offline regression suite, pinned tested Python environment, and a Windows GitHub Actions workflow. CI is configured locally; it has not run on GitHub in this session.

Remaining Phase 2 work: durable batch jobs, cancellation/retries, bounded model-worker concurrency, validated explainability, structured operational metrics and load testing. Synchronous batch/screen endpoints are intentionally bounded and are not a durable job queue.

## Phase 3 — Broad-data experiments and model improvement: benchmark complete

- Acquired the pinned full BindingDB snapshot and a ChEMBL 37 catalog of 2,999 classified human non-kinase reference sequences.
- Published 309,685 Ki and 21,219 Kd observations, covering 1,638 distinct non-kinase proteins, plus 11 supported family/endpoint views. Overlapping views are not independent datasets.
- Completed 57 new benchmark runs: 42 initial broad/family/original-panel comparisons, three corrected-KIBA models and 12 nuclear-receptor repeat seeds.
- Added feature-matched hybrid CNN/GINE branches, frozen ESM-2 representations, train-only normalization, reusable integrity-checked feature matrices and validated Intel XPU execution.
- Locked validation-based architecture selection before held-out evaluation of all 57 artifacts. KIBA test RMSE: GINE 0.3689, CNN 0.3730, XGBoost 0.4189. Broad Ki: GINE 0.7519, XGBoost 0.7633. All stronger and weaker results are published in `results/reports/final_benchmark_v1.md`.
- Registered every new benchmark artifact and preserved fixed seed-42 representatives for defaults; original KIBA and broad Ki use GNN. Defaults were not changed using test results.

Remaining research controls: repeated seeds beyond the nuclear-receptor endpoints; full scaffold/homology/family and assay/publication holdout experiments; graph/descriptor ablations; calibrated predictive intervals. The implemented scaffold/family split functions do not mean every model has been evaluated under those protocols. Do not retune against the released test results.

## Phase 4 — Next.js research dashboard: initial implementation verified

The `dashboard/` TypeScript App Router workspace includes overview/family coverage, dataset provenance, scoped model selection, real single predictions, CSV batch upload with row-level failures and export, model catalog and completed experiment records. Backend catalogs verify dataset file hashes and omit incomplete runs. Type checking and production build pass. Browser checks cover a real non-kinase prediction and a mixed-validity CSV batch. Initial preview used an isolated QA registry. Four validated local research artifacts are now registered for normal startup, with XGBoost as the provisional default for each task. Legacy HTML remains available through the API root.

Saved batch jobs now use SQLite persistence, atomic claims, renewable leases, bounded retries, cooperative cancellation, recent-job history and refresh recovery. Training matrix/checkpoint monitoring and distinct held-out scores are visible. The bounded synchronous API remains available. Remaining deployment work includes authentication, sustained capacity testing, worker/process resource policy and hosted artifact distribution. See `DASHBOARD.md` for startup.

## Phase 5 — Release hardening: planned

Add worker deployment and persistent storage, authentication if exposed beyond a local trusted environment, resource limits, concurrency/load checks, structured logs/metrics, artifact distribution/rollback, end-to-end browser checks and final reproducibility documentation. Security and deployment checks must match the actual deployment environment. Research metrics and biological claims remain separately reviewed.

## Verified local outputs

See `results/reports/phase3_validation.json` and `phase3_experiments.json` for the earlier foundation verification, real training metrics and browser/API checks. The earlier `implementation_validation.json` records foundation source checksums and corrected KIBA split sizes. Raw and prepared data live in ignored `data/raw/` and `data/processed/prepared/` directories. Keep or archive those directories separately from Git when reproducing the work.

Latest verification: `results/reports/expanded_validation.json`; final research scores: `results/reports/final_benchmark_v1.json`.
