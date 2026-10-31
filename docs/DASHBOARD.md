# Next.js research workspace

Use Node 24 and pnpm 11.25.0. From the repository root, start the Python API:

```powershell
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd dashboard
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
pnpm start
```

Open http://127.0.0.1:3000. The server proxies `/api` to the local Python service. Set `PHARMALENS_API_URL` before building to change the upstream. No browser secret is required. Use `pnpm dev` during UI development.

The model selector lists actual registered artifacts scoped by dataset, release and endpoint. A validation experiment alone does not register or promote a model. Set `PHARMALENS_REGISTRY_PATH` when using a separate registry; never treat a QA default as a final research champion.

For the existing local verification artifacts, the isolated preview registry is `experiments/runs/dashboard_preview_qa/registry.json`. Set the environment variable to its absolute path before starting the API. This ignored directory must be reproduced or archived separately; it is not distributed in Git. The local registry contains all 57 new trained candidates, prior baselines and repeat-seed alternatives. Corrected KIBA and broad Ki use GNN defaults; other task defaults follow their validation evidence. All are labeled provisional validation models; final research selection remains pending.

CSV batches require `smiles,sequence` headers with optional `id`, at most 1,000 pairs and 2 MB. Invalid chemistry/sequence rows return individual errors. Exports preserve identifiers, failures, score scale and model provenance and escape spreadsheet formulas. Jobs are stored in SQLite (`data/jobs/predictions.sqlite3`, configurable with `PHARMALENS_JOBS_DB`). They survive page refreshes and have saved progress, history, result export, cancellation between rows and manual retries. Each lineage permits at most three execution attempts; the queue accepts at most 50 pending/running jobs. Atomic worker claims and renewable 120-second leases recover abandoned work after a process crash. Recovery may repeat computation; it is not an exactly-once execution guarantee. Keep the local database with the installation; use SQLite-aware backups. The existing synchronous `/predict/batch` endpoint remains supported.

Verified locally: strict TypeScript check, production build, API proxy/catalogs, browser single prediction, and a two-row CSV with one successful prediction and one invalid-molecule failure. A 1,000-row saved job was cancelled, retried to completion and exported with identifiers and artifact checksums preserved; page refresh restored the saved result. This is functional verification, not a capacity or accessibility certification. Do not expose the service publicly until deployment-specific authentication, worker isolation and resource limits are implemented.

The Experiments view refreshes saved matrix/checkpoint state every 15 seconds. Intermediate normalized training MSE is labeled separately from final validation/test RMSE. Held-out metrics are linked to exact artifact hashes through `results/reports/final_benchmark_v1.json`; defaults were locked before test evaluation.
