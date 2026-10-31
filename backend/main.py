"""
PharmaLens — FastAPI Backend

REST API for drug-target interaction prediction, screening, and exploration.
"""

import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.utils.serialization import json_safe

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("pharmalens.api")


# ─── FastAPI App ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(application):
    await startup_event()
    from backend.jobs import BatchJobStore, BatchJobWorker
    from src.utils.config import PROJECT_ROOT

    application.state.job_store = None
    application.state.job_worker = None
    try:
        store = BatchJobStore(
            os.getenv("PHARMALENS_JOBS_DB", str(PROJECT_ROOT / "data/jobs/predictions.sqlite3"))
        )
        worker = BatchJobWorker(store, execute_prediction_job)
        application.state.job_store, application.state.job_worker = store, worker
        worker.start()
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        logger.error("Prediction queue startup failed: %s", exc)
    try:
        yield
    finally:
        if application.state.job_worker:
            application.state.job_worker.stop()
        load_registered.cache_clear()


app = FastAPI(
    title="PharmaLens API",
    description="Explainable AI Framework for Drug–Target Interaction Prediction",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.exception_handler(sqlite3.Error)
async def job_storage_error(request, exc):
    logger.error("Prediction job database is unavailable: %s", type(exc).__name__)
    return JSONResponse(
        status_code=503,
        content={"detail": "Prediction job storage is temporarily unavailable; retry shortly"},
        headers={"Retry-After": "2"},
    )


# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "PHARMALENS_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────────


class ModelScope(BaseModel):
    model_id: Optional[str] = Field(None, max_length=200)
    dataset: str = Field("KIBA", min_length=1, max_length=100)
    endpoint: str = Field("KIBA", pattern="^(KIBA|Ki|Kd|IC50)$")
    dataset_version: Optional[str] = Field(None, max_length=100)


class PairInput(BaseModel):
    smiles: str = Field(
        ..., min_length=1, max_length=10000, description="Drug SMILES string", examples=["CCO"]
    )
    sequence: str = Field(
        ..., min_length=10, max_length=10000, description="Protein amino acid sequence"
    )


class PredictionRequest(ModelScope, PairInput):
    explain: bool = Field(False, description="Include SHAP explanations")


class PredictionResponse(BaseModel):
    score: float
    smiles: str
    sequence_length: int
    inference_time: float
    shap_values: Optional[dict] = None
    model_id: str
    dataset: str
    dataset_version: str
    label: dict


class BatchPairInput(PairInput):
    # Biological validity belongs to each row, while request bounds protect resources.
    sequence: str = Field(..., min_length=1, max_length=10000)


class BatchPredictionRequest(ModelScope):
    pairs: list[BatchPairInput] = Field(..., min_length=1, max_length=1000)


class BatchJobRequest(BatchPredictionRequest):
    ids: Optional[list[str]] = Field(None, max_length=1000)
    idempotency_key: Optional[str] = Field(None, min_length=1, max_length=100)


class ScreeningRequest(ModelScope):
    smiles_list: list[str] = Field(..., min_length=1, max_length=1000)
    sequence_list: list[str] = Field(..., min_length=1, max_length=1000)
    drug_ids: Optional[list[str]] = None
    target_ids: Optional[list[str]] = None


class ScreeningResult(BaseModel):
    drug_id: str
    target_id: str
    smiles: str
    predicted_score: Optional[float]
    rank: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str]
    uptime: float
    jobs_ready: bool = False


class DatasetStatsResponse(BaseModel):
    total_interactions: int
    unique_drugs: int
    unique_targets: int
    score_range: list[float]
    split_sizes: dict


class ModelComparisonEntry(BaseModel):
    model: str
    rmse: Optional[float]
    mae: Optional[float]
    pearson: Optional[float]
    spearman: Optional[float]
    r2: Optional[float]
    ci: Optional[float]
    training_time: Optional[float]


# ─── Global State ──────────────────────────────────────────────────────────────

pipeline = None
start_time = time.time()


def get_registry():
    from src.inference.model_registry import ModelRegistry

    return ModelRegistry(os.getenv("PHARMALENS_REGISTRY_PATH"))


@lru_cache(maxsize=4)
def load_registered(registry_path, name, checksum, endpoint, preprocessing_version):
    from src.inference.model_registry import ModelRegistry

    return ModelRegistry(registry_path).load(name)


def resolve_model(scope):
    registry = get_registry()
    try:
        entry = (
            registry.get(scope.model_id)
            if scope.model_id
            else registry.get_best_model(scope.dataset, scope.endpoint, scope.dataset_version)
        )
        if entry is None:
            raise HTTPException(
                status_code=404 if scope.model_id else 503,
                detail="Requested model is unavailable"
                if scope.model_id
                else "No verified default model is available for this dataset and endpoint",
            )
        if (entry.dataset, entry.endpoint) != (scope.dataset, scope.endpoint) or (
            scope.dataset_version is not None and entry.dataset_version != scope.dataset_version
        ):
            raise HTTPException(
                status_code=422,
                detail="Model does not match the requested dataset, endpoint or release",
            )
        loaded = load_registered(
            str(registry.registry_path),
            entry.name,
            entry.artifact_sha256,
            entry.endpoint,
            entry.preprocessing_version,
        )
        return entry, loaded
    except (ValueError, OSError, RuntimeError) as exc:
        logger.warning("Model resolution failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="Model artifact is unavailable or failed verification"
        ) from exc


def model_metadata(entry, loaded):
    return {
        "model_id": entry.name,
        "dataset": entry.dataset,
        "dataset_version": entry.dataset_version,
        "label": loaded.label_contract.as_dict(),
    }


# ─── Startup Event ─────────────────────────────────────────────────────────────


async def startup_event():
    """Load the best model on startup."""
    global pipeline

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    # Explicit registry defaults supersede hardcoded XGBoost filenames.
    try:
        _, pipeline = resolve_model(ModelScope())
    except HTTPException:
        pipeline = None
        logger.info(
            "No default KIBA artifact registered; other endpoint models may be selected explicitly"
        )


# ─── API Routes ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if pipeline is not None else "degraded",
        model_loaded=pipeline is not None,
        model_name=pipeline.model_type if pipeline else None,
        uptime=time.time() - start_time,
        jobs_ready=getattr(app.state, "job_store", None) is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict interaction score for a drug-target pair."""
    entry, selected = resolve_model(request)
    if request.explain and entry.model_type != "xgboost":
        raise HTTPException(
            status_code=422, detail="SHAP explanations are supported for XGBoost only"
        )

    try:
        result = selected.predict(request.smiles, request.sequence, explain=request.explain)
        return PredictionResponse(
            score=result["score"],
            smiles=result["smiles"],
            sequence_length=result.get("sequence_length", len(request.sequence)),
            inference_time=result["inference_time"],
            shap_values=result.get("shap_values"),
            **model_metadata(entry, selected),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict/batch")
def batch_predict(request: BatchPredictionRequest):
    """Predict interaction scores for multiple drug-target pairs."""
    entry, selected = resolve_model(request)

    results = []
    for row_id, pair in enumerate(request.pairs):
        try:
            pred = selected.predict(pair.smiles, pair.sequence)
            results.append(
                {
                    "row_id": row_id,
                    "smiles": pair.smiles,
                    "score": pred["score"],
                    "inference_time": pred["inference_time"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "row_id": row_id,
                    "smiles": pair.smiles,
                    "score": None,
                    "error": str(e),
                }
            )

    return json_safe({"results": results, "total": len(results), **model_metadata(entry, selected)})


def job_store():
    store = getattr(app.state, "job_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Prediction job storage is unavailable")
    return store


def execute_prediction_job(payload, progress):
    request = BatchPredictionRequest.model_validate(payload)
    entry, selected = resolve_model(request)
    if entry.artifact_sha256 != payload["artifact_sha256"]:
        raise ValueError("The submitted model artifact changed")
    results = []
    for index, pair in enumerate(request.pairs):
        progress(index)
        try:
            prediction = selected.predict(pair.smiles, pair.sequence)
            results.append(
                {
                    "row_id": index,
                    "smiles": pair.smiles,
                    "score": prediction["score"],
                    "inference_time": prediction["inference_time"],
                }
            )
        except Exception as exc:
            results.append(
                {"row_id": index, "smiles": pair.smiles, "score": None, "error": str(exc)}
            )
    progress(len(request.pairs))
    return json_safe(
        {
            "results": results,
            "total": len(results),
            **model_metadata(entry, selected),
            "artifact_sha256": entry.artifact_sha256,
            "inputs": [
                {**pair.model_dump(), "id": payload["row_ids"][index]}
                for index, pair in enumerate(request.pairs)
            ],
        }
    )


@app.post("/jobs", status_code=202)
def submit_prediction_job(request: BatchJobRequest):
    if request.ids is not None and (
        len(request.ids) != len(request.pairs) or any(len(value) > 200 for value in request.ids)
    ):
        raise HTTPException(
            status_code=422,
            detail="Provide one identifier per input pair, each at most 200 characters",
        )
    entry, selected = resolve_model(request)
    payload = {
        **request.model_dump(exclude={"ids", "idempotency_key"}),
        "model_id": entry.name,
        "dataset": entry.dataset,
        "dataset_version": entry.dataset_version,
        "endpoint": entry.endpoint,
        "artifact_sha256": entry.artifact_sha256,
        "row_ids": request.ids or [""] * len(request.pairs),
    }
    try:
        return job_store().submit(payload, request.idempotency_key)
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/jobs")
def recent_prediction_jobs(limit: int = Query(20, ge=1, le=100)):
    return {"jobs": job_store().recent(limit)}


@app.get("/jobs/{job_id}")
def prediction_job_status(job_id: str):
    try:
        return job_store().get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction job not found") from exc


@app.get("/jobs/{job_id}/result")
def prediction_job_result(job_id: str):
    try:
        job = job_store().get(job_id, include_result=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction job not found") from exc
    if not job["result_available"]:
        raise HTTPException(status_code=409, detail="This job has no completed result")
    return job["result"]


@app.post("/jobs/{job_id}/cancel")
def cancel_prediction_job(job_id: str):
    try:
        return job_store().cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction job not found") from exc


@app.post("/jobs/{job_id}/retry", status_code=202)
def retry_prediction_job(job_id: str):
    try:
        return job_store().retry(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Prediction job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OverflowError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@app.post("/screen")
def screen(request: ScreeningRequest):
    """Screen all drug × target combinations."""
    if len(request.smiles_list) * len(request.sequence_list) > 10000:
        raise HTTPException(
            status_code=422,
            detail="Synchronous screening supports at most 10,000 pairs; divide the workload",
        )
    if any(not s or len(s) > 10000 for s in request.smiles_list) or any(
        not 10 <= len(s) <= 10000 for s in request.sequence_list
    ):
        raise HTTPException(status_code=422, detail="Screening inputs exceed supported lengths")
    entry, selected = resolve_model(request)

    try:
        df = selected.screen(
            request.smiles_list,
            request.sequence_list,
            request.drug_ids,
            request.target_ids,
        )
        return json_safe(
            {
                "results": df.to_dict(orient="records"),
                "total_combinations": len(df),
                **model_metadata(entry, selected),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Screening error: {e}")
        raise HTTPException(status_code=500, detail="Screening failed")


@app.get("/dataset/stats", response_model=DatasetStatsResponse)
def dataset_stats():
    """Get KIBA dataset statistics."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.utils.config import PROCESSED_DATA_DIR

    try:
        import pandas as pd

        df = pd.read_csv(PROCESSED_DATA_DIR / "kiba_clean.csv")

        split_sizes = {}
        splits_dir = PROCESSED_DATA_DIR / "splits"
        for split_file in splits_dir.glob("*.csv"):
            split_df = pd.read_csv(split_file)
            split_sizes[split_file.stem] = len(split_df)

        return DatasetStatsResponse(
            total_interactions=len(df),
            unique_drugs=df["drug_id"].nunique(),
            unique_targets=df["target_id"].nunique(),
            score_range=[float(df["score"].min()), float(df["score"].max())],
            split_sizes=split_sizes,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail="Dataset not found. Run preprocessing notebooks first."
        )


@app.get("/models/comparison")
def model_comparison():
    """Get comparison of all trained models."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.evaluation.experiment_tracker import ExperimentTracker

    try:
        tracker = ExperimentTracker()
        df = tracker.get_results_dataframe()

        if len(df) == 0:
            return {"models": [], "message": "No experiments logged yet."}

        return json_safe(
            {
                "models": df.to_dict(orient="records"),
                "total": len(df),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def architecture_label(model_type, params, pretrained=False):
    label = {
        "xgboost": "XGBoost",
        "deeptda": "CNN",
        "gnn": params.get("gnn_type", "gcn").upper(),
    }.get(model_type, model_type)
    if params.get("auxiliary_dim"):
        label = "Hybrid " + label
    return label + (" + ESM-2" if pretrained else "")


@lru_cache(maxsize=4)
def read_benchmark(path, modified_ns, size):
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        data.get("runs", {})
        if isinstance(data, dict) and isinstance(data.get("runs", {}), dict)
        else {}
    )


def benchmark_runs():
    from src.utils.config import PROJECT_ROOT

    path = Path(
        os.getenv(
            "PHARMALENS_BENCHMARK_REPORT",
            str(PROJECT_ROOT / "results/reports/final_benchmark_v1.json"),
        )
    )
    try:
        stat = path.stat()
        return read_benchmark(str(path), stat.st_mtime_ns, stat.st_size)
    except (OSError, ValueError, TypeError):
        return {}


@app.get("/models/list")
def list_models():
    """List all registered models."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    try:
        registry = get_registry()
        models = registry.list_models()
        for model in models:
            params, pretrained = model.get("hyperparameters", {}), False
            try:
                import json

                report = json.loads(
                    (Path(model["path"]).parent / "run.json").read_text(encoding="utf-8")
                )
                if report["artifact"]["sha256"] == model["artifact_sha256"]:
                    params = report["config"]["model_params"]
                    pretrained = bool(
                        report.get("protein_feature_contract")
                        or report["config"].get("protein_embedding_cache")
                    )
            except (OSError, ValueError, TypeError, KeyError):
                pass
            model["architecture"] = architecture_label(model["model_type"], params, pretrained)
        return json_safe({"models": models})
    except (OSError, ValueError):
        raise HTTPException(status_code=503, detail="Model registry is unavailable")


# ─── Serve Frontend ────────────────────────────────────────────────────────────


@lru_cache(maxsize=128)
def verified_file(path, expected_hash, modified_ns, size):
    from src.data.acquisition import digest_file

    return digest_file(path) == expected_hash


@app.get("/datasets")
def dataset_catalog():
    """Describe actual prepared versions and verify their published bytes."""
    import json

    from src.utils.config import PROJECT_ROOT

    root = Path(os.getenv("PHARMALENS_PREPARED_DATA_DIR", PROJECT_ROOT / "data/processed/prepared"))
    datasets, issues = [], []
    for manifest_path in sorted(root.rglob("manifest.json")):
        if any(part.endswith(".part") for part in manifest_path.relative_to(root).parts):
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not manifest.get("outputs"):
                continue
            endpoints = []
            for endpoint, spec in manifest.get("outputs", {}).items():
                path = manifest_path.parent / spec["path"]
                if path.resolve().parent != manifest_path.parent.resolve():
                    raise ValueError("Unexpected dataset member path")
                exists = path.is_file()
                stat = path.stat() if exists else None
                available = exists and verified_file(
                    str(path), spec["sha256"], stat.st_mtime_ns, stat.st_size
                )
                count = manifest.get("counts", {}).get(endpoint, manifest.get("observations", 0))
                endpoints.append({"endpoint": endpoint, "rows": count, "available": available})
            datasets.append(
                {
                    "id": manifest_path.parent.relative_to(root).as_posix(),
                    "endpoints": endpoints,
                    "family_counts": manifest.get("family_counts", {}),
                    "excluded": manifest.get("excluded", {}),
                    "non_kinase_only": manifest.get("non_kinase_only", False),
                    "source_sample": manifest.get("source_manifest", {}).get("status") == "sample",
                    "unique_pairs": manifest.get("unique_pairs"),
                    "limitations": manifest.get("limitations", []),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            issues.append(
                {
                    "id": manifest_path.parent.name,
                    "reason": "Manifest or dataset integrity check failed",
                }
            )
    return json_safe({"datasets": datasets, "issues": issues})


@app.get("/experiments")
def completed_experiments():
    """Expose completed, auditable runs; partial checkpoints are not successes."""
    import json

    from src.utils.config import PROJECT_ROOT

    root = Path(os.getenv("PHARMALENS_RUNS_DIR", PROJECT_ROOT / "experiments/runs"))
    runs, issues = [], []
    benchmark = benchmark_runs()
    paths = sorted(
        (path for path in root.glob("*/run.json") if not path.parent.name.endswith(".part")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:100]
    for path in paths:
        if path.parent.name.endswith(".part"):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            config = report["config"]
            evaluated = benchmark.get(report["run_id"], {})
            if not isinstance(evaluated, dict):
                evaluated = {}
            test_metrics = (
                evaluated.get("test_metrics")
                if evaluated.get("artifact_sha256") == report.get("artifact", {}).get("sha256")
                else report.get("test_metrics")
            )
            runs.append(
                {
                    "id": report["run_id"],
                    "model": config["model"],
                    "architecture": architecture_label(
                        config["model"],
                        config["model_params"],
                        bool(
                            report.get("protein_feature_contract")
                            or config.get("protein_embedding_cache")
                        ),
                    ),
                    "dataset": config["dataset"],
                    "dataset_version": config["dataset_version"],
                    "endpoint": config["endpoint"],
                    "split": config["split"],
                    "seed": config["seed"],
                    "smoke": config.get("smoke", False),
                    "status": report["status"],
                    "completed_at": report["completed_at"],
                    "validation_metrics": report["validation_metrics"],
                    "test_evaluated": test_metrics is not None,
                    "test_metrics": test_metrics,
                    "seconds": report["training_and_evaluation_seconds"],
                    "serving_parity": report["serving_parity"]["verified"],
                    "iteration_budget_reached": report.get("hit_iteration_budget", False),
                }
            )
        except (OSError, ValueError, KeyError, TypeError):
            issues.append({"id": path.parent.name, "reason": "Run metadata unavailable"})
    checkpoints, matrices = [], []
    for path in root.glob("*.part/progress.json"):
        try:
            progress = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(progress, dict):
                continue
            age = max(0, time.time() - path.stat().st_mtime)
            if not isinstance(progress.get("epoch"), int):
                continue
            checkpoints.append(
                {
                    "id": path.parent.name.rsplit(".", 2)[0],
                    "epoch": progress["epoch"],
                    "max_epochs": progress.get("max_epochs"),
                    "reported_status": "failed"
                    if (path.parent / "failure.json").exists()
                    else progress.get("status", "unknown"),
                    "age_seconds": round(age),
                    "stale": age > 600,
                    "validation_loss": progress.get("val_loss"),
                    "loss_space": progress.get("loss_space", "unknown"),
                }
            )
        except (OSError, ValueError, TypeError):
            continue
    for path in sorted(root.glob("*_status.json")):
        try:
            status = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(status.get("total_jobs"), int) or not isinstance(
                status.get("results"), dict
            ):
                continue
            results = status["results"]
            matrices.append(
                {
                    "id": path.stem,
                    "total_jobs": status["total_jobs"],
                    "completed_jobs": sum(
                        row.get("status") in {"completed", "verified_existing"}
                        for row in results.values()
                    ),
                    "failed_jobs": sum(row.get("status") == "failed" for row in results.values()),
                    "registration_errors": len(status.get("registration_errors", [])),
                    "updated_at": status.get("updated_at"),
                }
            )
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return json_safe(
        {
            "runs": runs,
            "issues": issues,
            "checkpoints": sorted(checkpoints, key=lambda row: row["age_seconds"]),
            "matrices": matrices,
        }
    )


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    # Serve CSS and JS directly (no /static prefix needed by the HTML)
    @app.get("/style.css", include_in_schema=False)
    async def serve_css():
        return FileResponse(FRONTEND_DIR / "style.css", media_type="text/css")

    @app.get("/app.js", include_in_schema=False)
    async def serve_js():
        return FileResponse(FRONTEND_DIR / "app.js", media_type="application/javascript")


# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("\n  🔬 PharmaLens API starting...")
    print("  📡 API docs:  http://localhost:8000/docs")
    print("  🌐 Frontend:  http://localhost:8000/")
    print()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
