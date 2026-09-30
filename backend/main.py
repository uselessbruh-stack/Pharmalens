"""
PharmaLens — FastAPI Backend

REST API for drug-target interaction prediction, screening, and exploration.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pharmalens.api")

# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="PharmaLens API",
    description="Explainable AI Framework for Drug–Target Interaction Prediction",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    smiles: str = Field(..., description="Drug SMILES string", examples=["CCO"])
    sequence: str = Field(..., description="Protein amino acid sequence", examples=["MKTFVLLL"])
    explain: bool = Field(False, description="Include SHAP explanations")


class PredictionResponse(BaseModel):
    score: float
    smiles: str
    sequence_length: int
    inference_time: float
    shap_values: Optional[dict] = None


class BatchPredictionRequest(BaseModel):
    pairs: list[dict] = Field(..., description="List of {smiles, sequence} dicts")


class ScreeningRequest(BaseModel):
    smiles_list: list[str]
    sequence_list: list[str]
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


# ─── Startup Event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Load the best model on startup."""
    global pipeline

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.inference.pipeline import PharmaLensPipeline
    from src.utils.config import XGBOOST_MODELS_DIR

    # Try to load the best XGBoost model
    model_paths = [
        XGBOOST_MODELS_DIR / "xgb_tuned_best.json",
        XGBOOST_MODELS_DIR / "xgb_best.json",
    ]

    for path in model_paths:
        if path.exists():
            try:
                pipeline = PharmaLensPipeline.from_pretrained(str(path), "xgboost")
                logger.info(f"Model loaded: {path.name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    logger.warning("No trained model found. Run the research notebooks first.")


# ─── API Routes ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=pipeline is not None,
        model_name=pipeline.model_type if pipeline else None,
        uptime=time.time() - start_time,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict interaction score for a drug-target pair."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run research notebooks first.")

    try:
        result = pipeline.predict(request.smiles, request.sequence, explain=request.explain)
        return PredictionResponse(
            score=result["score"],
            smiles=result["smiles"],
            sequence_length=result.get("sequence_length", len(request.sequence)),
            inference_time=result["inference_time"],
            shap_values=result.get("shap_values"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Predict interaction scores for multiple drug-target pairs."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for pair in request.pairs:
        try:
            pred = pipeline.predict(pair["smiles"], pair["sequence"])
            results.append({
                "smiles": pair["smiles"],
                "score": pred["score"],
                "inference_time": pred["inference_time"],
            })
        except Exception as e:
            results.append({
                "smiles": pair.get("smiles", ""),
                "score": None,
                "error": str(e),
            })

    return {"results": results, "total": len(results)}


@app.post("/screen")
async def screen(request: ScreeningRequest):
    """Screen all drug × target combinations."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        df = pipeline.screen(
            request.smiles_list,
            request.sequence_list,
            request.drug_ids,
            request.target_ids,
        )
        return {
            "results": df.to_dict(orient="records"),
            "total_combinations": len(df),
        }
    except Exception as e:
        logger.error(f"Screening error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dataset/stats", response_model=DatasetStatsResponse)
async def dataset_stats():
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
        raise HTTPException(status_code=404, detail="Dataset not found. Run preprocessing notebooks first.")


@app.get("/models/comparison")
async def model_comparison():
    """Get comparison of all trained models."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.evaluation.experiment_tracker import ExperimentTracker

    try:
        tracker = ExperimentTracker()
        df = tracker.get_results_dataframe()

        if len(df) == 0:
            return {"models": [], "message": "No experiments logged yet."}

        return {
            "models": df.to_dict(orient="records"),
            "total": len(df),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/list")
async def list_models():
    """List all registered models."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from src.inference.model_registry import ModelRegistry

    try:
        registry = ModelRegistry()
        return {"models": registry.list_models()}
    except Exception:
        return {"models": []}


# ─── Serve Frontend ────────────────────────────────────────────────────────────

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

