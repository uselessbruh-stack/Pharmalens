"""
PharmaLens — Model Registry

Central registry that tracks, loads, and serves trained models.
Used by the inference pipeline and the FastAPI backend.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pharmalens.inference.registry")


@dataclass
class ModelEntry:
    """Metadata for a registered model."""

    name: str
    model_type: str  # 'xgboost', 'deeptda', 'gnn'
    path: str
    description: str = ""
    metrics: dict = field(default_factory=dict)
    hyperparameters: dict = field(default_factory=dict)
    is_best: bool = False
    dataset: str = "KIBA"
    split: str = "random"
    created_at: str = ""
    endpoint: str = "KIBA"
    dataset_version: str = "legacy"
    validation_metrics: dict = field(default_factory=dict)
    artifact_sha256: str = ""
    preprocessing_version: str = "legacy_v1"


class ModelRegistry:
    """
    Central model registry for PharmaLens.

    Tracks all trained models, their paths, metadata, and metrics.
    Supports loading models for inference and comparing model performance.

    Usage:
        registry = ModelRegistry("models/registry.json")
        registry.register("XGBoost_v1", "xgboost", "models/xgboost/best.json", metrics={...})
        model = registry.load("XGBoost_v1")
        best = registry.get_best_model()
    """

    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            from src.utils.config import MODELS_DIR

            registry_path = MODELS_DIR / "registry.json"

        self.registry_path = Path(registry_path)
        self.entries: dict[str, ModelEntry] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
            for name, entry_data in data.items():
                self.entries[name] = ModelEntry(**entry_data)
            logger.info(f"Loaded {len(self.entries)} models from registry")
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self):
        """Persist registry to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(entry) for name, entry in self.entries.items()}
        from src.data.acquisition import atomic_json
        from src.utils.serialization import json_safe

        atomic_json(self.registry_path, json_safe(data))

    def register(
        self,
        name: str,
        model_type: str,
        path: str,
        description: str = "",
        metrics: Optional[dict] = None,
        hyperparameters: Optional[dict] = None,
        is_best: bool = False,
        dataset: str = "KIBA",
        split: str = "random",
        endpoint: str = "KIBA",
        dataset_version: str = "legacy",
        validation_metrics: Optional[dict] = None,
        preprocessing_version: str = "legacy_v1",
    ):
        """
        Register a trained model.

        Args:
            name: Unique model name.
            model_type: 'xgboost', 'deeptda', or 'gnn'.
            path: Path to model file.
            description: Human-readable description.
            metrics: Test set metrics dict.
            hyperparameters: Training hyperparameters.
            is_best: Whether this is the best model.
            dataset: Dataset name.
            split: Split strategy used.
        """
        from datetime import datetime, timezone

        from src.data.acquisition import digest_file
        from src.data.labels import CONTRACTS

        if name in self.entries:
            raise ValueError("Model IDs are immutable; register a new version under a new ID")
        if model_type not in {"xgboost", "deeptda", "gnn"} or endpoint not in CONTRACTS:
            raise ValueError("Unsupported model type or endpoint")
        if preprocessing_version not in {"legacy_v1", "standardized_v1"}:
            raise ValueError("Unsupported preprocessing version")
        artifact = Path(path).resolve()
        if not artifact.is_file():
            raise ValueError("A model artifact must exist before registration")
        validation_metrics = validation_metrics or {}
        score = validation_metrics.get("rmse")
        if is_best and (not isinstance(score, (int, float)) or not math.isfinite(score)):
            raise ValueError("Promoting a default requires a finite validation RMSE")

        # If marking as best, unmark previous best
        if is_best:
            for entry in self.entries.values():
                if (entry.dataset, entry.endpoint, entry.dataset_version, entry.split) == (
                    dataset,
                    endpoint,
                    dataset_version,
                    split,
                ):
                    entry.is_best = False

        self.entries[name] = ModelEntry(
            name=name,
            model_type=model_type,
            path=str(artifact),
            description=description,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            is_best=is_best,
            dataset=dataset,
            split=split,
            created_at=datetime.now(timezone.utc).isoformat(),
            endpoint=endpoint,
            dataset_version=dataset_version,
            validation_metrics=validation_metrics,
            artifact_sha256=digest_file(artifact),
            preprocessing_version=preprocessing_version,
        )

        self._save_registry()
        logger.info(f"Registered model: {name} (type={model_type}, best={is_best})")

    def get(self, name: str) -> Optional[ModelEntry]:
        """Get model entry by name."""
        return self.entries.get(name)

    def get_best_model(
        self, dataset="KIBA", endpoint="KIBA", dataset_version=None, split="random"
    ) -> Optional[ModelEntry]:
        """Get the model marked as best."""
        candidates = [
            entry
            for entry in self.entries.values()
            if entry.dataset == dataset
            and entry.endpoint == endpoint
            and entry.split == split
            and (dataset_version is None or entry.dataset_version == dataset_version)
            and Path(entry.path).is_file()
            and entry.artifact_sha256
        ]
        versions = {entry.dataset_version for entry in candidates}
        if dataset_version is None and len(versions) > 1:
            raise ValueError(
                "Specify dataset_version when multiple dataset releases are registered"
            )
        for entry in candidates:
            if entry.is_best:
                return entry

        # If none marked as best, return model with lowest RMSE
        candidates = [
            entry
            for entry in candidates
            if isinstance(entry.validation_metrics.get("rmse"), (int, float))
            and math.isfinite(entry.validation_metrics["rmse"])
        ]
        if candidates:
            return min(
                candidates,
                key=lambda e: e.validation_metrics["rmse"],
            )
        return None

    def load(self, name: str):
        """
        Load a model by name using the inference pipeline.

        Returns:
            PharmaLensPipeline instance.
        """
        entry = self.get(name)
        if entry is None:
            raise KeyError(f"Model not found: {name}")
        from src.data.acquisition import digest_file

        if not entry.artifact_sha256 or digest_file(entry.path) != entry.artifact_sha256:
            raise ValueError(
                "Model artifact integrity/provenance verification failed; re-register verified artifacts"
            )

        from src.inference.pipeline import PharmaLensPipeline

        return PharmaLensPipeline.from_pretrained(
            entry.path, entry.model_type, entry.endpoint, entry.preprocessing_version
        )

    def load_best(self, **scope):
        """Load the best model."""
        best = self.get_best_model(**scope)
        if best is None:
            raise RuntimeError("No models registered")
        return self.load(best.name)

    def list_models(self) -> list[dict]:
        """List all registered models as dicts."""
        return [
            {
                **asdict(entry),
                "available": Path(entry.path).is_file() and bool(entry.artifact_sha256),
            }
            for entry in self.entries.values()
        ]

    def compare(self, metric: str = "rmse") -> list[dict]:
        """Compare all models by a metric, sorted best-first."""
        ascending = metric in {"rmse", "mae", "mse"}

        missing = float("inf") if ascending else float("-inf")
        models = [{"name": name, **entry.metrics} for name, entry in self.entries.items()]

        return sorted(
            models,
            key=lambda x: (
                x.get(metric)
                if isinstance(x.get(metric), (int, float)) and math.isfinite(x[metric])
                else missing
            ),
            reverse=not ascending,
        )

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"ModelRegistry({len(self.entries)} models)"
