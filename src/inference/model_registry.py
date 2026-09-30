"""
PharmaLens — Model Registry

Central registry that tracks, loads, and serves trained models.
Used by the inference pipeline and the FastAPI backend.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
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
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

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
        from datetime import datetime

        # If marking as best, unmark previous best
        if is_best:
            for entry in self.entries.values():
                entry.is_best = False

        self.entries[name] = ModelEntry(
            name=name,
            model_type=model_type,
            path=str(path),
            description=description,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            is_best=is_best,
            dataset=dataset,
            split=split,
            created_at=datetime.now().isoformat(),
        )

        self._save_registry()
        logger.info(f"Registered model: {name} (type={model_type}, best={is_best})")

    def get(self, name: str) -> Optional[ModelEntry]:
        """Get model entry by name."""
        return self.entries.get(name)

    def get_best_model(self) -> Optional[ModelEntry]:
        """Get the model marked as best."""
        for entry in self.entries.values():
            if entry.is_best:
                return entry

        # If none marked as best, return model with lowest RMSE
        if self.entries:
            return min(
                self.entries.values(),
                key=lambda e: e.metrics.get("rmse", float("inf")),
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

        from src.inference.pipeline import PharmaLensPipeline
        return PharmaLensPipeline.from_pretrained(entry.path, entry.model_type)

    def load_best(self):
        """Load the best model."""
        best = self.get_best_model()
        if best is None:
            raise RuntimeError("No models registered")
        return self.load(best.name)

    def list_models(self) -> list[dict]:
        """List all registered models as dicts."""
        return [asdict(entry) for entry in self.entries.values()]

    def compare(self, metric: str = "rmse") -> list[dict]:
        """Compare all models by a metric, sorted best-first."""
        ascending = metric in {"rmse", "mae", "mse"}

        models = [
            {"name": name, metric: entry.metrics.get(metric, float("inf")), **entry.metrics}
            for name, entry in self.entries.items()
        ]

        return sorted(models, key=lambda x: x.get(metric, float("inf")),
                       reverse=not ascending)

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"ModelRegistry({len(self.entries)} models)"
