"""
PharmaLens — Experiment Tracker

Records and manages experiment results for reproducibility.
Every model training run generates a tracked record with:
  - Experiment ID, timestamp, model, dataset, split, seed
  - Hyperparameters, metrics, timing information
  - Saved to both CSV and JSON for easy access
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import RESULTS_DIR, ensure_dirs, save_json, load_json


class ExperimentTracker:
    """
    Tracks experiment results across all model training runs.

    Usage:
        tracker = ExperimentTracker()

        # Log a new experiment
        tracker.log_experiment(
            model="XGBoost",
            dataset="KIBA",
            representation="descriptors+fingerprints",
            split="random",
            seed=42,
            hyperparameters={"n_estimators": 500, "max_depth": 7},
            metrics={"rmse": 0.85, "mae": 0.63, "pearson": 0.89, ...},
            training_time=120.5,
            inference_time=0.3,
            num_parameters=15000,
            notes="Baseline XGBoost run"
        )

        # Load all experiments
        df = tracker.load_experiments()

        # Get best model by metric
        best = tracker.get_best("rmse", minimize=True)
    """

    def __init__(self, results_dir: Optional[Path] = None):
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        ensure_dirs()
        self.csv_path = self.results_dir / "experiment_log.csv"
        self.json_path = self.results_dir / "experiment_log.json"

    def _generate_id(self) -> str:
        """Generate a short unique experiment ID."""
        return f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def log_experiment(
        self,
        model: str,
        dataset: str = "KIBA",
        representation: str = "",
        split: str = "random",
        seed: int = 42,
        hyperparameters: Optional[dict] = None,
        metrics: Optional[dict] = None,
        training_time: Optional[float] = None,
        inference_time: Optional[float] = None,
        num_parameters: Optional[int] = None,
        notes: str = "",
        extra: Optional[dict] = None,
    ) -> dict:
        """
        Log a single experiment result.

        Args:
            model: Model name (e.g., 'XGBoost', 'DeepDTA', 'GNN-GCN')
            dataset: Dataset name ('KIBA' or 'Davis')
            representation: Input representation description
            split: Split strategy ('random', 'cold_drug', 'cold_target')
            seed: Random seed used
            hyperparameters: Model hyperparameters dict
            metrics: Evaluation metrics dict
            training_time: Training time in seconds
            inference_time: Inference time in seconds
            num_parameters: Number of model parameters
            notes: Free-text notes
            extra: Any additional key-value pairs

        Returns:
            dict: The complete experiment record
        """
        record = {
            "experiment_id": self._generate_id(),
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "dataset": dataset,
            "representation": representation,
            "split": split,
            "seed": seed,
            "hyperparameters": json.dumps(hyperparameters or {}),
            "training_time_sec": training_time,
            "inference_time_sec": inference_time,
            "num_parameters": num_parameters,
            "notes": notes,
        }

        # Flatten metrics into record
        if metrics:
            for key, value in metrics.items():
                record[key] = value

        # Add any extra fields
        if extra:
            record.update(extra)

        # Append to CSV
        df_new = pd.DataFrame([record])
        if self.csv_path.exists():
            df_existing = pd.read_csv(self.csv_path)
            df_all = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(self.csv_path, index=False)

        # Append to JSON
        if self.json_path.exists():
            all_records = load_json(self.json_path)
        else:
            all_records = []
        all_records.append(record)
        save_json(all_records, self.json_path)

        print(f"✓ Logged experiment: {record['experiment_id']} | {model} | {split} | seed={seed}")
        return record

    def load_experiments(self) -> pd.DataFrame:
        """Load all experiment records as a DataFrame."""
        if not self.csv_path.exists():
            print("No experiments logged yet.")
            return pd.DataFrame()
        return pd.read_csv(self.csv_path)

    def load_experiments_json(self) -> list[dict]:
        """Load all experiment records as a list of dicts."""
        if not self.json_path.exists():
            return []
        return load_json(self.json_path)

    def get_experiments_by_model(self, model: str) -> pd.DataFrame:
        """Filter experiments by model name."""
        df = self.load_experiments()
        if df.empty:
            return df
        return df[df["model"] == model]

    def get_experiments_by_split(self, split: str) -> pd.DataFrame:
        """Filter experiments by split strategy."""
        df = self.load_experiments()
        if df.empty:
            return df
        return df[df["split"] == split]

    def get_best(self, metric: str, minimize: bool = True, split: str = "random") -> Optional[dict]:
        """
        Get the best experiment for a given metric and split.

        Args:
            metric: Metric column name (e.g., 'rmse', 'pearson')
            minimize: If True, lower is better (RMSE, MAE). If False, higher is better (Pearson, CI).
            split: Filter by split strategy.

        Returns:
            dict: Best experiment record, or None if no experiments.
        """
        df = self.load_experiments()
        if df.empty or metric not in df.columns:
            return None

        df = df[df["split"] == split]
        if df.empty:
            return None

        if minimize:
            best_idx = df[metric].idxmin()
        else:
            best_idx = df[metric].idxmax()

        return df.loc[best_idx].to_dict()

    def get_comparison_table(
        self,
        split: str = "random",
        metrics: list[str] = ["rmse", "mae", "pearson", "spearman", "r2", "ci"],
    ) -> pd.DataFrame:
        """
        Create a model comparison table for a given split.

        Groups by model and computes mean ± std for each metric across seeds.

        Args:
            split: Split strategy to filter by.
            metrics: List of metric columns to include.

        Returns:
            pd.DataFrame: Comparison table with mean ± std.
        """
        df = self.load_experiments()
        if df.empty:
            return pd.DataFrame()

        df = df[df["split"] == split]
        if df.empty:
            return pd.DataFrame()

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in df.columns]

        # Group by model and compute stats
        grouped = df.groupby("model")[available_metrics]
        means = grouped.mean().round(4)
        stds = grouped.std().round(4)

        # Format as mean ± std
        comparison = pd.DataFrame(index=means.index)
        for metric in available_metrics:
            comparison[metric] = means[metric].astype(str) + " ± " + stds[metric].fillna(0).astype(str)

        # Add extra columns
        if "training_time_sec" in df.columns:
            comparison["train_time"] = grouped.first()["training_time_sec"].round(1) if "training_time_sec" in df.columns else ""
        if "num_parameters" in df.columns:
            comparison["params"] = df.groupby("model")["num_parameters"].first()

        return comparison

    def save_model_results(self, model_name: str, results: dict, filename: Optional[str] = None):
        """
        Save a model's complete results to a standalone JSON file.

        Args:
            model_name: Name of the model
            results: Complete results dictionary
            filename: Optional custom filename
        """
        if filename is None:
            filename = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_results.json"

        path = self.results_dir / filename
        save_json(results, path)
        print(f"✓ Model results saved: {path}")

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get all results as a DataFrame with standardized column names.

        This is the primary method used by notebooks and the API
        to retrieve experiment results.

        Returns:
            pd.DataFrame with columns: model, split, rmse, mae, pearson,
            spearman, r2, ci, training_time, etc.
        """
        df = self.load_experiments()
        if df.empty:
            return df

        # Standardize column names
        if "training_time_sec" in df.columns and "training_time" not in df.columns:
            df["training_time"] = df["training_time_sec"]

        return df

