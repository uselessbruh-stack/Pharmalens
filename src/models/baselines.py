"""
PharmaLens — Baseline Models

Implements simple baseline models to establish lower-bound performance:
    1. MeanPredictor — always predicts training set mean
    2. Ridge Regression — linear model with L2 regularization
    3. Random Forest — ensemble of decision trees
"""

import logging
import time
from typing import Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger("pharmalens.models.baselines")


# ==============================================================================
# Mean Predictor (Trivial Baseline)
# ==============================================================================

class MeanPredictor(BaseEstimator, RegressorMixin):
    """
    Always predicts the mean of the training set target values.

    This establishes the absolute lower bound of model performance.
    Any useful model should significantly outperform this baseline.
    """

    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)

    def get_params(self, deep=True):
        return {}

    def __repr__(self):
        return f"MeanPredictor(mean={self.mean_})"


# ==============================================================================
# Training Utility
# ==============================================================================

def train_and_evaluate_baseline(
    model,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Train a sklearn-compatible model and evaluate on val and test sets.

    Args:
        model: sklearn-compatible regressor.
        model_name: Name for logging.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.

    Returns:
        dict with training info and metrics.
    """
    from src.evaluation.metrics import compute_regression_metrics

    logger.info(f"Training {model_name}...")

    # Train
    t_start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t_start
    logger.info(f"  Training time: {training_time:.2f}s")

    # Predict
    t_start = time.time()
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    inference_time = time.time() - t_start

    # Evaluate
    train_metrics = compute_regression_metrics(y_train, y_train_pred)
    val_metrics = compute_regression_metrics(y_val, y_val_pred)
    test_metrics = compute_regression_metrics(y_test, y_test_pred)

    results = {
        "model_name": model_name,
        "training_time": round(training_time, 2),
        "inference_time": round(inference_time, 4),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "predictions": {
            "y_train_pred": y_train_pred,
            "y_val_pred": y_val_pred,
            "y_test_pred": y_test_pred,
        },
    }

    # Log key metrics
    logger.info(
        f"  {model_name} Results:\n"
        f"    Train — RMSE: {train_metrics['rmse']:.4f}, Pearson: {train_metrics['pearson']:.4f}\n"
        f"    Val   — RMSE: {val_metrics['rmse']:.4f}, Pearson: {val_metrics['pearson']:.4f}\n"
        f"    Test  — RMSE: {test_metrics['rmse']:.4f}, Pearson: {test_metrics['pearson']:.4f}, CI: {test_metrics['ci']:.4f}"
    )

    return results


# ==============================================================================
# Model Factory
# ==============================================================================

def get_baseline_models() -> dict:
    """
    Get all baseline models with default hyperparameters.

    Returns:
        dict: {model_name: model_instance}
    """
    return {
        "MeanPredictor": MeanPredictor(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        ),
    }


def save_baseline_model(model, model_name: str, save_dir=None):
    """Save a trained baseline model using joblib."""
    import joblib
    from src.utils.config import BASELINE_MODELS_DIR

    if save_dir is None:
        save_dir = BASELINE_MODELS_DIR

    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    path = save_dir / f"{model_name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved {model_name} to {path}")
    return path


def load_baseline_model(model_name: str, save_dir=None):
    """Load a trained baseline model."""
    import joblib
    from src.utils.config import BASELINE_MODELS_DIR

    if save_dir is None:
        save_dir = BASELINE_MODELS_DIR

    from pathlib import Path
    path = Path(save_dir) / f"{model_name.lower().replace(' ', '_')}.joblib"
    return joblib.load(path)
