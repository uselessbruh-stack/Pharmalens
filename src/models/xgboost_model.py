"""
PharmaLens — XGBoost Model

XGBoost wrapper with SHAP explainability support for DTI prediction.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger("pharmalens.models.xgboost_model")


class XGBoostDTI(BaseEstimator, RegressorMixin):
    """
    XGBoost model for Drug-Target Interaction prediction.

    Wraps xgb.XGBRegressor with convenience methods for
    training, evaluation, feature importance, and SHAP explanation.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 7,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.01,
        reg_lambda: float = 1.0,
        min_child_weight: int = 5,
        random_state: int = 42,
        n_jobs: int = -1,
        early_stopping_rounds: int = 50,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds

        self.model = None
        self.training_time = None
        self.best_iteration = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """Train the XGBoost model with optional early stopping."""
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            min_child_weight=self.min_child_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            tree_method="hist",
            verbosity=0,
        )

        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_train, y_train), (X_val, y_val)]
            fit_params["verbose"] = 100 if verbose else 0

        t_start = time.time()
        self.model.fit(X_train, y_train, **fit_params)
        self.training_time = time.time() - t_start

        self.best_iteration = getattr(self.model, "best_iteration", self.n_estimators)
        logger.info(f"XGBoost trained in {self.training_time:.1f}s (best_iter={self.best_iteration})")

        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self, importance_type="weight") -> dict:
        """Get feature importance scores."""
        return self.model.get_booster().get_score(importance_type=importance_type)

    def save(self, path):
        """Save the model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info(f"XGBoost model saved to {path}")

    def load(self, path):
        """Load a saved model."""
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))
        logger.info(f"XGBoost model loaded from {path}")
        return self

    def get_hyperparams(self) -> dict:
        """Get current hyperparameters as a dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
        }
