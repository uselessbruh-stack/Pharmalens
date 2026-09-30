"""
PharmaLens — Evaluation Metrics

Unified metrics computation for DTI prediction.
Includes regression metrics, ranking metrics, and the Concordance Index (CI).
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from typing import Optional


# ==============================================================================
# Concordance Index
# ==============================================================================

def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Concordance Index (CI) for DTI prediction.

    CI measures the probability that for a randomly chosen pair of
    drug-target interactions, the predicted values are ranked in the
    same order as the true values.

    CI = (number of concordant pairs) / (total comparable pairs)

    A CI of 0.5 indicates random prediction.
    A CI of 1.0 indicates perfect ranking.

    Reference:
        Gönen & Heller, 2005. "Concordance probability and discriminatory power
        in proportional hazards regression."

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        float: Concordance Index (0.0 to 1.0)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    concordant = 0
    discordant = 0
    tied = 0
    total = 0

    n = len(y_true)
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if y_true[i] > y_true[j]:
                    if y_pred[i] > y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        tied += 1
                    else:
                        discordant += 1
                elif y_true[i] < y_true[j]:
                    if y_pred[i] < y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        tied += 1
                    else:
                        discordant += 1

    if total == 0:
        return 0.0

    return (concordant + 0.5 * tied) / total


def concordance_index_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fast vectorized approximation of Concordance Index.
    Uses sampling for large datasets (>5000 samples) to avoid O(n²) computation.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        float: Concordance Index (0.0 to 1.0)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    n = len(y_true)

    # For small datasets, use exact computation
    if n <= 5000:
        return concordance_index(y_true, y_pred)

    # For large datasets, sample pairs
    rng = np.random.RandomState(42)
    n_pairs = min(500_000, n * (n - 1) // 2)

    idx_i = rng.randint(0, n, size=n_pairs)
    idx_j = rng.randint(0, n, size=n_pairs)

    # Remove self-pairs
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    true_diff = y_true[idx_i] - y_true[idx_j]
    pred_diff = y_pred[idx_i] - y_pred[idx_j]

    # Only consider pairs where true values differ
    nonzero = true_diff != 0
    true_diff = true_diff[nonzero]
    pred_diff = pred_diff[nonzero]

    if len(true_diff) == 0:
        return 0.0

    concordant = np.sum(np.sign(true_diff) == np.sign(pred_diff))
    tied = np.sum(pred_diff == 0)
    total = len(true_diff)

    return (concordant + 0.5 * tied) / total


# ==============================================================================
# Regression Metrics
# ==============================================================================

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_fast_ci: bool = True,
) -> dict:
    """
    Compute all regression metrics for DTI prediction.

    Args:
        y_true: Ground truth binding affinity values.
        y_pred: Predicted binding affinity values.
        use_fast_ci: If True, use fast CI for large datasets.

    Returns:
        dict with keys: rmse, mae, mse, pearson, spearman, r2, ci
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    assert len(y_true) == len(y_pred), (
        f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
    )

    # Remove NaN entries
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {k: float("nan") for k in ["rmse", "mae", "mse", "pearson", "spearman", "r2", "ci"]}

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

    ci_func = concordance_index_fast if use_fast_ci else concordance_index
    ci = ci_func(y_true, y_pred)

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "pearson": round(float(pearson_r), 4),
        "pearson_pvalue": float(pearson_p),
        "spearman": round(float(spearman_r), 4),
        "spearman_pvalue": float(spearman_p),
        "r2": round(r2, 4),
        "ci": round(ci, 4),
        "n_samples": len(y_true),
    }


# ==============================================================================
# Ranking Metrics
# ==============================================================================

def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: Optional[float] = None) -> float:
    """
    Precision@K: fraction of top-K predicted items that are truly relevant.

    Args:
        y_true: Ground truth scores (higher = better interaction).
        y_pred: Predicted scores.
        k: Number of top items to consider.
        threshold: If provided, items with y_true >= threshold are "relevant".
                   If None, uses median of y_true.

    Returns:
        float: Precision@K
    """
    if threshold is None:
        threshold = np.median(y_true)

    top_k_idx = np.argsort(y_pred)[::-1][:k]
    relevant = y_true[top_k_idx] >= threshold
    return float(np.mean(relevant))


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: Optional[float] = None) -> float:
    """
    Recall@K: fraction of all relevant items that appear in top-K predictions.
    """
    if threshold is None:
        threshold = np.median(y_true)

    total_relevant = np.sum(y_true >= threshold)
    if total_relevant == 0:
        return 0.0

    top_k_idx = np.argsort(y_pred)[::-1][:k]
    hits = np.sum(y_true[top_k_idx] >= threshold)
    return float(hits / total_relevant)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    NDCG = DCG@K / ideal_DCG@K

    Uses the relevance values directly (not binary).
    """
    # Get top-K by predicted score
    top_k_idx = np.argsort(y_pred)[::-1][:k]
    top_k_true = y_true[top_k_idx]

    # DCG
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(top_k_true / discounts)

    # Ideal DCG
    ideal_top_k = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_top_k / discounts)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


def hit_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: Optional[float] = None) -> float:
    """
    Hit Rate@K: 1 if at least one relevant item appears in top-K, else 0.
    """
    if threshold is None:
        threshold = np.median(y_true)

    top_k_idx = np.argsort(y_pred)[::-1][:k]
    return float(np.any(y_true[top_k_idx] >= threshold))


def enrichment_factor_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int, threshold: Optional[float] = None) -> float:
    """
    Enrichment Factor@K: ratio of relevant items in top-K vs. random expectation.

    EF@K = (hits_in_top_K / K) / (total_relevant / N)
    """
    if threshold is None:
        threshold = np.median(y_true)

    n = len(y_true)
    total_relevant = np.sum(y_true >= threshold)

    if total_relevant == 0 or k == 0:
        return 0.0

    top_k_idx = np.argsort(y_pred)[::-1][:k]
    hits = np.sum(y_true[top_k_idx] >= threshold)

    expected = total_relevant / n
    observed = hits / k

    if expected == 0:
        return 0.0

    return float(observed / expected)


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_list: list[int] = [5, 10, 20],
    threshold: Optional[float] = None,
) -> dict:
    """
    Compute all ranking metrics at multiple K values.

    Args:
        y_true: Ground truth scores.
        y_pred: Predicted scores.
        k_list: List of K values for @K metrics.
        threshold: Relevance threshold.

    Returns:
        dict with all ranking metrics.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    results = {}
    for k in k_list:
        if k > len(y_true):
            continue
        results[f"precision@{k}"] = round(precision_at_k(y_true, y_pred, k, threshold), 4)
        results[f"recall@{k}"] = round(recall_at_k(y_true, y_pred, k, threshold), 4)
        results[f"ndcg@{k}"] = round(ndcg_at_k(y_true, y_pred, k), 4)
        results[f"hit_rate@{k}"] = round(hit_rate_at_k(y_true, y_pred, k, threshold), 4)
        results[f"ef@{k}"] = round(enrichment_factor_at_k(y_true, y_pred, k, threshold), 4)

    return results


# ==============================================================================
# Summary Formatting
# ==============================================================================

def format_metrics_table(metrics: dict, model_name: str = "") -> str:
    """Format metrics as a readable table string."""
    lines = []
    if model_name:
        lines.append(f"Model: {model_name}")
        lines.append("─" * 40)

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key:<20s} {value:.4f}")
        else:
            lines.append(f"  {key:<20s} {value}")

    return "\n".join(lines)


def metrics_to_row(metrics: dict, model_name: str, **extra) -> dict:
    """Convert metrics dict to a flat row for DataFrame/CSV."""
    row = {"model": model_name}
    row.update(extra)
    row.update(metrics)
    return row
