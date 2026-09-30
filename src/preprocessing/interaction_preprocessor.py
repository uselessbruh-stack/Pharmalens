"""
PharmaLens — Interaction Data Preprocessing

Handles cleaning, deduplication, and transformation of interaction scores.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.preprocessing.interaction")


def remove_duplicate_interactions(
    df: pd.DataFrame,
    drug_col: str = "drug_id",
    target_col: str = "target_id",
    score_col: str = "score",
    strategy: str = "mean",
) -> tuple[pd.DataFrame, int]:
    """
    Handle duplicate (drug, target) pairs.

    Args:
        df: Interaction DataFrame.
        drug_col: Drug identifier column.
        target_col: Target identifier column.
        score_col: Score column.
        strategy: How to handle duplicates - 'mean', 'first', 'max', 'min'.

    Returns:
        tuple: (deduplicated_df, n_duplicates_removed)
    """
    n_before = len(df)
    n_dupes = df.duplicated(subset=[drug_col, target_col]).sum()

    if n_dupes == 0:
        return df, 0

    if strategy == "mean":
        # Average scores for duplicate pairs
        non_score_cols = [c for c in df.columns if c != score_col]
        group_cols = [drug_col, target_col]

        # Keep first occurrence of non-score columns, average the score
        df_deduped = df.groupby(group_cols, as_index=False).agg(
            {score_col: "mean", **{c: "first" for c in non_score_cols if c not in group_cols}}
        )
    elif strategy == "first":
        df_deduped = df.drop_duplicates(subset=[drug_col, target_col], keep="first")
    elif strategy == "max":
        df_deduped = df.sort_values(score_col, ascending=False).drop_duplicates(
            subset=[drug_col, target_col], keep="first"
        )
    elif strategy == "min":
        df_deduped = df.sort_values(score_col, ascending=True).drop_duplicates(
            subset=[drug_col, target_col], keep="first"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'mean', 'first', 'max', or 'min'.")

    n_removed = n_before - len(df_deduped)
    logger.info(f"  Removed {n_removed} duplicate interactions (strategy='{strategy}')")

    return df_deduped.reset_index(drop=True), n_removed


def handle_missing_scores(
    df: pd.DataFrame,
    score_col: str = "score",
) -> tuple[pd.DataFrame, int]:
    """
    Remove rows with missing interaction scores.

    Args:
        df: Interaction DataFrame.
        score_col: Score column name.

    Returns:
        tuple: (cleaned_df, n_removed)
    """
    n_before = len(df)
    df = df.dropna(subset=[score_col]).copy()
    n_removed = n_before - len(df)

    if n_removed > 0:
        logger.info(f"  Removed {n_removed} rows with missing scores")

    return df, n_removed


def detect_outliers(
    df: pd.DataFrame,
    score_col: str = "score",
    n_sigma: float = 3.0,
) -> pd.Series:
    """
    Detect score outliers using the z-score method.

    Args:
        df: Interaction DataFrame.
        score_col: Score column name.
        n_sigma: Number of standard deviations for outlier threshold.

    Returns:
        pd.Series: Boolean mask where True = outlier.
    """
    scores = df[score_col]
    mean = scores.mean()
    std = scores.std()

    z_scores = np.abs((scores - mean) / std)
    outlier_mask = z_scores > n_sigma

    n_outliers = outlier_mask.sum()
    if n_outliers > 0:
        logger.info(
            f"  Detected {n_outliers} outliers (>{n_sigma}σ from mean). "
            f"Score range of outliers: [{scores[outlier_mask].min():.2f}, {scores[outlier_mask].max():.2f}]"
        )

    return outlier_mask


def transform_scores(
    df: pd.DataFrame,
    score_col: str = "score",
    dataset_name: str = "kiba",
) -> pd.DataFrame:
    """
    Apply dataset-specific score transformations.

    For KIBA: No transformation needed (KIBA scores are already unified).
    For Davis: Apply -log10(Kd/1e9) transformation if needed.

    Args:
        df: Interaction DataFrame.
        score_col: Score column name.
        dataset_name: 'kiba' or 'davis'.

    Returns:
        DataFrame with transformed scores.
    """
    df = df.copy()

    if dataset_name.lower() == "davis":
        # Check if already transformed (scores > 0 and < ~15 suggest already log-transformed)
        max_score = df[score_col].max()
        if max_score > 100:
            # Likely raw Kd values in nM — transform
            logger.info("  Applying -log10(Kd/1e9) transformation for Davis")
            # Kd values: replace 0 with small value to avoid log(0)
            df[score_col] = df[score_col].clip(lower=1e-10)
            df[score_col] = -np.log10(df[score_col] / 1e9)
        else:
            logger.info("  Davis scores appear already transformed, skipping")

    elif dataset_name.lower() == "kiba":
        logger.info("  KIBA scores: no transformation needed")

    return df


def preprocess_interactions(
    df: pd.DataFrame,
    dataset_name: str = "kiba",
    drug_col: str = "drug_id",
    target_col: str = "target_id",
    score_col: str = "score",
    remove_outliers: bool = False,
    outlier_sigma: float = 3.0,
    dedupe_strategy: str = "mean",
) -> tuple[pd.DataFrame, dict]:
    """
    Full interaction preprocessing pipeline.

    Steps:
        1. Remove missing scores
        2. Remove duplicate interactions
        3. Detect outliers (optional removal)
        4. Transform scores (dataset-specific)

    Args:
        df: Raw interaction DataFrame.
        dataset_name: 'kiba' or 'davis'.
        drug_col: Drug ID column.
        target_col: Target ID column.
        score_col: Score column.
        remove_outliers: Whether to remove detected outliers.
        outlier_sigma: Sigma threshold for outlier detection.
        dedupe_strategy: Strategy for handling duplicates.

    Returns:
        tuple: (cleaned_df, report_dict)
    """
    report = {
        "original_count": len(df),
        "dataset": dataset_name,
        "steps": [],
    }

    logger.info(f"Starting interaction preprocessing on {len(df)} records...")

    # Step 1: Missing scores
    df, n_missing = handle_missing_scores(df, score_col)
    report["steps"].append({"step": "remove_missing_scores", "removed": n_missing})

    # Step 2: Duplicates
    df, n_dupes = remove_duplicate_interactions(df, drug_col, target_col, score_col, dedupe_strategy)
    report["steps"].append({"step": "remove_duplicates", "removed": n_dupes, "strategy": dedupe_strategy})

    # Step 3: Outliers
    outlier_mask = detect_outliers(df, score_col, outlier_sigma)
    n_outliers = int(outlier_mask.sum())
    report["steps"].append({
        "step": "outlier_detection",
        "n_outliers": n_outliers,
        "sigma": outlier_sigma,
        "removed": n_outliers if remove_outliers else 0,
    })
    if remove_outliers and n_outliers > 0:
        df = df[~outlier_mask].reset_index(drop=True)
        logger.info(f"  Removed {n_outliers} outlier interactions")

    # Step 4: Score transformation
    df = transform_scores(df, score_col, dataset_name)
    report["steps"].append({"step": "score_transformation", "dataset": dataset_name})

    report["final_count"] = len(df)
    report["total_removed"] = report["original_count"] - report["final_count"]
    report["score_stats_after"] = {
        "mean": round(float(df[score_col].mean()), 4),
        "std": round(float(df[score_col].std()), 4),
        "min": round(float(df[score_col].min()), 4),
        "max": round(float(df[score_col].max()), 4),
        "median": round(float(df[score_col].median()), 4),
    }

    logger.info(
        f"  Interaction preprocessing complete: {report['original_count']} → {report['final_count']} "
        f"({report['total_removed']} removed)"
    )

    return df, report
