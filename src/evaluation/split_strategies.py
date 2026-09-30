"""
PharmaLens — Data Split Strategies

Implements random, cold-drug, cold-target, and cold-both split strategies
for evaluating DTI prediction models on different generalization scenarios.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("pharmalens.evaluation.splits")


def random_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Standard random split of interaction pairs.

    Each (drug, target) pair is randomly assigned to train/val/test.
    The same drug and target can appear in multiple splits.

    Args:
        df: Interaction DataFrame.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_ratio, random_state=seed, shuffle=True
    )

    # Second split: train vs val
    val_frac = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val, test_size=val_frac, random_state=seed, shuffle=True
    )

    logger.info(
        f"Random split (seed={seed}): "
        f"train={len(train)}, val={len(val)}, test={len(test)}"
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def cold_drug_split(
    df: pd.DataFrame,
    drug_col: str = "drug_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-drug split: test drugs are completely unseen during training.

    Drugs are partitioned into train/val/test groups.
    All interactions involving test drugs go to the test set.

    This evaluates: "Can the model predict interactions for NEW drugs?"

    Args:
        df: Interaction DataFrame.
        drug_col: Drug identifier column.
        train_ratio: Fraction of drugs for training.
        val_ratio: Fraction of drugs for validation.
        test_ratio: Fraction of drugs for testing.
        seed: Random seed.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    unique_drugs = df[drug_col].unique()
    n_drugs = len(unique_drugs)

    # Split drugs (not interactions)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_drugs)

    n_test = max(1, int(n_drugs * test_ratio))
    n_val = max(1, int(n_drugs * val_ratio))
    n_train = n_drugs - n_test - n_val

    train_drugs = set(unique_drugs[:n_train])
    val_drugs = set(unique_drugs[n_train:n_train + n_val])
    test_drugs = set(unique_drugs[n_train + n_val:])

    train = df[df[drug_col].isin(train_drugs)].reset_index(drop=True)
    val = df[df[drug_col].isin(val_drugs)].reset_index(drop=True)
    test = df[df[drug_col].isin(test_drugs)].reset_index(drop=True)

    logger.info(
        f"Cold-drug split (seed={seed}): "
        f"train={len(train)} ({len(train_drugs)} drugs), "
        f"val={len(val)} ({len(val_drugs)} drugs), "
        f"test={len(test)} ({len(test_drugs)} drugs)"
    )

    # Verify no overlap
    assert len(train_drugs & test_drugs) == 0, "Drug leakage between train and test!"
    assert len(train_drugs & val_drugs) == 0, "Drug leakage between train and val!"
    assert len(val_drugs & test_drugs) == 0, "Drug leakage between val and test!"

    return train, val, test


def cold_target_split(
    df: pd.DataFrame,
    target_col: str = "target_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-target split: test protein targets are completely unseen during training.

    Proteins are partitioned into train/val/test groups.
    All interactions involving test proteins go to the test set.

    This evaluates: "Can the model predict interactions for NEW protein targets?"

    Args:
        df: Interaction DataFrame.
        target_col: Target identifier column.
        train_ratio: Fraction of targets for training.
        val_ratio: Fraction of targets for validation.
        test_ratio: Fraction of targets for testing.
        seed: Random seed.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    unique_targets = df[target_col].unique()
    n_targets = len(unique_targets)

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_targets)

    n_test = max(1, int(n_targets * test_ratio))
    n_val = max(1, int(n_targets * val_ratio))
    n_train = n_targets - n_test - n_val

    train_targets = set(unique_targets[:n_train])
    val_targets = set(unique_targets[n_train:n_train + n_val])
    test_targets = set(unique_targets[n_train + n_val:])

    train = df[df[target_col].isin(train_targets)].reset_index(drop=True)
    val = df[df[target_col].isin(val_targets)].reset_index(drop=True)
    test = df[df[target_col].isin(test_targets)].reset_index(drop=True)

    logger.info(
        f"Cold-target split (seed={seed}): "
        f"train={len(train)} ({len(train_targets)} targets), "
        f"val={len(val)} ({len(val_targets)} targets), "
        f"test={len(test)} ({len(test_targets)} targets)"
    )

    # Verify no overlap
    assert len(train_targets & test_targets) == 0, "Target leakage between train and test!"
    assert len(train_targets & val_targets) == 0, "Target leakage between train and val!"
    assert len(val_targets & test_targets) == 0, "Target leakage between val and test!"

    return train, val, test


def cold_both_split(
    df: pd.DataFrame,
    drug_col: str = "drug_id",
    target_col: str = "target_id",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cold-both split: test set contains interactions where BOTH drug AND target
    are unseen during training.

    This is the most challenging evaluation setting.

    Args:
        df: Interaction DataFrame.
        drug_col: Drug identifier column.
        target_col: Target identifier column.
        seed: Random seed.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    unique_drugs = df[drug_col].unique()
    unique_targets = df[target_col].unique()

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_drugs)
    rng.shuffle(unique_targets)

    n_test_drugs = max(1, int(len(unique_drugs) * test_ratio))
    n_val_drugs = max(1, int(len(unique_drugs) * val_ratio))
    n_test_targets = max(1, int(len(unique_targets) * test_ratio))
    n_val_targets = max(1, int(len(unique_targets) * val_ratio))

    test_drugs = set(unique_drugs[:n_test_drugs])
    val_drugs = set(unique_drugs[n_test_drugs:n_test_drugs + n_val_drugs])
    train_drugs = set(unique_drugs[n_test_drugs + n_val_drugs:])

    test_targets = set(unique_targets[:n_test_targets])
    val_targets = set(unique_targets[n_test_targets:n_test_targets + n_val_targets])
    train_targets = set(unique_targets[n_test_targets + n_val_targets:])

    # Test: both drug AND target are unseen
    test_mask = df[drug_col].isin(test_drugs) & df[target_col].isin(test_targets)
    test = df[test_mask].reset_index(drop=True)

    # Val: both drug AND target are in val sets
    val_mask = df[drug_col].isin(val_drugs) & df[target_col].isin(val_targets)
    val = df[val_mask].reset_index(drop=True)

    # Train: everything else (may include some interactions with val/test drugs/targets
    # as long as not BOTH are unseen)
    train_mask = ~(test_mask | val_mask)
    train = df[train_mask].reset_index(drop=True)

    logger.info(
        f"Cold-both split (seed={seed}): "
        f"train={len(train)}, val={len(val)}, test={len(test)}"
    )

    return train, val, test


def get_split(
    df: pd.DataFrame,
    strategy: str = "random",
    seed: int = 42,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Get train/val/test split using the specified strategy.

    Args:
        df: Interaction DataFrame.
        strategy: 'random', 'cold_drug', 'cold_target', or 'cold_both'.
        seed: Random seed.
        **kwargs: Additional arguments passed to the split function.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    strategies = {
        "random": random_split,
        "cold_drug": cold_drug_split,
        "cold_target": cold_target_split,
        "cold_both": cold_both_split,
    }

    if strategy not in strategies:
        raise ValueError(f"Unknown split strategy: '{strategy}'. Choose from {list(strategies.keys())}")

    return strategies[strategy](df, seed=seed, **kwargs)


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir,
    split_name: str = "random",
):
    """
    Save train/val/test DataFrames to CSV files.

    Args:
        train, val, test: Split DataFrames.
        output_dir: Directory to save to.
        split_name: Name prefix for files.
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(output_dir / f"{split_name}_train.csv", index=False)
    val.to_csv(output_dir / f"{split_name}_val.csv", index=False)
    test.to_csv(output_dir / f"{split_name}_test.csv", index=False)

    logger.info(f"Splits saved to {output_dir}/{split_name}_*.csv")


def load_splits(
    input_dir,
    split_name: str = "random",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load previously saved splits from CSV.

    Args:
        input_dir: Directory containing the split files.
        split_name: Name prefix used when saving.

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    from pathlib import Path
    input_dir = Path(input_dir)

    train = pd.read_csv(input_dir / f"{split_name}_train.csv")
    val = pd.read_csv(input_dir / f"{split_name}_val.csv")
    test = pd.read_csv(input_dir / f"{split_name}_test.csv")

    return train, val, test


def summarize_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    drug_col: str = "drug_id",
    target_col: str = "target_id",
    split_name: str = "random",
) -> dict:
    """
    Generate a summary of a data split, including overlap analysis.

    Returns:
        dict: Summary statistics for the split.
    """
    train_drugs = set(train[drug_col].unique())
    val_drugs = set(val[drug_col].unique())
    test_drugs = set(test[drug_col].unique())

    train_targets = set(train[target_col].unique())
    val_targets = set(val[target_col].unique())
    test_targets = set(test[target_col].unique())

    return {
        "split_name": split_name,
        "train_interactions": len(train),
        "val_interactions": len(val),
        "test_interactions": len(test),
        "train_drugs": len(train_drugs),
        "val_drugs": len(val_drugs),
        "test_drugs": len(test_drugs),
        "train_targets": len(train_targets),
        "val_targets": len(val_targets),
        "test_targets": len(test_targets),
        "drug_overlap_train_test": len(train_drugs & test_drugs),
        "target_overlap_train_test": len(train_targets & test_targets),
        "drug_overlap_train_val": len(train_drugs & val_drugs),
        "target_overlap_train_val": len(train_targets & val_targets),
        "unseen_test_drugs": len(test_drugs - train_drugs),
        "unseen_test_targets": len(test_targets - train_targets),
    }
