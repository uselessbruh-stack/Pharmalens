"""
PharmaLens — Data Split Strategies

Implements random, cold-drug, cold-target, and cold-both split strategies
for evaluating DTI prediction models on different generalization scenarios.
"""

import logging
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("pharmalens.evaluation.splits")


@lru_cache(maxsize=100000)
def scaffold_identity(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Scaffold split requires valid standardized SMILES")
    # Acyclic compounds share the empty Murcko scaffold; never disperse that
    # group across partitions just to obtain convenient row fractions.
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def grouped_domain_split(df, strategy, seed=42, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    if strategy == "scaffold":
        groups = df.smiles.map(scaffold_identity)
    elif strategy == "family":
        if (
            "family" not in df
            or df.family.isin(["", "unclassified"]).any()
            or df.family.isna().any()
        ):
            raise ValueError("Family holdout requires verified family annotations for every row")
        groups = df.family
    else:
        raise ValueError("Unknown grouped domain strategy")
    identities = _partition(groups, train_ratio, val_ratio, test_ratio, seed)
    parts = tuple(df[groups.isin(values)].reset_index(drop=True) for values in identities)
    validate_split_integrity(*parts, strategy)
    return parts


def _validate_ratios(train_ratio, val_ratio, test_ratio):
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=float)
    if not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("Train, validation and test ratios must be positive and sum to 1")


def _identity_column(df, kind, fallback):
    # Data must be chemically standardized before splitting. Clean SMILES and
    # exact sequences prevent aliases in source IDs from leaking across splits.
    column = "smiles" if kind == "drug" else "sequence"
    column = column if column in df.columns else fallback
    if (
        column not in df.columns
        or df[column].isna().any()
        or df[column].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"Missing or empty {kind} identity column: {column}")
    return column


def _partition(values, train_ratio, val_ratio, test_ratio, seed):
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    values = np.array(sorted(set(values)), dtype=object)
    if len(values) < 3:
        raise ValueError("At least three distinct identity groups are required")
    np.random.RandomState(seed).shuffle(values)
    n_test = max(1, int(len(values) * test_ratio))
    n_val = max(1, int(len(values) * val_ratio))
    n_train = len(values) - n_test - n_val
    if n_train < 1:
        raise ValueError("Ratios leave no training identity groups")
    return (
        set(values[:n_train]),
        set(values[n_train : n_train + n_val]),
        set(values[n_train + n_val :]),
    )


def validate_split_integrity(
    train, val, test, strategy="random", drug_col="drug_id", target_col="target_id"
):
    """Fail on repeated pair identities or held-out entity leakage."""
    frames = [train, val, test]
    if any(frame.empty for frame in frames):
        raise ValueError("A split partition is empty; revise group allocation or collect more data")
    dc = _identity_column(train, "drug", drug_col)
    tc = _identity_column(train, "target", target_col)
    for left, right in [(0, 1), (0, 2), (1, 2)]:
        a, b = frames[left], frames[right]
        if set(zip(a[dc], a[tc])) & set(zip(b[dc], b[tc])):
            raise ValueError("Interaction identity leakage across partitions")
        if strategy in {"cold_drug", "cold_both"} and set(a[dc]) & set(b[dc]):
            raise ValueError("Molecule identity leakage across partitions")
        if strategy in {"cold_target", "cold_both"} and set(a[tc]) & set(b[tc]):
            raise ValueError("Protein identity leakage across partitions")
        if strategy == "scaffold" and set(a[dc].map(scaffold_identity)) & set(
            b[dc].map(scaffold_identity)
        ):
            raise ValueError("Molecular scaffold leakage across partitions")
        if strategy == "family" and set(a.family) & set(b.family):
            raise ValueError("Protein family leakage across partitions")


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
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    dc = _identity_column(df, "drug", "drug_id")
    tc = _identity_column(df, "target", "target_id")
    # Split unique pair identities, then recover all replicates/aliases. This
    # intentionally supersedes the historical leaky row-wise random split.
    groups = df[[dc, tc]].drop_duplicates().sort_values([dc, tc]).reset_index(drop=True)

    # First split: train+val vs test
    train_val, test = train_test_split(
        groups, test_size=test_ratio, random_state=seed, shuffle=True
    )

    # Second split: train vs val
    val_frac = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=val_frac, random_state=seed, shuffle=True)

    pair_index = pd.MultiIndex.from_frame(df[[dc, tc]])
    train, val, test = [
        df[pair_index.isin(pd.MultiIndex.from_frame(part[[dc, tc]]))].reset_index(drop=True)
        for part in (train, val, test)
    ]
    validate_split_integrity(train, val, test)

    logger.info(f"Random split (seed={seed}): train={len(train)}, val={len(val)}, test={len(test)}")

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
    drug_col = _identity_column(df, "drug", drug_col)
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    unique_drugs = np.array(sorted(df[drug_col].unique()), dtype=object)
    n_drugs = len(unique_drugs)
    if n_drugs < 3:
        raise ValueError("At least three distinct molecules are required")

    # Split drugs (not interactions)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_drugs)

    n_test = max(1, int(n_drugs * test_ratio))
    n_val = max(1, int(n_drugs * val_ratio))
    n_train = n_drugs - n_test - n_val

    train_drugs = set(unique_drugs[:n_train])
    val_drugs = set(unique_drugs[n_train : n_train + n_val])
    test_drugs = set(unique_drugs[n_train + n_val :])

    train = df[df[drug_col].isin(train_drugs)].reset_index(drop=True)
    val = df[df[drug_col].isin(val_drugs)].reset_index(drop=True)
    test = df[df[drug_col].isin(test_drugs)].reset_index(drop=True)
    validate_split_integrity(train, val, test, "cold_drug", drug_col=drug_col)

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
    target_col = _identity_column(df, "target", target_col)
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    unique_targets = np.array(sorted(df[target_col].unique()), dtype=object)
    n_targets = len(unique_targets)
    if n_targets < 3:
        raise ValueError("At least three distinct proteins are required")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_targets)

    n_test = max(1, int(n_targets * test_ratio))
    n_val = max(1, int(n_targets * val_ratio))
    n_train = n_targets - n_test - n_val

    train_targets = set(unique_targets[:n_train])
    val_targets = set(unique_targets[n_train : n_train + n_val])
    test_targets = set(unique_targets[n_train + n_val :])

    train = df[df[target_col].isin(train_targets)].reset_index(drop=True)
    val = df[df[target_col].isin(val_targets)].reset_index(drop=True)
    test = df[df[target_col].isin(test_targets)].reset_index(drop=True)
    validate_split_integrity(train, val, test, "cold_target", target_col=target_col)

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
    drug_col = _identity_column(df, "drug", drug_col)
    target_col = _identity_column(df, "target", target_col)
    train_drugs, val_drugs, test_drugs = _partition(
        df[drug_col], train_ratio, val_ratio, test_ratio, seed
    )
    train_targets, val_targets, test_targets = _partition(
        df[target_col], train_ratio, val_ratio, test_ratio, seed + 1
    )

    # Test: both drug AND target are unseen
    test_mask = df[drug_col].isin(test_drugs) & df[target_col].isin(test_targets)
    test = df[test_mask].reset_index(drop=True)

    # Val: both drug AND target are in val sets
    val_mask = df[drug_col].isin(val_drugs) & df[target_col].isin(val_targets)
    val = df[val_mask].reset_index(drop=True)

    # Cross-block pairs are excluded: the complement would expose test entities.
    train_mask = df[drug_col].isin(train_drugs) & df[target_col].isin(train_targets)
    train = df[train_mask].reset_index(drop=True)
    dropped = len(df) - len(train) - len(val) - len(test)
    validate_split_integrity(train, val, test, "cold_both", drug_col, target_col)
    for part in (train, val, test):
        part.attrs["excluded_cross_block_rows"] = dropped

    logger.info(
        f"Cold-both split (seed={seed}): train={len(train)}, val={len(val)}, test={len(test)}"
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
    if strategy in {"scaffold", "family"}:
        return grouped_domain_split(df, strategy, seed=seed, **kwargs)
    strategies = {
        "random": random_split,
        "cold_drug": cold_drug_split,
        "cold_target": cold_target_split,
        "cold_both": cold_both_split,
    }

    if strategy not in strategies:
        raise ValueError(
            f"Unknown split strategy: '{strategy}'. Choose from {list(strategies.keys())}"
        )

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
