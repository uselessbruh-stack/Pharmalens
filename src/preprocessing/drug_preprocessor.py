"""
PharmaLens — Drug Preprocessing

Validates, canonicalizes, and cleans drug SMILES strings using RDKit.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.preprocessing.drug")


def validate_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string represents a valid molecule.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if the SMILES is valid, False otherwise.
    """
    from rdkit import Chem
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to its canonical (standardized) form.

    Canonical SMILES ensures the same molecule always has the same
    string representation, regardless of how it was originally written.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES string, or None if invalid.
    """
    from rdkit import Chem
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def remove_salts_and_fragments(smiles: str) -> Optional[str]:
    """
    Remove salts and keep only the largest fragment from a SMILES string.

    Some SMILES contain multiple fragments separated by '.', for example:
    'CC.O' contains two fragments. We keep only the largest one.

    Args:
        smiles: Input SMILES string (possibly with fragments).

    Returns:
        SMILES of the largest fragment, or None if invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if not smiles or not isinstance(smiles, str):
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # If no fragments, return as-is
    if '.' not in smiles:
        return Chem.MolToSmiles(mol, canonical=True)

    # Split into fragments and keep the largest
    fragments = smiles.split('.')
    largest = None
    largest_size = 0

    for frag in fragments:
        frag_mol = Chem.MolFromSmiles(frag)
        if frag_mol is not None:
            size = frag_mol.GetNumHeavyAtoms()
            if size > largest_size:
                largest_size = size
                largest = frag_mol

    if largest is None:
        return None

    return Chem.MolToSmiles(largest, canonical=True)


def preprocess_drugs(df: pd.DataFrame, smiles_col: str = "smiles") -> tuple[pd.DataFrame, dict]:
    """
    Full drug preprocessing pipeline.

    Steps:
        1. Validate SMILES (remove invalid molecules)
        2. Remove salts/fragments (keep largest fragment)
        3. Canonicalize SMILES
        4. Remove duplicates (by canonical SMILES)

    Args:
        df: DataFrame with a SMILES column.
        smiles_col: Name of the SMILES column.

    Returns:
        tuple: (cleaned_df, report_dict)
            - cleaned_df: DataFrame with valid, canonical SMILES
            - report_dict: Summary of cleaning steps
    """
    report = {
        "original_count": len(df),
        "steps": [],
    }

    logger.info(f"Starting drug preprocessing on {len(df)} records...")

    # Step 1: Remove rows with missing SMILES
    n_before = len(df)
    df = df.dropna(subset=[smiles_col]).copy()
    n_missing = n_before - len(df)
    report["steps"].append({"step": "remove_missing_smiles", "removed": n_missing})
    if n_missing > 0:
        logger.info(f"  Removed {n_missing} rows with missing SMILES")

    # Step 2: Validate SMILES
    n_before = len(df)
    df["_valid"] = df[smiles_col].apply(validate_smiles)
    n_invalid = (~df["_valid"]).sum()
    df = df[df["_valid"]].drop(columns=["_valid"])
    report["steps"].append({"step": "validate_smiles", "removed": int(n_invalid)})
    if n_invalid > 0:
        logger.info(f"  Removed {n_invalid} rows with invalid SMILES")

    # Step 3: Remove salts/fragments
    df["_clean_smiles"] = df[smiles_col].apply(remove_salts_and_fragments)
    n_failed = df["_clean_smiles"].isna().sum()
    df = df.dropna(subset=["_clean_smiles"])
    df[smiles_col] = df["_clean_smiles"]
    df = df.drop(columns=["_clean_smiles"])
    report["steps"].append({"step": "remove_salts", "removed": int(n_failed)})

    # Step 4: Canonicalize
    df[smiles_col] = df[smiles_col].apply(canonicalize_smiles)
    df = df.dropna(subset=[smiles_col])

    # Step 5: Report duplicates (but do NOT remove interaction rows — 
    # the same drug can interact with different proteins)
    n_unique_drugs = df[smiles_col].nunique()
    report["unique_drugs_after"] = int(n_unique_drugs)

    report["final_count"] = len(df)
    report["total_removed"] = report["original_count"] - report["final_count"]

    logger.info(
        f"  Drug preprocessing complete: {report['original_count']} → {report['final_count']} "
        f"({report['total_removed']} removed, {n_unique_drugs} unique drugs)"
    )

    return df, report


def validate_and_clean_smiles(smiles: str) -> Optional[str]:
    """
    Validate, clean, and canonicalize a single SMILES string.

    Convenience function for the inference pipeline. Combines:
    validation, salt/fragment removal, and canonicalization.

    Args:
        smiles: Raw SMILES string.

    Returns:
        Canonical cleaned SMILES, or None if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    # Remove leading/trailing whitespace
    smiles = smiles.strip()

    # Remove salts/fragments and canonicalize
    cleaned = remove_salts_and_fragments(smiles)
    return cleaned

