"""
PharmaLens — Protein Preprocessing

Validates and cleans protein amino-acid sequences.
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.preprocessing.protein")

# Standard 20 amino acid single-letter codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Extended codes (including ambiguous)
EXTENDED_AA = set("ACDEFGHIKLMNPQRSTVWYBJOUXZ")

# Minimum and maximum sequence lengths to keep
MIN_SEQ_LENGTH = 50
MAX_SEQ_LENGTH = 5000


def validate_sequence(sequence: str, allow_extended: bool = False) -> bool:
    """
    Check if a protein sequence contains only valid amino acid characters.

    Args:
        sequence: Amino acid sequence string.
        allow_extended: If True, allow ambiguous codes (B, J, O, U, X, Z).

    Returns:
        True if the sequence is valid.
    """
    if not sequence or not isinstance(sequence, str):
        return False

    seq_upper = sequence.upper().strip()
    valid_chars = EXTENDED_AA if allow_extended else STANDARD_AA

    return all(c in valid_chars for c in seq_upper)


def normalize_sequence(sequence: str) -> Optional[str]:
    """
    Normalize a protein sequence.

    Steps:
        1. Convert to uppercase
        2. Remove whitespace and newlines
        3. Remove non-letter characters

    Args:
        sequence: Raw amino acid sequence.

    Returns:
        Normalized sequence string, or None if empty after cleaning.
    """
    if not sequence or not isinstance(sequence, str):
        return None

    # Uppercase and strip
    seq = sequence.upper().strip()

    # Remove whitespace, numbers, special characters
    seq = re.sub(r'[^A-Z]', '', seq)

    if len(seq) == 0:
        return None

    return seq


def replace_nonstandard_aa(sequence: str) -> str:
    """
    Replace non-standard amino acid codes with standard ones.

    Replacements:
        B (Asx = Asp or Asn) → N (Asn, more common)
        J (Xle = Leu or Ile) → L (Leu, more common)
        O (Pyrrolysine) → K (Lysine, structurally similar)
        U (Selenocysteine) → C (Cysteine, structurally similar)
        X (Unknown) → A (Alanine, smallest/neutral)
        Z (Glx = Glu or Gln) → Q (Gln, more common)
    """
    replacements = {
        'B': 'N',
        'J': 'L',
        'O': 'K',
        'U': 'C',
        'X': 'A',
        'Z': 'Q',
    }
    for old, new in replacements.items():
        sequence = sequence.replace(old, new)
    return sequence


def preprocess_proteins(
    df: pd.DataFrame,
    sequence_col: str = "sequence",
    min_length: int = MIN_SEQ_LENGTH,
    max_length: int = MAX_SEQ_LENGTH,
) -> tuple[pd.DataFrame, dict]:
    """
    Full protein preprocessing pipeline.

    Steps:
        1. Normalize sequences (uppercase, strip)
        2. Replace non-standard amino acids
        3. Validate sequences
        4. Filter by length
        5. Report duplicates

    Args:
        df: DataFrame with a sequence column.
        sequence_col: Name of the sequence column.
        min_length: Minimum allowed sequence length.
        max_length: Maximum allowed sequence length.

    Returns:
        tuple: (cleaned_df, report_dict)
    """
    report = {
        "original_count": len(df),
        "steps": [],
    }

    logger.info(f"Starting protein preprocessing on {len(df)} records...")

    # Step 1: Remove missing sequences
    n_before = len(df)
    df = df.dropna(subset=[sequence_col]).copy()
    n_missing = n_before - len(df)
    report["steps"].append({"step": "remove_missing_sequences", "removed": n_missing})
    if n_missing > 0:
        logger.info(f"  Removed {n_missing} rows with missing sequences")

    # Step 2: Normalize
    df[sequence_col] = df[sequence_col].apply(normalize_sequence)
    n_before = len(df)
    df = df.dropna(subset=[sequence_col])
    n_failed = n_before - len(df)
    report["steps"].append({"step": "normalize", "removed": int(n_failed)})

    # Step 3: Replace non-standard amino acids
    n_nonstandard = df[sequence_col].apply(
        lambda s: any(c not in STANDARD_AA for c in s)
    ).sum()
    df[sequence_col] = df[sequence_col].apply(replace_nonstandard_aa)
    report["steps"].append({"step": "replace_nonstandard_aa", "affected": int(n_nonstandard)})
    if n_nonstandard > 0:
        logger.info(f"  Replaced non-standard AA in {n_nonstandard} sequences")

    # Step 4: Validate (should all pass after normalization + replacement)
    n_before = len(df)
    df["_valid"] = df[sequence_col].apply(validate_sequence)
    n_invalid = (~df["_valid"]).sum()
    df = df[df["_valid"]].drop(columns=["_valid"])
    report["steps"].append({"step": "validate", "removed": int(n_invalid)})
    if n_invalid > 0:
        logger.info(f"  Removed {n_invalid} sequences with invalid characters")

    # Step 5: Filter by length
    df["_seq_len"] = df[sequence_col].str.len()
    n_before = len(df)
    df = df[(df["_seq_len"] >= min_length) & (df["_seq_len"] <= max_length)]
    n_filtered = n_before - len(df)
    df = df.drop(columns=["_seq_len"])
    report["steps"].append({
        "step": "filter_by_length",
        "removed": int(n_filtered),
        "min_length": min_length,
        "max_length": max_length,
    })
    if n_filtered > 0:
        logger.info(f"  Removed {n_filtered} sequences outside length range [{min_length}, {max_length}]")

    # Step 6: Report duplicates
    n_unique = df[sequence_col].nunique()
    report["unique_targets_after"] = int(n_unique)

    report["final_count"] = len(df)
    report["total_removed"] = report["original_count"] - report["final_count"]

    logger.info(
        f"  Protein preprocessing complete: {report['original_count']} → {report['final_count']} "
        f"({report['total_removed']} removed, {n_unique} unique targets)"
    )

    return df, report


def validate_and_clean_sequence(sequence: str) -> Optional[str]:
    """
    Validate and clean a single protein sequence.

    Convenience function for the inference pipeline. Combines:
    normalization, non-standard AA replacement, and validation.

    Args:
        sequence: Raw amino acid sequence.

    Returns:
        Cleaned sequence, or None if invalid.
    """
    normalized = normalize_sequence(sequence)
    if normalized is None:
        return None

    # Replace non-standard amino acids
    cleaned = replace_nonstandard_aa(normalized)

    # Validate
    if not validate_sequence(cleaned):
        return None

    # Check length
    if len(cleaned) < 10:
        return None

    return cleaned

