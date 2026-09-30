"""
PharmaLens — Protein Feature Extraction

Generates sequence-derived features for protein targets.
"""

import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.features.protein")


# Standard amino acid alphabet
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
NUM_AA = len(AMINO_ACIDS)


# ==============================================================================
# Amino Acid Composition
# ==============================================================================

def amino_acid_composition(sequence: str) -> np.ndarray:
    """
    Compute amino acid composition (AAC) — the fraction of each amino acid.

    AAC is a 20-dimensional vector where each element represents
    the frequency of a standard amino acid in the sequence.

    Args:
        sequence: Amino acid sequence string.

    Returns:
        np.ndarray of shape (20,) — normalized frequencies.
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return np.zeros(NUM_AA, dtype=np.float32)

    counts = Counter(sequence)
    composition = np.array(
        [counts.get(aa, 0) / seq_len for aa in AMINO_ACIDS],
        dtype=np.float32
    )
    return composition


# ==============================================================================
# Dipeptide Composition
# ==============================================================================

def dipeptide_composition(sequence: str) -> np.ndarray:
    """
    Compute dipeptide composition (DPC) — the frequency of each pair of amino acids.

    DPC is a 400-dimensional vector (20×20) that captures local sequence
    patterns by counting consecutive amino acid pairs.

    Args:
        sequence: Amino acid sequence string.

    Returns:
        np.ndarray of shape (400,) — normalized dipeptide frequencies.
    """
    n_dipeptides = len(sequence) - 1
    if n_dipeptides <= 0:
        return np.zeros(NUM_AA * NUM_AA, dtype=np.float32)

    # Count dipeptides
    counts = Counter()
    for i in range(n_dipeptides):
        dipep = sequence[i:i + 2]
        counts[dipep] += 1

    # Build feature vector in fixed order
    composition = np.zeros(NUM_AA * NUM_AA, dtype=np.float32)
    for i, aa1 in enumerate(AMINO_ACIDS):
        for j, aa2 in enumerate(AMINO_ACIDS):
            idx = i * NUM_AA + j
            composition[idx] = counts.get(aa1 + aa2, 0) / n_dipeptides

    return composition


# ==============================================================================
# Conjoint Triad Features
# ==============================================================================

# Group amino acids into 7 categories based on physicochemical properties
# Following Shen et al., 2007
AA_GROUPS = {
    'A': 0, 'G': 0, 'V': 0,  # Small, hydrophobic
    'I': 1, 'L': 1, 'F': 1, 'P': 1,  # Hydrophobic
    'Y': 2, 'M': 2, 'T': 2, 'S': 2,  # Polar, uncharged
    'H': 3, 'N': 3, 'Q': 3, 'W': 3,  # Polar, uncharged (larger)
    'R': 4, 'K': 4,  # Positively charged
    'D': 5, 'E': 5,  # Negatively charged
    'C': 6,  # Cysteine (special)
}


def conjoint_triad(sequence: str) -> np.ndarray:
    """
    Compute Conjoint Triad (CT) features.

    Groups amino acids into 7 categories based on physicochemical properties,
    then counts all possible triad patterns (7³ = 343 features).

    Reference:
        Shen et al., 2007. "Predicting protein-protein interactions based only
        on sequences information."

    Args:
        sequence: Amino acid sequence string.

    Returns:
        np.ndarray of shape (343,) — normalized triad frequencies.
    """
    n_triads = len(sequence) - 2
    if n_triads <= 0:
        return np.zeros(343, dtype=np.float32)

    # Convert sequence to group indices
    groups = [AA_GROUPS.get(aa, 0) for aa in sequence]

    # Count triads
    counts = np.zeros(343, dtype=np.float32)
    for i in range(n_triads):
        idx = groups[i] * 49 + groups[i + 1] * 7 + groups[i + 2]
        counts[idx] += 1

    # Normalize
    counts /= n_triads

    return counts


# ==============================================================================
# CTD Features (Composition, Transition, Distribution)
# ==============================================================================

# Property groups for CTD
# Each property divides AAs into 3 classes
CTD_PROPERTIES = {
    "hydrophobicity": {
        0: set("RKEDQN"),       # Polar
        1: set("GASTPHY"),      # Neutral
        2: set("CLVIMFW"),      # Hydrophobic
    },
    "charge": {
        0: set("KR"),           # Positive
        1: set("ANCQGHILMFPSTWYV"),  # Neutral
        2: set("DE"),           # Negative
    },
    "polarity": {
        0: set("LIFWCMVY"),     # Nonpolar
        1: set("PATGS"),        # Slightly polar
        2: set("HQRKNED"),      # Polar
    },
}


def ctd_composition(sequence: str) -> np.ndarray:
    """
    CTD Composition: fraction of residues in each class for each property.

    Returns 3 properties × 3 classes = 9 features.
    """
    n = len(sequence)
    if n == 0:
        return np.zeros(9, dtype=np.float32)

    features = []
    for prop_name, groups in CTD_PROPERTIES.items():
        for group_id in [0, 1, 2]:
            count = sum(1 for aa in sequence if aa in groups[group_id])
            features.append(count / n)

    return np.array(features, dtype=np.float32)


# ==============================================================================
# Sequence Encoding (for DeepDTA-style models)
# ==============================================================================

# Protein sequence character set (25 chars: 20 standard + padding)
PROTEIN_CHARSET = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-indexed, 0 = padding


def encode_sequence(sequence: str, max_length: int = 1000) -> np.ndarray:
    """
    Encode a protein sequence as a sequence of integers.

    Used by DeepDTA-style models where protein sequence is processed as 1D input.

    Args:
        sequence: Amino acid sequence.
        max_length: Maximum length (pad/truncate).

    Returns:
        np.ndarray of shape (max_length,) with integer indices.
    """
    encoded = np.zeros(max_length, dtype=np.int32)
    for i, aa in enumerate(sequence[:max_length]):
        encoded[i] = PROTEIN_CHARSET.get(aa, 0)
    return encoded


def encode_sequences_batch(sequences: list[str], max_length: int = 1000) -> np.ndarray:
    """Encode a batch of protein sequences."""
    return np.array([encode_sequence(s, max_length) for s in sequences])


# ==============================================================================
# Combined Protein Feature Vector
# ==============================================================================

def get_protein_feature_vector(
    sequence: str,
    include_aac: bool = True,
    include_dpc: bool = True,
    include_ct: bool = False,
    include_ctd: bool = False,
) -> np.ndarray:
    """
    Create a combined feature vector for a protein target.

    Concatenates selected sequence-derived features.
    Used for traditional ML models (RF, XGBoost).

    Args:
        sequence: Amino acid sequence.
        include_aac: Include amino acid composition (20D).
        include_dpc: Include dipeptide composition (400D).
        include_ct: Include conjoint triad (343D).
        include_ctd: Include CTD composition (9D).

    Returns:
        np.ndarray: Combined feature vector.
    """
    features = []

    if include_aac:
        features.append(amino_acid_composition(sequence))
    if include_dpc:
        features.append(dipeptide_composition(sequence))
    if include_ct:
        features.append(conjoint_triad(sequence))
    if include_ctd:
        features.append(ctd_composition(sequence))

    if not features:
        return np.array([], dtype=np.float32)

    return np.concatenate(features)


def get_protein_feature_names(
    include_aac: bool = True,
    include_dpc: bool = True,
    include_ct: bool = False,
    include_ctd: bool = False,
) -> list[str]:
    """Get feature names for the protein feature vector."""
    names = []
    if include_aac:
        names.extend([f"aac_{aa}" for aa in AMINO_ACIDS])
    if include_dpc:
        names.extend([f"dpc_{aa1}{aa2}" for aa1 in AMINO_ACIDS for aa2 in AMINO_ACIDS])
    if include_ct:
        names.extend([f"ct_{i}" for i in range(343)])
    if include_ctd:
        for prop in CTD_PROPERTIES:
            for c in range(3):
                names.append(f"ctd_{prop}_c{c}")
    return names
