"""
PharmaLens — Drug Feature Extraction

Generates molecular descriptors, fingerprints, and graph representations
from drug SMILES strings using RDKit.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.features.drug")


# ==============================================================================
# Molecular Descriptors
# ==============================================================================

def compute_molecular_descriptors(smiles: str) -> Optional[dict]:
    """
    Compute physicochemical descriptors for a single molecule.

    Descriptors:
        - molecular_weight: Molecular weight (Da)
        - heavy_atom_count: Number of non-hydrogen atoms
        - ring_count: Number of rings
        - hbd: Hydrogen bond donors
        - hba: Hydrogen bond acceptors
        - rotatable_bonds: Number of rotatable bonds
        - tpsa: Topological polar surface area (Å²)
        - logp: Estimated octanol-water partition coefficient (Wildman-Crippen)
        - num_aromatic_rings: Number of aromatic rings
        - fraction_csp3: Fraction of carbon atoms that are sp3 hybridized

    Args:
        smiles: SMILES string.

    Returns:
        dict of descriptor values, or None if molecule is invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "heavy_atom_count": Descriptors.HeavyAtomCount(mol),
            "ring_count": Descriptors.RingCount(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "logp": round(Descriptors.MolLogP(mol), 2),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "fraction_csp3": round(Descriptors.FractionCSP3(mol), 4),
        }
    except Exception as e:
        logger.warning(f"Failed to compute descriptors for SMILES={smiles[:50]}: {e}")
        return None


def compute_descriptors_batch(smiles_list: list[str]) -> pd.DataFrame:
    """
    Compute molecular descriptors for a batch of SMILES.

    Args:
        smiles_list: List of SMILES strings.

    Returns:
        pd.DataFrame with one row per molecule and descriptor columns.
    """
    from tqdm import tqdm

    records = []
    for smiles in tqdm(smiles_list, desc="Computing descriptors"):
        desc = compute_molecular_descriptors(smiles)
        if desc is not None:
            desc["smiles"] = smiles
            records.append(desc)
        else:
            records.append({"smiles": smiles})

    df = pd.DataFrame(records)
    logger.info(f"Computed descriptors for {len(df)} molecules ({df.dropna().shape[0]} valid)")
    return df


# ==============================================================================
# Molecular Fingerprints
# ==============================================================================

def compute_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    """
    Compute Morgan (circular) fingerprint for a molecule.

    Morgan fingerprints encode circular substructure patterns around each atom.
    They are similar to ECFP (Extended Connectivity Fingerprints).

    Args:
        smiles: SMILES string.
        radius: Fingerprint radius (2 = ECFP4, 3 = ECFP6).
        n_bits: Length of the bit vector.

    Returns:
        np.ndarray of shape (n_bits,) with 0/1 values, or None if invalid.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)


def compute_fingerprints_batch(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 1024,
) -> np.ndarray:
    """
    Compute Morgan fingerprints for a batch of SMILES.

    Args:
        smiles_list: List of SMILES strings.
        radius: Fingerprint radius.
        n_bits: Fingerprint length.

    Returns:
        np.ndarray of shape (n_molecules, n_bits).
        Invalid molecules get all-zero fingerprints.
    """
    from tqdm import tqdm

    fps = []
    n_failed = 0

    for smiles in tqdm(smiles_list, desc="Computing fingerprints"):
        fp = compute_morgan_fingerprint(smiles, radius, n_bits)
        if fp is not None:
            fps.append(fp)
        else:
            fps.append(np.zeros(n_bits, dtype=np.int8))
            n_failed += 1

    if n_failed > 0:
        logger.warning(f"Failed to compute fingerprints for {n_failed} molecules")

    return np.array(fps)


def compute_maccs_keys(smiles: str) -> Optional[np.ndarray]:
    """
    Compute MACCS (Molecular ACCess System) keys fingerprint.

    MACCS keys are a set of 166 predefined structural key patterns.

    Args:
        smiles: SMILES string.

    Returns:
        np.ndarray of shape (167,) with 0/1 values.
    """
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp, dtype=np.int8)


# ==============================================================================
# SMILES Tokenization (for DeepDTA-style models)
# ==============================================================================

# Character set for SMILES encoding
SMILES_CHARSET = {
    '#': 1, '%': 2, '(': 3, ')': 4, '+': 5, '-': 6, '.': 7,
    '/': 8, '0': 9, '1': 10, '2': 11, '3': 12, '4': 13, '5': 14,
    '6': 15, '7': 16, '8': 17, '9': 18, '=': 19, '@': 20,
    'A': 21, 'B': 22, 'C': 23, 'F': 24, 'G': 25, 'H': 26,
    'I': 27, 'K': 28, 'L': 29, 'M': 30, 'N': 31, 'O': 32,
    'P': 33, 'R': 34, 'S': 35, 'T': 36, 'V': 37, 'W': 38,
    'X': 39, 'Y': 40, 'Z': 41, '[': 42, '\\': 43, ']': 44,
    'a': 45, 'b': 46, 'c': 47, 'd': 48, 'e': 49, 'f': 50,
    'g': 51, 'h': 52, 'i': 53, 'l': 54, 'm': 55, 'n': 56,
    'o': 57, 'p': 58, 'r': 59, 's': 60, 't': 61, 'u': 62,
}


def encode_smiles(smiles: str, max_length: int = 100) -> np.ndarray:
    """
    Encode a SMILES string as a sequence of integers (character-level).

    Used by DeepDTA-style models where SMILES is processed as a 1D sequence.

    Args:
        smiles: SMILES string.
        max_length: Maximum sequence length (pad/truncate to this).

    Returns:
        np.ndarray of shape (max_length,) with integer indices.
        0 is used for padding.
    """
    encoded = np.zeros(max_length, dtype=np.int32)
    for i, char in enumerate(smiles[:max_length]):
        encoded[i] = SMILES_CHARSET.get(char, 0)
    return encoded


def encode_smiles_batch(smiles_list: list[str], max_length: int = 100) -> np.ndarray:
    """
    Encode a batch of SMILES strings.

    Args:
        smiles_list: List of SMILES strings.
        max_length: Maximum sequence length.

    Returns:
        np.ndarray of shape (n_molecules, max_length).
    """
    return np.array([encode_smiles(s, max_length) for s in smiles_list])


# ==============================================================================
# Combined Drug Feature Vector
# ==============================================================================

def get_drug_feature_vector(
    smiles: str,
    include_descriptors: bool = True,
    include_fingerprints: bool = True,
    fp_radius: int = 2,
    fp_bits: int = 1024,
) -> Optional[np.ndarray]:
    """
    Create a combined feature vector for a drug.

    Concatenates molecular descriptors and Morgan fingerprint.
    Used for traditional ML models (RF, XGBoost).

    Args:
        smiles: SMILES string.
        include_descriptors: Include physicochemical descriptors.
        include_fingerprints: Include Morgan fingerprint.
        fp_radius: Fingerprint radius.
        fp_bits: Fingerprint length.

    Returns:
        np.ndarray: Combined feature vector, or None if invalid.
    """
    features = []

    if include_descriptors:
        desc = compute_molecular_descriptors(smiles)
        if desc is None:
            return None
        # Ordered list of descriptor values (excluding smiles key)
        desc_values = [
            desc["molecular_weight"], desc["heavy_atom_count"],
            desc["ring_count"], desc["hbd"], desc["hba"],
            desc["rotatable_bonds"], desc["tpsa"], desc["logp"],
            desc["num_aromatic_rings"], desc["fraction_csp3"],
        ]
        features.extend(desc_values)

    if include_fingerprints:
        fp = compute_morgan_fingerprint(smiles, fp_radius, fp_bits)
        if fp is None:
            return None
        features.extend(fp.tolist())

    return np.array(features, dtype=np.float32)


# Descriptor names for feature importance interpretation
DESCRIPTOR_NAMES = [
    "molecular_weight", "heavy_atom_count", "ring_count",
    "hbd", "hba", "rotatable_bonds", "tpsa", "logp",
    "num_aromatic_rings", "fraction_csp3",
]


def get_feature_names(include_descriptors: bool = True, include_fingerprints: bool = True, fp_bits: int = 1024) -> list[str]:
    """Get feature names for the combined drug feature vector."""
    names = []
    if include_descriptors:
        names.extend(DESCRIPTOR_NAMES)
    if include_fingerprints:
        names.extend([f"morgan_bit_{i}" for i in range(fp_bits)])
    return names
