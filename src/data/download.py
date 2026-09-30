"""
PharmaLens — Dataset Download and Management

Downloads KIBA (and optionally Davis) benchmark datasets for DTI prediction.
Source: DeepDTA repository (https://github.com/hkmztrk/DeepDTA)
"""

import json
import os
import pickle
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    METADATA_DIR,
    ensure_dirs,
    save_json,
    setup_logging,
)

logger = setup_logging("pharmalens.data")

# ==============================================================================
# Download URLs
# ==============================================================================

# DeepDTA GitHub raw content URLs for KIBA and Davis datasets
DEEPDTA_BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"

DATASET_URLS = {
    "kiba": {
        "ligands": f"{DEEPDTA_BASE}/kiba/ligands_can.txt",
        "proteins": f"{DEEPDTA_BASE}/kiba/proteins.txt",
        "Y": f"{DEEPDTA_BASE}/kiba/Y",
    },
    "davis": {
        "ligands": f"{DEEPDTA_BASE}/davis/ligands_can.txt",
        "proteins": f"{DEEPDTA_BASE}/davis/proteins.txt",
        "Y": f"{DEEPDTA_BASE}/davis/Y",
    },
}


# ==============================================================================
# Download Functions
# ==============================================================================

def download_file(url: str, dest_path: Path, desc: str = "") -> bool:
    """
    Download a file from URL to destination path with progress bar.

    Args:
        url: URL to download from.
        dest_path: Local file path to save to.
        desc: Description for progress bar.

    Returns:
        True if download successful, False otherwise.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        logger.info(f"File already exists: {dest_path.name}")
        return True

    try:
        logger.info(f"Downloading {desc or dest_path.name} from {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc or dest_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded: {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
        return True

    except requests.RequestException as e:
        logger.error(f"Download failed for {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_dataset(dataset_name: str = "kiba") -> bool:
    """
    Download a complete DTI dataset (KIBA or Davis).

    Args:
        dataset_name: 'kiba' or 'davis'

    Returns:
        True if all files downloaded successfully.
    """
    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'kiba' or 'davis'.")

    ensure_dirs()
    dataset_dir = RAW_DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    urls = DATASET_URLS[dataset_name]
    success = True

    for file_key, url in urls.items():
        filename = url.split("/")[-1]
        dest = dataset_dir / filename
        if not download_file(url, dest, desc=f"{dataset_name}/{filename}"):
            success = False

    if success:
        logger.info(f"✓ Dataset '{dataset_name}' downloaded successfully to {dataset_dir}")
    else:
        logger.error(f"✗ Some files failed to download for '{dataset_name}'")

    return success


# ==============================================================================
# Data Loading Functions
# ==============================================================================

def load_ligands(dataset_name: str = "kiba") -> dict:
    """
    Load drug SMILES from the ligands file.

    Returns:
        dict: {drug_id: SMILES_string}
    """
    filepath = RAW_DATA_DIR / dataset_name / "ligands_can.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Ligands file not found: {filepath}\n"
            f"Run download_dataset('{dataset_name}') first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # The file is a JSON-like dictionary
    ligands = json.loads(content)
    logger.info(f"Loaded {len(ligands)} drugs from {dataset_name}")
    return ligands


def load_proteins(dataset_name: str = "kiba") -> dict:
    """
    Load protein sequences from the proteins file.

    Returns:
        dict: {protein_id: amino_acid_sequence}
    """
    filepath = RAW_DATA_DIR / dataset_name / "proteins.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Proteins file not found: {filepath}\n"
            f"Run download_dataset('{dataset_name}') first."
        )

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    proteins = json.loads(content)
    logger.info(f"Loaded {len(proteins)} proteins from {dataset_name}")
    return proteins


def load_interaction_matrix(dataset_name: str = "kiba") -> np.ndarray:
    """
    Load the drug-target interaction matrix.

    The Y matrix has shape (num_drugs, num_targets).
    NaN values indicate unmeasured interactions.

    Returns:
        np.ndarray: Interaction score matrix
    """
    filepath = RAW_DATA_DIR / dataset_name / "Y"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Interaction matrix not found: {filepath}\n"
            f"Run download_dataset('{dataset_name}') first."
        )

    # The Y file is a pickled numpy array (Python 2 format)
    with open(filepath, "rb") as f:
        Y = pickle.load(f, encoding="latin1")

    Y = np.array(Y, dtype=np.float64)

    # Replace infinity with NaN for consistent handling
    Y[np.isinf(Y)] = np.nan

    logger.info(f"Loaded interaction matrix: shape={Y.shape} from {dataset_name}")
    return Y


def load_raw_dataset(dataset_name: str = "kiba") -> tuple:
    """
    Load all components of a raw dataset.

    Returns:
        tuple: (ligands_dict, proteins_dict, interaction_matrix)
    """
    ligands = load_ligands(dataset_name)
    proteins = load_proteins(dataset_name)
    Y = load_interaction_matrix(dataset_name)
    return ligands, proteins, Y


def create_interaction_dataframe(
    ligands: dict,
    proteins: dict,
    Y: np.ndarray,
    dataset_name: str = "kiba",
) -> pd.DataFrame:
    """
    Convert the raw data into a tidy DataFrame of (drug, target, score) triples.

    Only includes measured interactions (non-NaN values).

    Args:
        ligands: {drug_id: SMILES}
        proteins: {protein_id: sequence}
        Y: Interaction matrix (drugs × targets)

    Returns:
        pd.DataFrame with columns: drug_id, target_id, smiles, sequence, score
    """
    drug_ids = list(ligands.keys())
    target_ids = list(proteins.keys())

    records = []
    for i, drug_id in enumerate(drug_ids):
        for j, target_id in enumerate(target_ids):
            score = Y[i, j]
            if not np.isnan(score):
                records.append({
                    "drug_id": drug_id,
                    "target_id": target_id,
                    "smiles": ligands[drug_id],
                    "sequence": proteins[target_id],
                    "score": float(score),
                })

    df = pd.DataFrame(records)
    logger.info(
        f"Created interaction DataFrame: {len(df):,} rows, "
        f"{df['drug_id'].nunique()} drugs, {df['target_id'].nunique()} targets"
    )
    return df


# ==============================================================================
# Dataset Summary
# ==============================================================================

def generate_dataset_summary(
    df: pd.DataFrame,
    Y: np.ndarray,
    dataset_name: str = "kiba",
) -> dict:
    """
    Generate comprehensive dataset statistics and save to JSON.

    Args:
        df: Interaction DataFrame
        Y: Raw interaction matrix
        dataset_name: Name of dataset

    Returns:
        dict: Summary statistics
    """
    summary = {
        "dataset_name": dataset_name,
        "generated_at": pd.Timestamp.now().isoformat(),
        "matrix_shape": list(Y.shape),
        "total_matrix_entries": int(Y.size),
        "measured_interactions": int(np.sum(~np.isnan(Y))),
        "unmeasured_interactions": int(np.sum(np.isnan(Y))),
        "sparsity_pct": round(float(np.sum(np.isnan(Y)) / Y.size * 100), 2),
        "num_records": len(df),
        "unique_drugs": int(df["drug_id"].nunique()),
        "unique_targets": int(df["target_id"].nunique()),
        "duplicate_pairs": int(df.duplicated(subset=["drug_id", "target_id"]).sum()),
        "missing_smiles": int(df["smiles"].isna().sum()),
        "missing_sequences": int(df["sequence"].isna().sum()),
        "missing_scores": int(df["score"].isna().sum()),
        "score_statistics": {
            "mean": round(float(df["score"].mean()), 4),
            "std": round(float(df["score"].std()), 4),
            "min": round(float(df["score"].min()), 4),
            "max": round(float(df["score"].max()), 4),
            "median": round(float(df["score"].median()), 4),
            "q25": round(float(df["score"].quantile(0.25)), 4),
            "q75": round(float(df["score"].quantile(0.75)), 4),
        },
        "smiles_length_stats": {
            "mean": round(float(df["smiles"].str.len().mean()), 1),
            "min": int(df["smiles"].str.len().min()),
            "max": int(df["smiles"].str.len().max()),
        },
        "sequence_length_stats": {
            "mean": round(float(df["sequence"].str.len().mean()), 1),
            "min": int(df["sequence"].str.len().min()),
            "max": int(df["sequence"].str.len().max()),
        },
    }

    # Save to metadata
    ensure_dirs()
    summary_path = METADATA_DIR / f"{dataset_name}_summary.json"
    save_json(summary, summary_path)
    logger.info(f"Dataset summary saved to {summary_path}")

    return summary


def print_dataset_summary(summary: dict):
    """Pretty-print the dataset summary."""
    print("=" * 60)
    print(f"  DATASET SUMMARY: {summary['dataset_name'].upper()}")
    print("=" * 60)
    print(f"  Matrix shape:           {summary['matrix_shape']}")
    print(f"  Total entries:          {summary['total_matrix_entries']:,}")
    print(f"  Measured interactions:  {summary['measured_interactions']:,}")
    print(f"  Unmeasured (NaN):       {summary['unmeasured_interactions']:,}")
    print(f"  Sparsity:               {summary['sparsity_pct']}%")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  DataFrame records:      {summary['num_records']:,}")
    print(f"  Unique drugs:           {summary['unique_drugs']}")
    print(f"  Unique targets:         {summary['unique_targets']}")
    print(f"  Duplicate pairs:        {summary['duplicate_pairs']}")
    print(f"  ─────────────────────────────────────────────────")
    stats = summary["score_statistics"]
    print(f"  Score mean ± std:       {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  Score range:            [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"  Score median:           {stats['median']:.4f}")
    print(f"  Score Q25/Q75:          {stats['q25']:.4f} / {stats['q75']:.4f}")
    print(f"  ─────────────────────────────────────────────────")
    smiles_stats = summary["smiles_length_stats"]
    seq_stats = summary["sequence_length_stats"]
    print(f"  SMILES length:          {smiles_stats['min']}–{smiles_stats['max']} (avg {smiles_stats['mean']})")
    print(f"  Sequence length:        {seq_stats['min']}–{seq_stats['max']} (avg {seq_stats['mean']})")
    print("=" * 60)


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Download and inspect KIBA
    print("Downloading KIBA dataset...")
    download_dataset("kiba")

    print("\nLoading dataset...")
    ligands, proteins, Y = load_raw_dataset("kiba")

    print("\nCreating DataFrame...")
    df = create_interaction_dataframe(ligands, proteins, Y, "kiba")

    print("\nGenerating summary...")
    summary = generate_dataset_summary(df, Y, "kiba")
    print_dataset_summary(summary)

    print(f"\nFirst 5 rows:")
    print(df.head())
