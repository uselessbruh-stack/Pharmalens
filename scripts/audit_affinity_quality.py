"""Describe assay/replicate limits without deleting inconvenient measurements."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file


def audit(path):
    frame = pd.read_csv(path, keep_default_na=False)
    groups = frame.groupby(["drug_id", "target_id", "endpoint"])
    statistics = groups.score.agg(["size", "min", "max", "mean"])
    repeated = statistics[statistics["size"] > 1]
    residual = frame.score - groups.score.transform("mean")
    families = []
    for family, rows in frame.groupby("family"):
        families.append(
            {
                "family": family,
                "observations": len(rows),
                "molecules": rows.drug_id.nunique(),
                "proteins": rows.target_id.nunique(),
                "label_quantiles": rows.score.quantile([0, 0.01, 0.5, 0.99, 1]).to_dict(),
            }
        )
    return {
        "file": str(Path(path).resolve()),
        "sha256": digest_file(path),
        "endpoint": frame.endpoint.unique().tolist(),
        "observations": len(frame),
        "molecules": frame.drug_id.nunique(),
        "proteins": frame.target_id.nunique(),
        "unique_pairs": len(statistics),
        "repeated_pairs": len(repeated),
        "repeated_pairs_spread_over_half_log_unit": int(
            ((repeated["max"] - repeated["min"]) > 0.5).sum()
        ),
        "repeated_pairs_spread_over_one_log_unit": int(
            ((repeated["max"] - repeated["min"]) > 1).sum()
        ),
        "within_pair_rms_dispersion": float(np.sqrt(np.mean(residual.to_numpy() ** 2))),
        "dispersion_interpretation": "Descriptive all-data inconsistency for predictors using only molecule/protein identities; not a trained or held-out baseline",
        "missing_publication_fraction": float(frame.publication_id.eq("").mean())
        if "publication_id" in frame
        else None,
        "protein_rows_above_1500_residues": int((frame.sequence.str.len() > 1500).sum()),
        "smiles_rows_above_200_characters": int((frame.smiles.str.len() > 200).sum()),
        "families": families,
        "data_filtered_or_labels_changed": False,
        "limitations": "Assays, constructs, publication reuse and source selection require further review; repeated observations are not independent evidence",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable quality report")
    atomic_json(args.output, {"datasets": [audit(path) for path in args.inputs]})
    print(f"Audited {len(args.inputs)} endpoint datasets")
