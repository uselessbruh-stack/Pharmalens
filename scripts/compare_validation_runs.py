"""Paired, target-group bootstrap of compatible validation predictions."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file


def compare(reference_dir, candidate_dir, draws=2000, seed=2026):
    runs, predictions = [], []
    for directory in (Path(reference_dir), Path(candidate_dir)):
        report = json.loads((directory / "run.json").read_text())
        if report["config"].get("smoke") or report["status"] != "validation_complete":
            raise ValueError("Only completed non-smoke validation runs can be compared")
        runs.append(report)
        predictions.append(pd.read_csv(directory / "validation_predictions.csv"))
    contracts = [
        (
            r["config"]["dataset"],
            r["config"]["dataset_version"],
            r["config"]["endpoint"],
            r["config"]["split"],
            r["data_sha256"],
            r["split_membership"]["val"]["sha256"],
        )
        for r in runs
    ]
    if contracts[0] != contracts[1]:
        raise ValueError("Runs have different data, labels or validation membership")
    left, right = predictions
    if left.observation_id.duplicated().any() or right.observation_id.duplicated().any():
        raise ValueError("Validation observations are not unique")
    joined = left.merge(
        right, on="observation_id", suffixes=("_reference", "_candidate"), validate="one_to_one"
    )
    if (
        len(joined) != len(left)
        or len(joined) != len(right)
        or not np.array_equal(joined.score_reference, joined.score_candidate)
        or not np.array_equal(joined.target_id_reference, joined.target_id_candidate)
    ):
        raise ValueError("Paired observations, targets or labels differ")
    joined["error_reference"] = (joined.prediction_reference - joined.score_reference) ** 2
    joined["error_candidate"] = (joined.prediction_candidate - joined.score_candidate) ** 2
    grouped = joined.groupby("target_id_reference").agg(
        error_reference=("error_reference", "sum"),
        error_candidate=("error_candidate", "sum"),
        rows=("observation_id", "size"),
    )
    if len(grouped) < 3 or draws < 100:
        raise ValueError("Need at least three target groups and 100 bootstrap draws")
    values = grouped.to_numpy()
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        sampled = values[rng.integers(0, len(values), len(values))].sum(axis=0)
        deltas.append(float(np.sqrt(sampled[1] / sampled[2]) - np.sqrt(sampled[0] / sampled[2])))
    return {
        "reference_run": runs[0]["run_id"],
        "candidate_run": runs[1]["run_id"],
        "rows": len(joined),
        "target_groups": len(grouped),
        "draws": draws,
        "seed": seed,
        "candidate_minus_reference_rmse": runs[1]["validation_metrics"]["rmse"]
        - runs[0]["validation_metrics"]["rmse"],
        "conditional_target_bootstrap_95_percentile_interval": np.quantile(
            deltas, [0.025, 0.975]
        ).tolist(),
        "source_hashes": [r["source_sha256"] for r in runs],
        "prediction_file_hashes": [
            digest_file(Path(p) / "validation_predictions.csv")
            for p in (reference_dir, candidate_dir)
        ],
        "scope": "validation only; negative differences favor candidate",
        "limitations": "Conditional on this validation set; does not correct tuning/selection bias or measure initialization uncertainty; not final test evidence",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new immutable comparison output")
    atomic_json(args.output, compare(args.reference, args.candidate))
