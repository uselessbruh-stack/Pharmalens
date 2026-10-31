"""Build new identity-safe splits without overwriting historical experiments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.acquisition import ROOT, atomic_json, digest_file
from src.evaluation.split_strategies import get_split, save_splits, summarize_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/kiba_clean.csv")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/datasets/kiba_identity_v2/splits"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new version instead of overwriting split provenance")
    frame = pd.read_csv(args.input)
    args.output.mkdir(parents=True)
    report = {
        "schema_version": 2,
        "input_sha256": digest_file(args.input),
        "seed": args.seed,
        "identity_policy": "prestandardized_smiles_and_exact_sequence",
        "historical_metrics_valid_for_these_splits": False,
        "splits": {},
    }
    for strategy in ("random", "cold_drug", "cold_target", "cold_both"):
        train, val, test = get_split(frame, strategy, seed=args.seed)
        save_splits(train, val, test, args.output, strategy)
        report["splits"][strategy] = {
            "summary": summarize_split(train, val, test, split_name=strategy),
            "excluded_cross_block_rows": train.attrs.get("excluded_cross_block_rows", 0),
            "files": {
                part: {
                    "path": f"{strategy}_{part}.csv",
                    "sha256": digest_file(args.output / f"{strategy}_{part}.csv"),
                }
                for part in ("train", "val", "test")
            },
        }
        print(strategy, len(train), len(val), len(test))
    atomic_json(args.output / "manifest.json", report)


if __name__ == "__main__":
    main()
