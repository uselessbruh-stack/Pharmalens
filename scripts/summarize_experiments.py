"""Summarize validation runs without ranking incompatible tasks or smoke runs."""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.data.acquisition import atomic_json


def summarize(root):
    groups = defaultdict(list)
    smokes = []
    for path in sorted(Path(root).glob("*/run.json")):
        if path.parent.name.endswith(".part"):
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["config"].get("smoke"):
            smokes.append(
                {
                    "run": path.parent.name,
                    "status": run["status"],
                    "serving_parity": run["serving_parity"]["verified"],
                }
            )
            continue
        config = run["config"]
        contract = {
            "dataset": config["dataset"],
            "dataset_version": config["dataset_version"],
            "endpoint": config["endpoint"],
            "split": config["split"],
            "split_seed": config["split_seed"],
            "data_sha256": run["data_sha256"],
            "validation_membership_sha256": run["split_membership"]["val"]["sha256"],
            "source_sha256": run["source_sha256"],
            "model": config["model"],
            "model_params": config["model_params"],
            "training": config.get("training", {}),
            "protein_feature_contract": run.get(
                "protein_feature_contract", config.get("protein_embedding_cache")
            ),
        }
        key = json.dumps(contract, sort_keys=True)
        groups[key].append((path.parent.name, run))
    summaries = []
    for key, rows in groups.items():
        # Re-running the same model seed does not count as another replicate.
        seeds = {}
        for name, run in rows:
            seed = run["config"]["seed"]
            if seed in seeds:
                raise ValueError(f"Duplicate seed within one experiment contract: {seed}")
            seeds[seed] = (name, run)
        metrics = {}
        for metric in ("rmse", "mae", "ci", "pearson", "r2"):
            values = [row[1]["validation_metrics"][metric] for row in seeds.values()]
            values = [value for value in values if value is not None]
            metrics[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        summaries.append(
            {
                "contract_id": hashlib.sha256(key.encode()).hexdigest(),
                "contract": json.loads(key),
                "model_seeds": sorted(seeds),
                "runs": [row[0] for row in seeds.values()],
                "validation_metrics": metrics,
                "replicates": len(seeds),
                "all_serving_parity_passed": all(
                    row[1]["serving_parity"]["verified"] for row in seeds.values()
                ),
                "test_evaluations": sum(
                    row[1]["test_metrics"] is not None for row in seeds.values()
                ),
                "iteration_budget_reached": [
                    row[0] for row in seeds.values() if row[1].get("hit_iteration_budget")
                ],
            }
        )
    return {
        "scope": "Validation summaries grouped by data, split, code and model contract",
        "groups": summaries,
        "smoke_checks_excluded_from_model_ranking": smokes,
        "primary_model_selected": False,
        "limitations": "Seed variation is not a confidence interval or proof of architecture superiority; different label scales and splits are not ranked together",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("experiments/runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/reports/phase3_experiments.json")
    )
    args = parser.parse_args()
    report = summarize(args.runs)
    atomic_json(args.output, report)
    print(f"Summarized {len(report['groups'])} compatible experiment groups; no primary promoted")


if __name__ == "__main__":
    main()
