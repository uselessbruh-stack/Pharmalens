"""Audit committed research evidence using only the Python standard library.

Run from any directory: python scripts/audit_research.py
This checks data and saved records; it does not rerun or certify model metrics.
"""

import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def strict_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    return value


def main():
    path = ROOT / "data/processed/kiba_clean.csv"
    rows = read_csv(path)
    drugs = {row["drug_id"]: row["smiles"] for row in rows}
    targets = {row["target_id"]: row["sequence"] for row in rows}
    identities = Counter((row["smiles"], row["sequence"]) for row in rows)
    scores = [float(row["score"]) for row in rows]
    result = {
        "scope": "Static source review and committed data/record audit; no model inference or retraining",
        "dataset": {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
            "drug_ids": len(drugs),
            "distinct_clean_smiles": len(set(drugs.values())),
            "target_ids": len(targets),
            "distinct_sequences": len(set(targets.values())),
            "duplicate_identity_pair_rows": sum(n - 1 for n in identities.values()),
            "nonfinite_scores": sum(not math.isfinite(score) for score in scores),
            "drugs_over_100_characters": sum(len(s) > 100 for s in drugs.values()),
            "targets_over_1000_residues": sum(len(s) > 1000 for s in targets.values()),
            "interactions_with_truncated_drugs": sum(len(r["smiles"]) > 100 for r in rows),
            "interactions_with_truncated_targets": sum(len(r["sequence"]) > 1000 for r in rows),
        },
        "splits": {},
    }
    for strategy in ("random", "cold_drug", "cold_target"):
        parts = {}
        for part in ("train", "val", "test"):
            records = read_csv(ROOT / f"data/processed/splits/{strategy}_{part}.csv")
            parts[part] = {
                "rows": len(records),
                "drug_ids": {r["drug_id"] for r in records},
                "target_ids": {r["target_id"] for r in records},
                "smiles": {r["smiles"] for r in records},
                "sequences": {r["sequence"] for r in records},
                "id_pairs": {(r["drug_id"], r["target_id"]) for r in records},
                "identity_pairs": {(r["smiles"], r["sequence"]) for r in records},
            }
        result["splits"][strategy] = {
            "sizes": {name: p["rows"] for name, p in parts.items()},
            "overlap": {
                f"{left}_{right}": {
                    key: len(parts[left][key] & parts[right][key])
                    for key in (
                        "drug_ids",
                        "target_ids",
                        "smiles",
                        "sequences",
                        "id_pairs",
                        "identity_pairs",
                    )
                }
                for left, right in combinations(parts, 2)
            },
        }
    log = json.loads((ROOT / "experiments/results/experiment_log.json").read_text())
    result["recorded_random_results"] = sorted(
        [
            {
                key: row.get(key)
                for key in (
                    "model",
                    "seed",
                    "rmse",
                    "mae",
                    "ci",
                    "mse",
                    "training_time_sec",
                    "n_samples",
                )
            }
            for row in log
            if row["split"] == "random"
        ],
        key=lambda row: row["rmse"],
    )
    result["recorded_seeds"] = sorted({row["seed"] for row in log})
    result["model_artifacts"] = [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "models").rglob("*"))
        if p.is_file() and p.name != ".gitkeep"
    ]
    result["python_dependencies_available"] = {
        name: importlib.util.find_spec(name) is not None
        for name in ("numpy", "pandas", "torch", "rdkit", "xgboost", "fastapi", "pytest")
    }
    result["notebooks"] = []
    for notebook in sorted((ROOT / "notebooks").glob("*.ipynb")):
        document = json.loads(notebook.read_text(encoding="utf-8"))
        result["notebooks"].append(
            {
                "name": notebook.name,
                "code_cells": sum(c["cell_type"] == "code" for c in document["cells"]),
                "saved_errors": [
                    {"name": output.get("ename"), "message": output.get("evalue")}
                    for cell in document["cells"]
                    for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ],
            }
        )
    output = ROOT / "results/reports/research_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(strict_json(result), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}")
    print(
        f"Rows: {len(rows)}; distinct molecules: {len(set(drugs.values()))}; duplicate identity rows: {sum(n - 1 for n in identities.values())}"
    )
    for name, split in result["splits"].items():
        print(f"{name} train/test overlap: {split['overlap']['train_test']}")
    print(f"Available model artifacts: {len(result['model_artifacts'])}")


if __name__ == "__main__":
    main()
