"""Lock validation-only model selection, then evaluate saved models on untouched test rows."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file, utc_now
from src.evaluation.metrics import compute_regression_metrics
from src.inference.pipeline import PharmaLensPipeline
from src.training.experiment import read_partitions, source_hash, tabular_features
from src.utils.serialization import json_safe


def freeze(plans, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Selection locks are immutable; choose a new output")
    groups, candidates = defaultdict(list), {}
    for plan_file in plans:
        plan = json.loads(Path(plan_file).read_text())
        for job in plan["jobs"]:
            directory = Path(job["output"])
            report_file = directory / "run.json"
            report = json.loads(report_file.read_text())
            if (
                directory.name.endswith(".part")
                or report["status"] != "validation_complete"
                or report["config"].get("smoke")
                or report.get("test_metrics") is not None
            ):
                raise ValueError("Require completed, non-smoke validation-only runs")
            if (
                report["source_sha256"] != plan["source_sha256"]
                or not report["serving_parity"]["verified"]
            ):
                raise ValueError("Run provenance/parity does not match its plan")
            if digest_file(directory / report["artifact"]["path"]) != report["artifact"]["sha256"]:
                raise ValueError("Artifact checksum mismatch")
            name = report["run_id"]
            candidates[name] = {
                "directory": str(directory.resolve()),
                "report_sha256": digest_file(report_file),
                "artifact_sha256": report["artifact"]["sha256"],
            }
            config = report["config"]
            scope = (
                config["dataset"],
                config["dataset_version"],
                config["endpoint"],
                config["split"],
            )
            groups[scope].append(report)
    selections = []
    for scope, rows in groups.items():
        if (
            len({r["data_sha256"] for r in rows}) != 1
            or len({r["split_membership"]["val"]["sha256"] for r in rows}) != 1
        ):
            raise ValueError("Selection requires matching data and validation membership")
        architectures = defaultdict(list)
        for report in rows:
            architectures[report["config"]["model"]].append(report)
        summaries = {}
        for architecture, repetitions in architectures.items():
            seeds = [r["config"]["seed"] for r in repetitions]
            if len(set(seeds)) != len(seeds):
                raise ValueError("Duplicate initialization seed within a task/architecture")
            scores = [r["validation_metrics"]["rmse"] for r in repetitions]
            summaries[architecture] = {
                "validation_mean_rmse": float(np.mean(scores)),
                "validation_seed_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
                "seeds": seeds,
                "runs": [r["run_id"] for r in repetitions],
            }
        winner = min(summaries, key=lambda key: summaries[key]["validation_mean_rmse"])
        representative = min(
            architectures[winner], key=lambda r: (r["config"]["seed"] != 42, r["config"]["seed"])
        )
        selections.append(
            {
                "scope": scope,
                "selected_architecture": winner,
                "representative_run": representative["run_id"],
                "architectures": summaries,
            }
        )
    lock = {
        "created_at": utc_now(),
        "selection_policy": "Lowest mean validation RMSE across available initialization seeds; seed 42 is the fixed serving representative, never best-seed selection",
        "test_used_for_selection": False,
        "candidates": candidates,
        "selections": selections,
        "limitations": "Most tasks have one initialization; nuclear-receptor endpoints have three; random identity-grouped evaluation does not establish cold-target/scaffold or external validity",
    }
    atomic_json(output, lock)
    return lock


def test_loader(frame, config, features):
    params = config["model_params"]
    if config["model"] == "deeptda":
        from torch.utils.data import DataLoader
        from src.models.deeptda_dataset import DTIDataset, create_encoding_dicts

        dl, pl = params.get("drug_max_len", 100), params.get("protein_max_len", 1000)
        drugs, proteins = create_encoding_dicts(frame, drug_max_len=dl, target_max_len=pl)
        dataset = DTIDataset(
            frame.drug_id, frame.target_id, frame.score, drugs, proteins, dl, pl, auxiliary=features
        )
    else:
        from torch_geometric.loader import DataLoader
        from src.models.gnn_dataset import CachedGraphPairDataset

        dataset = CachedGraphPairDataset(
            frame, params.get("protein_max_len", 1000), auxiliary=features
        )
    return DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)


def evaluate(selection_file, output):
    selection_file, output = Path(selection_file), Path(output)
    if output.exists():
        raise ValueError("Evaluation outputs are immutable; choose a new directory")
    lock = json.loads(selection_file.read_text())
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    results = {}
    for name, evidence in lock["candidates"].items():
        root = Path(evidence["directory"])
        if digest_file(root / "run.json") != evidence["report_sha256"]:
            raise ValueError("Run changed after validation selection was locked")
        report = json.loads((root / "run.json").read_text())
        config = report["config"]
        artifact = root / report["artifact"]["path"]
        if digest_file(artifact) != evidence["artifact_sha256"]:
            raise ValueError("Selected artifact changed")
        _, parts, fingerprint = read_partitions(config)
        frame = parts[2]
        if (
            fingerprint != report["data_sha256"]
            or digest_file(root / "test_membership.csv")
            != report["split_membership"]["test"]["sha256"]
        ):
            raise ValueError("Test dataset/membership changed")
        columns = ["observation_id", "drug_id", "target_id"]
        if (
            not frame[columns]
            .astype(str)
            .reset_index(drop=True)
            .equals(pd.read_csv(root / "test_membership.csv", dtype=str)[columns])
        ):
            raise ValueError("Reconstructed test rows do not match frozen membership")
        pipeline = PharmaLensPipeline.from_pretrained(
            artifact, config["model"], config["endpoint"], config["preprocessing_version"]
        )
        needs_auxiliary = config["model"] == "xgboost" or config["model_params"].get(
            "auxiliary_dim", 0
        )
        features = (
            tabular_features([frame], pipeline.embedding_store)[0] if needs_auxiliary else None
        )
        if config["model"] == "xgboost":
            prediction = pipeline.model.predict(features)
        else:
            values = []
            with torch.inference_mode():
                for batch in test_loader(frame, config, features):
                    if hasattr(batch, "y"):
                        values.append(pipeline.model(batch).cpu().numpy())
                    else:
                        values.append(pipeline.model(*batch[:-1]).cpu().numpy())
            prediction = (
                np.concatenate(values) * pipeline.target_scaler["scale"]
                + pipeline.target_scaler["mean"]
            )
        served = pipeline.predict(frame.iloc[0].smiles, frame.iloc[0].sequence)["score"]
        if not np.isclose(served, prediction[0], atol=1e-5, rtol=1e-5):
            raise ValueError("Test batching and serving parity failed")
        exported = frame[[*columns, "score"]].copy()
        exported["prediction"] = prediction
        exported.to_csv(output / f"{name}.csv", index=False)
        results[name] = {
            "test_metrics": compute_regression_metrics(frame.score.to_numpy(), prediction),
            "prediction_sha256": digest_file(output / f"{name}.csv"),
            "test_rows": len(frame),
            "serving_parity": True,
        }
        atomic_json(
            output / "progress.json",
            {"completed": len(results), "total": len(lock["candidates"]), "last_run": name},
        )
        print(f"{name}: test RMSE {results[name]['test_metrics']['rmse']:.5f}", flush=True)
    atomic_json(
        output / "report.json",
        json_safe(
            {
                "selection_lock_sha256": digest_file(selection_file),
                "evaluator_source_sha256": source_hash(),
                "test_used_for_selection": False,
                "results": results,
                "locked_selections": lock["selections"],
                "limitations": lock["limitations"],
            }
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", nargs="+", type=Path)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plans:
        freeze(args.plans, args.selection)
    if args.output:
        evaluate(args.selection, args.output)
