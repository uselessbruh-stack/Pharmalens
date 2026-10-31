"""Publish every locked benchmark result, keeping validation selection separate from test."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file, utc_now


def compile_report(selection_file, evaluation_dir, output):
    selection_file, evaluation_dir, output = (
        Path(selection_file),
        Path(evaluation_dir),
        Path(output),
    )
    if output.exists():
        raise ValueError("Benchmark reports are immutable; choose a new output")
    lock = json.loads(selection_file.read_text())
    evaluation = json.loads((evaluation_dir / "report.json").read_text())
    if evaluation["selection_lock_sha256"] != digest_file(selection_file) or set(
        evaluation["results"]
    ) != set(lock["candidates"]):
        raise ValueError("Evaluation does not match the complete locked candidate set")
    runs = {}
    for name, evidence in lock["candidates"].items():
        directory = Path(evidence["directory"])
        if digest_file(directory / "run.json") != evidence["report_sha256"]:
            raise ValueError("Training report changed after selection")
        original = json.loads((directory / "run.json").read_text())
        tested = evaluation["results"][name]
        if digest_file(evaluation_dir / f"{name}.csv") != tested["prediction_sha256"]:
            raise ValueError("Test predictions changed")
        runs[name] = {
            "artifact_sha256": evidence["artifact_sha256"],
            "validation_metrics": original["validation_metrics"],
            "test_metrics": tested["test_metrics"],
            "seed": original["config"]["seed"],
            "model": original["config"]["model"],
            "serving_parity": tested["serving_parity"],
        }
    tasks = []
    for selection in lock["selections"]:
        architecture_results = {}
        for architecture, summary in selection["architectures"].items():
            values = [runs[name]["test_metrics"]["rmse"] for name in summary["runs"]]
            architecture_results[architecture] = {
                **summary,
                "test_mean_rmse": float(np.mean(values)),
                "test_seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        interval = None
        if selection["selected_architecture"] != "xgboost":
            reference_names = selection["architectures"]["xgboost"]["runs"]
            reference = next(
                (name for name in reference_names if runs[name]["seed"] == 42), reference_names[0]
            )
            candidate = selection["representative_run"]
            left = pd.read_csv(evaluation_dir / f"{reference}.csv")
            right = pd.read_csv(evaluation_dir / f"{candidate}.csv")
            paired = left.merge(
                right,
                on="observation_id",
                suffixes=("_reference", "_candidate"),
                validate="one_to_one",
            )
            if len(paired) != len(left) or not np.array_equal(
                paired.score_reference, paired.score_candidate
            ):
                raise ValueError("Test comparison rows/labels differ")
            paired["a"] = (paired.prediction_reference - paired.score_reference) ** 2
            paired["b"] = (paired.prediction_candidate - paired.score_candidate) ** 2
            groups = (
                paired.groupby("target_id_reference")
                .agg(a=("a", "sum"), b=("b", "sum"), n=("a", "size"))
                .to_numpy()
            )
            if len(groups) >= 3:
                rng = np.random.default_rng(2026)
                differences = []
                for _ in range(2000):
                    totals = groups[rng.integers(0, len(groups), len(groups))].sum(0)
                    differences.append(
                        float(np.sqrt(totals[1] / totals[2]) - np.sqrt(totals[0] / totals[2]))
                    )
                interval = {
                    "candidate_minus_xgboost_rmse": runs[candidate]["test_metrics"]["rmse"]
                    - runs[reference]["test_metrics"]["rmse"],
                    "target_group_percentile_95_interval": np.quantile(
                        differences, [0.025, 0.975]
                    ).tolist(),
                    "target_groups": len(groups),
                    "limitations": "Conditional test-cohort interval; does not correct multiple comparisons or chemical-cluster dependence",
                }
        tasks.append(
            {
                **selection,
                "architectures": architecture_results,
                "representative_test_comparison": interval,
            }
        )
    report = {
        "created_at": utc_now(),
        "selection_lock_sha256": digest_file(selection_file),
        "evaluation_report_sha256": digest_file(evaluation_dir / "report.json"),
        "selection_policy": lock["selection_policy"],
        "test_used_for_selection": False,
        "runs": runs,
        "tasks": tasks,
        "limitations": lock["limitations"],
    }
    atomic_json(output, report)
    lines = [
        "# Frozen affinity benchmark",
        "",
        "Defaults were selected using validation only. Test scores did not change the selections.",
        "",
        "| Dataset | Endpoint | Model | Seeds | Mean validation RMSE | Mean test RMSE |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task in tasks:
        for model, values in task["architectures"].items():
            lines.append(
                f"| {task['scope'][0]} | {task['scope'][2]} | {model} | {len(values['seeds'])} | {values['validation_mean_rmse']:.4f} | {values['test_mean_rmse']:.4f} |"
            )
    lines += [
        "",
        "These are identity-grouped random splits. Family views overlap broad datasets; they are not independent external evidence. Most tasks have one model seed. The nuclear-receptor tasks have three. Neural gains are task-dependent and some validation gains do not carry over to test.",
    ]
    output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compile_report(args.selection, args.evaluation, args.output)
