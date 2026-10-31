"""Register complete compatible matrix groups as provisional local models."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.register_experiment import register_run
from src.data.acquisition import digest_file
from src.inference.model_registry import ModelRegistry
from src.training.experiment import load_config


def register_complete_groups(plan, registry_path=None):
    groups = defaultdict(list)
    for job in plan["jobs"]:
        if digest_file(job["config"]) != job["config_sha256"]:
            raise ValueError("Matrix configuration changed")
        config = json.loads(Path(job["config"]).read_text())
        groups[
            (
                config["dataset"],
                config["dataset_version"],
                config["endpoint"],
                config.get("split", "random"),
            )
        ].append(job)
    completed = []
    for scope, jobs in groups.items():
        if not all((Path(job["output"]) / "run.json").is_file() for job in jobs):
            continue
        reports = [json.loads((Path(job["output"]) / "run.json").read_text()) for job in jobs]
        if any(
            report["config"] != load_config(job["config"]) for job, report in zip(jobs, reports)
        ):
            raise ValueError("Matrix output configuration differs from the frozen plan")
        if any(
            report["source_sha256"] != plan["source_sha256"]
            or report["config"].get("smoke")
            or not report["serving_parity"]["verified"]
            for report in reports
        ):
            raise ValueError("Matrix output failed source/smoke/parity requirements")
        if (
            len({report["split_membership"]["val"]["sha256"] for report in reports}) != 1
            or len({report["data_sha256"] for report in reports}) != 1
        ):
            raise ValueError("Cannot select a default across different validation data")
        winner = min(
            range(len(reports)), key=lambda index: reports[index]["validation_metrics"]["rmse"]
        )
        previous = ModelRegistry(registry_path).get_best_model(
            dataset=scope[0], dataset_version=scope[1], endpoint=scope[2], split=scope[3]
        )
        preserve_previous = False
        if previous:
            previous_report = Path(previous.path).parent / "run.json"
            if not previous_report.is_file():
                preserve_previous = True  # Do not replace an incomparable existing default.
            else:
                prior = json.loads(previous_report.read_text())
                comparable = (
                    prior["data_sha256"] == reports[0]["data_sha256"]
                    and prior["split_membership"]["val"]["sha256"]
                    == reports[0]["split_membership"]["val"]["sha256"]
                )
                preserve_previous = (
                    not comparable
                    or prior["validation_metrics"]["rmse"]
                    <= reports[winner]["validation_metrics"]["rmse"]
                )
        for index, (job, report) in enumerate(zip(jobs, reports)):
            name = Path(job["output"]).name
            registry = ModelRegistry(registry_path)
            if name in registry.entries:
                if registry.entries[name].artifact_sha256 != report["artifact"]["sha256"]:
                    raise ValueError("Existing registration differs from the matrix artifact")
                continue
            register_run(
                job["output"],
                name,
                registry_path,
                default=index == winner and not preserve_previous,
            )
        completed.append(
            {
                "scope": scope,
                "provisional_default": previous.name
                if preserve_previous
                else Path(jobs[winner]["output"]).name,
                "selection_metric": "validation_rmse",
                "final_research_champion": False,
            }
        )
    return completed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            register_complete_groups(json.loads(args.plan.read_text()), args.registry), indent=2
        )
    )
