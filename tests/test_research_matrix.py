import json

import pandas as pd
import pytest

from scripts.build_research_cohorts import publish_views
from scripts.run_experiment_matrix import run_matrix
from src.data.acquisition import atomic_json, digest_file
from src.training.experiment import source_hash


@pytest.mark.parametrize("prior_rmse", [None, 0.7])
def test_matrix_selection_waits_for_complete_group_and_preserves_stronger_default(
    tmp_path, prior_rmse
):
    from scripts.register_experiment import register_run
    from scripts.register_matrix_models import register_complete_groups
    from src.inference.model_registry import ModelRegistry
    from src.training.experiment import load_config

    def make_run(name, rmse):
        directory = tmp_path / name
        directory.mkdir()
        artifact = directory / "model.json"
        artifact.write_text("test-only artifact, never loaded")
        config_file = tmp_path / f"{name}.json"
        atomic_json(
            config_file,
            {
                "dataset": "scope",
                "dataset_version": "v1",
                "endpoint": "Ki",
                "input": "unused.csv",
                "model": "xgboost",
                "preprocessing_version": "standardized_v1",
            },
        )
        config = load_config(config_file)
        atomic_json(
            directory / "run.json",
            {
                "run_id": name,
                "status": "validation_complete",
                "config": config,
                "source_sha256": "frozen",
                "data_sha256": "same",
                "split_membership": {"val": {"sha256": "same"}},
                "validation_metrics": {"rmse": rmse},
                "serving_parity": {"verified": True},
                "effective_model_config": {},
                "artifact": {"path": artifact.name, "sha256": digest_file(artifact)},
            },
        )
        return {
            "id": name,
            "config": str(config_file),
            "config_sha256": digest_file(config_file),
            "output": str(directory),
        }

    first, second = make_run("first", 1.2), make_run("second", 0.9)
    registry = tmp_path / "registry.json"
    if prior_rmse is not None:
        prior = make_run("prior", prior_rmse)
        register_run(prior["output"], "prior", registry, default=True)
    plan = {"source_sha256": "frozen", "jobs": [first, second]}
    report = register_complete_groups(plan, registry)
    assert report[0]["provisional_default"] == ("prior" if prior_rmse is not None else "second")
    selected = ModelRegistry(registry).get_best_model(
        dataset="scope", dataset_version="v1", endpoint="Ki"
    )
    assert selected.name == ("prior" if prior_rmse is not None else "second")
    assert not report[0]["final_research_champion"]


def test_paired_comparison_rejects_different_labels_or_membership(tmp_path):
    from scripts.compare_validation_runs import compare

    runs = []
    for name, error in [("reference", 1.0), ("candidate", 0.5)]:
        folder = tmp_path / name
        folder.mkdir()
        frame = pd.DataFrame(
            {
                "observation_id": [str(i) for i in range(6)],
                "target_id": [str(i // 2) for i in range(6)],
                "score": list(range(6)),
                "prediction": [i + error for i in range(6)],
            }
        )
        frame.to_csv(folder / "validation_predictions.csv", index=False)
        atomic_json(
            folder / "run.json",
            {
                "run_id": name,
                "status": "validation_complete",
                "config": {
                    "dataset": "same",
                    "dataset_version": "v1",
                    "endpoint": "Ki",
                    "split": "random",
                    "smoke": False,
                },
                "data_sha256": "same",
                "source_sha256": "code",
                "split_membership": {"val": {"sha256": "same"}},
                "validation_metrics": {"rmse": error},
            },
        )
        runs.append(folder)
    result = compare(*runs, draws=100)
    assert result["candidate_minus_reference_rmse"] == -0.5
    assert result["conditional_target_bootstrap_95_percentile_interval"] == [-0.5, -0.5]
    frame.loc[0, "score"] = 100
    frame.to_csv(runs[1] / "validation_predictions.csv", index=False)
    with pytest.raises(ValueError, match="labels differ"):
        compare(*runs, draws=100)


def test_cohorts_preserve_labels_and_disclose_shared_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    outputs = {}
    for endpoint in ("Ki", "Kd"):
        data = source / f"{endpoint}.csv"
        pd.DataFrame(
            [
                {
                    "drug_id": str(i),
                    "target_id": str(j),
                    "family": "gpcr",
                    "endpoint": endpoint,
                    "score": i + j,
                }
                for i in range(4)
                for j in range(3)
            ]
        ).to_csv(data, index=False)
        outputs[endpoint] = {"path": data.name, "sha256": digest_file(data)}
    atomic_json(source / "manifest.json", {"outputs": outputs})
    output = tmp_path / "views"
    publish_views(source, output, min_rows=10, min_drugs=3, min_targets=3)
    for endpoint in ("Ki", "Kd"):
        manifest = json.loads((output / f"gpcr_{endpoint.lower()}" / "manifest.json").read_text())
        assert not manifest["independent_dataset_claim"]
        assert manifest["counts"][endpoint] == 12
    with pytest.raises(ValueError, match="new cohort"):
        publish_views(source, output)


def test_matrix_refuses_changed_configuration_without_launching(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{}")
    plan = tmp_path / "plan.json"
    atomic_json(
        plan,
        {
            "source_sha256": source_hash(),
            "jobs": [
                {
                    "id": "tampered",
                    "config": str(config),
                    "config_sha256": "0" * 64,
                    "output": str(tmp_path / "output"),
                }
            ],
        },
    )
    status = tmp_path / "status.json"
    assert run_matrix(plan, status, workers=1) == 1
    assert "Config changed" in json.loads(status.read_text())["results"]["tampered"]["error"]
    assert not (tmp_path / "output").exists()
