import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import concordance_index, concordance_index_exact
from src.training.experiment import load_config, read_partitions, run_experiment


def test_registration_checks_completed_run_and_artifact_integrity(tmp_path):
    from scripts.register_experiment import register_run
    from src.inference.model_registry import ModelRegistry

    config = experiment_config(tmp_path)
    config["smoke"] = False
    output = tmp_path / "completed"
    report = run_experiment(config, output)
    registry_path = tmp_path / "registry.json"
    register_run(output, "verified-model", registry_path, default=True)
    entry = ModelRegistry(registry_path).entries["verified-model"]
    assert entry.validation_metrics == report["validation_metrics"]
    assert entry.is_best and "Provisional" in entry.description
    artifact = output / report["artifact"]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        register_run(output, "tampered-model", registry_path)
    assert "tampered-model" not in ModelRegistry(registry_path).entries


@pytest.mark.parametrize("size", [3, 7, 20, 50])
def test_exact_scalable_concordance_matches_pair_enumeration(size):
    rng = np.random.default_rng(size)
    for _ in range(10):
        truth, pred = rng.integers(0, 4, size), rng.integers(0, 4, size)
        if np.ptp(truth):
            assert concordance_index_exact(truth, pred) == pytest.approx(
                concordance_index(truth, pred)
            )
    assert np.isnan(concordance_index_exact(np.ones(size), np.ones(size)))


def experiment_config(tmp_path, model="xgboost"):
    use_gine = model == "gine"
    model = "gnn" if use_gine else model
    rows = []
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    for drug in range(1, 11):
        for target in range(8):
            rows.append(
                {
                    "drug_id": f"d{drug}",
                    "target_id": f"t{target}",
                    "smiles": "C" * drug,
                    "sequence": alphabet[target:] + alphabet[:target],
                    "score": 3 + drug / 10 + target / 20,
                    "endpoint": "Ki",
                }
            )
    data = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data, index=False)
    params = {
        "xgboost": {"n_estimators": 15, "max_depth": 2, "early_stopping_rounds": 3, "n_jobs": 1},
        "deeptda": {
            "drug_max_len": 20,
            "protein_max_len": 30,
            "drug_embed_dim": 8,
            "protein_embed_dim": 8,
            "num_filters": 4,
            "fc_dims": [8],
        },
        "gnn": {
            "gnn_hidden_dim": 8,
            "gnn_output_dim": 8,
            "protein_max_len": 30,
            "protein_embed_dim": 8,
            "protein_num_filters": 4,
            "fc_dims": [8],
        },
    }[model]
    if use_gine:
        params.update(gnn_type="gine", mask_padding=True)
    if model == "deeptda":
        params["mask_padding"] = True
    config = {
        "dataset": "synthetic",
        "dataset_version": "test",
        "endpoint": "Ki",
        "input": str(data),
        "preprocessing_version": "standardized_v1",
        "model": model,
        "model_params": params,
        "training": {"threads": 1}
        if model == "xgboost"
        else {"epochs": 1, "batch_size": 15, "threads": 1},
        "seed": 42,
        "split_seed": 10,
        "smoke": True,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return load_config(path)


@pytest.mark.parametrize("model", ["xgboost", "deeptda", "gnn", "gine"])
def test_experiment_trains_reloads_and_preserves_observation_ids(tmp_path, model):
    config = experiment_config(tmp_path, model)
    output = tmp_path / "run"
    report = run_experiment(config, output)
    assert report["status"] == "smoke_complete"
    assert report["serving_parity"]["verified"]
    assert report["test_metrics"] is None
    assert not (output / "test_predictions.csv").exists()
    assert (
        len(pd.read_csv(output / "validation_predictions.csv"))
        == report["split_membership"]["val"]["rows"]
    )
    assert report["validation_controls"]["targets"] > 0
    assert (output / "source_snapshot.zip").is_file()
    if model != "xgboost":
        training_rows = read_partitions(config)[1][0]
        assert report["target_scaler"]["mean"] == pytest.approx(training_rows.score.mean())
        assert report["target_scaler"]["scale"] == pytest.approx(training_rows.score.std(ddof=0))
    with pytest.raises(ValueError, match="exists"):
        run_experiment(config, output)


def test_model_seeds_do_not_change_data_partitions(tmp_path):
    config = experiment_config(tmp_path)
    parts_a = read_partitions(config)[1]
    parts_b = read_partitions({**config, "seed": 999})[1]
    assert parts_a[2].observation_id.tolist() == parts_b[2].observation_id.tolist()
    changed = pd.read_csv(config["input"])
    changed.loc[0, "endpoint"] = "Kd"
    changed.to_csv(config["input"], index=False)
    with pytest.raises(ValueError, match="endpoint"):
        read_partitions(config)


@pytest.mark.parametrize("model", ["deeptda", "gine"])
def test_hybrid_features_use_training_statistics_and_reload(tmp_path, model):
    import torch
    from src.training.experiment import tabular_features

    config = experiment_config(tmp_path, model)
    config["model_params"]["auxiliary_dim"] = 1454
    training = read_partitions(config)[1][0]
    output = tmp_path / "hybrid"
    report = run_experiment(config, output)
    checkpoint = torch.load(output / report["artifact"]["path"], weights_only=True)
    expected = tabular_features([training])[0].mean(axis=0)
    assert np.allclose(
        checkpoint["model_state_dict"]["auxiliary_encoder.mean"].numpy(), expected, atol=1e-5
    )
    assert report["serving_parity"]["verified"]


def test_feature_cache_ignores_labels_but_preserves_rows_and_rejects_corruption(
    tmp_path, monkeypatch
):
    from unittest.mock import patch
    from src.training.experiment import tabular_features

    monkeypatch.setenv("PHARMALENS_FEATURE_CACHE", str(tmp_path / "features"))
    config = experiment_config(tmp_path)
    parts = read_partitions(config)[1]
    expected = tabular_features(parts)
    changed_labels = [part.assign(score=part.score + 100) for part in parts]
    with patch(
        "src.training.experiment._uncached_tabular_features",
        side_effect=AssertionError("must build"),
    ):
        actual = tabular_features(changed_labels)
        assert all(np.array_equal(a, b) for a, b in zip(expected, actual))
        with pytest.raises(AssertionError, match="must build"):
            tabular_features([part.iloc[::-1] for part in parts])
    cache = next((tmp_path / "features").glob("*/features.npz"))
    cache.write_bytes(b"broken")
    with pytest.raises(ValueError, match="integrity"):
        tabular_features(parts)


def test_training_settings_are_strict_and_smoke_is_explicit(tmp_path):
    config = experiment_config(tmp_path)
    config.update(training={"epochz": 1})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="settings"):
        load_config(path)
    config.update(training={}, smoke=False, max_rows=5)
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="smoke"):
        load_config(path)
