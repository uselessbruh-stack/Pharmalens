import json

import pytest
import torch
from fastapi.testclient import TestClient

from src.inference.model_registry import ModelRegistry
from src.models.deeptda import DeepDTA


def write_artifact(path):
    model = DeepDTA(
        drug_max_len=35,
        protein_max_len=45,
        drug_embed_dim=8,
        protein_embed_dim=8,
        num_filters=4,
        fc_dims=[8],
    )
    torch.save({"config": model.config, "model_state_dict": model.state_dict()}, path)


def test_registry_scopes_defaults_and_uses_validation_not_test(tmp_path):
    path = tmp_path / "model.pt"
    write_artifact(path)
    registry = ModelRegistry(tmp_path / "registry.json")
    for name, endpoint, validation, test in [
        ("ki1", "Ki", 0.8, 0.1),
        ("ki2", "Ki", 0.5, 0.9),
        ("kd", "Kd", 0.01, 0.01),
    ]:
        registry.register(
            name,
            "deeptda",
            path,
            dataset="broad",
            dataset_version="v1",
            endpoint=endpoint,
            metrics={"rmse": test},
            validation_metrics={"rmse": validation},
        )
    assert registry.get_best_model("broad", "Ki").name == "ki2"
    assert registry.get_best_model("broad", "Kd").name == "kd"
    with pytest.raises(ValueError, match="immutable"):
        registry.register("ki1", "deeptda", path)
    path.write_text("modified artifact")
    with pytest.raises(ValueError, match="integrity"):
        registry.load("ki2")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from backend.main import app, load_registered

    path = tmp_path / "model.pt"
    write_artifact(path)
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    for endpoint in ("Ki", "Kd"):
        registry.register(
            endpoint.lower(),
            "deeptda",
            path,
            dataset="broad",
            dataset_version="v1",
            endpoint=endpoint,
            is_best=True,
            validation_metrics={"rmse": 0.5},
        )
    monkeypatch.setenv("PHARMALENS_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("PHARMALENS_JOBS_DB", str(tmp_path / "jobs.sqlite3"))
    load_registered.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    load_registered.cache_clear()


def payload(endpoint="Ki"):
    return {
        "dataset": "broad",
        "endpoint": endpoint,
        "dataset_version": "v1",
        "smiles": "CCO",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
    }


def test_model_selection_is_request_scoped(client):
    ki = client.post("/predict", json=payload()).json()
    kd = client.post("/predict", json=payload("Kd")).json()
    assert ki["model_id"] == "ki" and ki["label"]["label_name"] == "pKi"
    assert kd["model_id"] == "kd" and kd["label"]["label_name"] == "pKd"
    assert client.post("/predict", json={**payload(), "model_id": "kd"}).status_code == 422
    assert client.post("/predict", json={**payload(), "model_id": "missing"}).status_code == 404


def test_batch_and_screen_keep_failures_as_json_null(client):
    scope = {key: value for key, value in payload().items() if key not in {"smiles", "sequence"}}
    pair = {key: payload()[key] for key in ("smiles", "sequence")}
    batch = client.post(
        "/predict/batch", json={**scope, "pairs": [pair, {**pair, "smiles": "invalid"}]}
    ).json()
    assert batch["results"][1]["score"] is None
    assert batch["results"][1]["row_id"] == 1
    screen = client.post(
        "/screen",
        json={**scope, "smiles_list": ["CCO", "invalid"], "sequence_list": [pair["sequence"]]},
    )
    assert screen.status_code == 200
    assert screen.json()["results"][-1]["rank"] is None
    assert (
        client.post("/predict/batch", json={**scope, "pairs": [{"smiles": "CCO"}]}).status_code
        == 422
    )
    assert (
        client.post(
            "/screen",
            json={
                **scope,
                "smiles_list": ["CCO"],
                "sequence_list": [pair["sequence"]],
                "drug_ids": [],
            },
        ).status_code
        == 422
    )


def test_historical_comparison_is_strict_json(client):
    response = client.get("/models/comparison")
    assert response.status_code == 200
    records = response.json()["models"]
    assert records
    json.dumps(records, allow_nan=False)
    assert client.get("/models/list").json()["models"][0]["endpoint"] in {"Ki", "Kd"}


def test_concurrent_endpoint_selection_does_not_cross_contaminate(client):
    from concurrent.futures import ThreadPoolExecutor

    endpoints = ["Ki", "Kd"] * 8
    with ThreadPoolExecutor(max_workers=4) as workers:
        responses = list(
            workers.map(lambda endpoint: client.post("/predict", json=payload(endpoint)), endpoints)
        )
    assert all(response.status_code == 200 for response in responses)
    assert [response.json()["label"]["endpoint"] for response in responses] == endpoints


def test_dataset_catalog_verifies_files_and_reports_corruption(client, tmp_path, monkeypatch):
    from src.data.acquisition import atomic_json, digest_file

    folder = tmp_path / "catalog" / "prepared_v1"
    folder.mkdir(parents=True)
    data = folder / "Ki.csv"
    data.write_text("score\n1\n")
    atomic_json(
        folder / "manifest.json",
        {"counts": {"Ki": 1}, "outputs": {"Ki": {"path": "Ki.csv", "sha256": digest_file(data)}}},
    )
    monkeypatch.setenv("PHARMALENS_PREPARED_DATA_DIR", str(folder.parent))
    assert client.get("/datasets").json()["datasets"][0]["endpoints"][0]["available"]
    data.write_text("score\n99999999\n")
    assert not client.get("/datasets").json()["datasets"][0]["endpoints"][0]["available"]
    atomic_json(
        folder / "manifest.json", {"outputs": {"Ki": {"path": "../../outside.csv", "sha256": "x"}}}
    )
    assert client.get("/datasets").json()["issues"]


def test_experiment_catalog_does_not_publish_partial_checkpoints(client, tmp_path, monkeypatch):
    from src.data.acquisition import atomic_json

    root = tmp_path / "runs"
    partial = root / "unfinished.part"
    partial.mkdir(parents=True)
    atomic_json(partial / "run.json", {"status": "incomplete"})
    monkeypatch.setenv("PHARMALENS_RUNS_DIR", str(root))
    assert client.get("/experiments").json()["runs"] == []


def test_dataset_catalog_lists_nested_views_but_never_staging(client, tmp_path, monkeypatch):
    from src.data.acquisition import atomic_json, digest_file

    root = tmp_path / "datasets"
    for name in ("views_v1/gpcr_ki", "unfinished.part/gpcr_ki"):
        folder = root / name
        folder.mkdir(parents=True)
        data = folder / "Ki.csv"
        data.write_text("score\n1\n")
        atomic_json(
            folder / "manifest.json",
            {
                "counts": {"Ki": 1},
                "outputs": {"Ki": {"path": "Ki.csv", "sha256": digest_file(data)}},
            },
        )
    monkeypatch.setenv("PHARMALENS_PREPARED_DATA_DIR", str(root))
    datasets = client.get("/datasets").json()["datasets"]
    assert [row["id"] for row in datasets] == ["views_v1/gpcr_ki"]


def test_experiment_progress_is_separate_from_completed_evidence(client, tmp_path, monkeypatch):
    from src.data.acquisition import atomic_json

    root = tmp_path / "progress_runs"
    stage = root / "candidate.123.part"
    stage.mkdir(parents=True)
    atomic_json(
        stage / "progress.json",
        {
            "status": "running",
            "epoch": 2,
            "max_epochs": 4,
            "val_loss": 0.3,
            "loss_space": "normalized_target",
        },
    )
    invalid = root / "invalid.123.part"
    invalid.mkdir()
    atomic_json(invalid / "progress.json", [])
    atomic_json(
        root / "matrix_status.json",
        {"total_jobs": 3, "results": {"a": {"status": "completed"}, "b": {"status": "failed"}}},
    )
    monkeypatch.setenv("PHARMALENS_RUNS_DIR", str(root))
    result = client.get("/experiments").json()
    assert result["runs"] == []
    assert result["checkpoints"][0]["id"] == "candidate"
    assert result["checkpoints"][0]["validation_loss"] == 0.3
    assert result["matrices"][0]["completed_jobs"] == 1
    assert result["matrices"][0]["failed_jobs"] == 1


def test_durable_job_keeps_partial_failures_ids_and_frozen_model(client):
    import time

    scope = {key: value for key, value in payload().items() if key not in {"smiles", "sequence"}}
    pair = {key: payload()[key] for key in ("smiles", "sequence")}
    request = {
        **scope,
        "pairs": [pair, {**pair, "sequence": "A"}],
        "ids": ["valid", "short-sequence"],
        "idempotency_key": "job-idempotency-test",
    }
    submitted = client.post("/jobs", json=request)
    assert submitted.status_code == 202
    job_id = submitted.json()["id"]
    assert client.post("/jobs", json=request).json()["id"] == job_id
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.05)
    assert status["status"] == "completed"
    assert status["model_id"] == "ki"
    assert status["completed_rows"] == 2
    assert "payload" not in status and "result" not in status
    result = client.get(f"/jobs/{job_id}/result").json()
    assert result["results"][0]["score"] is not None
    assert result["results"][1]["score"] is None
    assert [row["id"] for row in result["inputs"]] == ["valid", "short-sequence"]
    assert result["artifact_sha256"]
    assert client.post(f"/jobs/{job_id}/retry").status_code == 409
    assert client.get("/jobs").json()["jobs"][0]["id"] == job_id
    assert client.get("/jobs/missing").status_code == 404


def test_test_metrics_require_matching_model_artifact(client, tmp_path, monkeypatch):
    from src.data.acquisition import atomic_json

    root = tmp_path / "evaluated_runs"
    (root / "record").mkdir(parents=True)
    atomic_json(
        root / "record/run.json",
        {
            "run_id": "record",
            "config": {
                "model": "xgboost",
                "model_params": {},
                "dataset": "broad",
                "dataset_version": "v1",
                "endpoint": "Ki",
                "split": "random",
                "seed": 42,
            },
            "status": "validation_complete",
            "completed_at": "2026-10-31T00:00:00Z",
            "validation_metrics": {"rmse": 0.5},
            "training_and_evaluation_seconds": 1,
            "serving_parity": {"verified": True},
            "artifact": {"sha256": "expected"},
        },
    )
    benchmark = tmp_path / "benchmark.json"
    monkeypatch.setenv("PHARMALENS_RUNS_DIR", str(root))
    monkeypatch.setenv("PHARMALENS_BENCHMARK_REPORT", str(benchmark))
    atomic_json(
        benchmark,
        {"runs": {"record": {"artifact_sha256": "different", "test_metrics": {"rmse": 0.1}}}},
    )
    assert not client.get("/experiments").json()["runs"][0]["test_evaluated"]
    atomic_json(
        benchmark,
        {"runs": {"record": {"artifact_sha256": "expected", "test_metrics": {"rmse": 0.6}}}},
    )
    run = client.get("/experiments").json()["runs"][0]
    assert run["test_evaluated"] and run["test_metrics"]["rmse"] == 0.6
