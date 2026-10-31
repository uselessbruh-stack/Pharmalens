"""Live HTTP integration check using actual validation-run model artifacts.

Creates an isolated QA registry and a hidden local test server; never promotes
the research registry or leaves the server running after verification.
"""

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.acquisition import ROOT, atomic_json, digest_file
from src.inference.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new QA output directory")
    args.output.mkdir(parents=True)
    registry_path = (args.output / "registry.json").resolve()
    registry = ModelRegistry(registry_path)
    probes = []
    for index, run in enumerate(args.runs):
        report = json.loads((run / "run.json").read_text(encoding="utf-8"))
        config = report["config"]
        artifact = run / report["artifact"]["path"]
        if digest_file(artifact) != report["artifact"]["sha256"]:
            raise ValueError("Run artifact checksum failed")
        name = f"qa_{index}_{run.name}"
        registry.register(
            name,
            config["model"],
            artifact,
            dataset=config["dataset"],
            endpoint=config["endpoint"],
            dataset_version=config["dataset_version"],
            preprocessing_version=config["preprocessing_version"],
            validation_metrics=report["validation_metrics"],
            is_best=True,
            description="Isolated HTTP QA candidate; not a research promotion",
        )
        input_path = Path(config["input"])
        input_path = input_path if input_path.is_absolute() else ROOT / input_path
        frame = pd.read_csv(input_path, keep_default_na=False)
        validation = pd.read_csv(run / "validation_predictions.csv", keep_default_na=False).iloc[0]
        row = frame[
            (frame.drug_id == validation.drug_id) & (frame.target_id == validation.target_id)
        ].iloc[0]
        probes.append(
            {
                "model_id": name,
                "dataset": config["dataset"],
                "dataset_version": config["dataset_version"],
                "endpoint": config["endpoint"],
                "smiles": row.smiles,
                "sequence": row.sequence,
                "expected": float(validation.prediction),
            }
        )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    environment = {
        **os.environ,
        "PHARMALENS_REGISTRY_PATH": str(registry_path),
        "PHARMALENS_JOBS_DB": str((args.output / "jobs.sqlite3").resolve()),
    }
    with (args.output / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        base = f"http://127.0.0.1:{port}"

        def request(path, payload=None):
            started = time.perf_counter()
            data = json.dumps(payload).encode() if payload is not None else None
            req = Request(base + path, data=data, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=30) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
                value = (
                    json.loads(
                        body, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
                    )
                    if "application/json" in content_type
                    else body.decode()
                )
                return value, time.perf_counter() - started

        try:
            deadline = time.monotonic() + 25
            while True:
                try:
                    health, _ = request("/health")
                    break
                except URLError:
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("Test server did not become ready; inspect server.log")
                    time.sleep(0.1)
            checks = {"health": health, "static_files": {}}
            for path in ("/", "/app.js", "/style.css"):
                body, _ = request(path)
                checks["static_files"][path] = {"nonempty": bool(body)}
            for path in ("/models/list", "/models/comparison", "/openapi.json"):
                request(path)
            for probe in probes:
                payload = {key: value for key, value in probe.items() if key != "expected"}
                prediction, _ = request("/predict", payload)
                assert prediction["model_id"] == probe["model_id"]
                assert np.isclose(prediction["score"], probe["expected"], atol=1e-5, rtol=1e-5)
                scope = {
                    key: probe[key]
                    for key in ("model_id", "dataset", "dataset_version", "endpoint")
                }
                pair = {key: probe[key] for key in ("smiles", "sequence")}
                batch, _ = request(
                    "/predict/batch", {**scope, "pairs": [pair, {**pair, "smiles": "invalid"}]}
                )
                assert (
                    batch["results"][0]["score"] is not None
                    and batch["results"][1]["score"] is None
                )
                screen, _ = request(
                    "/screen",
                    {
                        **scope,
                        "smiles_list": [probe["smiles"], "invalid"],
                        "sequence_list": [probe["sequence"]],
                    },
                )
                assert screen["results"][-1]["rank"] is None
            workloads = [probes[index % len(probes)] for index in range(16)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                replies = list(
                    pool.map(
                        lambda probe: request(
                            "/predict",
                            {key: value for key, value in probe.items() if key != "expected"},
                        ),
                        workloads,
                    )
                )
            for probe, (reply, _) in zip(workloads, replies):
                assert reply["model_id"] == probe["model_id"]
                assert np.isclose(reply["score"], probe["expected"], atol=1e-5, rtol=1e-5)
            checks.update(
                {
                    "status": "passed",
                    "actual_artifacts": len(probes),
                    "concurrent_requests": 16,
                    "concurrent_workers": 4,
                    "http_request_latency_seconds_p50": float(
                        np.percentile([reply[1] for reply in replies], 50)
                    ),
                    "http_request_latency_seconds_p95": float(
                        np.percentile([reply[1] for reply in replies], 95)
                    ),
                    "prediction_parity": True,
                    "partial_failure_json": True,
                    "production_registry_modified": False,
                    "browser_verified": False,
                    "limitations": "Local concurrent smoke test, not a capacity benchmark or browser-rendering test",
                }
            )
            atomic_json(args.output / "report.json", checks)
            print(json.dumps(checks, indent=2))
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
