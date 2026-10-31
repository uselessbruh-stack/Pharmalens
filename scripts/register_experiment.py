"""Register a completed non-smoke experiment as a provisional research model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import digest_file
from src.inference.model_registry import ModelRegistry


def register_run(run_dir, name, registry_path=None, default=False):
    root = Path(run_dir).resolve()
    report = json.loads((root / "run.json").read_text(encoding="utf-8"))
    config = report["config"]
    if root.name.endswith(".part") or config.get("smoke"):
        raise ValueError("Only completed non-smoke experiments can be registered")
    if report["status"] != "validation_complete" or not report["serving_parity"]["verified"]:
        raise ValueError("Completed validation and verified serving parity are required")
    artifact = (root / report["artifact"]["path"]).resolve()
    if not artifact.is_relative_to(root) or digest_file(artifact) != report["artifact"]["sha256"]:
        raise ValueError("Artifact location or checksum does not match the experiment")
    ModelRegistry(registry_path).register(
        name,
        config["model"],
        str(artifact),
        description=f"Provisional validation model from {report['run_id']}; final research selection pending",
        dataset=config["dataset"],
        dataset_version=config["dataset_version"],
        endpoint=config["endpoint"],
        split=config["split"],
        preprocessing_version=config["preprocessing_version"],
        hyperparameters=report["effective_model_config"],
        validation_metrics=report["validation_metrics"],
        is_best=default,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    register_run(args.run, args.name, args.registry, args.default)
    print(f"Registered {args.name}; provisional default={args.default}")


if __name__ == "__main__":
    main()
