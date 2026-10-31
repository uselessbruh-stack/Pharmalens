"""Register a real artifact with explicit validation evidence and label scope."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--type", required=True, choices=["xgboost", "deeptda", "gnn"])
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--endpoint", required=True, choices=["KIBA", "Ki", "Kd", "IC50"])
    parser.add_argument("--validation-metrics", required=True, type=Path)
    parser.add_argument("--preprocessing-version", required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    registry = ModelRegistry(args.registry)
    registry.register(
        args.name,
        args.type,
        args.artifact,
        dataset=args.dataset,
        endpoint=args.endpoint,
        dataset_version=args.dataset_version,
        preprocessing_version=args.preprocessing_version,
        validation_metrics=json.loads(args.validation_metrics.read_text(encoding="utf-8")),
        is_best=args.default,
    )
    print(f"Registered {args.name}; checksum recorded; default={args.default}")


if __name__ == "__main__":
    main()
