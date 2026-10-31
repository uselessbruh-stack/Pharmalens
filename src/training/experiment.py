"""Reproducible CLI experiments with identity-safe splits and honest run status."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import random
import subprocess
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.acquisition import ROOT, atomic_json, digest_file, utc_now, replace_with_retry
from src.data.labels import CONTRACTS
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.split_strategies import get_split, validate_split_integrity
from src.inference.pipeline import PharmaLensPipeline
from src.utils.serialization import json_safe

LOG = logging.getLogger("pharmalens.experiment")
ALLOWED = {
    "dataset",
    "dataset_version",
    "endpoint",
    "input",
    "preprocessing_version",
    "model",
    "model_params",
    "training",
    "split",
    "split_seed",
    "seed",
    "split_manifest",
    "smoke",
    "max_rows",
    "protein_embedding_cache",
}


def source_hash():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(config) - ALLOWED
    if unknown:
        raise ValueError(f"Unknown experiment fields: {sorted(unknown)}")
    for field in (
        "dataset",
        "dataset_version",
        "endpoint",
        "input",
        "model",
        "preprocessing_version",
    ):
        if not config.get(field):
            raise ValueError(f"Missing experiment field: {field}")
    if config["model"] not in {"xgboost", "deeptda", "gnn"}:
        raise ValueError("Unsupported experiment model")
    if config["endpoint"] not in CONTRACTS:
        raise ValueError("Unsupported endpoint")
    if config["preprocessing_version"] not in {"legacy_v1", "standardized_v1"}:
        raise ValueError("Unsupported preprocessing contract")
    if config.get("max_rows") and not config.get("smoke"):
        raise ValueError("Row-limited experiments must be marked smoke=true")
    config.setdefault("split", "random")
    config.setdefault("split_seed", 42)
    config.setdefault("seed", 42)
    config.setdefault("smoke", False)
    config.setdefault("training", {})
    config.setdefault("model_params", {})
    if not isinstance(config["smoke"], bool):
        raise ValueError("smoke must be a JSON boolean")
    if config.get("max_rows") is not None and (
        not isinstance(config["max_rows"], int) or config["max_rows"] < 3
    ):
        raise ValueError("max_rows must be an integer of at least three")
    allowed_training = (
        {"threads"}
        if config["model"] == "xgboost"
        else {
            "threads",
            "epochs",
            "patience",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "normalize_targets",
            "device",
        }
    )
    if set(config["training"]) - allowed_training:
        raise ValueError("Unknown or inapplicable training settings")
    if config["training"].get("device", "cpu") not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Training device must be cpu, cuda or xpu")
    if (
        config.get("protein_embedding_cache")
        and config["model"] != "xgboost"
        and config["model_params"].get("auxiliary_dim") != 1774
    ):
        raise ValueError("ESM hybrid models require 1774 auxiliary features")
    for field in ("threads", "epochs", "patience", "batch_size"):
        if field in config["training"] and (
            not isinstance(config["training"][field], int) or config["training"][field] < 1
        ):
            raise ValueError(f"{field} must be a positive integer")
    if "normalize_targets" in config["training"] and not isinstance(
        config["training"]["normalize_targets"], bool
    ):
        raise ValueError("normalize_targets must be a JSON boolean")
    if "random_state" in config["model_params"]:
        raise ValueError("Use the top-level seed rather than model_params.random_state")
    return config


def read_partitions(config):
    path = resolve_path(config["input"])
    fingerprint = digest_file(path)
    source_manifest = path.parent / "manifest.json"
    if source_manifest.exists():
        source_record = json.loads(source_manifest.read_text(encoding="utf-8"))
        output_spec = source_record.get("outputs", {}).get(config["endpoint"])
        if output_spec and (
            output_spec.get("path") != path.name or output_spec.get("sha256") != fingerprint
        ):
            raise ValueError("Prepared dataset checksum does not match its declared version")
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"smiles", "sequence", "score", "drug_id", "target_id"}
    if not required <= set(frame):
        raise ValueError(f"Dataset lacks required fields: {sorted(required - set(frame))}")
    if frame.empty or not np.isfinite(pd.to_numeric(frame["score"], errors="raise")).all():
        raise ValueError("Dataset must have nonempty finite labels")
    if "endpoint" in frame and set(frame["endpoint"]) != {config["endpoint"]}:
        raise ValueError("Dataset endpoint does not match experiment contract")
    if "observation_id" not in frame:
        frame["observation_id"] = [
            hashlib.sha256(f"{fingerprint}:{i}".encode()).hexdigest() for i in range(len(frame))
        ]
    if frame["observation_id"].duplicated().any():
        raise ValueError("Observation IDs must be unique; resolve duplicate source records first")
    for column, identity in [("drug_id", "smiles"), ("target_id", "sequence")]:
        if (frame.groupby(column)[identity].nunique() > 1).any():
            raise ValueError(f"A {column} maps to incompatible identities")
    if config["preprocessing_version"] == "standardized_v1":
        from src.data.preparation import AA, clean_molecule

        for smiles in frame.smiles.unique():
            if clean_molecule(smiles) != smiles:
                raise ValueError(
                    "Dataset molecules do not match standardized_v1; prepare the input first"
                )
        for sequence in frame.sequence.unique():
            if not 10 <= len(sequence) <= 10000 or not set(sequence) <= AA:
                raise ValueError("Dataset proteins do not match standardized_v1")
    if config.get("max_rows"):
        frame = frame.sample(
            n=min(int(config["max_rows"]), len(frame)), random_state=config["split_seed"]
        ).sort_index()
    if config.get("split_manifest"):
        if config.get("max_rows"):
            raise ValueError("A row-limited smoke run cannot reuse a full-data split manifest")
        manifest_path = resolve_path(config["split_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("input_sha256") != fingerprint:
            raise ValueError("Split manifest does not match the dataset bytes")
        if manifest.get("seed") != config["split_seed"]:
            raise ValueError("Split seed does not match the frozen manifest")
        spec = manifest["splits"][config["split"]]["files"]
        parts = []
        # Existing KIBA split files do not contain observation IDs. Match exact
        # source rows, including aliases and label, through deterministic queues.
        from collections import defaultdict, deque

        original = defaultdict(deque)
        keys = ["drug_id", "target_id", "smiles", "sequence", "score"]
        for i, values in enumerate(frame[keys].itertuples(index=False, name=None)):
            original[values].append(i)
        for part in ("train", "val", "test"):
            file = manifest_path.parent / spec[part]["path"]
            if digest_file(file) != spec[part]["sha256"]:
                raise ValueError(f"Frozen split checksum failed: {part}")
            data = pd.read_csv(file, keep_default_na=False)
            indices = []
            for values in data[keys].itertuples(index=False, name=None):
                if not original[values]:
                    raise ValueError("Split includes an unknown or repeated source row")
                indices.append(original[values].popleft())
            parts.append(frame.iloc[indices].copy().reset_index(drop=True))
    else:
        parts = list(get_split(frame, config["split"], seed=config["split_seed"]))
    validate_split_integrity(*parts, config["split"])
    return frame, parts, fingerprint


def tabular_features(parts, protein_store=None):
    """Reuse immutable label-free matrices across matched model experiments."""
    import os
    from rdkit import __version__ as rdkit_version
    from src.features.drug_features import get_feature_names
    from src.features.protein_features import get_protein_feature_names

    signature = hashlib.sha256()
    signature.update(
        json.dumps(
            [np.__version__, rdkit_version, protein_store.contract if protein_store else None],
            sort_keys=True,
        ).encode()
    )
    for source in [
        Path(__file__),
        ROOT / "src/features/drug_features.py",
        ROOT / "src/features/protein_features.py",
        ROOT / "src/features/protein_embeddings.py",
    ]:
        signature.update(source.read_bytes())
    for part in parts:
        signature.update(f"PART:{len(part)}\0".encode())
        for smiles, sequence in part[["smiles", "sequence"]].itertuples(index=False, name=None):
            signature.update(smiles.encode() + b"\0" + sequence.encode() + b"\0")
    key = signature.hexdigest()
    directory = (
        Path(os.getenv("PHARMALENS_FEATURE_CACHE", ROOT / "data/processed/features/tabular_v1"))
        / key
    )
    expected_shapes = [
        (
            len(part),
            len(get_feature_names())
            + len(get_protein_feature_names())
            + (protein_store.dimension if protein_store else 0),
        )
        for part in parts
    ]

    def read_cache():
        metadata = json.loads((directory / "manifest.json").read_text())
        data = directory / "features.npz"
        if metadata["signature"] != key or digest_file(data) != metadata["sha256"]:
            raise ValueError("Tabular feature cache integrity check failed")
        with np.load(data, allow_pickle=False) as archive:
            matrices = [archive[f"part{index}"] for index in range(len(parts))]
        if any(
            matrix.shape != shape or matrix.dtype != np.float32 or not np.isfinite(matrix).all()
            for matrix, shape in zip(matrices, expected_shapes)
        ):
            raise ValueError("Tabular feature cache schema mismatch")
        return matrices

    if directory.exists():
        LOG.info("Reusing verified tabular features %s", key[:12])
        return read_cache()
    matrices = _uncached_tabular_features(parts, protein_store)
    staging = directory.with_name(directory.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    np.savez_compressed(
        staging / "features.npz",
        **{f"part{index}": matrix for index, matrix in enumerate(matrices)},
    )
    atomic_json(
        staging / "manifest.json",
        {
            "signature": key,
            "sha256": digest_file(staging / "features.npz"),
            "shapes": expected_shapes,
            "affinity_labels_used": False,
            "normalization_fitted": False,
        },
    )
    try:
        staging.replace(directory)
    except OSError:
        if not directory.exists():
            raise
        # Another process may have published the same immutable feature contract.
        # Keep diagnostic staging; never overwrite the winning cache directory.
        metadata = json.loads((directory / "manifest.json").read_text())
        if (
            metadata["signature"] != key
            or digest_file(directory / "features.npz") != metadata["sha256"]
        ):
            raise ValueError("Concurrent tabular feature cache publication failed integrity")
    return matrices


def _uncached_tabular_features(parts, protein_store=None):
    from src.features.drug_features import get_drug_feature_vector
    from src.features.protein_features import get_protein_feature_vector

    joined = pd.concat(parts, ignore_index=True)
    drugs = {smiles: get_drug_feature_vector(smiles) for smiles in joined.smiles.unique()}
    proteins = {
        sequence: np.concatenate(
            [get_protein_feature_vector(sequence), protein_store.get(sequence)]
        )
        if protein_store
        else get_protein_feature_vector(sequence)
        for sequence in joined.sequence.unique()
    }
    if any(
        value is None or not np.isfinite(value).all()
        for value in [*drugs.values(), *proteins.values()]
    ):
        raise ValueError("Feature extraction failed; no observations are silently skipped")
    matrices = [
        np.stack(
            [
                np.concatenate([drugs[s], proteins[p]])
                for s, p in part[["smiles", "sequence"]].itertuples(index=False, name=None)
            ]
        ).astype(np.float32)
        for part in parts
    ]
    return matrices


def validation_controls(train, validation, prediction):
    """Training-only shortcut baselines and explicit per-target aggregation."""
    global_mean = float(train.score.mean())
    shortcuts = {}
    for column, label in [("smiles", "drug_mean"), ("sequence", "target_mean")]:
        means = train.groupby(column).score.mean()
        mapped = validation[column].map(means)
        shortcuts[label] = {
            "metrics": compute_regression_metrics(
                validation.score.to_numpy(), mapped.fillna(global_mean).to_numpy()
            ),
            "unknown_identity_fraction": float(mapped.isna().mean()),
        }
    evaluated = validation.copy()
    evaluated["prediction"] = prediction
    target_metrics = []
    for target, rows in evaluated.groupby("target_id"):
        target_metrics.append(
            {
                "target_id": target,
                "rows": len(rows),
                **compute_regression_metrics(rows.score.to_numpy(), rows.prediction.to_numpy()),
            }
        )
    return {
        "shortcuts": shortcuts,
        "per_family": [
            {
                "family": family,
                "rows": len(rows),
                **compute_regression_metrics(rows.score.to_numpy(), rows.prediction.to_numpy()),
            }
            for family, rows in evaluated.groupby("family")
        ]
        if "family" in evaluated
        else [],
        "per_target": target_metrics,
        "macro_mean_target_rmse": float(np.mean([r["rmse"] for r in target_metrics])),
        "targets": len(target_metrics),
    }


def neural_loaders(parts, config, protein_store=None, auxiliary_features=None):
    import torch
    from torch.utils.data import DataLoader

    joined = pd.concat(parts, ignore_index=True)
    params = config["model_params"]
    auxiliary = (
        auxiliary_features
        if auxiliary_features is not None
        else (
            tabular_features(parts, protein_store)
            if params.get("auxiliary_dim", 0)
            else [None] * len(parts)
        )
    )
    if auxiliary[0] is not None and auxiliary[0].shape[1] != params["auxiliary_dim"]:
        raise ValueError("auxiliary_dim does not match the feature contract")
    if config["model"] == "deeptda":
        from src.models.deeptda_dataset import DTIDataset, create_encoding_dicts

        dl, pl = params.get("drug_max_len", 100), params.get("protein_max_len", 1000)
        drugs, proteins = create_encoding_dicts(joined, drug_max_len=dl, target_max_len=pl)
        datasets = [
            DTIDataset(
                part.drug_id.tolist(),
                part.target_id.tolist(),
                part.score.tolist(),
                drugs,
                proteins,
                dl,
                pl,
                auxiliary=features,
            )
            for part, features in zip(parts, auxiliary)
        ]
    else:
        from torch_geometric.loader import DataLoader

        from src.models.gnn_dataset import CachedGraphPairDataset

        graph_cache, protein_cache = {}, {}
        datasets = [
            CachedGraphPairDataset(
                part,
                params.get("protein_max_len", 1000),
                graph_cache,
                protein_cache,
                auxiliary=features,
            )
            for part, features in zip(parts, auxiliary)
        ]
    batch_size = config["training"].get("batch_size", 64)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    generator = torch.Generator().manual_seed(config["seed"])
    loaders = [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=index == 0,
            num_workers=0,
            generator=generator if index == 0 else None,
        )
        for index, dataset in enumerate(datasets)
    ]
    for part, dataset in zip(parts, datasets):
        if len(part) != len(dataset):
            raise ValueError(
                "Model dataset omitted observations; all models must use identical retained rows"
            )
    return loaders


def run_experiment(config, output, evaluate_test=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Run directory exists; choose a new immutable run ID")
    frame, parts, data_sha = read_partitions(config)
    protein_store = None
    if config.get("protein_embedding_cache"):
        from src.features.protein_embeddings import load_store

        cache = config["protein_embedding_cache"]
        protein_store = load_store(cache if isinstance(cache, dict) else resolve_path(cache))
    code_sha = source_hash()
    staging = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    with zipfile.ZipFile(
        staging / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as snapshot:
        for source in sorted((ROOT / "src").rglob("*.py")):
            snapshot.write(source, source.relative_to(ROOT).as_posix())
    run_id = output.name
    try:
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(int(config["training"].get("threads", 4)))
        torch.use_deterministic_algorithms(True)
        device_name = config["training"].get("device", "cpu")
        if device_name == "xpu" and not torch.xpu.is_available():
            raise ValueError(
                "Requested XPU device is unavailable; select cpu or install the validated XPU environment"
            )
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("Requested CUDA device is unavailable")
        split_manifest = {}
        for name, part in zip(("train", "val", "test"), parts):
            membership = part[["observation_id", "drug_id", "target_id"]]
            membership.to_csv(staging / f"{name}_membership.csv", index=False)
            split_manifest[name] = {
                "rows": len(part),
                "sha256": digest_file(staging / f"{name}_membership.csv"),
            }
        started = time.perf_counter()
        target_scaler = {"mean": 0.0, "scale": 1.0}
        if config["model"] == "xgboost":
            from src.models.xgboost_model import XGBoostDTI

            params = {
                "n_jobs": 4,
                **config["model_params"],
                "random_state": seed,
                "protein_embedding_cache": protein_store.contract if protein_store else None,
            }
            model = XGBoostDTI(**params)
            LOG.info("Extracting tabular features for %d observations", len(frame))
            matrices = tabular_features(parts, protein_store)
            LOG.info(
                "Training XGBoost on %d rows and %d features", len(parts[0]), matrices[0].shape[1]
            )
            model.fit(
                matrices[0],
                parts[0].score.to_numpy(),
                matrices[1],
                parts[1].score.to_numpy(),
                verbose=False,
            )
            artifact = staging / "model.json"
            model.save(artifact)
            validation_prediction = model.predict(matrices[1])
            test_prediction = model.predict(matrices[2]) if evaluate_test else None
            history = {"best_iteration": model.best_iteration}
        else:
            from src.models.deeptda import DeepDTA
            from src.models.gnn_model import GraphDTA
            from src.training.trainer import Trainer

            model = (DeepDTA if config["model"] == "deeptda" else GraphDTA)(
                **{
                    **config["model_params"],
                    "protein_embedding_cache": protein_store.contract if protein_store else None,
                }
            )
            auxiliary = (
                tabular_features(parts, protein_store)
                if model.auxiliary_encoder is not None
                else None
            )
            if model.auxiliary_encoder is not None:
                # Fit input statistics on training identities only, never validation/test.
                model.auxiliary_encoder.fit_normalization(auxiliary[0])
            settings = config["training"]
            if settings.get("normalize_targets", True):
                target_scaler = {
                    "mean": float(parts[0].score.mean()),
                    "scale": max(float(parts[0].score.std(ddof=0)), 1e-8),
                }
            normalized_parts = [
                part.assign(score=(part.score - target_scaler["mean"]) / target_scaler["scale"])
                for part in parts
            ]
            train_loader, val_loader, test_loader = neural_loaders(
                normalized_parts, config, protein_store, auxiliary
            )
            # Move parameters before constructing the optimizer so its references
            # and eventual state tensors belong to the requested backend.
            model.to(torch.device(device_name))
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=settings.get("learning_rate", 0.001),
                weight_decay=settings.get("weight_decay", 0.0001),
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            trainer = Trainer(
                model,
                optimizer,
                scheduler=scheduler,
                early_stopping_patience=settings.get("patience", 15),
                checkpoint_dir=staging,
                model_name="model",
                target_scaler=target_scaler,
                device=torch.device(device_name),
            )
            history = trainer.fit(
                train_loader, val_loader, epochs=settings.get("epochs", 100), verbose=False
            )
            validation_prediction = (
                trainer.validate(val_loader)[1] * target_scaler["scale"] + target_scaler["mean"]
            )
            test_prediction = (
                trainer.validate(test_loader)[1] * target_scaler["scale"] + target_scaler["mean"]
                if evaluate_test
                else None
            )
            artifact = staging / "model_best.pt"
        training_seconds = time.perf_counter() - started
        validation_metrics = compute_regression_metrics(
            parts[1].score.to_numpy(), validation_prediction
        )
        test_metrics = (
            compute_regression_metrics(parts[2].score.to_numpy(), test_prediction)
            if evaluate_test
            else None
        )
        for label, part, predictions in [
            ("validation", parts[1], validation_prediction),
            ("test", parts[2], test_prediction),
        ]:
            if predictions is not None:
                exported = part[["observation_id", "drug_id", "target_id", "score"]].copy()
                exported["prediction"] = predictions
                exported.to_csv(staging / f"{label}_predictions.csv", index=False)
        # Verify the serialized artifact through the exact pipeline used by API.
        loaded = PharmaLensPipeline.from_pretrained(
            artifact, config["model"], config["endpoint"], config["preprocessing_version"]
        )
        first = parts[1].iloc[0]
        served = loaded.predict(first.smiles, first.sequence)["score"]
        if not np.isclose(served, validation_prediction[0], atol=1e-5, rtol=1e-5):
            raise ValueError("Reloaded serving prediction differs from offline validation")
        mean = float(parts[0].score.mean())
        baseline = compute_regression_metrics(
            parts[1].score.to_numpy(), np.full(len(parts[1]), mean)
        )
        try:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        report = {
            "run_id": run_id,
            "status": "smoke_complete" if config["smoke"] else "validation_complete",
            "config": config,
            "completed_at": utc_now(),
            "data_sha256": data_sha,
            "source_sha256": code_sha,
            "source_snapshot_sha256": digest_file(staging / "source_snapshot.zip"),
            "git_revision": revision,
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "rdkit",
                    "xgboost",
                    "torch",
                    "torch-geometric",
                    "fair-esm",
                ]
            },
            "split_membership": split_manifest,
            "input_rows": len(frame),
            "protein_feature_contract": protein_store.contract if protein_store else None,
            "execution_device": device_name,
            "excluded_cross_block_rows": len(frame) - sum(map(len, parts)),
            "validation_metrics": validation_metrics,
            "validation_controls": validation_controls(parts[0], parts[1], validation_prediction),
            "mean_baseline_validation": baseline,
            "test_metrics": test_metrics,
            "test_evaluation_requested": evaluate_test,
            "test_used_for_model_selection": False,
            "training_and_evaluation_seconds": training_seconds,
            "artifact": {"path": artifact.name, "sha256": digest_file(artifact)},
            "effective_model_config": model.get_hyperparams()
            if config["model"] == "xgboost"
            else model.config,
            "serving_parity": {"verified": True, "first_validation_score": served},
            "truncation": {
                "drug_rows": int(
                    (frame.smiles.str.len() > config["model_params"].get("drug_max_len", 100)).sum()
                )
                if config["model"] == "deeptda"
                else 0,
                "protein_rows": int(
                    (
                        frame.sequence.str.len()
                        > config["model_params"].get("protein_max_len", 1000)
                    ).sum()
                )
                if config["model"] != "xgboost"
                and config["model_params"].get("use_protein_cnn", True)
                else 0,
            },
            "research_status": "integration_smoke_only"
            if config["smoke"]
            else "single_run_baseline_requires_comparative_review",
            "default_promoted": False,
            "target_scaler": target_scaler,
            "hit_iteration_budget": bool(model.best_iteration + 1 >= model.n_estimators)
            if config["model"] == "xgboost"
            else bool(len(history["epoch"]) >= settings.get("epochs", 100)),
            "limitations": [
                "One model run is not evidence of architecture superiority",
                "Source/assay and external-domain validation remain required",
            ],
        }
        if source_hash() != code_sha:
            raise ValueError(
                "Source code changed during training; rerun to obtain consistent provenance"
            )
        atomic_json(staging / "run.json", json_safe(report))
        atomic_json(staging / "validation_metrics.json", json_safe(validation_metrics))
        atomic_json(staging / "history.json", json_safe(history))
        replace_with_retry(staging, output)
        LOG.info(
            "Completed %s: validation RMSE %.5f; no default promoted",
            run_id,
            validation_metrics["rmse"],
        )
        return report
    except Exception as exc:
        atomic_json(
            staging / "failure.json",
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "config": config,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Explicit final held-out evaluation; do not use for tuning",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        config = load_config(args.config)
        if args.seed is not None:
            config["seed"] = args.seed
        run_experiment(config, args.output, args.evaluate_test)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")
    return 0
