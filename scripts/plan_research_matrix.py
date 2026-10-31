"""Freeze coverage-driven Ki/Kd model comparisons before examining results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file
from src.training.experiment import source_hash


def build_plan(
    output,
    scope="all",
    models=("xgboost", "deeptda", "gnn"),
    seeds=(42,),
    splits=("random",),
    protein_cache=None,
    device="cpu",
):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new immutable plan directory")
    output.mkdir(parents=True)
    feature_contract = None
    if protein_cache:
        from src.features.protein_embeddings import load_store

        feature_contract = load_store(protein_cache).contract
    tasks = []
    if scope in {"all", "broad"}:
        for endpoint in ("Ki", "Kd"):
            tasks.append(
                (
                    f"broad_{endpoint.lower()}",
                    "NonKinaseBroad",
                    Path(
                        f"data/processed/prepared/nonkinase_{endpoint.lower()}_202610_v3/{endpoint}.csv"
                    ),
                    endpoint,
                )
            )
    if scope in {"all", "family"}:
        root = Path("data/processed/prepared/family_affinity_202610_v3")
        cohorts = json.loads((root / "manifest.json").read_text())["cohorts"]
        tasks += [
            (
                row["directory"],
                f"NonKinase_{row['family']}",
                root / row["directory"] / f"{row['endpoint']}.csv",
                row["endpoint"],
            )
            for row in cohorts
            if row["training_supported"]
        ]
    # Small tasks first makes integration failures visible before expensive runs;
    # ordering uses coverage only and cannot depend on model scores.
    tasks.sort(
        key=lambda task: json.loads((task[2].parent / "manifest.json").read_text()).get(
            "observations",
            json.loads((task[2].parent / "manifest.json").read_text())
            .get("counts", {})
            .get(task[3], 0),
        )
    )
    jobs = []
    for task_id, dataset, data, endpoint in tasks:
        if not data.is_file():
            raise ValueError(f"Prepared task is unavailable: {data}")
        for split in splits:
            if split == "family" and not task_id.startswith("broad_"):
                continue
            for model in models:
                for seed in seeds:
                    params = (
                        {
                            "n_estimators": 1800,
                            "max_depth": 9,
                            "learning_rate": 0.03,
                            "early_stopping_rounds": 60,
                            "n_jobs": 4,
                        }
                        if model == "xgboost"
                        else {
                            "protein_max_len": 1500,
                            "protein_embed_dim": 32,
                            "fc_dims": [512, 256],
                            "dropout": 0.1,
                            "mask_padding": True,
                            "deduplicate_proteins": True,
                            "auxiliary_dim": 1774 if protein_cache else 1454,
                            "auxiliary_interactions": True,
                            "use_protein_cnn": not bool(protein_cache),
                        }
                    )
                    if model == "deeptda":
                        params.update(drug_max_len=200, drug_embed_dim=32, num_filters=32)
                    elif model == "gnn":
                        params.update(
                            gnn_type="gine",
                            gnn_hidden_dim=128,
                            gnn_output_dim=128,
                            protein_num_filters=32,
                        )
                    config = {
                        "dataset": dataset,
                        "dataset_version": "202610_v3",
                        "endpoint": endpoint,
                        "input": data.as_posix(),
                        "preprocessing_version": "standardized_v1",
                        "model": model,
                        "model_params": params,
                        "seed": seed,
                        "split": split,
                        "split_seed": 42,
                        "smoke": False,
                        "training": {"threads": 4}
                        if model == "xgboost"
                        else {
                            "threads": 2,
                            "device": device,
                            "epochs": 120,
                            "patience": 15,
                            "batch_size": 128,
                            "learning_rate": 0.001 if model == "deeptda" else 0.0005,
                            "weight_decay": 0.0001,
                            "normalize_targets": True,
                        },
                    }
                    if protein_cache:
                        config["protein_embedding_cache"] = feature_contract
                    job_id = f"{task_id}_{split}_{model}_seed{seed}"
                    path = output / f"{job_id}.json"
                    atomic_json(path, config)
                    jobs.append(
                        {
                            "id": job_id,
                            "config": path.as_posix(),
                            "config_sha256": digest_file(path),
                            "output": f"experiments/runs/{output.name}_{job_id}",
                        }
                    )
    plan = {
        "source_sha256": source_hash(),
        "jobs": jobs,
        "test_evaluation_requested": False,
        "selection": "fixed coverage thresholds; all supported family/endpoint results retained",
        "note": "Exploratory matrix, not a guarantee of neural superiority; do not compare incompatible scales or splits",
    }
    atomic_json(output / "plan.json", plan)
    return len(jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", choices=["all", "broad", "family"], default="all")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgboost", "deeptda", "gnn"],
        default=["xgboost", "deeptda", "gnn"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["random", "cold_drug", "cold_target", "scaffold", "family"],
        default=["random"],
    )
    parser.add_argument("--protein-cache", type=Path)
    parser.add_argument("--device", choices=["cpu", "xpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(
        f"Planned {build_plan(args.output, args.scope, args.models, args.seeds, args.splits, args.protein_cache, args.device)} experiments"
    )
