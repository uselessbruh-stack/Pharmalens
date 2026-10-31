import json

import numpy as np
import pandas as pd
import pytest

from src.data.acquisition import atomic_json, digest_file
from src.data.preparation import stable_id
from src.features.protein_embeddings import POOLING_POLICY, ProteinEmbeddingStore
from src.training.experiment import run_experiment
from tests.test_experiments import experiment_config


def make_cache(tmp_path, sequences):
    folder = tmp_path / "protein_cache"
    folder.mkdir()
    weights = folder / "esm2_t6_8M_UR50D.pt"
    weights.write_bytes(b"test-only unused encoder")
    weights.with_name(weights.stem + "-contact-regression.pt").write_bytes(
        b"test-only unused regression"
    )
    data = folder / "embeddings.npz"
    np.savez_compressed(
        data,
        ids=np.array([stable_id(seq) for seq in sequences]),
        embeddings=np.random.default_rng(42).normal(size=(len(sequences), 320)).astype(np.float32),
    )
    atomic_json(
        folder / "manifest.json",
        {
            "encoder": "esm2_t6_8M_UR50D",
            "dimension": 320,
            "layer": 6,
            "labels_used": False,
            "encoder_finetuned": False,
            "pooling": POOLING_POLICY,
            "data_file": data.name,
            "sha256": digest_file(data),
            "protein_count": len(sequences),
            "weights": {"path": str(weights), "sha256": digest_file(weights)},
        },
    )
    return folder


def test_protein_cache_rejects_modified_bytes_and_supervised_features(tmp_path):
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    folder = make_cache(tmp_path, [sequence])
    store = ProteinEmbeddingStore(folder)
    assert store.get(sequence).shape == (320,)
    assert ProteinEmbeddingStore(store.contract).contract == store.contract
    (folder / "embeddings.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ProteinEmbeddingStore(store.contract)
    metadata = json.loads((folder / "manifest.json").read_text())
    metadata["labels_used"] = True
    atomic_json(folder / "manifest.json", metadata)
    with pytest.raises(ValueError, match="Unsupported"):
        ProteinEmbeddingStore(folder)


@pytest.mark.parametrize("model", ["xgboost", "deeptda", "gine", "deeptda_frozen", "gine_frozen"])
def test_shared_protein_features_survive_training_and_serving_reload(tmp_path, model):
    frozen = model.endswith("_frozen")
    model = model.removesuffix("_frozen")
    config = experiment_config(tmp_path, model)
    sequences = pd.read_csv(config["input"]).sequence.unique().tolist()
    cache = make_cache(tmp_path, sequences)
    config["protein_embedding_cache"] = str(cache)
    if model != "xgboost":
        config["model_params"].update(auxiliary_dim=1774, auxiliary_interactions=True)
        config["model_params"]["use_protein_cnn"] = not frozen
    output = tmp_path / "trained"
    result = run_experiment(config, output)
    assert result["serving_parity"]["verified"]
    assert result["protein_feature_contract"]["sha256"] == digest_file(cache / "manifest.json")
    assert result["test_metrics"] is None
    assert result["execution_device"] == "cpu"
