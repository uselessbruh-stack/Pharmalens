"""Integrity-checked frozen protein features shared by trees and neural models."""

import json
import argparse
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.data.acquisition import digest_file
from src.data.preparation import AA, stable_id

POOLING_POLICY = "residue mean; nonoverlapping 1022-residue chunks weighted by residue count; exclude BOS/EOS/padding"


class ProteinEmbeddingStore:
    def __init__(self, contract):
        expected = contract if isinstance(contract, dict) else None
        path = Path(expected["manifest"] if expected else contract).resolve()
        if path.is_dir():
            path = path / "manifest.json"
        manifest_sha = digest_file(path)
        if expected and manifest_sha != expected["sha256"]:
            raise ValueError("Protein feature manifest checksum mismatch")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (
            manifest["encoder"] != "esm2_t6_8M_UR50D"
            or manifest["dimension"] != 320
            or manifest["layer"] != 6
            or manifest.get("labels_used") is not False
            or manifest.get("encoder_finetuned") is not False
            or manifest.get("pooling") != POOLING_POLICY
        ):
            raise ValueError("Unsupported pretrained protein feature contract")
        data = (path.parent / manifest["data_file"]).resolve()
        if data.parent != path.parent or digest_file(data) != manifest["sha256"]:
            raise ValueError("Protein embedding cache location/checksum mismatch")
        self.weights = Path(manifest["weights"]["path"]).resolve()
        if digest_file(self.weights) != manifest["weights"]["sha256"]:
            raise ValueError("Pretrained protein weights checksum mismatch")
        self.regression = self.weights.with_name(self.weights.stem + "-contact-regression.pt")
        regression_sha = digest_file(self.regression)
        if expected and regression_sha != expected["regression_sha256"]:
            raise ValueError("Protein encoder regression weights checksum mismatch")
        with np.load(data, allow_pickle=False) as archive:
            ids, embeddings = archive["ids"], archive["embeddings"]
        if (
            embeddings.shape != (manifest["protein_count"], 320)
            or ids.shape != (len(embeddings),)
            or len(set(ids.tolist())) != len(ids)
            or not np.isfinite(embeddings).all()
            or embeddings.dtype != np.float32
        ):
            raise ValueError("Invalid protein feature matrix")
        self.vectors = {str(key): value.astype(np.float32) for key, value in zip(ids, embeddings)}
        self.contract = {
            "manifest": str(path),
            "sha256": manifest_sha,
            "regression_sha256": regression_sha,
        }
        self.dimension = 320
        self.model = None
        self.alphabet = None
        self._encoding_lock = threading.RLock()

    def get(self, sequence):
        vector = self.vectors.get(stable_id(sequence))
        return vector if vector is not None else self.encode_new_sequence(sequence)

    @lru_cache(maxsize=512)
    def encode_new_sequence(self, sequence):
        """Serve unseen proteins locally; never download weights in an API request."""
        with self._encoding_lock:
            return self._encode(sequence)

    def _encode(self, sequence):
        if (
            not isinstance(sequence, str)
            or not 10 <= len(sequence) <= 10000
            or not set(sequence) <= AA
        ):
            raise ValueError("Pretrained encoder requires standard-AA protein sequences")
        import esm
        import torch

        if self.model is None:
            # Official ESM checkpoints contain a plain argparse Namespace.
            # Keep weights-only loading and allow only that specific metadata type.
            with torch.serialization.safe_globals([argparse.Namespace]):
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(
                    str(self.weights)
                )
            self.model.eval().cpu()
        converter = self.alphabet.get_batch_converter()
        total = np.zeros(320, dtype=np.float64)
        count = 0
        with torch.inference_mode():
            for start in range(0, len(sequence), 1022):
                chunk = sequence[start : start + 1022]
                _, _, tokens = converter([("protein", chunk)])
                representation = self.model(tokens, repr_layers=[6], return_contacts=False)[
                    "representations"
                ][6]
                total += representation[0, 1 : len(chunk) + 1].sum(0).numpy()
                count += len(chunk)
        vector = (total / count).astype(np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Protein encoder returned non-finite features")
        return vector


@lru_cache(maxsize=8)
def _load_store(manifest, sha256, regression_sha256):
    return ProteinEmbeddingStore(
        {"manifest": manifest, "sha256": sha256, "regression_sha256": regression_sha256}
    )


def load_store(contract):
    if isinstance(contract, dict):
        return _load_store(contract["manifest"], contract["sha256"], contract["regression_sha256"])
    # Resolve/verify once and return the same immutable contract used in artifacts.
    store = ProteinEmbeddingStore(contract)
    return _load_store(**store.contract)
