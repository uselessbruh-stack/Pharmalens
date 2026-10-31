"""Cache frozen ESM-2 protein representations without using affinity labels."""

import argparse
import importlib.metadata
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import ROOT, atomic_json, digest_file
from src.data.preparation import AA, stable_id

MODEL = "esm2_t6_8M_UR50D"


def generate(inputs, output, device="cpu", weights_dir=None):
    import esm

    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new immutable embedding cache")
    sequences = set()
    sources = []
    for path in inputs:
        path = Path(path)
        for frame in pd.read_csv(path, usecols=["sequence"], chunksize=10000):
            sequences.update(frame.sequence.unique())
        sources.append({"path": str(path.resolve()), "sha256": digest_file(path)})
    if not sequences or any(
        not isinstance(seq, str) or not 10 <= len(seq) <= 10000 or not set(seq) <= AA
        for seq in sequences
    ):
        raise ValueError("Inputs must contain prepared standard-AA protein sequences")
    weights_dir = Path(weights_dir or ROOT / "models/protein/torch_hub").resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(weights_dir))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval().to(device)
    converter = alphabet.get_batch_converter()
    sequences = sorted(sequences, key=stable_id)
    chunks = [
        (i, seq[start : start + 1022])
        for i, seq in enumerate(sequences)
        for start in range(0, len(seq), 1022)
    ]
    chunks.sort(key=lambda item: (-len(item[1]), item[0]))
    sums = np.zeros((len(sequences), 320), dtype=np.float64)
    counts = np.zeros(len(sequences), dtype=np.int64)
    staging = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    position = 0
    with torch.inference_mode():
        while position < len(chunks):
            batch_size = min(8, max(1, 4096 // (len(chunks[position][1]) + 2)))
            batch = chunks[position : position + batch_size]
            _, _, tokens = converter([(str(index), seq) for index, seq in batch])
            representations = model(tokens.to(device), repr_layers=[6], return_contacts=False)[
                "representations"
            ][6]
            for row, (index, sequence) in enumerate(batch):
                pooled_sum = representations[row, 1 : len(sequence) + 1].sum(0).cpu().numpy()
                sums[index] += pooled_sum
                counts[index] += len(sequence)
            position += len(batch)
            if position % 100 < len(batch) or position == len(chunks):
                atomic_json(
                    staging / "progress.json",
                    {
                        "status": "encoding",
                        "chunks_completed": position,
                        "chunks_total": len(chunks),
                    },
                )
                print(f"ESM chunks: {position}/{len(chunks)}", flush=True)
    if not np.array_equal(counts, [len(seq) for seq in sequences]):
        raise ValueError("Not every protein residue was encoded exactly once")
    embeddings = (sums / counts[:, None]).astype(np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite protein representations")
    data = staging / "embeddings.npz"
    np.savez_compressed(
        data,
        ids=np.array([stable_id(seq) for seq in sequences]),
        embeddings=embeddings,
        lengths=counts,
    )
    weights = weights_dir / "checkpoints" / f"{MODEL}.pt"
    atomic_json(
        staging / "manifest.json",
        {
            "encoder": MODEL,
            "dimension": 320,
            "layer": 6,
            "protein_count": len(sequences),
            "device": device,
            "torch": torch.__version__,
            "fair_esm": importlib.metadata.version("fair-esm"),
            "data_file": data.name,
            "sha256": digest_file(data),
            "sources": sources,
            "weights": {"path": str(weights), "sha256": digest_file(weights)},
            "pooling": "residue mean; nonoverlapping 1022-residue chunks weighted by residue count; exclude BOS/EOS/padding",
            "labels_used": False,
            "encoder_finetuned": False,
            "limitations": "Frozen public pretraining may contain related proteins; chunking loses cross-chunk context; does not establish external-domain validation",
            "source": "https://github.com/facebookresearch/esm",
            "license": "MIT (ESM repository; preserve attribution)",
        },
    )
    staging.replace(output)
    print(f"Cached {len(sequences)} proteins in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "xpu", "cuda"], default="cpu")
    args = parser.parse_args()
    generate(args.inputs, args.output, args.device)
