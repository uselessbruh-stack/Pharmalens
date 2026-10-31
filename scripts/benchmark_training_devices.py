"""Measure neural training throughput; never interpret these timings as accuracy."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from src.data.acquisition import atomic_json
from src.models.deeptda import DeepDTA


def benchmark():
    results = []
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    for device in ["cpu", "xpu"]:
        if device == "xpu" and not torch.xpu.is_available():
            continue
        for batch_size in [128, 512]:
            torch.manual_seed(42)
            model = DeepDTA(
                drug_max_len=200,
                drug_embed_dim=32,
                protein_embed_dim=32,
                num_filters=32,
                fc_dims=[512, 256],
                dropout=0.1,
                mask_padding=True,
                auxiliary_dim=1774,
                auxiliary_interactions=True,
                use_protein_cnn=False,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            drug = torch.randint(1, 40, (batch_size, 200), device=device)
            protein = torch.ones(batch_size, 1500, dtype=torch.long, device=device)
            auxiliary = torch.randn(batch_size, 1774, device=device)
            target = torch.randn(batch_size, device=device)

            def step():
                prediction = model(drug, protein, auxiliary)
                if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
                    raise ValueError("Non-finite device benchmark values")
                loss = (prediction - target).square().mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
                return loss.item()

            step()
            if device == "xpu":
                torch.xpu.synchronize()
            started = time.perf_counter()
            for _ in range(5):
                step()
            if device == "xpu":
                torch.xpu.synchronize()
            elapsed = time.perf_counter() - started
            row = {
                "device": device,
                "batch_size": batch_size,
                "seconds_per_step": elapsed / 5,
                "pairs_per_second": batch_size * 5 / elapsed,
            }
            results.append(row)
            print(row, flush=True)
    atomic_json(
        "results/reports/device_throughput.json",
        {
            "results": results,
            "limitations": "Synthetic inputs; current concurrent machine workload; throughput only, not affinity performance",
        },
    )


if __name__ == "__main__":
    benchmark()
