"""Check real CPU/XPU neural parity and backward kernels before GPU training."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch_geometric.data import Batch, Data
from src.data.acquisition import atomic_json
from src.features.mol_graph import smiles_to_graph
from src.models.deeptda import DeepDTA
from src.models.gnn_model import GraphDTA


def verify():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    if not torch.xpu.is_available():
        raise RuntimeError("XPU runtime/device unavailable")
    report = {"torch": torch.__version__, "device": torch.xpu.get_device_name(0), "models": {}}
    for architecture in ("cnn", "gine"):
        model = (
            DeepDTA(
                drug_embed_dim=16,
                protein_embed_dim=16,
                num_filters=16,
                fc_dims=[32],
                mask_padding=True,
                auxiliary_dim=1454,
                deduplicate_proteins=True,
            )
            if architecture == "cnn"
            else GraphDTA(
                gnn_type="gine",
                gnn_hidden_dim=32,
                gnn_output_dim=32,
                protein_embed_dim=16,
                protein_num_filters=16,
                fc_dims=[32],
                mask_padding=True,
                auxiliary_dim=1454,
                deduplicate_proteins=True,
            )
        )
        model.eval()
        protein = torch.zeros((4, 500), dtype=torch.long)
        protein[:, :400] = torch.randint(1, 21, (1, 400)).expand(4, -1)
        auxiliary = torch.randn(4, 1454)
        if architecture == "cnn":
            drug = torch.randint(1, 40, (4, 100))
            inputs = (drug, protein, auxiliary)
        else:
            graph = smiles_to_graph("CCOc1ccccc1C(=O)N")
            batch = Batch.from_data_list(
                [
                    Data(
                        x=torch.tensor(graph["node_features"], dtype=torch.float),
                        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                        edge_attr=torch.tensor(graph["edge_features"], dtype=torch.float),
                        protein_input=protein[i : i + 1],
                        auxiliary=auxiliary[i : i + 1],
                    )
                    for i in range(4)
                ]
            )
            inputs = (batch,)
        with torch.no_grad():
            cpu_prediction = model(*inputs).detach()
        model.to("xpu")
        device_inputs = tuple(value.to("xpu") for value in inputs)
        with torch.no_grad():
            prediction = model(*device_inputs).cpu()
        if not torch.allclose(cpu_prediction, prediction, rtol=1e-4, atol=1e-5):
            raise ValueError(f"{architecture} CPU/XPU prediction mismatch")
        model.train()
        started = time.perf_counter()
        model(*device_inputs).square().mean().backward()
        torch.xpu.synchronize()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Non-finite accelerator gradients")
        report["models"][architecture] = {
            "prediction_parity": True,
            "backward_finite": True,
            "max_prediction_difference": float((cpu_prediction - prediction).abs().max()),
            "first_backward_seconds": time.perf_counter() - started,
        }
    report["status"] = "passed"
    atomic_json(Path("results/reports/xpu_verification.json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
