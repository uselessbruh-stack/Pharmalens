"""Exact group Shapley values for every active neural representation branch.

The coalition game masks learned representations to zero and auxiliary features
to their saved training means. It is not atom/residue attribution or a biological
counterfactual. No mutable hooks, gradients or explainer state are shared.
"""

import math

import numpy as np
import torch


def explain_neural(pipeline, smiles, sequence, served_score):
    from src.features.drug_features import encode_smiles
    from src.features.protein_features import encode_sequence

    model = pipeline.model
    device = next(model.parameters()).device
    smiles, sequence = pipeline._clean_smiles(smiles), pipeline._clean_sequence(sequence)
    protein = torch.as_tensor(
        encode_sequence(sequence, max_length=model.config["protein_max_len"]),
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        if pipeline.model_type == "deeptda":
            drug = model.encode_drug(torch.as_tensor(
                encode_smiles(smiles, max_length=model.drug_max_len),
                dtype=torch.long, device=device,
            ).unsqueeze(0))
            prot = model.encode_protein(protein) if model.use_protein_cnn else None
        else:
            from src.features.mol_graph import smiles_to_graph

            graph = smiles_to_graph(smiles)
            drug = model.drug_encoder(
                torch.as_tensor(graph["node_features"], dtype=torch.float32, device=device),
                torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device),
                torch.zeros(graph["num_atoms"], dtype=torch.long, device=device),
                torch.as_tensor(graph["edge_features"], dtype=torch.float32, device=device),
            )
            prot = model.protein_encoder(protein) if model.use_protein_cnn else None
        aux = pipeline._neural_auxiliary(smiles, sequence, device)
        groups = [("Learned molecular representation", "drug", None)]
        if prot is not None:
            groups.append(("Learned protein CNN representation", "protein", None))
        if aux is not None:
            for name, start, stop in [
                ("Molecular descriptors", 0, 10),
                ("Morgan fingerprint", 10, 1034),
                ("Protein amino acid composition", 1034, 1054),
                ("Protein dipeptide composition", 1054, 1454),
                ("Frozen ESM-2 protein representation", 1454, 1774),
            ]:
                if start < aux.shape[1]:
                    groups.append((name, "auxiliary", slice(start, min(stop, aux.shape[1]))))
        n = len(groups)
        masks = torch.as_tensor(
            [[bool(mask & (1 << j)) for j in range(n)] for mask in range(1 << n)],
            device=device,
        )
        size = len(masks)
        drugs = drug.expand(size, -1) * masks[:, 0:1]
        proteins = prot.expand(size, -1) * masks[:, 1:2] if prot is not None else None
        auxiliaries = None
        if aux is not None:
            mean = model.auxiliary_encoder.mean.unsqueeze(0)
            auxiliaries = mean.expand(size, -1).clone()
            for j, (_, kind, indices) in enumerate(groups):
                if kind == "auxiliary":
                    auxiliaries[:, indices] = torch.where(
                        masks[:, j:j+1], aux[:, indices], mean[:, indices]
                    )
        raw = model.predict_representations(drugs, proteins, auxiliaries).cpu().numpy()
    values = raw.astype(np.float64) * pipeline.target_scaler["scale"] + pipeline.target_scaler["mean"]
    if not np.isfinite(values).all() or not np.isclose(values[-1], served_score, atol=2e-4, rtol=1e-5):
        raise ValueError("Explanation forward pass did not match the served prediction")
    contributions = []
    for j, (name, _, _) in enumerate(groups):
        value = 0.0
        for mask in range(1 << n):
            if not mask & (1 << j):
                k = mask.bit_count()
                weight = 1.0 / (n * math.comb(n - 1, k))
                value += weight * (values[mask | (1 << j)] - values[mask])
        contributions.append({"feature": name, "shap_value": float(value)})
    reconstructed = float(values[0] + sum(row["shap_value"] for row in contributions))
    return {
        "method": "Exact grouped representation SHAP",
        "scope": "All active representation branches of this neural model",
        "base_value": float(values[0]),
        "top_features": sorted(contributions, key=lambda row: abs(row["shap_value"]), reverse=True),
        "other_features_contribution": 0.0,
        "reconstructed_score": reconstructed,
        "additivity_error": float(abs(reconstructed - served_score)),
        "coalitions_evaluated": size,
        "baseline": "Zero learned representations and saved training means for auxiliary features",
        "limitations": [
            "Representation-group attribution, not atom, bond or residue localization.",
            "Masked representations can be outside the training distribution.",
            "Contributions depend on the baseline and grouping; they are not causal or mechanistic evidence.",
            "The frozen protein encoder is explained as a group, not internally decomposed.",
        ],
    }
