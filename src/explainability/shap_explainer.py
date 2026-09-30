"""
PharmaLens — Explainability Module

SHAP-based and gradient-based explanations for DTI predictions.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("pharmalens.explainability")


class SHAPExplainer:
    """
    SHAP-based explainer for tree models (XGBoost, Random Forest).

    Provides global feature importance, local per-prediction explanations,
    and feature category analysis (drug descriptors vs fingerprints vs protein).
    """

    def __init__(self, model, feature_names: Optional[list] = None):
        """
        Args:
            model: Trained tree model (XGBoost or sklearn estimator).
            feature_names: List of feature names matching the input dimension.
        """
        import shap

        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
        self.base_value = float(self.explainer.expected_value)

    def explain_single(self, X: np.ndarray, top_k: int = 10) -> dict:
        """
        Explain a single prediction.

        Args:
            X: Feature vector (1, n_features).
            top_k: Number of top features to return.

        Returns:
            dict with prediction details and top contributing features.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        shap_values = self.explainer.shap_values(X)[0]
        prediction = float(self.model.predict(X)[0])

        # Top contributing features
        top_idx = np.argsort(np.abs(shap_values))[::-1][:top_k]
        top_features = []
        for idx in top_idx:
            name = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else f"feature_{idx}"
            top_features.append({
                "rank": len(top_features) + 1,
                "feature": name,
                "feature_index": int(idx),
                "shap_value": float(shap_values[idx]),
                "feature_value": float(X[0, idx]),
                "direction": "↑ increases score" if shap_values[idx] > 0 else "↓ decreases score",
                "abs_importance": float(abs(shap_values[idx])),
            })

        return {
            "prediction": prediction,
            "base_value": self.base_value,
            "top_features": top_features,
            "all_shap_values": shap_values.tolist(),
        }

    def explain_batch(self, X: np.ndarray) -> dict:
        """
        Compute global feature importance from a batch.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            dict with global importance analysis.
        """
        shap_values = self.explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)

        # Global ranking
        sorted_idx = np.argsort(mean_abs)[::-1]
        global_importance = []
        for rank, idx in enumerate(sorted_idx[:30]):
            name = self.feature_names[idx] if self.feature_names and idx < len(self.feature_names) else f"feature_{idx}"
            global_importance.append({
                "rank": rank + 1,
                "feature": name,
                "mean_abs_shap": float(mean_abs[idx]),
            })

        return {
            "global_importance": global_importance,
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs,
        }

    def category_importance(
        self,
        X: np.ndarray,
        n_descriptors: int = 10,
        n_fingerprint: int = 1024,
        n_aac: int = 20,
        n_dpc: int = 400,
    ) -> dict:
        """
        Aggregate SHAP importance by feature category.

        Args:
            X: Feature matrix.
            n_descriptors: Number of molecular descriptor features.
            n_fingerprint: Number of fingerprint bits.
            n_aac: Number of amino acid composition features.
            n_dpc: Number of dipeptide composition features.

        Returns:
            dict mapping category name to total SHAP importance.
        """
        shap_values = self.explainer.shap_values(X)
        mean_abs = np.abs(shap_values).mean(axis=0)

        boundaries = [0, n_descriptors, n_descriptors + n_fingerprint,
                       n_descriptors + n_fingerprint + n_aac,
                       n_descriptors + n_fingerprint + n_aac + n_dpc]
        categories = ["Molecular Descriptors", "Morgan Fingerprint",
                       "Amino Acid Composition", "Dipeptide Composition"]

        result = {}
        for i, cat in enumerate(categories):
            start, end = boundaries[i], boundaries[i + 1]
            if end <= len(mean_abs):
                result[cat] = {
                    "total_importance": float(mean_abs[start:end].sum()),
                    "mean_importance": float(mean_abs[start:end].mean()),
                    "n_features": end - start,
                    "top_feature_idx": int(start + np.argmax(mean_abs[start:end])),
                }

        # Drug vs protein split
        drug_end = n_descriptors + n_fingerprint
        drug_importance = float(mean_abs[:drug_end].sum())
        protein_importance = float(mean_abs[drug_end:].sum())
        total = drug_importance + protein_importance

        result["_summary"] = {
            "drug_importance": drug_importance,
            "protein_importance": protein_importance,
            "drug_pct": round(drug_importance / total * 100, 1) if total > 0 else 0,
            "protein_pct": round(protein_importance / total * 100, 1) if total > 0 else 0,
        }

        return result


def compute_integrated_gradients(model, drug_input, protein_input, n_steps: int = 50):
    """
    Compute Integrated Gradients for a DeepDTA model.

    Measures how much each input position contributes to the prediction
    by integrating gradients from a baseline (zero) input to the actual input.

    Args:
        model: DeepDTA model (PyTorch).
        drug_input: Drug tensor (1, max_len).
        protein_input: Protein tensor (1, max_len).
        n_steps: Number of interpolation steps.

    Returns:
        dict with drug and protein attributions.
    """
    import torch

    model.eval()

    # Baselines (zero inputs)
    drug_baseline = torch.zeros_like(drug_input)
    protein_baseline = torch.zeros_like(protein_input)

    # Interpolated inputs
    alphas = torch.linspace(0, 1, n_steps + 1).unsqueeze(1)

    # Drug attributions (keep protein fixed)
    drug_grads = []
    for alpha in alphas:
        interp = drug_baseline + alpha * (drug_input - drug_baseline)
        interp = interp.long()
        interp.requires_grad_(False)

        # Use embedding directly for gradient computation
        drug_emb = model.drug_embedding(interp)
        drug_emb.requires_grad_(True)

        # Forward through drug encoder manually
        x = drug_emb.permute(0, 2, 1)
        for conv in model.drug_convs:
            x = torch.relu(conv(x))
        drug_repr = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)

        prot_repr = model.encode_protein(protein_input)
        combined = torch.cat([drug_repr, prot_repr], dim=1)
        for fc, dropout in zip(model.fc_layers, model.fc_dropouts):
            combined = torch.relu(fc(combined))
        output = model.output(combined)

        output.backward()
        drug_grads.append(drug_emb.grad.detach().clone())
        drug_emb.grad = None

    # Average gradients × (input - baseline)
    avg_grads = torch.stack(drug_grads).mean(dim=0)
    drug_attr = (avg_grads * (model.drug_embedding(drug_input) - model.drug_embedding(drug_baseline))).sum(dim=-1)

    return {
        "drug_attributions": drug_attr.detach().numpy()[0],
    }
