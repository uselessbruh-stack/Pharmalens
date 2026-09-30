"""
PharmaLens — Inference Pipeline

End-to-end prediction pipeline that takes raw SMILES + protein sequences
and produces interaction scores with optional SHAP explanations.
"""

import logging
import pickle
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

logger = logging.getLogger("pharmalens.inference")


class PharmaLensPipeline:
    """
    Production inference pipeline for PharmaLens.

    Takes raw SMILES strings and protein sequences as input,
    computes features, runs the model, and returns predictions
    with optional SHAP explanations.

    Usage:
        pipeline = PharmaLensPipeline.from_pretrained("models/xgboost/xgb_best.json")
        result = pipeline.predict("CCO", "MKTFVLLL...")
        results = pipeline.batch_predict(smiles_list, sequence_list)
    """

    def __init__(self, model, model_type: str = "xgboost"):
        """
        Args:
            model: Trained model instance.
            model_type: One of 'xgboost', 'deeptda', 'gnn'.
        """
        self.model = model
        self.model_type = model_type
        self._shap_explainer = None

    @classmethod
    def from_pretrained(cls, model_path: str, model_type: str = "xgboost"):
        """
        Load a trained model and create a pipeline.

        Args:
            model_path: Path to saved model file.
            model_type: 'xgboost', 'deeptda', or 'gnn'.

        Returns:
            PharmaLensPipeline instance.
        """
        model_path = Path(model_path)

        if model_type == "xgboost":
            from src.models.xgboost_model import XGBoostDTI
            model = XGBoostDTI()
            model.load(model_path)

        elif model_type == "deeptda":
            import torch
            from src.models.deeptda import DeepDTA
            checkpoint = torch.load(model_path, map_location="cpu")
            config = checkpoint.get("config", {})
            model = DeepDTA(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        elif model_type == "gnn":
            import torch
            from src.models.gnn_model import GraphDTA
            checkpoint = torch.load(model_path, map_location="cpu")
            config = checkpoint.get("config", {})
            model = GraphDTA(**config)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Loaded {model_type} model from {model_path}")
        return cls(model, model_type)

    def predict(
        self,
        smiles: str,
        sequence: str,
        explain: bool = False,
    ) -> dict:
        """
        Predict interaction score for a single drug-target pair.

        Args:
            smiles: Drug SMILES string.
            sequence: Protein amino acid sequence.
            explain: If True, include SHAP explanations (XGBoost only).

        Returns:
            dict with 'score', 'drug_features', 'protein_features',
            and optionally 'shap_values'.
        """
        t_start = time.time()

        if self.model_type == "xgboost":
            result = self._predict_xgboost(smiles, sequence, explain)
        elif self.model_type == "deeptda":
            result = self._predict_deeptda(smiles, sequence)
        elif self.model_type == "gnn":
            result = self._predict_gnn(smiles, sequence)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        result["inference_time"] = time.time() - t_start
        return result

    def batch_predict(
        self,
        smiles_list: list[str],
        sequence_list: list[str],
    ) -> list[dict]:
        """
        Predict interaction scores for multiple drug-target pairs.

        Args:
            smiles_list: List of SMILES strings.
            sequence_list: List of protein sequences (same length as smiles_list).

        Returns:
            List of result dicts.
        """
        assert len(smiles_list) == len(sequence_list), \
            "smiles_list and sequence_list must have the same length"

        results = []
        for smiles, sequence in zip(smiles_list, sequence_list):
            try:
                result = self.predict(smiles, sequence, explain=False)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed prediction for {smiles[:30]}...: {e}")
                results.append({
                    "score": None,
                    "error": str(e),
                    "smiles": smiles,
                })

        return results

    def screen(
        self,
        smiles_list: list[str],
        sequence_list: list[str],
        drug_ids: Optional[list[str]] = None,
        target_ids: Optional[list[str]] = None,
    ) -> "pd.DataFrame":
        """
        Screen all combinations of drugs × targets and return ranked results.

        Args:
            smiles_list: List of drug SMILES.
            sequence_list: List of protein sequences.
            drug_ids: Optional drug identifiers.
            target_ids: Optional target identifiers.

        Returns:
            DataFrame with all combinations ranked by predicted score.
        """
        import pandas as pd

        if drug_ids is None:
            drug_ids = [f"drug_{i}" for i in range(len(smiles_list))]
        if target_ids is None:
            target_ids = [f"target_{i}" for i in range(len(sequence_list))]

        results = []
        for i, (smiles, did) in enumerate(zip(smiles_list, drug_ids)):
            for j, (seq, tid) in enumerate(zip(sequence_list, target_ids)):
                try:
                    pred = self.predict(smiles, seq)
                    results.append({
                        "drug_id": did,
                        "target_id": tid,
                        "smiles": smiles,
                        "predicted_score": pred["score"],
                        "inference_time": pred["inference_time"],
                    })
                except Exception as e:
                    results.append({
                        "drug_id": did,
                        "target_id": tid,
                        "smiles": smiles,
                        "predicted_score": None,
                        "error": str(e),
                    })

        df = pd.DataFrame(results)
        df = df.sort_values("predicted_score", ascending=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    # =========================================================================
    # Private prediction methods per model type
    # =========================================================================

    def _predict_xgboost(self, smiles: str, sequence: str, explain: bool = False) -> dict:
        """XGBoost prediction with optional SHAP."""
        from src.features.drug_features import get_drug_feature_vector
        from src.features.protein_features import get_protein_feature_vector
        from src.preprocessing.drug_preprocessor import validate_and_clean_smiles
        from src.preprocessing.protein_preprocessor import validate_and_clean_sequence

        # Preprocess
        clean_smiles = validate_and_clean_smiles(smiles)
        if clean_smiles is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        clean_seq = validate_and_clean_sequence(sequence)
        if clean_seq is None:
            raise ValueError(f"Invalid protein sequence")

        # Extract features
        drug_fv = get_drug_feature_vector(clean_smiles)
        if drug_fv is None:
            raise ValueError(f"Failed to compute drug features for: {clean_smiles}")

        prot_fv = get_protein_feature_vector(clean_seq)

        # Concatenate and predict
        combined = np.concatenate([drug_fv, prot_fv]).reshape(1, -1)
        score = float(self.model.predict(combined)[0])

        result = {
            "score": score,
            "smiles": clean_smiles,
            "sequence_length": len(clean_seq),
        }

        # SHAP explanation
        if explain:
            result["shap_values"] = self._compute_shap(combined)

        return result

    def _predict_deeptda(self, smiles: str, sequence: str) -> dict:
        """DeepDTA prediction."""
        import torch
        from src.features.drug_features import encode_smiles
        from src.features.protein_features import encode_sequence
        from src.preprocessing.drug_preprocessor import validate_and_clean_smiles
        from src.preprocessing.protein_preprocessor import validate_and_clean_sequence

        clean_smiles = validate_and_clean_smiles(smiles)
        if clean_smiles is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        clean_seq = validate_and_clean_sequence(sequence)
        if clean_seq is None:
            raise ValueError(f"Invalid protein sequence")

        drug_enc = encode_smiles(clean_smiles, max_length=100)
        prot_enc = encode_sequence(clean_seq, max_length=1000)

        drug_tensor = torch.tensor(drug_enc, dtype=torch.long).unsqueeze(0)
        prot_tensor = torch.tensor(prot_enc, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            score = float(self.model(drug_tensor, prot_tensor).item())

        return {
            "score": score,
            "smiles": clean_smiles,
            "sequence_length": len(clean_seq),
        }

    def _predict_gnn(self, smiles: str, sequence: str) -> dict:
        """GNN prediction."""
        import torch
        from torch_geometric.data import Data, Batch
        from src.features.mol_graph import smiles_to_graph
        from src.features.protein_features import encode_sequence
        from src.preprocessing.drug_preprocessor import validate_and_clean_smiles
        from src.preprocessing.protein_preprocessor import validate_and_clean_sequence

        clean_smiles = validate_and_clean_smiles(smiles)
        if clean_smiles is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        clean_seq = validate_and_clean_sequence(sequence)
        if clean_seq is None:
            raise ValueError(f"Invalid protein sequence")

        graph = smiles_to_graph(clean_smiles)
        if graph is None:
            raise ValueError(f"Failed to convert SMILES to graph: {clean_smiles}")

        prot_enc = encode_sequence(clean_seq, max_length=1000)

        data = Data(
            x=torch.tensor(graph["node_features"], dtype=torch.float),
            edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
            protein_input=torch.tensor(prot_enc, dtype=torch.long).unsqueeze(0),
        )

        batch = Batch.from_data_list([data])
        batch.protein_input = data.protein_input

        with torch.no_grad():
            score = float(self.model(batch).item())

        return {
            "score": score,
            "smiles": clean_smiles,
            "sequence_length": len(clean_seq),
            "num_atoms": graph["num_atoms"],
        }

    def _compute_shap(self, X: np.ndarray) -> dict:
        """Compute SHAP values for XGBoost."""
        import shap
        from src.features.drug_features import get_feature_names
        from src.features.protein_features import get_protein_feature_names

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self.model.model)

        sv = self._shap_explainer.shap_values(X)[0]
        feature_names = get_feature_names() + get_protein_feature_names()

        # Top 10 contributing features
        top_idx = np.argsort(np.abs(sv))[::-1][:10]
        top_features = []
        for idx in top_idx:
            name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            top_features.append({
                "feature": name,
                "shap_value": float(sv[idx]),
                "feature_value": float(X[0, idx]),
                "direction": "positive" if sv[idx] > 0 else "negative",
            })

        return {
            "base_value": float(self._shap_explainer.expected_value),
            "top_features": top_features,
        }
