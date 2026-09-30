"""
PharmaLens — Screening Module

High-level screening API for drug candidate prioritization.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("pharmalens.screening")


class DrugScreener:
    """
    Drug screening pipeline for candidate prioritization.

    Screens drug candidates against protein targets and produces
    a ranked list of predicted interactions, optionally with
    explainability analysis.

    Usage:
        screener = DrugScreener(pipeline)
        results = screener.screen_panel(smiles_list, sequence_list)
        top_hits = screener.get_top_hits(results, n=10)
    """

    def __init__(self, pipeline):
        """
        Args:
            pipeline: PharmaLensPipeline instance.
        """
        self.pipeline = pipeline

    def screen_panel(
        self,
        smiles_list: list[str],
        sequence_list: list[str],
        drug_ids: Optional[list[str]] = None,
        target_ids: Optional[list[str]] = None,
        explain_top_n: int = 0,
    ) -> pd.DataFrame:
        """
        Screen all drug × target combinations.

        Args:
            smiles_list: Drug SMILES strings.
            sequence_list: Protein sequences.
            drug_ids: Optional identifiers for drugs.
            target_ids: Optional identifiers for targets.
            explain_top_n: If > 0, add SHAP explanations for top N hits.

        Returns:
            DataFrame ranked by predicted binding score.
        """
        if drug_ids is None:
            drug_ids = [f"drug_{i}" for i in range(len(smiles_list))]
        if target_ids is None:
            target_ids = [f"target_{i}" for i in range(len(sequence_list))]

        results = []
        total = len(smiles_list) * len(sequence_list)
        logger.info(f"Screening {len(smiles_list)} drugs × {len(sequence_list)} targets = {total} combinations")

        for i, (smiles, did) in enumerate(zip(smiles_list, drug_ids)):
            for j, (seq, tid) in enumerate(zip(sequence_list, target_ids)):
                try:
                    pred = self.pipeline.predict(smiles, seq)
                    results.append({
                        "drug_id": did,
                        "target_id": tid,
                        "smiles": smiles[:60],
                        "predicted_score": pred["score"],
                        "inference_time_ms": pred["inference_time"] * 1000,
                    })
                except Exception as e:
                    results.append({
                        "drug_id": did,
                        "target_id": tid,
                        "smiles": smiles[:60],
                        "predicted_score": None,
                        "error": str(e),
                    })

        df = pd.DataFrame(results)
        df = df.sort_values("predicted_score", ascending=True, na_position="last")
        df["rank"] = range(1, len(df) + 1)

        # Classify binding strength
        df["binding_class"] = df["predicted_score"].apply(self._classify_binding)

        # Add SHAP for top hits if requested
        if explain_top_n > 0 and self.pipeline.model_type == "xgboost":
            df = self._add_explanations(df, smiles_list, sequence_list,
                                        drug_ids, target_ids, explain_top_n)

        logger.info(f"Screening complete: {len(df)} results, {df['predicted_score'].notna().sum()} successful")
        return df

    @staticmethod
    def _classify_binding(score):
        """Classify binding strength based on KIBA score."""
        if score is None or np.isnan(score):
            return "unknown"
        if score < 3:
            return "strong"
        elif score < 10:
            return "moderate"
        else:
            return "weak"

    def _add_explanations(self, df, smiles_list, sequence_list,
                          drug_ids, target_ids, top_n):
        """Add SHAP explanations for top N hits."""
        top_rows = df.head(top_n)

        smiles_dict = dict(zip(drug_ids, smiles_list))
        seq_dict = dict(zip(target_ids, sequence_list))

        explanations = []
        for _, row in top_rows.iterrows():
            try:
                smiles = smiles_dict.get(row["drug_id"], "")
                seq = seq_dict.get(row["target_id"], "")
                result = self.pipeline.predict(smiles, seq, explain=True)
                explanations.append(result.get("shap_values", {}))
            except Exception:
                explanations.append({})

        df.loc[df.index[:top_n], "shap_explanation"] = explanations
        return df

    def get_top_hits(self, results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top N strongest predicted binders."""
        return results_df.head(n)

    def generate_report(self, results_df: pd.DataFrame) -> dict:
        """Generate a summary report of screening results."""
        valid = results_df[results_df["predicted_score"].notna()]

        return {
            "total_combinations": len(results_df),
            "successful_predictions": len(valid),
            "failed_predictions": len(results_df) - len(valid),
            "score_stats": {
                "mean": float(valid["predicted_score"].mean()),
                "std": float(valid["predicted_score"].std()),
                "min": float(valid["predicted_score"].min()),
                "max": float(valid["predicted_score"].max()),
                "median": float(valid["predicted_score"].median()),
            },
            "binding_distribution": valid["binding_class"].value_counts().to_dict(),
            "top_5_hits": valid.head(5)[["drug_id", "target_id", "predicted_score", "binding_class"]].to_dict("records"),
        }
