"""
PharmaLens — Screening Module

High-level screening API for drug candidate prioritization.
"""

import logging
from typing import Optional

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
        # One ranking implementation keeps endpoint direction and failures aligned.
        df = self.pipeline.screen(smiles_list, sequence_list, drug_ids, target_ids)
        df["inference_time_ms"] = (
            df.get("inference_time", pd.Series(index=df.index, dtype=float)) * 1000
        )
        df["binding_class"] = "unclassified"
        if explain_top_n > 0 and self.pipeline.model_type == "xgboost":
            resolved_drugs = (
                drug_ids if drug_ids is not None else [f"drug_{i}" for i in range(len(smiles_list))]
            )
            resolved_targets = (
                target_ids
                if target_ids is not None
                else [f"target_{i}" for i in range(len(sequence_list))]
            )
            df = self._add_explanations(
                df, smiles_list, sequence_list, resolved_drugs, resolved_targets, explain_top_n
            )
        return df

    def _add_explanations(self, df, smiles_list, sequence_list, drug_ids, target_ids, top_n):
        """Add SHAP explanations for top N hits."""
        top_rows = df[df["predicted_score"].notna()].head(top_n)

        smiles_dict = dict(zip(drug_ids, smiles_list))
        seq_dict = dict(zip(target_ids, sequence_list))

        df["shap_explanation"] = pd.Series([None] * len(df), index=df.index, dtype=object)
        for index, row in top_rows.iterrows():
            try:
                smiles = smiles_dict.get(row["drug_id"], "")
                seq = seq_dict.get(row["target_id"], "")
                result = self.pipeline.predict(smiles, seq, explain=True)
                df.at[index, "shap_explanation"] = result.get("shap_values", {})
            except Exception:
                df.at[index, "shap_explanation"] = {}
        return df

    def get_top_hits(self, results_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Get top N strongest predicted binders."""
        return results_df[results_df["predicted_score"].notna()].head(n)

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
            "top_5_hits": valid.head(5)[
                ["drug_id", "target_id", "predicted_score", "binding_class"]
            ].to_dict("records"),
        }
