import pytest
import torch

from src.inference.pipeline import PharmaLensPipeline
from src.models.deeptda import DeepDTA
from src.models.gnn_model import GraphDTA
from src.utils.serialization import json_safe

SEQUENCE = "ACDEFGHIKLMNPQRSTVWY"


@pytest.mark.parametrize("model_type", ["deeptda", "gnn"])
def test_neural_checkpoint_roundtrip_uses_saved_config(tmp_path, model_type):
    if model_type == "deeptda":
        model = DeepDTA(
            drug_max_len=35,
            protein_max_len=45,
            drug_embed_dim=8,
            protein_embed_dim=8,
            num_filters=4,
            fc_dims=[8],
        )
    else:
        model = GraphDTA(
            gnn_hidden_dim=8,
            gnn_output_dim=8,
            protein_max_len=45,
            protein_embed_dim=8,
            protein_num_filters=4,
            fc_dims=[8],
        )
    model.eval()
    before = PharmaLensPipeline(model, model_type, "Ki").predict("CCO", SEQUENCE)
    path = tmp_path / "model.pt"
    torch.save({"config": model.config, "model_state_dict": model.state_dict()}, path)
    after = PharmaLensPipeline.from_pretrained(path, model_type, "Ki").predict("CCO", SEQUENCE)
    assert after["score"] == pytest.approx(before["score"])
    assert after["label"]["label_name"] == "pKi"


def test_screening_direction_partial_failure_and_input_lengths():
    pipeline = PharmaLensPipeline(None)

    def predict(smiles, sequence):
        if smiles == "bad":
            raise ValueError("invalid molecule")
        return {"score": float(smiles), "inference_time": 0.1}

    pipeline.predict = predict
    rows = json_safe(pipeline.screen(["3", "9", "bad"], [SEQUENCE]).to_dict("records"))
    assert rows[0]["predicted_score"] == 9
    assert rows[-1]["predicted_score"] is None
    assert rows[-1]["rank"] is None
    with pytest.raises(ValueError, match="drug_ids"):
        pipeline.screen(["3"], [SEQUENCE], drug_ids=[])
    with pytest.raises(ValueError):
        pipeline.screen([], [])


def test_screening_explanations_do_not_explain_failed_rows():
    from src.screening.screener import DrugScreener

    pipeline = PharmaLensPipeline(None)

    def predict(smiles, sequence, explain=False):
        if smiles == "bad":
            raise ValueError("invalid")
        return {"score": float(smiles), "inference_time": 0.1, "shap_values": {"base": 1}}

    pipeline.predict = predict
    screener = DrugScreener(pipeline)
    rows = screener.screen_panel(["3", "9", "bad"], [SEQUENCE], explain_top_n=5)
    assert rows.iloc[0]["shap_explanation"] == {"base": 1}
    assert rows.iloc[-1]["shap_explanation"] is None
    assert len(screener.get_top_hits(rows, 5)) == 2


def test_real_xgboost_shap_reconstructs_served_prediction():
    import numpy as np

    from src.features.drug_features import get_drug_feature_vector
    from src.features.protein_features import get_protein_feature_vector
    from src.models.xgboost_model import XGBoostDTI

    smiles = ["C" * n for n in range(1, 13)]
    x = np.stack(
        [
            np.concatenate([get_drug_feature_vector(s), get_protein_feature_vector(SEQUENCE)])
            for s in smiles
        ]
    )
    model = XGBoostDTI(n_estimators=20, max_depth=2, min_child_weight=1, n_jobs=1)
    model.fit(x, np.arange(len(x), dtype=float), verbose=False)
    result = PharmaLensPipeline(
        model, endpoint="Ki", preprocessing_version="standardized_v1"
    ).predict("CCCC", SEQUENCE, explain=True)
    explanation = result["shap_values"]
    reconstructed = (
        explanation["base_value"]
        + sum(row["shap_value"] for row in explanation["top_features"])
        + explanation["other_features_contribution"]
    )
    assert reconstructed == pytest.approx(result["score"], abs=1e-5)
