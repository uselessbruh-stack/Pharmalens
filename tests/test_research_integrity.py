import json

import numpy as np
import pandas as pd
import pytest

from src.data.labels import concentration_to_p
from src.evaluation.metrics import compute_regression_metrics
from src.evaluation.split_strategies import get_split, validate_split_integrity
from src.utils.serialization import json_safe


@pytest.mark.parametrize("strategy", ["scaffold", "family"])
def test_domain_holdouts_keep_entire_scaffolds_or_families_together(strategy):
    from src.evaluation.split_strategies import scaffold_identity

    smiles = ["c1ccccc1", "Oc1ccccc1", "C1CCCCC1", "c1ccncc1", "C1CCCC1", "C1CCC1", "C1CC1", "CCC"]
    rows = [
        {
            "drug_id": str(i),
            "target_id": str(j),
            "smiles": smi,
            "sequence": f"protein{j}",
            "family": f"family{j}",
            "score": i + j,
        }
        for i, smi in enumerate(smiles)
        for j in range(6)
    ]
    parts = get_split(pd.DataFrame(rows), strategy, seed=42)
    validate_split_integrity(*parts, strategy)
    assert sum(map(len, parts)) == len(rows)
    groups = [
        set(p.smiles.map(scaffold_identity)) if strategy == "scaffold" else set(p.family)
        for p in parts
    ]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


@pytest.fixture
def interactions():
    rows = [
        {
            "drug_id": f"d{i}",
            "target_id": f"t{j}",
            "smiles": f"mol{i}",
            "sequence": f"protein{j}",
            "score": i + j,
        }
        for i in range(20)
        for j in range(20)
    ]
    # Source IDs differ but molecule identity and observations are the same.
    rows.extend({**r, "drug_id": "alias" + r["drug_id"]} for r in rows[:30])
    return pd.DataFrame(rows)


@pytest.mark.parametrize("strategy", ["random", "cold_drug", "cold_target", "cold_both"])
def test_splits_keep_aliases_and_replicates_together(interactions, strategy):
    parts = get_split(interactions, strategy, seed=42)
    validate_split_integrity(*parts, strategy)
    if strategy == "cold_both":
        dropped = parts[0].attrs["excluded_cross_block_rows"]
        assert dropped > 0
        assert sum(map(len, parts)) + dropped == len(interactions)
    else:
        assert sum(map(len, parts)) == len(interactions)
    repeated = get_split(interactions.sample(frac=1, random_state=4), strategy, seed=42)
    assert set(zip(parts[2].smiles, parts[2].sequence)) == set(
        zip(repeated[2].smiles, repeated[2].sequence)
    )


def test_split_integrity_rejects_leakage(interactions):
    with pytest.raises(ValueError, match="leakage"):
        validate_split_integrity(interactions, interactions, interactions)


@pytest.mark.parametrize("strategy", ["random", "cold_drug", "cold_target", "cold_both"])
def test_invalid_split_ratios_are_rejected(interactions, strategy):
    with pytest.raises(ValueError):
        get_split(interactions, strategy, train_ratio=0.8, val_ratio=-0.1, test_ratio=0.3)


@pytest.mark.parametrize(
    "value,unit,expected", [(1, "nM", 9), (100, "nM", 7), (1, "uM", 6), (1e-9, "M", 9)]
)
def test_explicit_concentration_units(value, unit, expected):
    assert concentration_to_p(value, unit) == pytest.approx(expected)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_concentrations_are_not_clipped(value):
    with pytest.raises(ValueError):
        concentration_to_p(value)


def test_censored_measurement_is_not_an_exact_label():
    with pytest.raises(ValueError, match="Censored"):
        concentration_to_p(10, relation=">")


@pytest.mark.parametrize("prediction", [[0, np.nan], [0, np.inf], []])
def test_metrics_do_not_silently_exclude_bad_predictions(prediction):
    with pytest.raises(ValueError):
        compute_regression_metrics([1, 2], prediction)


def test_undefined_statistics_serialize_as_strict_null():
    metrics = compute_regression_metrics([1, 2], [1.5, 1.5])
    serialized = json.dumps(json_safe(metrics), allow_nan=False)
    assert json.loads(serialized)["pearson"] is None
    assert metrics["rmse"] == 0.5


def test_saved_metrics_are_atomic_strict_json(tmp_path):
    from src.utils.config import save_json

    path = tmp_path / "metrics.json"
    save_json({"values": np.array([1.0, np.nan]), "path": tmp_path}, path)
    document = json.loads(path.read_text())
    assert document["values"] == [1.0, None]
