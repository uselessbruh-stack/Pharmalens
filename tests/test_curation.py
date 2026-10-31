import json

import pandas as pd
import pytest

from src.data.acquisition import atomic_json, digest_file
from src.data.curation import combine
from src.data.preparation import normalize_observation


def source(tmp_path, name, record_id, value=10):
    folder = tmp_path / name
    folder.mkdir()
    row = normalize_observation(
        {
            "source": "BindingDB",
            "source_record_id": record_id,
            "smiles": "CCO",
            "sequence": "ACDEFGHIKLMNPQRSTVWY",
            "endpoint": "Ki",
            "original_value": value,
            "original_unit": "nM",
        }
    )
    path = folder / "Ki.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    atomic_json(
        folder / "manifest.json",
        {"outputs": {"Ki": {"path": path.name, "sha256": digest_file(path)}}},
    )
    return path


def test_overlapping_sources_are_reported_without_duplicate_weight(tmp_path):
    a, b = source(tmp_path, "a", "one"), source(tmp_path, "b", "one")
    report = combine([a, b], tmp_path / "combined", "Ki")
    assert report["observations"] == 1
    assert report["duplicate_source_rows_removed"] == 1
    assert report["source_pair_overlap"][0]["shared_pair_identities"] == 1
    assert not report["independent_external_validation_claim"]


def test_disagreeing_source_ids_are_quarantined_not_arbitrarily_selected(tmp_path):
    a, b, c = (
        source(tmp_path, "a", "one", 10),
        source(tmp_path, "b", "one", 100),
        source(tmp_path, "c", "two", 10),
    )
    report = combine([a, b, c], tmp_path / "combined", "Ki")
    assert report["conflicting_source_ids"] == 1
    assert report["quarantined_conflicting_rows"] == 2
    assert report["observations"] == 1


def test_unknown_families_are_not_called_non_kinase(tmp_path):
    path = source(tmp_path, "a", "one")
    with pytest.raises(ValueError, match="No observations"):
        combine([path], tmp_path / "nonkinase", "Ki", non_kinase_only=True)


def test_family_mapping_requires_verified_sequence_and_source_hash(tmp_path):
    path = source(tmp_path, "a", "one")
    reference = tmp_path / "reference"
    reference.mkdir()
    observations = reference / "observations.jsonl"
    observations.write_text(
        json.dumps({"family": "gpcr", "sequence": "ACDEFGHIKLMNPQRSTVWY"}) + "\n"
    )
    atomic_json(
        reference / "manifest.json", {"source": "ChEMBL", "sha256": digest_file(observations)}
    )
    report = combine([path], tmp_path / "nonkinase", "Ki", observations, non_kinase_only=True)
    assert report["family_counts"] == {"gpcr": 1}
    observations.write_text("tampered")
    with pytest.raises(ValueError, match="receipt"):
        combine([path], tmp_path / "bad", "Ki", observations)
