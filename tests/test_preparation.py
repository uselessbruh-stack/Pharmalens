import csv
import json

import pytest

from src.data.acquisition import atomic_json, digest_file
from src.data.preparation import normalize_observation, prepare


def observation(**changes):
    record = {
        "source": "test",
        "source_record_id": "1",
        "smiles": "OCC",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "endpoint": "Ki",
        "original_value": "1",
        "original_unit": "nM",
        "relation": "=",
    }
    return {**record, **changes}


def test_molecule_aliases_have_identical_identity():
    a = normalize_observation(observation(smiles="OCC"))
    b = normalize_observation(observation(smiles="CCO", source_record_id="2"))
    assert a["drug_id"] == b["drug_id"]
    assert a["observation_id"] != b["observation_id"]
    assert a["score"] == 9


def test_nonkinase_admission_requires_verified_reference(tmp_path):
    with pytest.raises(ValueError, match="verified family reference"):
        prepare(tmp_path / "unused.zip", tmp_path / "output", "bindingdb", non_kinase_only=True)


@pytest.mark.parametrize(
    "changes", [{"sequence": "ACDEFGHIKX"}, {"original_value": ">100"}, {"endpoint": "EC50"}]
)
def test_invalid_or_incompatible_observations_are_rejected(changes):
    with pytest.raises(ValueError):
        normalize_observation(observation(**changes))


def test_preparation_has_exclusion_ledger_and_endpoint_contract(tmp_path):
    raw = tmp_path / "observations.jsonl"
    base = {
        "family": "gpcr",
        "accession": "P14416",
        "sequence": "ACDEFGHIKLMNPQRSTVWY",
        "molecule_type": "Small molecule",
        "assay": {"confidence_score": 9},
        "activity": {
            "activity_id": 1,
            "canonical_smiles": "CCO",
            "standard_type": "Ki",
            "standard_value": "10",
            "standard_units": "nM",
            "standard_relation": "=",
            "assay_type": "B",
        },
    }
    raw.write_text(
        json.dumps(base) + "\n" + json.dumps({**base, "molecule_type": "Antibody"}) + "\n",
        encoding="utf-8",
    )
    atomic_json(tmp_path / "manifest.json", {"sha256": digest_file(raw), "status": "sample"})
    out = tmp_path / "prepared"
    prepare(raw, out, "chembl")
    report = json.loads((out / "manifest.json").read_text())
    assert report["counts"]["Ki"] == 1
    assert report["excluded"]["not_small_molecule"] == 1
    assert report["source_manifest"]["status"] == "sample"
    with (out / "Ki.csv").open() as handle:
        assert float(next(csv.DictReader(handle))["score"]) == 8
    with pytest.raises(ValueError, match="already exists"):
        prepare(raw, out, "chembl")


def test_actual_chembl_variant_field_is_excluded(tmp_path):
    from src.data.preparation import chembl_records

    raw = tmp_path / "variant.jsonl"
    raw.write_text(
        json.dumps(
            {
                "family": "gpcr",
                "accession": "P14416",
                "sequence": "ACDEFGHIKLMNPQRSTVWY",
                "molecule_type": "Small molecule",
                "assay": {"confidence_score": 9, "variant_sequence": {"mutation": "A1G"}},
                "activity": {"activity_id": 1, "assay_type": "B"},
            }
        )
        + "\n"
    )
    assert next(chembl_records(raw))[1] == "unresolved_variant_sequence"
