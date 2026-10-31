"""Combine endpoint-specific observations with an explicit source-overlap ledger."""

import argparse
import json
import uuid
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.data.acquisition import atomic_json, digest_file, utc_now
from src.data.preparation import stable_id

FAMILIES = {"gpcr", "protease", "nuclear_receptor", "ion_channel", "transporter", "other_enzyme"}


def verified_family_map(observations):
    observations = Path(observations)
    manifest = json.loads((observations.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("source") != "ChEMBL" or digest_file(observations) != manifest.get("sha256"):
        raise ValueError("Family reference does not match its ChEMBL acquisition receipt")
    mapping = {}
    with observations.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            family = record["family"]
            if family not in FAMILIES:
                raise ValueError("Unexpected family in reference catalog")
            mapping.setdefault(stable_id(record["sequence"]), set()).add(family)
    return {key: next(iter(value)) for key, value in mapping.items() if len(value) == 1}


def combine(inputs, output, endpoint, family_reference=None, non_kinase_only=False):
    if endpoint not in {"Ki", "Kd"}:
        raise ValueError("Combined affinity tasks must specify Ki or Kd")
    output = Path(output)
    if output.exists():
        raise ValueError("Dataset version already exists; choose a new output")
    required = {
        "observation_id",
        "drug_id",
        "target_id",
        "smiles",
        "sequence",
        "endpoint",
        "score",
        "source",
    }
    frames, evidence, identities = [], [], {}
    for index, raw in enumerate(inputs):
        path = Path(raw)
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        spec = manifest["outputs"][endpoint]
        if spec["path"] != path.name or spec["sha256"] != digest_file(path):
            raise ValueError("Prepared source checksum or endpoint contract failed")
        frame = pd.read_csv(path, keep_default_na=False)
        if not required <= set(frame) or (not frame.empty and set(frame.endpoint) != {endpoint}):
            raise ValueError("Prepared input schema or endpoint mismatch")
        source_id = f"{index}:{path.parent.name}"
        frame["source_dataset"] = source_id
        frame["pair_id"] = [
            stable_id(f"{smiles}\0{sequence}\0{endpoint}")
            for smiles, sequence in frame[["smiles", "sequence"]].itertuples(index=False, name=None)
        ]
        identities[source_id] = set(frame.pair_id)
        frames.append(frame)
        evidence.append(
            {
                "id": source_id,
                "input": str(path.resolve()),
                "sha256": spec["sha256"],
                "rows": len(frame),
            }
        )
    if not frames:
        raise ValueError("At least one prepared source is required")
    frame = pd.concat(frames, ignore_index=True)
    # Reused source record IDs with disagreeing normalized payloads are all
    # quarantined, rather than keeping whichever source happened to come first.
    signatures = frame[
        ["observation_id", "smiles", "sequence", "score", "endpoint"]
    ].drop_duplicates()
    conflicts = signatures.groupby("observation_id").size()
    conflict_ids = set(conflicts[conflicts > 1].index)
    conflicting = frame[frame.observation_id.isin(conflict_ids)].copy()
    retained = frame[~frame.observation_id.isin(conflict_ids)].copy()
    duplicates = retained[retained.observation_id.duplicated(keep="first")].copy()
    retained = retained.drop_duplicates("observation_id").copy()
    for row in retained.itertuples():
        if stable_id(row.smiles) != row.drug_id or stable_id(row.sequence) != row.target_id:
            raise ValueError("Prepared identity hashes do not match molecule/sequence values")
    mapping = verified_family_map(family_reference) if family_reference else {}
    retained["family"] = retained.target_id.map(lambda key: mapping.get(key, "unclassified"))
    retained["family_annotation_basis"] = retained.target_id.map(
        lambda key: "exact_verified_reference_sequence" if key in mapping else "unresolved"
    )
    excluded_family = int((retained.family == "unclassified").sum()) if non_kinase_only else 0
    if non_kinase_only:
        retained = retained[retained.family.isin(FAMILIES)].copy()
    if retained.empty:
        raise ValueError("No observations survive the requested source/family criteria")
    staging = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    retained.to_csv(staging / f"{endpoint}.csv", index=False)
    conflicting.to_csv(staging / "conflicting_source_records.csv", index=False)
    duplicates.to_csv(staging / "duplicate_source_records.csv", index=False)
    overlap = [
        {
            "left": left,
            "right": right,
            "shared_pair_identities": len(identities[left] & identities[right]),
        }
        for left, right in combinations(identities, 2)
    ]
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "endpoint": endpoint,
        "sources": evidence,
        "observations": len(retained),
        "unique_pairs": retained.pair_id.nunique(),
        "conflicting_source_ids": len(conflict_ids),
        "quarantined_conflicting_rows": len(conflicting),
        "duplicate_source_rows_removed": len(duplicates),
        "source_pair_overlap": overlap,
        "family_counts": retained.family.value_counts().to_dict(),
        "unknown_family_rows_excluded": excluded_family,
        "non_kinase_only": non_kinase_only,
        "family_reference_sha256": digest_file(family_reference) if family_reference else None,
        "outputs": {
            endpoint: {
                "path": f"{endpoint}.csv",
                "sha256": digest_file(staging / f"{endpoint}.csv"),
            }
        },
        "replicates": "Different source observations retained; pair identities must remain grouped in splits",
        "independent_external_validation_claim": False,
        "limitations": [
            "Exact pair overlap does not resolve assay/publication reuse or sequence/scaffold similarity",
            "Unknown families are not assumed to be non-kinase",
            "Assay-heterogeneous replicate observations are not silently averaged",
        ],
    }
    atomic_json(staging / "manifest.json", manifest)
    staging.replace(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--endpoint", choices=["Ki", "Kd"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family-reference", type=Path)
    parser.add_argument("--non-kinase-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = combine(
            args.inputs, args.output, args.endpoint, args.family_reference, args.non_kinase_only
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Curation failed: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0
