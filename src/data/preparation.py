"""Normalize verified raw affinity observations without conflating endpoints."""

from __future__ import annotations

import argparse
import csv
import io
import json
import uuid
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

from src.data.acquisition import atomic_json, digest_file, utc_now
from src.data.labels import CONTRACTS, concentration_to_p

FIELDS = [
    "observation_id",
    "source",
    "source_record_id",
    "drug_id",
    "target_id",
    "smiles",
    "sequence",
    "accession",
    "family",
    "endpoint",
    "score",
    "original_value",
    "original_unit",
    "relation",
    "assay_id",
    "publication_id",
    "organism",
]
AA = set("ACDEFGHIKLMNPQRSTVWY")


def stable_id(value):
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


@lru_cache(maxsize=100_000)
def clean_molecule(smiles):
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    if not smiles:
        raise ValueError("missing_smiles")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumHeavyAtoms() == 0:
        raise ValueError("invalid_smiles")
    # Preserve stereochemistry and charge; select the largest organic fragment.
    # Tautomers are NOT silently collapsed; the policy is versioned in manifest.
    molecule = rdMolStandardize.LargestFragmentChooser(preferOrganic=True).choose(molecule)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def normalize_observation(record):
    sequence = "".join(record["sequence"].split()).upper()
    if not 10 <= len(sequence) <= 10000 or not set(sequence) <= AA:
        raise ValueError("invalid_or_ambiguous_sequence")
    endpoint = record["endpoint"]
    if endpoint not in {"Ki", "Kd"}:
        raise ValueError("unsupported_endpoint")
    raw = str(record["original_value"]).strip()
    if record.get("relation", "=") != "=" or raw.startswith(("<", ">", "~")):
        raise ValueError("censored_measurement")
    try:
        score = concentration_to_p(raw, record["original_unit"])
    except (ValueError, TypeError):
        raise ValueError("invalid_concentration") from None
    smiles = clean_molecule(record["smiles"])
    result = {key: record.get(key, "") for key in FIELDS}
    result.update(
        smiles=smiles,
        sequence=sequence,
        drug_id=stable_id(smiles),
        target_id=stable_id(sequence),
        score=score,
        relation="=",
    )
    result["observation_id"] = stable_id(
        f"{record['source']}:{record['source_record_id']}:{endpoint}"
    )
    return result


def chembl_records(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            observation = json.loads(line)
            activity = observation["activity"]
            assay = observation.get("assay") or {}
            record = {
                "source": "ChEMBL",
                "source_record_id": str(activity["activity_id"]),
                "smiles": activity.get("canonical_smiles"),
                "sequence": observation["sequence"],
                "endpoint": activity.get("standard_type"),
                "original_value": activity.get("standard_value"),
                "original_unit": activity.get("standard_units"),
                "relation": activity.get("standard_relation"),
                "accession": observation["accession"],
                "family": observation["family"],
                "assay_id": activity.get("assay_chembl_id"),
                "publication_id": activity.get("document_chembl_id"),
                "organism": activity.get("target_organism"),
            }
            reason = None
            if not observation.get("assay"):
                reason = "missing_assay_metadata"
            elif activity.get("assay_type") != "B" or assay.get("confidence_score") != 9:
                reason = "not_high_confidence_binding"
            elif observation.get("molecule_type") != "Small molecule":
                reason = "not_small_molecule"
            elif activity.get("potential_duplicate"):
                reason = "source_duplicate_flag"
            elif activity.get("data_validity_comment") not in (None, "", "Manually validated"):
                reason = "source_validity_flag"
            elif activity.get("assay_variant_mutation") or assay.get("variant_sequence"):
                reason = "unresolved_variant_sequence"
            yield record, reason


def bindingdb_records(path):
    csv.field_size_limit(10_000_000)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".tsv")]
        if len(members) != 1:
            raise ValueError("Expected one BindingDB TSV member; no files are extracted")
        with io.TextIOWrapper(archive.open(members[0]), encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "Ligand SMILES",
                "Ki (nM)",
                "Kd (nM)",
                "BindingDB Target Chain Sequence 1",
                "BindingDB Reactant_set_id",
            }
            if not required <= set(reader.fieldnames or []):
                raise ValueError("BindingDB schema changed: required columns are missing")
            for row in reader:
                for endpoint in ("Ki", "Kd"):
                    raw = (row.get(f"{endpoint} (nM)") or "").strip()
                    if not raw:
                        continue
                    record = {
                        "source": "BindingDB",
                        "source_record_id": row["BindingDB Reactant_set_id"],
                        "smiles": row["Ligand SMILES"],
                        "sequence": row["BindingDB Target Chain Sequence 1"],
                        "endpoint": endpoint,
                        "original_value": raw,
                        "original_unit": "nM",
                        "relation": "=",
                        "accession": row.get(
                            "UniProt (SwissProt) Primary ID of Target Chain 1", ""
                        ),
                        "family": "unclassified",
                        "publication_id": row.get("Article DOI", ""),
                        "assay_id": "",  # Main TSV does not resolve assay context; never fabricate it.
                        "organism": row.get(
                            "Target Source Organism According to Curator or DataSource", ""
                        ),
                    }
                    chain_count = row.get(
                        "Number of Protein Chains in Target (>1 implies a multichain complex)"
                    )
                    reason = None if chain_count == "1" else "unresolved_or_multichain_target"
                    yield record, reason


def prepare(input_path, output_dir, source, family_reference=None, non_kinase_only=False):
    if non_kinase_only and family_reference is None:
        raise ValueError("Non-kinase filtering requires a verified family reference")
    mapping = {}
    if family_reference is not None:
        from src.data.curation import verified_family_map

        mapping = verified_family_map(family_reference)
    input_path, output_dir = Path(input_path), Path(output_dir)
    manifest_path = (
        input_path.with_suffix(input_path.suffix + ".manifest.json")
        if source == "bindingdb"
        else input_path.parent / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sha256") != digest_file(input_path):
        raise ValueError("Raw file checksum does not match acquisition manifest")
    if source == "bindingdb" and manifest.get("status") != "verified":
        raise ValueError("BindingDB archive is not verified")
    if output_dir.exists():
        raise ValueError(
            "Output directory already exists; choose a new version to preserve provenance"
        )
    # Publish the whole prepared dataset atomically, including its report.
    staging = output_dir.with_name(output_dir.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    counts, exclusions, families = Counter(), Counter(), Counter()
    seen = set()
    try:
        with (
            (staging / "Ki.csv").open("w", newline="", encoding="utf-8") as ki,
            (staging / "Kd.csv").open("w", newline="", encoding="utf-8") as kd,
            (staging / "excluded.jsonl").open("w", encoding="utf-8") as rejected,
        ):
            writers = {"Ki": csv.DictWriter(ki, FIELDS), "Kd": csv.DictWriter(kd, FIELDS)}
            for writer in writers.values():
                writer.writeheader()
            iterator = (
                bindingdb_records(input_path)
                if source == "bindingdb"
                else chembl_records(input_path)
            )
            for record, reason in iterator:
                counts["source_endpoint_observations"] += 1
                if mapping:
                    key = stable_id("".join(record["sequence"].split()).upper())
                    record["family"] = mapping.get(key, "unclassified")
                    if non_kinase_only and record["family"] == "unclassified":
                        reason = "unresolved_nonkinase_family"
                try:
                    if reason:
                        raise ValueError(reason)
                    normalized = normalize_observation(record)
                    if normalized["observation_id"] in seen:
                        raise ValueError("duplicate_source_record")
                    seen.add(normalized["observation_id"])
                    writers[normalized["endpoint"]].writerow(normalized)
                    counts[normalized["endpoint"]] += 1
                    families[normalized["family"]] += 1
                except ValueError as exc:
                    reason = str(exc)
                    exclusions[reason] += 1
                    rejected.write(
                        json.dumps(
                            {
                                "source_record_id": record["source_record_id"],
                                "endpoint": record["endpoint"],
                                "reason": reason,
                            }
                        )
                        + "\n"
                    )
                if counts["source_endpoint_observations"] % 10000 == 0:
                    atomic_json(
                        staging / "progress.json",
                        {
                            "status": "preparing",
                            "counts": dict(counts),
                            "excluded": dict(exclusions),
                        },
                    )
        if not counts["Ki"] + counts["Kd"]:
            raise ValueError("No admissible exact Ki/Kd observations; see source/quality criteria")
        atomic_json(
            staging / "manifest.json",
            {
                "source_file": str(input_path.resolve()),
                "source_sha256": manifest["sha256"],
                "source_manifest": manifest,
                "prepared_at": utc_now(),
                "schema_version": 1,
                "counts": dict(counts),
                "excluded": dict(exclusions),
                "family_counts": dict(families),
                "family_reference_sha256": digest_file(family_reference)
                if family_reference
                else None,
                "non_kinase_only": non_kinase_only,
                "labels": {endpoint: CONTRACTS[endpoint].as_dict() for endpoint in ("Ki", "Kd")},
                "preprocessing": "largest_organic_fragment_preserve_charge_stereochemistry_v1; exact_standard_AA_sequences",
                "preprocessing_version": "standardized_v1",
                "replicates": "retained_with_source_context; identity_groups_must_be_split_together",
                "limitations": [
                    "BindingDB family unresolved until curated accession mapping",
                    "BindingDB assay mapping and cross-source overlap checks required before final research claims",
                ],
                "outputs": {
                    endpoint: {
                        "path": f"{endpoint}.csv",
                        "sha256": digest_file(staging / f"{endpoint}.csv"),
                    }
                    for endpoint in ("Ki", "Kd")
                },
            },
        )
        atomic_json(
            staging / "progress.json",
            {"status": "complete", "counts": dict(counts), "excluded": dict(exclusions)},
        )
        staging.replace(output_dir)
    except Exception:
        # Keep partial diagnostics under an explicit .part name; never publish.
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source", required=True, choices=["bindingdb", "chembl"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--family-reference", type=Path)
    parser.add_argument("--non-kinase-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(
            prepare(
                args.input, args.output, args.source, args.family_reference, args.non_kinase_only
            )
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
