"""Acquire a release-pinned catalog of human non-kinase target sequences."""

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import (
    CHEMBL,
    PANELS,
    ChemblClient,
    atomic_json,
    classify_component,
    digest_file,
    request,
    utc_now,
)


def discover(output, release="ChEMBL_37"):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new catalog version")
    with request(CHEMBL + "status.json") as response:
        if json.load(response)["chembl_db_version"] != release:
            raise ValueError("ChEMBL release mismatch")
    client = ChemblClient(output.parent / release / "catalog_responses", release)
    classes = {
        row["protein_class_id"]: row
        for page in client.pages("protein_classification.json", "protein_classifications")
        for row in page
    }
    staging = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    counts, excluded, panels = Counter(), Counter(), {key: [] for key in PANELS}
    records = {}
    for page in client.pages(
        "target_component.json",
        "target_components",
        organism="Homo sapiens",
        only="accession,component_id,organism,protein_classifications,sequence",
        limit=1000,
    ):
        for row in page:
            if row.get("organism") != "Homo sapiens":
                raise ValueError("API did not honor human-organism filter")
            family = classify_component(row, classes)
            if family not in PANELS or not row.get("accession") or not row.get("sequence"):
                excluded[family] += 1
                continue
            record = {
                "accession": row["accession"],
                "sequence": row["sequence"],
                "family": family,
                "component_id": row["component_id"],
            }
            key = (record["accession"], record["sequence"], family)
            records[key] = record
    with request(CHEMBL + "status.json") as response:
        if json.load(response)["chembl_db_version"] != release:
            raise ValueError("ChEMBL release changed during acquisition")
    if not records:
        raise ValueError("No family-classified sequences returned")
    data = staging / "observations.jsonl"
    with data.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            record = records[key]
            handle.write(json.dumps(record) + "\n")
            counts[record["family"]] += 1
            panels[record["family"]].append(record["accession"])
    # Stable accession ordering, independent of activity labels/model results.
    panels = {
        family: list(dict.fromkeys([*PANELS[family], *sorted(set(values))]))
        for family, values in panels.items()
    }
    atomic_json(staging / "panels.json", panels)
    atomic_json(
        staging / "manifest.json",
        {
            "source": "ChEMBL",
            "release": release,
            "created_at": utc_now(),
            "status": "human_family_reference_catalog",
            "sha256": digest_file(data),
            "sequences_by_family": dict(counts),
            "excluded_components": dict(excluded),
            "selection": "all returned human components matching verified family ancestry; no activity/model selection",
            "requests": list(client.used.values()),
            "license": "ChEMBL CC BY-SA 3.0; preserve release/source attribution",
            "limitations": "Sequence annotations do not establish assay independence or resolve constructs",
        },
    )
    staging.replace(output)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default="ChEMBL_37")
    args = parser.parse_args()
    print(json.dumps(discover(args.output, args.release), indent=2))
