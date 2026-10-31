"""Publish endpoint/family views without claiming independent data sources."""

import argparse
import json
import sys
import uuid
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import atomic_json, digest_file
from src.data.curation import FAMILIES


def publish_views(source, output, min_rows=500, min_drugs=100, min_targets=3):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError("Choose a new cohort version")
    manifest = json.loads((source / "manifest.json").read_text())
    staging = output.with_name(output.name + f".{uuid.uuid4().hex}.part")
    staging.mkdir(parents=True)
    cohorts = []
    for endpoint in ("Ki", "Kd"):
        if endpoint not in manifest["outputs"]:
            continue
        spec = manifest["outputs"][endpoint]
        data = (source / spec["path"]).resolve()
        if not data.is_relative_to(source.resolve()) or digest_file(data) != spec["sha256"]:
            raise ValueError("Source endpoint path/checksum mismatch")
        frame = pd.read_csv(data, keep_default_na=False)
        for family in sorted(FAMILIES):
            rows = frame[frame.family == family]
            statistics = {
                "rows": len(rows),
                "drugs": rows.drug_id.nunique(),
                "proteins": rows.target_id.nunique(),
                "pairs": len(rows[["drug_id", "target_id"]].drop_duplicates()),
            }
            supported = (
                statistics["rows"] >= min_rows
                and statistics["drugs"] >= min_drugs
                and statistics["proteins"] >= min_targets
            )
            entry = {
                "family": family,
                "endpoint": endpoint,
                **statistics,
                "training_supported": bool(supported),
            }
            if supported:
                name = f"{family}_{endpoint.lower()}"
                directory = staging / name
                directory.mkdir()
                rows.to_csv(directory / f"{endpoint}.csv", index=False)
                atomic_json(
                    directory / "manifest.json",
                    {
                        "endpoint": endpoint,
                        "family": family,
                        "non_kinase_only": True,
                        "counts": {endpoint: len(rows)},
                        "family_counts": {family: len(rows)},
                        "source_dataset": str(source.resolve()),
                        "source_manifest_sha256": digest_file(source / "manifest.json"),
                        "source_data_sha256": spec["sha256"],
                        "statistics": statistics,
                        "preprocessing_version": "standardized_v1",
                        "independent_dataset_claim": False,
                        "outputs": {
                            endpoint: {
                                "path": f"{endpoint}.csv",
                                "sha256": digest_file(directory / f"{endpoint}.csv"),
                            }
                        },
                    },
                )
                entry["directory"] = name
            cohorts.append(entry)
    atomic_json(
        staging / "manifest.json",
        {
            "cohorts": cohorts,
            "min_rows": min_rows,
            "min_drugs": min_drugs,
            "min_targets": min_targets,
            "selection": "fixed coverage criteria; no model performance selection",
            "limitations": "Overlapping views over the same source, not independent external datasets",
        },
    )
    staging.replace(output)
    return cohorts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(publish_views(args.input, args.output, args.min_rows), indent=2))
