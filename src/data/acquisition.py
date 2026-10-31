"""Reproducible, dependency-free acquisition of public affinity datasets.

Raw files are immutable unless explicitly refreshed. No archive is extracted and
no downloaded Python/pickle content is executed. ChEMBL responses are cached and
checksummed, with a release guard and explicit partial-download metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

LOG = logging.getLogger("pharmalens.data.acquisition")
ROOT = Path(__file__).resolve().parents[2]
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/"
RELEASE = "202610"
SOURCES = {
    "bindingdb_articles": (
        f"BindingDB_BindingDB_Articles_{RELEASE}_tsv.zip",
        "4b1d2f7c90e83dc6fd13e7b106a1f545",
    ),
    "bindingdb_pdsp": (f"BindingDB_PDSPKi_{RELEASE}_tsv.zip", None),
    "bindingdb_all": (f"BindingDB_All_{RELEASE}_tsv.zip", "ab7ca3dc152d395b773ff987d8e09f64"),
}
# Deliberately selected research starter panel, not an exhaustive family sample.
# Every accession is verified against the downloaded ChEMBL family hierarchy.
PANELS = {
    "gpcr": ("P14416", "P28223", "P07550"),
    "protease": ("P00734", "P00742", "P07858"),
    "nuclear_receptor": ("P03372", "P10275", "P04150"),
    "ion_channel": ("Q12809", "Q14524", "Q13936"),
    "transporter": ("P31645", "Q01959", "P23975"),
    "other_enzyme": ("P00918", "P22303", "O76074"),
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest_file(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_retry(source, destination, attempts=8):
    """Retry transient Windows sharing locks without overwriting published directories."""
    source, destination = Path(source), Path(destination)
    directory = source.is_dir()
    if directory and source.resolve().parent != destination.resolve().parent:
        raise ValueError("Directory publication must stay within its intended parent")
    for attempt in range(attempts):
        if directory and destination.exists():
            raise FileExistsError("Published directory already exists")
        try:
            source.replace(destination)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(0.1 * 2**attempt, 1))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.part")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def request(url, timeout=60, attempts=4):
    if urlparse(url).scheme != "https":
        raise ValueError("Dataset sources must use HTTPS")
    for attempt in range(attempts):
        try:
            response = urlopen(
                Request(
                    url,
                    headers={
                        "User-Agent": "PharmaLens-Research/0.2 (+dataset-provenance)",
                        "Accept": "application/json, application/zip, */*",
                    },
                ),
                timeout=timeout,
            )
            if urlparse(response.geturl()).scheme != "https":
                response.close()
                raise ValueError("Refusing insecure download redirect")
            return response
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, HTTPError) and exc.code not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt + 1 == attempts:
                raise
            retry_after = getattr(exc, "headers", {}).get("Retry-After", "")
            delay = min(float(retry_after), 30) if retry_after.isdigit() else 2**attempt
            LOG.warning("Request failed (%s); retrying in %ss", type(exc).__name__, delay)
            time.sleep(delay)
    raise RuntimeError("Unreachable request state")


def download_archive(url, destination, expected_md5=None, max_bytes=2_000_000_000):
    """Download atomically; verify existing bytes instead of trusting existence."""
    destination = Path(destination)
    metadata_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists():
        if not metadata_path.exists():
            raise ValueError(
                f"Unverified existing file: {destination}; move it aside before retrying"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("url") != url or digest_file(destination) != metadata.get("sha256"):
            raise ValueError(
                f"Cached artifact failed provenance/integrity verification: {destination}"
            )
        if expected_md5 and digest_file(destination, "md5") != expected_md5:
            raise ValueError(f"Cached artifact failed publisher checksum: {destination}")
        return metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid.uuid4().hex}.part")
    try:
        with request(url) as response, temporary.open("wb") as handle:
            size = 0
            length = response.headers.get("Content-Length")
            if length and int(length) > max_bytes:
                raise ValueError("Download exceeds configured byte limit")
            for block in iter(lambda: response.read(1024 * 1024), b""):
                size += len(block)
                if size > max_bytes:
                    raise ValueError("Download exceeds configured byte limit")
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
            if length and size != int(length):
                raise ValueError("Truncated download: content length mismatch")
            final_url = response.geturl()
        if not size or not zipfile.is_zipfile(temporary):
            raise ValueError("Expected a ZIP archive; source may have returned a login/error page")
        actual_md5 = digest_file(temporary, "md5")
        if expected_md5 and actual_md5 != expected_md5:
            raise ValueError("Publisher MD5 mismatch; the source release may have changed")
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"Corrupted ZIP member: {bad}")
            members = [member.filename for member in archive.infolist()]
        metadata = {
            "url": url,
            "resolved_url": final_url,
            "downloaded_at": utc_now(),
            "bytes": size,
            "sha256": digest_file(temporary),
            "md5": actual_md5,
            "publisher_md5": expected_md5,
            "members": members,
            "status": "verified",
        }
        temporary.replace(destination)
        atomic_json(metadata_path, metadata)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


class ChemblClient:
    def __init__(self, cache_dir, release):
        self.cache_dir = Path(cache_dir)
        self.release = release
        self.used = {}
        self.entity_cache = {}

    def get(self, resource, **params):
        url = urljoin(CHEMBL, resource)
        if urlparse(url).netloc != urlparse(CHEMBL).netloc or not url.startswith(CHEMBL):
            raise ValueError("Unexpected ChEMBL pagination origin")
        if params:
            url += ("&" if "?" in url else "?") + urlencode(sorted(params.items()))
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        receipt = self.cache_dir / f"{key}.receipt.json"
        if path.exists():
            if not receipt.exists():
                raise ValueError(f"Missing cache receipt: {path}")
            record = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                record.get("release") != self.release
                or record.get("url") != url
                or digest_file(path) != record.get("sha256")
            ):
                raise ValueError(f"ChEMBL cache integrity failure: {path}")
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            with request(url) as response:
                document = json.load(response)
            atomic_json(path, document)
            record = {
                "url": url,
                "release": self.release,
                "retrieved_at": utc_now(),
                "sha256": digest_file(path),
            }
            atomic_json(receipt, record)
            time.sleep(0.1)
        self.used[key] = record
        return document

    def pages(self, resource, collection, **params):
        params.setdefault("limit", 1000)
        seen = set()
        while resource:
            if resource in seen:
                raise ValueError("ChEMBL pagination loop detected")
            seen.add(resource)
            document = self.get(resource, **params)
            if collection not in document:
                raise ValueError(f"Missing collection {collection} in ChEMBL response")
            yield document[collection]
            next_url = document.get("page_meta", {}).get("next")
            resource = urljoin("https://www.ebi.ac.uk", next_url) if next_url else None
            params = {}

    def entities(self, resource, collection, identity_field, identities, fields):
        """Reuse verified entity metadata across pages, targets and reruns."""
        if resource not in self.entity_cache:
            cache = {}
            for receipt in self.cache_dir.glob("*.receipt.json"):
                record = json.loads(receipt.read_text(encoding="utf-8"))
                if urlparse(record.get("url", "")).path.rsplit("/", 1)[-1] != resource:
                    continue
                if record.get("release") != self.release:
                    continue
                path = receipt.with_name(receipt.name.replace(".receipt.json", ".json"))
                if not path.exists() or digest_file(path) != record.get("sha256"):
                    raise ValueError(f"Cached metadata failed integrity verification: {path}")
                document = json.loads(path.read_text(encoding="utf-8"))
                for row in document.get(collection, []):
                    # Reuse only entries containing the fields that matter for
                    # admission. Optional/null variant data can be absent.
                    if set(fields) <= set(row):
                        cache[row[identity_field]] = (
                            row,
                            receipt.name.removesuffix(".receipt.json"),
                            record,
                        )
            self.entity_cache[resource] = cache
        cache = self.entity_cache[resource]
        missing = sorted(set(identities) - set(cache))
        for start in range(0, len(missing), 100):
            params = {
                f"{identity_field}__in": ",".join(missing[start : start + 100]),
                "limit": 1000,
                "only": ",".join(fields),
            }
            document = self.get(resource, **params)
            # get() added the exact request receipt to this run's provenance.
            for row in document[collection]:
                cache[row[identity_field]] = (row, None, None)
        result = {}
        for identity in identities:
            if identity in cache:
                row, key, receipt = cache[identity]
                if key:
                    self.used[key] = receipt
                result[identity] = row
        return result


def classify_component(component, classes):
    names = set()
    for link in component.get("protein_classifications", []):
        current = link.get("protein_classification_id")
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            item = classes.get(current)
            if item is None:
                raise ValueError(f"Unknown protein classification ID: {current}")
            names.add(item["pref_name"].lower())
            current = item.get("parent_id")
    if "kinase" in names:
        return "kinase"
    if any("g protein-coupled receptor" in name for name in names):
        return "gpcr"
    for family, label in [
        ("protease", "protease"),
        ("nuclear_receptor", "nuclear receptor"),
        ("ion_channel", "ion channel"),
        ("transporter", "transporter"),
    ]:
        if label in names:
            return family
    return "other_enzyme" if "enzyme" in names else "unclassified"


def iter_chembl_observations(
    client, classes, families, max_targets, max_activities, exclusions, panels=None
):
    panels = PANELS if panels is None else panels
    for family in families:
        accessions = panels[family][: max_targets or None]
        for accession in accessions:
            components = client.get("target_component.json", accession=accession)[
                "target_components"
            ]
            eligible = [
                c
                for c in components
                if c.get("organism") == "Homo sapiens" and classify_component(c, classes) == family
            ]
            if len(eligible) != 1:
                raise ValueError(
                    f"Expected one verified human {family} component for {accession}, got {len(eligible)}"
                )
            component = eligible[0]
            component_url = f"target_component/{component['component_id']}.json"
            component = client.get(component_url)
            target_ids = ",".join(ref["target_chembl_id"] for ref in component["targets"])
            target_rows = client.get(
                "target.json",
                target_chembl_id__in=target_ids,
                target_type="SINGLE PROTEIN",
                limit=1000,
            )["targets"]
            if not target_rows:
                exclusions.append({"accession": accession, "reason": "no_single_protein_target"})
            for target in target_rows:
                target_id = target["target_chembl_id"]
                if (
                    target.get("target_type") != "SINGLE PROTEIN"
                    or len(target.get("target_components", [])) != 1
                ):
                    exclusions.append({"target_id": target_id, "reason": "not_single_protein"})
                    continue
                if target["target_components"][0].get("component_id") != component["component_id"]:
                    raise ValueError("Target/component identity mismatch in ChEMBL response")
                pages = client.pages(
                    "activity.json",
                    "activities",
                    target_chembl_id=target_id,
                    assay_type="B",
                    standard_type__in="Ki,Kd",
                    standard_units="nM",
                    standard_relation="=",
                    order_by="activity_id",
                    limit=min(max_activities, 1000) if max_activities else 1000,
                )
                count = 0
                for page in pages:
                    remaining = max_activities - count if max_activities else len(page)
                    chosen = page[:remaining]
                    assay_ids = sorted({a["assay_chembl_id"] for a in chosen})
                    molecule_ids = sorted({a["molecule_chembl_id"] for a in chosen})
                    assays = client.entities(
                        "assay.json",
                        "assays",
                        "assay_chembl_id",
                        assay_ids,
                        [
                            "assay_chembl_id",
                            "confidence_score",
                            "target_chembl_id",
                            "variant_sequence",
                        ],
                    )
                    molecules = client.entities(
                        "molecule.json",
                        "molecules",
                        "molecule_chembl_id",
                        molecule_ids,
                        ["molecule_chembl_id", "molecule_type"],
                    )
                    # Cache full source observations, target and sequence; preparation
                    # performs stricter assay/molecule/validity filtering separately.
                    for activity in chosen:
                        yield {
                            "family": family,
                            "accession": accession,
                            "target_id": target_id,
                            "component_id": component["component_id"],
                            "sequence": component["sequence"],
                            "activity": activity,
                            "assay": assays.get(activity["assay_chembl_id"]),
                            "molecule_type": molecules.get(activity["molecule_chembl_id"], {}).get(
                                "molecule_type"
                            ),
                        }
                    count += len(chosen)
                    LOG.info("ChEMBL %s %s: collected %s observations", family, target_id, count)
                    if max_activities and count >= max_activities:
                        break
                LOG.info("ChEMBL %s %s: %s source observations", family, target_id, count)


def acquire_chembl(
    output, families, expected_release=None, max_targets=0, max_activities=0, panels=None
):
    panels = PANELS if panels is None else panels
    # Query live status without a cache so a changed release cannot contaminate it.
    with request(CHEMBL + "status.json") as response:
        status = json.load(response)
    release = status["chembl_db_version"]
    if expected_release and release != expected_release:
        raise ValueError(
            f"Expected {expected_release}, but API serves {release}; choose the release explicitly"
        )
    directory = Path(output) / "chembl" / release
    client = ChemblClient(directory / "responses", release)
    classes = {
        row["protein_class_id"]: row
        for page in client.pages("protein_classification.json", "protein_classifications")
        for row in page
    }
    selected_accessions = {
        family: list(panels[family][: max_targets or None]) for family in families
    }
    signature = hashlib.sha256(
        json.dumps([4, selected_accessions, max_activities], sort_keys=True).encode()
    ).hexdigest()[:12]
    run_dir = directory / signature
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "observations.jsonl"
    temporary = output_path.with_name(output_path.name + f".{uuid.uuid4().hex}.part")
    exclusions = []
    count = 0
    family_counts = {family: 0 for family in families}
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in iter_chembl_observations(
                client, classes, families, max_targets, max_activities, exclusions, panels
            ):
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                count += 1
                family_counts[record["family"]] += 1
            handle.flush()
            os.fsync(handle.fileno())
        with request(CHEMBL + "status.json") as response:
            final_status = json.load(response)
        if final_status["chembl_db_version"] != release:
            raise ValueError("ChEMBL release changed during collection; refusing to mark complete")
        if not count:
            raise ValueError(
                "No ChEMBL observations were collected; refusing to publish an empty dataset"
            )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_json(
        run_dir / "manifest.json",
        {
            "source": "ChEMBL",
            "release": release,
            "completed_at": utc_now(),
            "status": "sample" if max_activities or max_targets else "starter_panel_complete",
            "scope": "curated_accession_panel_not_all_family_targets",
            "families": families,
            "selected_accessions": selected_accessions,
            "max_activities_per_target": max_activities,
            "observations": count,
            "raw_observations_by_family": family_counts,
            "sha256": digest_file(output_path),
            "requests": list(client.used.values()),
            "excluded_targets": exclusions,
            "prepared_for_training": False,
            "terms_url": "https://www.ebi.ac.uk/chembl/",
            "license": "ChEMBL CC BY-SA 3.0; preserve source attribution and release terms",
        },
    )
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*SOURCES, "chembl"],
        default=["bindingdb_articles", "bindingdb_pdsp", "chembl"],
    )
    parser.add_argument("--families", nargs="+", choices=[*PANELS, "all"], default=["all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--panel-json",
        type=Path,
        help="Explicit verified accession lists by supported family, e.g. the discovered catalog panels.json",
    )
    parser.add_argument(
        "--chembl-release",
        default="ChEMBL_37",
        help="Fail if the live API serves a different release",
    )
    parser.add_argument(
        "--max-targets-per-family",
        type=int,
        default=0,
        help="0 downloads the entire configured starter panel",
    )
    parser.add_argument(
        "--max-activities-per-target",
        type=int,
        default=0,
        help="0 downloads all matching activities; positive values mark a SAMPLE",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without downloading or writing files"
    )
    args = parser.parse_args(argv)
    if args.max_targets_per_family < 0 or args.max_activities_per_target < 0:
        parser.error("Download limits must be nonnegative")
    families = list(PANELS) if "all" in args.families else sorted(set(args.families))
    panels = PANELS
    if args.panel_json:
        panels = json.loads(args.panel_json.read_text(encoding="utf-8"))
        if not isinstance(panels, dict) or set(panels) - set(PANELS):
            parser.error("Panel must be an object with supported family names")
        for family in families:
            values = panels.get(family)
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value.isalnum() or not 6 <= len(value) <= 10
                    for value in values
                )
                or len(set(values)) != len(values)
            ):
                parser.error(f"Invalid or duplicated accessions for {family}")
    datasets = list(dict.fromkeys(args.datasets))
    if args.dry_run:
        print(
            json.dumps(
                {
                    "datasets": datasets,
                    "families": families,
                    "bindingdb_release": RELEASE,
                    "chembl_release": args.chembl_release,
                    "output": str(args.output_dir),
                    "scope": f"{sum(len(panels[family][: args.max_targets_per_family or None]) for family in families)} configured non-kinase accessions plus selected broad BindingDB archives",
                    "sample": bool(args.max_targets_per_family or args.max_activities_per_target),
                },
                indent=2,
            )
        )
        return 0
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    failures = []
    for name in datasets:
        try:
            if name == "chembl":
                location = acquire_chembl(
                    args.output_dir,
                    families,
                    args.chembl_release,
                    args.max_targets_per_family,
                    args.max_activities_per_target,
                    panels,
                )
                LOG.info("ChEMBL source data saved: %s", location)
            else:
                filename, checksum = SOURCES[name]
                url = f"https://www.bindingdb.org/rwd/bind/downloads/{filename}"
                LOG.info("Downloading %s", filename)
                download_archive(url, args.output_dir / "bindingdb" / RELEASE / filename, checksum)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            LOG.error("%s failed: %s", name, exc)
            failures.append(name)
    if failures:
        LOG.error(
            "Incomplete acquisition: %s. Successful files remain verified and reusable.",
            ", ".join(failures),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
