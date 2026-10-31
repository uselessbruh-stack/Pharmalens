import io
import zipfile
from unittest.mock import patch

import pytest

from src.data.acquisition import ChemblClient, classify_component, download_archive


class Response(io.BytesIO):
    def __init__(self, payload):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def geturl(self):
        return "https://example.org/data.zip"


def zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../never-extracted.tsv", "value\n1\n")
    return buffer.getvalue()


def test_atomic_verified_download_and_cache_tampering(tmp_path):
    destination = tmp_path / "data.zip"
    with patch("src.data.acquisition.request", return_value=Response(zip_bytes())):
        receipt = download_archive("https://example.org/data.zip", destination)
    assert receipt["status"] == "verified"
    assert not (tmp_path.parent / "never-extracted.tsv").exists()
    with patch(
        "src.data.acquisition.request", side_effect=AssertionError("must use verified cache")
    ):
        assert (
            download_archive("https://example.org/data.zip", destination)["sha256"]
            == receipt["sha256"]
        )
    destination.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        download_archive("https://example.org/data.zip", destination)


def test_error_page_or_bad_checksum_never_published(tmp_path):
    destination = tmp_path / "data.zip"
    for data, checksum in [(b"<html>sign in</html>", None), (zip_bytes(), "0" * 32)]:
        with (
            patch("src.data.acquisition.request", return_value=Response(data)),
            pytest.raises(ValueError),
        ):
            download_archive("https://example.org/data.zip", destination, checksum)
        assert not destination.exists()
        assert not list(tmp_path.glob("*.part"))


def test_family_mapping_uses_ancestry_and_excludes_kinases():
    classes = {
        1: {"pref_name": "Enzyme", "parent_id": None},
        2: {"pref_name": "Kinase", "parent_id": 1},
        3: {"pref_name": "Tyrosine kinase", "parent_id": 2},
    }
    assert (
        classify_component({"protein_classifications": [{"protein_classification_id": 3}]}, classes)
        == "kinase"
    )


def test_api_cache_rejects_changed_bytes_and_foreign_pagination(tmp_path):
    client = ChemblClient(tmp_path, "test")
    with patch("src.data.acquisition.request", return_value=Response(b'{"targets": []}')):
        client.get("target.json")
    path = next(p for p in tmp_path.glob("*.json") if not p.name.endswith("receipt.json"))
    path.write_text('{"targets": [1]}')
    with pytest.raises(ValueError, match="integrity"):
        client.get("target.json")
    with pytest.raises(ValueError, match="origin"):
        client.get("https://example.org/private")


def test_verified_entity_cache_is_reused_across_client_runs(tmp_path):
    client = ChemblClient(tmp_path, "test")
    payload = b'{"molecules":[{"molecule_chembl_id":"M1","molecule_type":"Small molecule"}]}'
    with patch("src.data.acquisition.request", return_value=Response(payload)):
        first = client.entities(
            "molecule.json",
            "molecules",
            "molecule_chembl_id",
            ["M1"],
            ["molecule_chembl_id", "molecule_type"],
        )
    fresh = ChemblClient(tmp_path, "test")
    with patch(
        "src.data.acquisition.request", side_effect=AssertionError("should reuse verified data")
    ):
        second = fresh.entities(
            "molecule.json",
            "molecules",
            "molecule_chembl_id",
            ["M1"],
            ["molecule_chembl_id", "molecule_type"],
        )
    assert first == second
    assert fresh.used


def test_publication_retries_sharing_locks_but_never_replaces_existing_directory(tmp_path):
    from src.data.acquisition import replace_with_retry
    from pathlib import Path

    source, destination = tmp_path / "ready.part", tmp_path / "published"
    source.mkdir()
    (source / "evidence.txt").write_text("verified")
    original = Path.replace
    calls = []

    def locked_once(path, target):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("simulated sharing lock")
        return original(path, target)

    with patch.object(Path, "replace", locked_once), patch("src.data.acquisition.time.sleep"):
        replace_with_retry(source, destination)
    assert len(calls) == 2 and (destination / "evidence.txt").read_text() == "verified"
    source.mkdir()
    with pytest.raises(FileExistsError):
        replace_with_retry(source, destination)
