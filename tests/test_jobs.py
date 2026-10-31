from concurrent.futures import ThreadPoolExecutor

import pytest

from backend.jobs import BatchJobStore, BatchJobWorker, JobCancelled


def payload():
    return {"model_id": "frozen", "pairs": [{"smiles": "CCO", "sequence": "ACDEFGHIKLMNPQRSTVWY"}]}


def test_submission_is_idempotent_and_capacity_is_bounded(tmp_path):
    store = BatchJobStore(tmp_path / "jobs.sqlite", capacity=1)
    first = store.submit(payload(), "same-key")
    assert store.submit(payload(), "same-key")["id"] == first["id"]
    with pytest.raises(ValueError, match="different inputs"):
        store.submit({**payload(), "model_id": "different"}, "same-key")
    with pytest.raises(OverflowError):
        store.submit(payload())
    store.cancel(first["id"])
    assert store.submit(payload())["status"] == "queued"


def test_concurrent_claim_has_one_owner_and_expired_owner_cannot_overwrite(tmp_path):
    clock = [1000.0]
    store = BatchJobStore(tmp_path / "jobs.sqlite", clock=lambda: clock[0], lease_seconds=10)
    job = store.submit(payload())
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(store.claim, ["a", "b", "c", "d"]))
    assert sum(claim is not None for claim in claims) == 1
    owner = ["a", "b", "c", "d"][[claim is not None for claim in claims].index(True)]
    clock[0] += 11
    recovered = store.claim("recovered")
    assert recovered[0] == job["id"]
    assert not store.finish(job["id"], owner, result={"wrong": True})
    assert store.finish(job["id"], "recovered", result={"correct": True})
    assert store.get(job["id"], include_result=True)["result"] == {"correct": True}
    assert store.get(job["id"])["attempts"] == 2


def test_cancellation_and_retry_keep_frozen_payload_and_attempt_limits(tmp_path):
    store = BatchJobStore(tmp_path / "jobs.sqlite")
    job = store.submit(payload())
    for attempt in range(1, 4):
        claimed = store.claim("worker")
        assert claimed[0] == job["id"]
        store.cancel(job["id"])
        with pytest.raises(JobCancelled):
            store.heartbeat(job["id"], "worker", 0)
        store.finish(job["id"], "worker", cancelled=True)
        assert store.get(job["id"])["attempts"] == attempt
        if attempt < 3:
            previous = job["id"]
            job = store.retry(previous)
            assert job["retry_of"] == previous
            assert store.get(job["id"], include_payload=True)["payload"] == payload()
        else:
            with pytest.raises(ValueError, match="Retry limit"):
                store.retry(job["id"])


def test_worker_completes_and_restart_recovers_queued_work(tmp_path):
    path = tmp_path / "jobs.sqlite"
    job = BatchJobStore(path).submit(payload())
    store = BatchJobStore(path)

    def execute(data, progress):
        progress(0)
        assert data == payload()
        progress(1)
        return {"results": [{"score": 1.0}]}

    assert BatchJobWorker(store, execute).process_once()
    assert store.get(job["id"])["status"] == "completed"
    assert store.get(job["id"])["completed_rows"] == 1


def test_shutdown_releases_work_and_failures_are_explicit(tmp_path):
    store = BatchJobStore(tmp_path / "jobs.sqlite")
    job = store.submit(payload())
    worker = BatchJobWorker(store, lambda data, progress: progress(0))
    worker.stopping.set()
    worker.process_once()
    assert store.get(job["id"])["status"] == "queued"

    def fail(data, progress):
        raise RuntimeError("simulated failure")

    BatchJobWorker(store, fail).process_once()
    assert store.get(job["id"])["status"] == "failed"
    assert not store.get(job["id"])["result_available"]
