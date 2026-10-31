"""Local durable prediction jobs with atomic claims and renewable worker leases."""

import hashlib
import json
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("pharmalens.jobs")
TERMINAL = {"completed", "failed", "cancelled"}


class JobCancelled(Exception):
    pass


class WorkerStopping(JobCancelled):
    pass


class BatchJobStore:
    def __init__(self, path, clock=time.time, lease_seconds=120, capacity=50):
        self.path, self.clock = Path(path), clock
        self.lease_seconds, self.capacity = lease_seconds, capacity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
        with self.connection(transaction=True) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS prediction_jobs (
                id TEXT PRIMARY KEY, request_key TEXT UNIQUE, payload TEXT NOT NULL,
                payload_hash TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, updated_at REAL NOT NULL,
                total_rows INTEGER NOT NULL, completed_rows INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
                worker_id TEXT, lease_expires REAL, result TEXT, error TEXT, retry_of TEXT,
                scope TEXT NOT NULL DEFAULT '{}'
            )""")
            if "scope" not in {row[1] for row in db.execute("PRAGMA table_info(prediction_jobs)")}:
                db.execute(
                    "ALTER TABLE prediction_jobs ADD COLUMN scope TEXT NOT NULL DEFAULT '{}'"
                )
                for row in db.execute("SELECT id,payload FROM prediction_jobs").fetchall():
                    payload = json.loads(row["payload"])
                    db.execute(
                        "UPDATE prediction_jobs SET scope=? WHERE id=?",
                        (
                            json.dumps(
                                {
                                    key: payload[key]
                                    for key in [
                                        "model_id",
                                        "dataset",
                                        "dataset_version",
                                        "endpoint",
                                    ]
                                    if key in payload
                                }
                            ),
                            row["id"],
                        ),
                    )
            db.execute(
                "CREATE INDEX IF NOT EXISTS jobs_queue ON prediction_jobs(status, created_at)"
            )
            if "owner_uid" not in {row[1] for row in db.execute("PRAGMA table_info(prediction_jobs)")}:
                db.execute("ALTER TABLE prediction_jobs ADD COLUMN owner_uid TEXT NOT NULL DEFAULT 'local'")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_owner ON prediction_jobs(owner_uid,created_at)")

    @contextmanager
    def connection(self, transaction=False):
        db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            if transaction:
                db.execute("BEGIN IMMEDIATE")
            yield db
            if transaction:
                db.commit()
        except Exception:
            if transaction:
                db.rollback()
            raise
        finally:
            db.close()

    def submit(self, payload, request_key=None, retry_of=None, initial_attempts=0):
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        checksum = hashlib.sha256(encoded.encode()).hexdigest()
        now, job_id = self.clock(), uuid.uuid4().hex
        with self.connection(transaction=True) as db:
            if request_key:
                prior = db.execute(
                    "SELECT id,payload_hash FROM prediction_jobs WHERE request_key=?",
                    (request_key,),
                ).fetchone()
                if prior:
                    if prior["payload_hash"] != checksum:
                        raise ValueError("Idempotency key was already used for different inputs")
                    return self.get(prior["id"])
            pending = db.execute(
                "SELECT count(*) FROM prediction_jobs WHERE status IN ('queued','running')"
            ).fetchone()[0]
            if pending >= self.capacity:
                raise OverflowError("Prediction queue is at capacity")
            db.execute(
                "INSERT INTO prediction_jobs(id,request_key,payload,payload_hash,status,created_at,updated_at,total_rows,retry_of,attempts,scope,owner_uid) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    request_key,
                    encoded,
                    checksum,
                    "queued",
                    now,
                    now,
                    len(payload["pairs"]),
                    retry_of,
                    initial_attempts,
                    json.dumps(
                        {
                            key: payload[key]
                            for key in ["model_id", "dataset", "dataset_version", "endpoint"]
                            if key in payload
                        }
                    ),
                    payload.get("owner_uid", "local"),
                ),
            )
        return self.get(job_id)

    def get(self, job_id, include_payload=False, include_result=False):
        columns = "id,status,created_at,updated_at,total_rows,completed_rows,attempts,error,retry_of,cancel_requested,scope,owner_uid,(result IS NOT NULL) AS has_result"
        if include_payload:
            columns += ",payload"
        if include_result:
            columns += ",result"
        with self.connection() as db:
            row = db.execute(
                f"SELECT {columns} FROM prediction_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Prediction job not found")
        result = {
            key: row[key]
            for key in [
                "id",
                "status",
                "created_at",
                "updated_at",
                "total_rows",
                "completed_rows",
                "attempts",
                "error",
                "retry_of",
                "owner_uid",
            ]
        }
        result.update(json.loads(row["scope"]))
        result["cancel_requested"] = bool(row["cancel_requested"])
        result["result_available"] = bool(row["has_result"]) and row["status"] == "completed"
        if include_payload:
            result["payload"] = json.loads(row["payload"])
        if include_result:
            result["result"] = json.loads(row["result"]) if row["result"] else None
        return result

    def recent(self, limit=20, owner_uid=None):
        with self.connection() as db:
            ids = [
                row[0]
                for row in db.execute(
                    "SELECT id FROM prediction_jobs " + ("WHERE owner_uid=? " if owner_uid is not None else "") + "ORDER BY created_at DESC LIMIT ?",
                    ((owner_uid,) if owner_uid is not None else ()) + (min(max(limit, 1), 100),),
                )
            ]
        return [self.get(job_id) for job_id in ids]

    def claim(self, worker_id):
        now = self.clock()
        with self.connection(transaction=True) as db:
            db.execute(
                "UPDATE prediction_jobs SET status='cancelled', updated_at=? WHERE status='running' AND lease_expires<? AND cancel_requested=1",
                (now, now),
            )
            db.execute(
                "UPDATE prediction_jobs SET status='failed',error='Worker lease expired after three attempts',updated_at=? WHERE status='running' AND lease_expires<? AND attempts>=3",
                (now, now),
            )
            db.execute(
                "UPDATE prediction_jobs SET status='queued',worker_id=NULL,lease_expires=NULL,completed_rows=0,updated_at=? WHERE status='running' AND lease_expires<? AND attempts<3",
                (now, now),
            )
            row = db.execute(
                "SELECT id,payload FROM prediction_jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE prediction_jobs SET status='running',worker_id=?,lease_expires=?,attempts=attempts+1,updated_at=?,error=NULL WHERE id=? AND status='queued'",
                (worker_id, now + self.lease_seconds, now, row["id"]),
            )
        return row["id"], json.loads(row["payload"])

    def heartbeat(self, job_id, worker_id, completed_rows):
        now = self.clock()
        with self.connection(transaction=True) as db:
            row = db.execute(
                "SELECT status,worker_id,cancel_requested FROM prediction_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if (
                row is None
                or row["status"] != "running"
                or row["worker_id"] != worker_id
                or row["cancel_requested"]
            ):
                raise JobCancelled()
            db.execute(
                "UPDATE prediction_jobs SET completed_rows=?,updated_at=?,lease_expires=? WHERE id=?",
                (completed_rows, now, now + self.lease_seconds, job_id),
            )

    def finish(self, job_id, worker_id, result=None, error=None, cancelled=False):
        with self.connection(transaction=True) as db:
            row = db.execute(
                "SELECT status,worker_id,cancel_requested,total_rows FROM prediction_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] != "running" or row["worker_id"] != worker_id:
                return False  # An expired worker may not overwrite a newer claim.
            status = (
                "cancelled"
                if cancelled or row["cancel_requested"]
                else "failed"
                if error
                else "completed"
            )
            if status == "completed" and not isinstance(result, dict):
                raise ValueError("Completed prediction jobs require a result object")
            encoded = json.dumps(result, allow_nan=False) if status == "completed" else None
            db.execute(
                "UPDATE prediction_jobs SET status=?,updated_at=?,result=?,error=?,lease_expires=NULL,completed_rows=? WHERE id=?",
                (
                    status,
                    self.clock(),
                    encoded,
                    error,
                    row["total_rows"]
                    if status == "completed"
                    else self.get(job_id)["completed_rows"],
                    job_id,
                ),
            )
        return True

    def cancel(self, job_id):
        with self.connection(transaction=True) as db:
            row = db.execute("SELECT status FROM prediction_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("Prediction job not found")
            if row["status"] not in TERMINAL:
                db.execute(
                    "UPDATE prediction_jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated_at=? WHERE id=?",
                    (self.clock(), job_id),
                )
        return self.get(job_id)

    def release(self, job_id, worker_id):
        with self.connection(transaction=True) as db:
            row = db.execute(
                "SELECT status,worker_id,cancel_requested,attempts FROM prediction_jobs WHERE id=?",
                (job_id,),
            ).fetchone()
            if row and row["status"] == "running" and row["worker_id"] == worker_id:
                status = (
                    "cancelled"
                    if row["cancel_requested"]
                    else "failed"
                    if row["attempts"] >= 3
                    else "queued"
                )
                db.execute(
                    "UPDATE prediction_jobs SET status=?,worker_id=NULL,lease_expires=NULL,completed_rows=0,updated_at=?,error=? WHERE id=?",
                    (
                        status,
                        self.clock(),
                        "Worker stopped after three attempts" if status == "failed" else None,
                        job_id,
                    ),
                )

    def retry(self, job_id):
        original = self.get(job_id, include_payload=True)
        if original["status"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled jobs can be retried")
        if original["attempts"] >= 3:
            raise ValueError(
                "Retry limit reached; review the failure before creating a new submission"
            )
        return self.submit(
            original["payload"], retry_of=job_id, initial_attempts=original["attempts"]
        )


class BatchJobWorker:
    def __init__(self, store, execute):
        self.store, self.execute = store, execute
        self.worker_id = uuid.uuid4().hex
        self.stopping = threading.Event()
        self.thread = None

    def process_once(self):
        claim = self.store.claim(self.worker_id)
        if claim is None:
            return False
        job_id, payload = claim

        def progress(rows):
            if self.stopping.is_set():
                raise WorkerStopping()
            self.store.heartbeat(job_id, self.worker_id, rows)

        try:
            result = self.execute(payload, progress)
            self.store.finish(job_id, self.worker_id, result=result)
        except WorkerStopping:
            self.store.release(job_id, self.worker_id)
        except JobCancelled:
            self.store.finish(job_id, self.worker_id, cancelled=True)
        except Exception:
            LOG.exception("Prediction worker failed for job %s", job_id)
            self.store.finish(
                job_id,
                self.worker_id,
                error="Prediction worker failed; check model availability and retry",
            )
        return True

    def start(self):
        def loop():
            while not self.stopping.is_set():
                try:
                    if not self.process_once():
                        self.stopping.wait(0.25)
                except Exception:
                    LOG.exception("Prediction queue storage is unavailable")
                    self.stopping.wait(1)

        self.thread = threading.Thread(target=loop, name="pharmalens-predictions", daemon=True)
        self.thread.start()

    def stop(self):
        self.stopping.set()
        if self.thread:
            self.thread.join(timeout=5)
