"""Execute/resume an explicit experiment matrix with bounded worker processes."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.acquisition import ROOT, atomic_json, digest_file, utc_now
from src.training.experiment import load_config, source_hash


def run_matrix(plan_path, status_path, workers=2, register=False):
    plan_path, status_path = Path(plan_path), Path(status_path)
    plan = json.loads(plan_path.read_text())
    if not 1 <= workers <= 3 or len({job["id"] for job in plan["jobs"]}) != len(plan["jobs"]):
        raise ValueError("Require 1–3 workers and unique job IDs")
    if plan["source_sha256"] != source_hash():
        raise ValueError("Source changed since the matrix was frozen; create a new reviewed plan")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    logs = status_path.parent / (status_path.stem + "_logs")
    logs.mkdir(exist_ok=True)
    results = {}
    registration_errors = []

    def execute(job):
        config_path, output = Path(job["config"]), Path(job["output"])
        if digest_file(config_path) != job["config_sha256"]:
            raise ValueError("Config changed after plan creation")
        config = load_config(config_path)
        if output.exists():
            existing = json.loads((output / "run.json").read_text())
            if existing["config"] != config or existing["source_sha256"] != plan["source_sha256"]:
                raise ValueError("Existing output does not match the frozen experiment")
            if existing["artifact"]["sha256"] != digest_file(output / existing["artifact"]["path"]):
                raise ValueError("Existing model artifact has changed")
            return {"status": "verified_existing", "output": str(output)}
        if source_hash() != plan["source_sha256"]:
            raise ValueError("Source changed during the matrix")
        with (logs / f"{job['id']}.log").open("a", encoding="utf-8") as handle:
            process = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_model.py",
                    "--config",
                    str(config_path),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        return {
            "status": "completed" if process.returncode == 0 else "failed",
            "exit_code": process.returncode,
            "output": str(output),
        }

    def save():
        atomic_json(
            status_path,
            {
                "updated_at": utc_now(),
                "plan_sha256": digest_file(plan_path),
                "total_jobs": len(plan["jobs"]),
                "workers": workers,
                "results": results,
                "test_evaluation_requested": False,
                "registration_errors": registration_errors,
            },
        )

    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(execute, job): job for job in plan["jobs"]}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results[job["id"]] = future.result()
            except Exception as exc:
                results[job["id"]] = {"status": "failed", "error": str(exc)}
            save()
            print(f"{job['id']}: {results[job['id']]['status']}", flush=True)
            if register:
                from scripts.register_matrix_models import register_complete_groups

                try:
                    register_complete_groups(plan)
                except Exception as exc:
                    registration_errors.append({"after_job": job["id"], "error": str(exc)})
                    save()
                    print(f"Registration requires review: {exc}", flush=True)
    return (
        1
        if registration_errors or any(row["status"] == "failed" for row in results.values())
        else 0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--after-status",
        type=Path,
        help="Wait for a prerequisite matrix to finish successfully before consuming another worker",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register all alternatives and the provisional validation default after each complete task group",
    )
    args = parser.parse_args()
    if args.after_status:
        plan = json.loads(args.plan.read_text())
        atomic_json(
            args.status,
            {
                "total_jobs": len(plan["jobs"]),
                "results": {},
                "waiting_for": str(args.after_status),
                "updated_at": utc_now(),
            },
        )
        print(f"Waiting for {args.after_status}", flush=True)
        while True:
            prior = json.loads(args.after_status.read_text())
            if len(prior.get("results", {})) == prior["total_jobs"]:
                if any(row["status"] == "failed" for row in prior["results"].values()):
                    raise SystemExit(
                        "Prerequisite matrix has failures; review it before continuing"
                    )
                break
            time.sleep(5)
    raise SystemExit(run_matrix(args.plan, args.status, args.workers, args.register))
