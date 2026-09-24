from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from .index import LocalIndex
from .jobs import JobStore
from .config import settings
from .metrics import index_jobs_total


def process_once() -> bool:
    store = JobStore(settings.job_db_path)
    claimed = store.claim()
    if claimed is None:
        return False
    job, payload = claimed
    try:
        if job.job_type != "index_documents":
            raise ValueError(f"unsupported job type: {job.job_type}")
        LocalIndex().build(Path(job.path))
        store.finish(job.id, True)
        index_jobs_total.labels(status="succeeded").inc()
    except Exception as exc:  # noqa: BLE001
        store.finish(job.id, False, str(exc))
        index_jobs_total.labels(status="failed").inc()
        raise
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalRAG background indexing worker")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=float(os.getenv("WORKER_POLL_SECONDS", str(settings.worker_poll_seconds))))
    args = parser.parse_args()
    if args.once:
        process_once(); return
    while True:
        if not process_once(): time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
