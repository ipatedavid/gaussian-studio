"""Job history that survives restarts: one SQLite file next to the app.

Each job is one row holding a JSON record (title, engine, GPU, timings,
outcome, where its log and results are). The log itself stays in its own
file under logs/, so a record is small and the full log can always be read.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    started_at REAL,
    updated_at REAL,
    data       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""
LOG_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_([0-9a-f]{6,32})\.log$")


class JobHistory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        # Jobs save from their own threads; one connection behind a lock.
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._db:
            self._db.executescript(SCHEMA)

    def save(self, record: dict) -> None:
        record = dict(record, updated_at=time.time())
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO jobs (job_id, started_at, updated_at, data) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET started_at = excluded.started_at, "
                "updated_at = excluded.updated_at, data = excluded.data",
                (record["job_id"], record.get("started_at"), record["updated_at"],
                 json.dumps(record)),
            )

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT data FROM jobs ORDER BY COALESCE(started_at, updated_at) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        records = []
        for (data,) in rows:
            try:
                records.append(json.loads(data))
            except ValueError:
                continue
        return records

    def delete(self, job_id: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # One-time import of jobs that ran before history existed
    # ------------------------------------------------------------------

    def import_logs(self, log_dir: Path, titles: dict[str, str]) -> int:
        """Create records for job logs that have none. Returns how many were added."""
        with self._lock:
            known = {row[0] for row in self._db.execute("SELECT job_id FROM jobs")}
        added = 0
        for path in sorted(Path(log_dir).glob("*.log")):
            match = LOG_NAME.match(path.name)
            if not match or match.group(2) in known:
                continue
            record = record_from_log(path, match.group(1), match.group(2), titles)
            self.save(record)
            added += 1
        return added


def record_from_log(path: Path, stamp: str, job_id: str, titles: dict[str, str]) -> dict:
    """Best-effort record rebuilt from a job's log file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    started = time.mktime(time.strptime(stamp, "%Y-%m-%d_%H-%M-%S"))

    def find(pattern: str) -> str | None:
        found = re.search(pattern, text)
        return found.group(1).strip() if found else None

    results_dir = find(r"Results saved to (.+)")
    if results_dir:
        outcome = "finished"
    elif re.search(r"Job stopped|Stop requested|stop request", text):
        outcome = "stopped"
    elif re.search(r"\[ERROR\]|Pipeline failed", text):
        outcome = "failed"
    else:
        outcome = "interrupted"
    engine_label = find(r"Processing engine: (.+?)\.")
    engine = {"Spirula Studio": "spirula", "Nerfstudio": "nerfstudio"}.get(engine_label or "", "")
    return {
        "job_id": job_id,
        "title": titles.get(job_id) or f"Job {job_id}",
        "context": engine_label or "",
        "engine": engine,
        "gpu_label": find(r"Selected GPU: (.+?) \(") or "",
        "cost_per_hour": None,
        "pod_id": find(r"Pod created: (\S+)") or find(r"Pod (\w+) is starting"),
        "started_at": started,
        "finished_at": path.stat().st_mtime,
        "outcome": outcome,
        "status_text": {"finished": "Finished", "stopped": "Stopped", "failed": "Failed"}.get(
            outcome, "Interrupted"),
        "results_dir": results_dir,
        "log_path": str(path),
        "running": False,
        "imported": True,
    }
