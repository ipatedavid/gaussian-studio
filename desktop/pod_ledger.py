"""A small on-disk record of every pod this app asks RunPod to create.

An entry is written *before* the creation request is sent, so a crash between
"request sent" and "pod ID received" still leaves the pod name behind. The
app compares open entries with RunPod's pod list to find pods nobody is
tracking any more.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

# States after which the pod is known not to exist (or never existed).
CLOSED_STATES = frozenset({"terminated", "not_created", "gone"})


class PodLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._entries = {
                    str(key): value for key, value in data.items() if isinstance(value, dict)
                }
        except (OSError, ValueError):
            pass

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            entry = self._entries.setdefault(job_id, {"job_id": job_id, "created_at": time.time()})
            entry.update(fields)
            entry["updated_at"] = time.time()
            self._save()

    def entries(self) -> list[dict]:
        with self._lock:
            return [dict(entry) for entry in self._entries.values()]

    def open_entries(self) -> list[dict]:
        return [entry for entry in self.entries() if entry.get("state") not in CLOSED_STATES]

    def by_pod_name(self) -> dict[str, dict]:
        return {entry["pod_name"]: entry for entry in self.entries() if entry.get("pod_name")}

    def _save(self) -> None:
        # Closed entries older than 30 days are of no further use.
        cutoff = time.time() - 30 * 86400
        self._entries = {
            key: entry for key, entry in self._entries.items()
            if entry.get("state") not in CLOSED_STATES or entry.get("updated_at", 0) >= cutoff
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(self._entries, indent=2), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError:
            # The ledger is a safety net; the RunPod pod list check still runs.
            pass
