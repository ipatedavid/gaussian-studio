from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))

from PySide6 import QtCore

from config import AppConfig
from job_controller import JobController
from job_history import JobHistory, record_from_log
from job_manager import JobManager

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

FINISHED_LOG = """[16:20:20] Job 327910e9282d; its pod will be named gaussian-327910e9282d.
[16:20:20] Processing engine: Spirula Studio.
[16:20:21] Selected GPU: RTX 3090 (COMMUNITY, Low stock, $0.220/hr quoted)
[16:20:23] Pod 26irww7nk03ngh is starting. Waiting for RunPod to expose its public SSH port.
[17:33:01] Results saved to /home/me/results/MVI/2026-10-04_16-20-20_327910
"""
STOPPED_LOG = """[02:06:39] Job b40516fc7a16; its pod will be named gaussian-b40516fc7a16.
[02:06:39] Processing engine: Nerfstudio.
[02:06:51] Job interrupted after stop request: Stop requested from the app.
"""


class HistoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.history = JobHistory(self.root / "jobs.db")

    def tearDown(self):
        self.history.close()
        self.tmp.cleanup()

    def test_save_update_order_and_delete(self):
        self.history.save({"job_id": "a", "title": "old", "started_at": 100})
        self.history.save({"job_id": "b", "title": "new", "started_at": 200})
        self.history.save({"job_id": "a", "title": "old, renamed", "started_at": 100})
        self.assertEqual([r["title"] for r in self.history.recent()], ["new", "old, renamed"])
        self.history.delete("b")
        self.assertEqual([r["job_id"] for r in self.history.recent()], ["a"])

    def test_records_survive_reopening(self):
        self.history.save({"job_id": "a", "title": "kept", "started_at": 1})
        self.history.close()
        self.history = JobHistory(self.root / "jobs.db")
        self.assertEqual(self.history.recent()[0]["title"], "kept")

    def test_import_from_logs(self):
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "2026-10-04_16-20-20_327910e9282d.log").write_text(FINISHED_LOG)
        (logs / "2026-10-05_02-06-38_b40516fc7a16.log").write_text(STOPPED_LOG)
        (logs / "app.log").write_text("not a job")
        added = self.history.import_logs(logs, {"327910e9282d": "PXL_20261004.mp4"})
        self.assertEqual(added, 2)
        records = {r["job_id"]: r for r in self.history.recent()}
        finished = records["327910e9282d"]
        self.assertEqual(finished["title"], "PXL_20261004.mp4")
        self.assertEqual(finished["outcome"], "finished")
        self.assertEqual(finished["results_dir"], "/home/me/results/MVI/2026-10-04_16-20-20_327910")
        self.assertEqual(finished["engine"], "spirula")
        self.assertEqual(finished["gpu_label"], "RTX 3090")
        self.assertEqual(finished["pod_id"], "26irww7nk03ngh")
        self.assertEqual(records["b40516fc7a16"]["outcome"], "stopped")
        self.assertEqual(records["b40516fc7a16"]["title"], "Job b40516fc7a16")
        # Importing again adds nothing.
        self.assertEqual(self.history.import_logs(logs, {}), 0)

    def test_users_real_logs_parse(self):
        real = ROOT / "desktop" / "logs"
        files = sorted(p for p in real.glob("*_*.log")) if real.is_dir() else []
        if not files:
            self.skipTest("no real job logs on this machine")
        for path in files:
            stamp, job_id = path.stem.rsplit("_", 1)
            record = record_from_log(path, stamp, job_id, {})
            self.assertIn(record["outcome"], {"finished", "stopped", "failed", "interrupted"})
            self.assertEqual(record["log_path"], str(path))


class ManagerHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        config = AppConfig()
        config.runpod_api_key = ""
        self.root = root
        (root / "logs").mkdir()
        self.manager = JobManager(config, ledger_path=root / "ledger.json", log_dir=root / "logs",
                                  history_path=root / "jobs.db")

    def tearDown(self):
        self.manager.history.close()
        self.tmp.cleanup()

    def test_running_job_left_behind_comes_back_interrupted(self):
        self.manager.history.save({"job_id": "x1", "title": "walk", "running": True,
                                   "outcome": None, "started_at": 50, "pod_id": "podx"})
        jobs = self.manager.load_history()
        self.assertEqual(len(jobs), 1)
        job = jobs[0]
        self.assertTrue(job.historic)
        self.assertEqual(job.outcome, "interrupted")
        self.assertIsNotNone(job.finished_at)
        self.assertFalse(job.busy)
        self.assertEqual(job.last_pod_id, "podx")

    def test_live_job_is_saved_and_reloaded_after_restart(self):
        release = threading.Event()

        def fake_run(job, *args, **kwargs):
            with job._lock:
                job.active_pod_id = "pod9"
            job.active_pod_changed.emit("pod9")
            job.status_changed.emit("Processing images...")
            release.wait(5)
            with job._lock:
                job.active_pod_id = None
                job.running = False
                job.outcome = "finished"
                job.finished_at = time.time()
            job.finished.emit(True)

        with patch.object(JobController, "_run_job", fake_run):
            job = self.manager.start_job(title="garden", gpu_label="A40", context="Spirula Studio",
                                         rate="$0.390/hr", run_args=(Path("/tmp/x"), ("A40", "S")),
                                         run_kwargs={})
            deadline = time.time() + 5
            while (self.manager.history.recent()[0].get("pod_id") != "pod9"
                   and time.time() < deadline):
                time.sleep(0.02)
            mid = self.manager.history.recent()[0]
            self.assertTrue(mid["running"])
            self.assertEqual(mid["pod_id"], "pod9")
            release.set()
            deadline = time.time() + 5
            while self.manager.history.recent()[0].get("outcome") != "finished" and time.time() < deadline:
                time.sleep(0.02)
        final = self.manager.history.recent()[0]
        self.assertEqual((final["title"], final["outcome"], final["running"], final["pod_id"]),
                         ("garden", "finished", False, "pod9"))
        # A fresh manager (the app restarted) shows it as a past job.
        again = JobManager(self.manager.config, ledger_path=self.root / "ledger.json",
                           log_dir=self.root / "logs", history_path=self.root / "jobs.db")
        past = again.load_history()
        self.assertEqual([j.title for j in past], ["garden"])
        self.assertEqual(past[0].context, "Spirula Studio")
        again.history.close()

    def test_remove_deletes_history_but_keeps_files(self):
        log = self.root / "logs" / "2026-10-05_02-06-38_b40516fc7a16.log"
        log.write_text(STOPPED_LOG)
        self.manager.history.set_meta("imported_logs", "1")
        self.manager.history.save({"job_id": "b40516fc7a16", "title": "t", "outcome": "stopped",
                                   "log_path": str(log), "started_at": 1})
        job = self.manager.load_history()[0]
        self.assertTrue(self.manager.remove_job(job))
        self.assertEqual(self.manager.history.recent(), [])
        self.assertTrue(log.is_file())


if __name__ == "__main__":
    unittest.main()
