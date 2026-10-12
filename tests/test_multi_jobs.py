from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from PySide6 import QtCore

from config import AppConfig
from job_controller import JobController
from job_manager import JobManager
from pod_ledger import PodLedger

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


class ManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        config = AppConfig()
        config.runpod_api_key = ""
        config.max_concurrent_pods = 2
        self.manager = JobManager(config, ledger_path=root / "ledger.json", log_dir=root / "logs",
                                  history_path=root / "jobs.db")
        self.release = threading.Event()

    def tearDown(self):
        self.release.set()
        self.tmp.cleanup()

    def start(self, title="job"):
        release = self.release

        def fake_run(job, *args, **kwargs):
            release.wait(5)
            with job._lock:
                job.running = False

        with patch.object(JobController, "_run_job", fake_run):
            return self.manager.start_job(
                title=title, gpu_label="A40",
                run_args=(Path("/tmp/photos"), ("A40", "SECURE")), run_kwargs={},
            )


class PodLimitTests(ManagerTestCase):
    def test_refuses_jobs_beyond_the_limit(self):
        first = self.start("one")
        second = self.start("two")
        with self.assertRaisesRegex(ValueError, "limit is 2"):
            self.start("three")
        self.assertEqual(len(self.manager.jobs), 2)
        self.assertNotEqual(first.pod_name, second.pod_name)

    def test_untracked_pods_count_toward_the_limit(self):
        self.manager.untracked = [{"id": "p1", "name": "gaussian-old", "cost": 0.5}]
        self.start("one")
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.start("two")

    def test_slot_frees_when_a_job_ends(self):
        self.start("one")
        self.start("two")
        self.release.set()
        deadline = time.time() + 5
        while any(job.running for job in self.manager.jobs) and time.time() < deadline:
            time.sleep(0.02)
        self.release.clear()
        self.start("three")

    def test_jobs_get_independent_config_copies(self):
        job = self.start("one")
        self.manager.config.max_gpu_price = 99
        self.assertNotEqual(job.config.max_gpu_price, 99)

    def test_busy_job_cannot_be_removed(self):
        job = self.start("one")
        self.assertFalse(self.manager.remove_job(job))


class UntrackedPodTests(ManagerTestCase):
    def test_reports_only_unowned_gaussian_pods(self):
        job = self.start("mine")
        job.pod_requested = True   # creation request sent, no pod ID yet
        received = []
        self.manager.untracked_pods_changed.connect(received.append)
        self.manager._reconcile([
            {"id": "a", "name": job.pod_name, "desiredStatus": "RUNNING", "costPerHr": 0.4},
            {"id": "b", "name": "gaussian-crashed", "desiredStatus": "RUNNING", "costPerHr": 0.5},
            {"id": "c", "name": "someone-elses-pod", "desiredStatus": "RUNNING"},
            {"id": "d", "name": "gaussian-gone", "desiredStatus": "TERMINATED"},
        ])
        self.assertEqual([pod["id"] for pod in received[-1]], ["b"])
        self.assertEqual(received[-1][0]["cost"], 0.5)

    def test_settles_ledger_entries_whose_pod_is_gone(self):
        ledger = self.manager.ledger
        ledger.update("old", pod_name="gaussian-old", state="running", pod_id="x")
        ledger._entries["old"]["updated_at"] = time.time() - 3600
        ledger.update("new", pod_name="gaussian-new", state="creating")
        self.manager._reconcile([])
        states = {entry["job_id"]: entry["state"] for entry in ledger.entries()}
        self.assertEqual(states["old"], "gone")
        # A creation request that may still be in flight is left open.
        self.assertEqual(states["new"], "creating")


class LedgerTests(unittest.TestCase):
    def test_entries_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            PodLedger(path).update("j1", pod_name="gaussian-j1", state="creating")
            reloaded = PodLedger(path)
            self.assertEqual(reloaded.open_entries()[0]["pod_name"], "gaussian-j1")
            reloaded.update("j1", state="terminated")
            self.assertEqual(PodLedger(path).open_entries(), [])


class CancelTests(unittest.TestCase):
    def test_cancel_before_pod_exists(self):
        job = JobController(AppConfig())
        job.running = True
        job.stop_pod()
        self.assertTrue(job._stop_requested.is_set())
        self.assertTrue(job._stop_done.is_set())


if __name__ == "__main__":
    unittest.main()
