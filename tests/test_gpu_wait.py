"""Jobs that wait for an out-of-stock GPU, and the pacing that keeps RunPod happy."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "tests"))

import gpu_wait
import test_job_flow as flow
from runpod_client import RunPodCapacityError, RunPodClient, RunPodError, RunPodRateLimitError
from test_job_flow import FakeRunPod, FakeScp


def offer(available: bool) -> list[dict]:
    return [{"id": "A40", "_cloud_type": "SECURE", "_available": available,
             "_display_name": "A40", "_stock_status": "High" if available else "None",
             "_hourly_price": 0.4}]


class PacingTests(unittest.TestCase):
    def test_backoff_doubles_up_to_the_cap_and_honours_retry_after(self):
        self.assertEqual(gpu_wait.next_delay(60, failed=False), gpu_wait.POLL_S)
        self.assertEqual(gpu_wait.next_delay(60, failed=True), 120)
        self.assertEqual(gpu_wait.next_delay(500, failed=True), gpu_wait.MAX_BACKOFF_S)
        self.assertEqual(gpu_wait.next_delay(60, failed=True, retry_after=900), 900)

    def test_jitter_stays_within_bounds(self):
        for _ in range(200):
            self.assertTrue(48 <= gpu_wait.jittered(60) <= 72)

    def test_checks_from_every_job_are_spaced_apart(self):
        stop = threading.Event()
        with patch.object(gpu_wait, "MIN_SPACING_S", 0.05), patch.object(gpu_wait, "_next_slot", 0.0):
            started = time.monotonic()
            for _ in range(4):
                self.assertTrue(gpu_wait.wait_turn(stop))
            self.assertGreaterEqual(time.monotonic() - started, 0.14)

    def test_stop_ends_the_wait(self):
        stop = threading.Event()
        stop.set()
        with patch.object(gpu_wait, "_next_slot", time.monotonic() + 30):
            self.assertFalse(gpu_wait.wait_turn(stop))


class RateLimitTests(unittest.TestCase):
    def test_429_raises_a_rate_limit_error_with_retry_after(self):
        client = RunPodClient("key", "https://api.example/graphql")
        response = Mock(status_code=429, headers={"Retry-After": "120"})
        with patch.object(client.session, "request", return_value=response):
            with self.assertRaises(RunPodRateLimitError) as caught:
                client.list_pods()
        self.assertEqual(caught.exception.retry_after, 120.0)
        self.assertIsInstance(caught.exception, RunPodError)


class WaitingJobTests(flow.JobFlowTests):
    """The real job flow with a RunPod whose GPU is busy at first."""

    def setUp(self):
        super().setUp()
        for name, value in (("POLL_S", 0.0), ("MIN_SPACING_S", 0.0), ("NOTE_EVERY_S", 0.0)):
            patcher = patch.object(gpu_wait, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_waiting(self, offers, *, wait=True, create_errors=0):
        calls = {"select": 0, "create": 0}
        sequence = list(offers)

        def select(*args, **kwargs):
            calls["select"] += 1
            return sequence.pop(0) if len(sequence) > 1 else sequence[0]

        real_create = FakeRunPod.create_pod

        def create(client, **kwargs):
            calls["create"] += 1
            if calls["create"] <= create_errors:
                raise RunPodCapacityError("There are no longer any instances available.")
            return real_create(client, **kwargs)

        job = self.make_job()
        with patch.object(FakeRunPod, "select_gpu_types", lambda self, *a, **k: select()), \
                patch.object(FakeRunPod, "create_pod", create):
            job._run_job(self.photos, ("A40", "SECURE"), wait_for_gpu=wait)
        return job, calls

    def test_waits_until_the_gpu_is_free(self):
        job, calls = self.run_waiting([offer(False), offer(False), offer(False), offer(True)])
        self.assertEqual(job.outcome, "finished")
        self.assertEqual((calls["select"], calls["create"]), (4, 1))
        log = (job.results_dir / "job.log").read_text()
        self.assertIn("is free after", log)

    def test_without_waiting_a_busy_gpu_fails_before_any_pod(self):
        job, calls = self.run_waiting([offer(False)], wait=False)
        self.assertNotEqual(job.outcome, "finished")
        self.assertEqual(calls["create"], 0)

    def test_a_card_taken_at_creation_sends_the_job_back_to_waiting(self):
        job, calls = self.run_waiting([offer(True), offer(False), offer(True)], create_errors=1)
        self.assertEqual(job.outcome, "finished")
        self.assertEqual(calls["create"], 2)

    def test_rate_limits_and_errors_do_not_end_the_wait(self):
        answers = [offer(False), RunPodRateLimitError("slow down", 0), RunPodError("offline"),
                   offer(True)]

        def select(*args, **kwargs):
            answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(answer, Exception):
                raise answer
            return answer

        job = self.make_job()
        with patch.object(FakeRunPod, "select_gpu_types", lambda self, *a, **k: select()), \
                patch.object(gpu_wait, "next_delay", lambda delay, **kw: 0.0):
            job._run_job(self.photos, ("A40", "SECURE"), wait_for_gpu=True)
        self.assertEqual(job.outcome, "finished")

    def test_stop_while_waiting_creates_no_pod(self):
        job = self.make_job()
        calls = []

        def select(*args, **kwargs):
            calls.append(1)
            if len(calls) == 3:
                job.stop_pod()  # what the Stop button does
            return offer(False)

        with patch.object(FakeRunPod, "select_gpu_types", lambda self, *a, **k: select()):
            job._run_job(self.photos, ("A40", "SECURE"), wait_for_gpu=True)
        self.assertEqual(job.outcome, "stopped")
        self.assertNotIn(("upload", 4), FakeScp.calls)
        self.assertEqual(FakeRunPod.terminated, [])


for _name in [n for n in dir(flow.JobFlowTests) if n.startswith("test_")]:
    setattr(WaitingJobTests, _name, None)


WINDOW_SCRIPT = r"""
import os, sys, tempfile
from pathlib import Path
os.environ["RUNPOD_API_KEY"] = ""
sys.path.insert(0, sys.argv[1])
tmp = Path(tempfile.mkdtemp())
import config
config.STATE_FILE = tmp / "app_state.json"
import job_manager
real = job_manager.JobManager
job_manager.JobManager = lambda cfg: real(cfg, ledger_path=tmp / "ledger.json",
                                          log_dir=tmp / "logs", history_path=tmp / "jobs.db")
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
import app as studio
studio.JobManager = job_manager.JobManager

capture = tmp / "capture"
capture.mkdir()
for index in range(4):
    (capture / f"p{index}.jpg").write_bytes(b"x" * 100)
window = studio.MainWindow()
window.input_folder.setText(str(capture))
window._local_folder_changed()
plan = window._source_plan()
window._current_plan = plan
window._gpu_lookup_settings = window._gpu_signature(plan)
gpus = [
    {"id": "RTX 4090", "_cloud_type": "SECURE", "_available": True, "_display_name": "RTX 4090",
     "_stock_status": "High", "_hourly_price": 0.69, "memoryInGb": 24},
    {"id": "RTX 3090", "_cloud_type": "COMMUNITY", "_available": False, "_display_name": "RTX 3090",
     "_stock_status": "No verified SSH/disk capacity", "_hourly_price": 0.22, "memoryInGb": 24},
]
window._show_gpus(gpus)
busy = window.gpu_list.item(1)
assert not busy.flags() & QtCore.Qt.ItemIsEnabled  # off: out-of-stock cards can't be picked
window.wait_for_gpu.setChecked(True)
busy = window.gpu_list.item(1)
assert busy.flags() & QtCore.Qt.ItemIsEnabled
window.gpu_list.setCurrentItem(busy)
assert window._selected_gpu() == ("RTX 3090", "COMMUNITY"), window._selected_gpu()
assert window.run_button.text() == "Wait and start", window.run_button.text()
window.wait_for_gpu.setChecked(False)  # back off: falls back to a free card
assert window._selected_gpu() == ("RTX 4090", "SECURE"), window._selected_gpu()
assert window.run_button.text() == "Start"
print("OK")
"""


class WindowTests(unittest.TestCase):
    def test_waiting_makes_busy_cards_pickable(self):
        result = subprocess.run(
            [sys.executable, "-c", WINDOW_SCRIPT, str(ROOT / "desktop")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        self.assertIn("OK", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
