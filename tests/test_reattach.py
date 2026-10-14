from __future__ import annotations

import json
import subprocess
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

import job_controller
from config import AppConfig
from job_controller import JobController
from job_manager import JobManager
from scp_client import ConnectionLostError, ScpClient

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


class FakeTransfer:
    def __init__(self, states):
        self.states = list(states)
        self.finished_with = None

    def remote_job_state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def finish_remote(self, engine):
        self.finished_with = engine
        return "finished from /workspace/job/nerfstudio_outputs/run/splat.ply"


def state(job, trainer, ready, step=None, **times):
    return {"job_running": job, "trainer_running": trainer, "ply_ready": ready,
            "summary_ready": ready, "step": step, "total_steps": 60000, "spirula": True, **times}


class FollowRemoteJobTests(unittest.TestCase):
    def follow(self, states, live_log=None):
        job = JobController(AppConfig())
        logs = []
        job.log_line.connect(logs.append, QtCore.Qt.DirectConnection)
        transfer = FakeTransfer(states)
        with patch.object(job._stop_requested, "wait", return_value=False), \
                patch.object(job_controller, "STUCK_JOB_SECONDS", 0):
            job._follow_remote_job(transfer, "spirula", live_log)
        return transfer, logs

    def test_job_that_finishes_on_its_own(self):
        transfer, logs = self.follow([
            state(True, True, False, 54000), state(True, True, False, 58000),
            state(True, False, True, 60000), state(False, False, True, 60000),
        ])
        self.assertIsNone(transfer.finished_with)
        self.assertTrue(any("step 58,000 of 60,000 (96%)" in line for line in logs))
        self.assertIn("The pod finished the job on its own.", logs)

    def test_checkpoints_tell_the_time_left(self):
        # 6,000 steps in 10 minutes: 54,000 more take 90 minutes.
        _, logs = self.follow([
            state(True, True, False, 6000, train_started=1000.0, step_time=1600.0),
            state(False, False, True, 60000),
        ])
        self.assertTrue(any("step 6,000 of 60,000 (10%), about 1 h 30 min left" in line
                            for line in logs))

    def test_no_checkpoint_lines_while_the_pod_output_streams(self):
        live = threading.Event()
        live.set()
        _, logs = self.follow([state(True, True, False, 6000), state(False, False, True, 60000)], live)
        self.assertFalse(any("training checkpoint" in line for line in logs))

    def test_dead_job_script_is_finished_by_the_app(self):
        transfer, logs = self.follow([state(False, True, False, 58000), state(False, False, False, 60000)])
        self.assertEqual(transfer.finished_with, "spirula")

    def test_stuck_job_script_is_finished_by_the_app(self):
        transfer, _ = self.follow([state(True, False, False, 60000)] * 3)
        self.assertEqual(transfer.finished_with, "spirula")


class RejoinAfterDropTests(unittest.TestCase):
    """The SSH session running the job drops (ssh exit 255) while the pod keeps going."""

    def rejoin(self, transfer):
        job = JobController(AppConfig())
        logs = []
        job.log_line.connect(logs.append, QtCore.Qt.DirectConnection)
        transfer.has_job_log = lambda: False
        with patch.object(job._stop_requested, "wait", return_value=False):
            job._rejoin_job(transfer, "spirula", ConnectionLostError("dropped"))
        return logs

    def test_a_running_job_is_followed_to_the_end(self):
        transfer = FakeTransfer([state(True, True, False, 5600), state(False, False, True, 63000)])
        logs = self.rejoin(transfer)
        self.assertIn("Reconnected; the job is still going.", logs)
        self.assertIn("The pod finished the job on its own.", logs)

    def test_an_unreachable_pod_keeps_the_original_error(self):
        class Unreachable(FakeTransfer):
            def remote_job_state(self):
                raise OSError("no route")
        with self.assertRaises(ConnectionLostError):
            self.rejoin(Unreachable([]))

    def test_a_job_that_died_with_no_result_keeps_the_original_error(self):
        with self.assertRaises(ConnectionLostError):
            self.rejoin(FakeTransfer([state(False, False, False, 5600)]))

    def test_ssh_exit_255_is_a_lost_connection(self):
        client = ScpClient(Path("/nonexistent"), "127.0.0.1", 22)
        with patch("scp_client._stream_output"), \
                patch("subprocess.Popen") as popen:
            popen.return_value.__enter__.return_value.wait.return_value = 255
            popen.return_value.__enter__.return_value.returncode = 255
            with self.assertRaises(ConnectionLostError):
                client.run_pipeline(1000, 0, lambda line: None)
            popen.return_value.__enter__.return_value.wait.return_value = 1
            popen.return_value.__enter__.return_value.returncode = 1
            with self.assertRaises(Exception) as failed:
                client.run_pipeline(1000, 0, lambda line: None)
            self.assertNotIsInstance(failed.exception, ConnectionLostError)


class CloseWhileRunningTests(unittest.TestCase):
    def test_only_jobs_processing_on_their_pod_may_be_left(self):
        job = JobController(AppConfig())
        self.assertFalse(job.detachable)  # creating or uploading: the app must stay
        job._on_pod = True
        self.assertTrue(job.detachable)
        job.detach()
        self.assertTrue(job._stop_log.is_set())
        job._stop_requested.set()  # stopping: the app must see the pod terminated
        self.assertFalse(job.detachable)


class WorkerLogTests(unittest.TestCase):
    def test_output_reaches_the_log_after_the_app_disconnects(self):
        sys.path.insert(0, str(ROOT / "worker"))
        import job

        class GoneSession:
            def write(self, text):
                raise OSError(5, "Input/output error")

            def flush(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            with open(Path(tmp) / "job.log", "a", encoding="utf-8") as log:
                tee = job._Tee(GoneSession(), log)
                print("step 100/60000", file=tee, flush=True)
                print("step 200/60000", file=tee, flush=True)
                tee.close()
            self.assertEqual((Path(tmp) / "job.log").read_text().split(),
                             ["step", "100/60000", "step", "200/60000"])

    def test_a_stalled_session_does_not_hang_the_job(self):
        # The app's computer vanished without closing the connection: writes block.
        sys.path.insert(0, str(ROOT / "worker"))
        import job

        stuck = threading.Event()

        class StalledSession:
            def write(self, text):
                stuck.wait()

            def flush(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            with open(Path(tmp) / "job.log", "a", encoding="utf-8") as log:
                tee = job._Tee(StalledSession(), log)
                started = time.monotonic()
                for step in range(job._Tee.QUEUE_SIZE + 100):
                    print(f"step {step}", file=tee, flush=True)
                tee.close(timeout=0.2)
                self.assertLess(time.monotonic() - started, 5)
                self.assertIsNone(tee.stream)
                stuck.set()
            lines = (Path(tmp) / "job.log").read_text().splitlines()
            self.assertEqual(len(lines), job._Tee.QUEUE_SIZE + 100)

    def test_a_live_session_gets_every_line(self):
        sys.path.insert(0, str(ROOT / "worker"))
        import job

        class Session:
            def __init__(self):
                self.text = ""

            def write(self, text):
                self.text += text

            def flush(self):
                pass

        session = Session()
        with tempfile.TemporaryDirectory() as tmp:
            with open(Path(tmp) / "job.log", "a", encoding="utf-8") as log:
                tee = job._Tee(session, log)
                for step in range(50):
                    print(f"step {step}", file=tee, flush=True)
                tee.close()
        self.assertEqual(session.text.splitlines(), [f"step {n}" for n in range(50)])


    def test_a_failing_job_reports_its_error_to_the_app(self):
        sys.path.insert(0, str(ROOT / "worker"))
        import job

        class Session:
            def __init__(self):
                self.text = ""

            def write(self, text):
                self.text += text

            def flush(self):
                pass

        def failing_pipeline():
            raise RuntimeError("Command failed with exit code 1: spirula sfm")

        session = Session()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            with patch.object(job, "INPUT_DIR", root / "input"), \
                 patch.object(job, "JOB_LOG", root / "job.log"), \
                 patch.object(job, "run_pipeline", failing_pipeline), \
                 patch.object(job.signal, "signal"), \
                 patch.object(sys, "stdout", session), patch.object(sys, "stderr", session):
                with self.assertRaises(SystemExit):
                    job.main()
            self.assertIn("exit code 1: spirula sfm", (root / "job.log").read_text())
        self.assertIn("exit code 1: spirula sfm", session.text)


class ReattachEntryTests(unittest.TestCase):
    def test_manager_reattach_takes_over_an_untracked_pod(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig()
            config.runpod_api_key = ""
            manager = JobManager(config, ledger_path=Path(tmp) / "l.json", log_dir=Path(tmp) / "logs",
                                 history_path=Path(tmp) / "jobs.db")
            manager.ledger.update("ac1605f0b85d", pod_name="gaussian-ac1605f0b85d", state="running",
                                  pod_id="bf8", title="MVI_4394.MP4", engine="spirula")
            pod = {"id": "bf8", "name": "gaussian-ac1605f0b85d", "status": "running", "cost": 0.27}
            manager.untracked = [pod]
            seen = []

            def fake_attach(job, entry, results_root):
                seen.append((job.job_id, job.title, job.engine, entry["pod_id"], entry["cost"]))
                with job._lock:
                    job.running = False

            with patch.object(JobController, "_attach_job", fake_attach):
                job = manager.reattach(pod)
                job._thread and job._thread.join(5)
            self.assertEqual(seen, [("ac1605f0b85d", "MVI_4394.MP4", "spirula", "bf8", 0.27)])
            self.assertEqual(manager.untracked, [])
            self.assertIn(job, manager.jobs)
            self.assertEqual(job.pod_name, "gaussian-ac1605f0b85d")


class PodScriptTests(unittest.TestCase):
    def test_state_script_runs_and_reports_json(self):
        result = subprocess.run(
            [sys.executable, "-c", ScpClient._STATE_SCRIPT], capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(set(data), {"job_running", "trainer_running", "ply_ready", "summary_ready",
                                     "step", "step_time", "train_started", "total_steps",
                                     "spirula"})
        self.assertFalse(data["job_running"])

    def test_finish_script_compiles(self):
        compile(ScpClient._FINISH_SCRIPT, "finish", "exec")


if __name__ == "__main__":
    unittest.main()
