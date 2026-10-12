from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from config import AppConfig
from job_controller import JobController, PodMissingError, PodStoppedError
from runpod_client import RunPodClient
from scp_client import ScpClient


class StartupMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.controller = JobController(AppConfig())
        self.logs = []
        self.controller.log_line.connect(self.logs.append)
        self.now = [0.0]

    def _clock(self):
        return patch.multiple(
            "job_controller.time",
            monotonic=lambda: self.now[0],
            sleep=lambda seconds: self.now.__setitem__(0, self.now[0] + seconds),
        )

    def test_reports_status_and_system_log_while_pod_starts(self):
        client = Mock()
        client.pod_snapshot.side_effect = [
            {"id": "pod1", "status": "PROVISIONING", "ssh": {"direct": None}},
            {"id": "pod1", "status": "STARTING", "ssh": {"direct": None}},
            {"id": "pod1", "status": "RUNNING", "ssh": {"direct": {"host": "example", "port": 1234}}},
        ]
        client.direct_ssh_endpoint.side_effect = RunPodClient.direct_ssh_endpoint
        client.recent_system_logs.return_value = ["Pulling image layers"]
        with self._clock():
            endpoint = self.controller._wait_for_public_ssh(client, "pod1")
        self.assertEqual(endpoint, ("example", 1234))
        self.assertTrue(any("provisioning" in line for line in self.logs))
        self.assertTrue(any("Pulling image layers" in line for line in self.logs))
        self.assertTrue(any("Public SSH port assigned" in line for line in self.logs))

    def test_temporary_404_does_not_abort_visible_pod(self):
        client = Mock()
        client.pod_snapshot.side_effect = [
            None,
            {"id": "pod1", "status": "RUNNING", "ssh": {"direct": {"host": "example", "port": 1234}}},
        ]
        client.list_pods.return_value = [{"id": "pod1", "desiredStatus": "RUNNING"}]
        client.direct_ssh_endpoint.side_effect = RunPodClient.direct_ssh_endpoint
        client.recent_system_logs.return_value = []
        with self._clock():
            endpoint = self.controller._wait_for_public_ssh(client, "pod1")
        self.assertEqual(endpoint, ("example", 1234))
        self.assertTrue(any("still listed" in line for line in self.logs))

    def test_stopped_pod_is_reported_without_waiting(self):
        client = Mock()
        client.pod_snapshot.return_value = {"id": "pod1", "status": "EXITED"}
        with self._clock():
            with self.assertRaisesRegex(PodStoppedError, "exited"):
                self.controller._wait_for_public_ssh(client, "pod1")

    def test_repeated_absence_is_reported(self):
        client = Mock()
        client.pod_snapshot.return_value = None
        client.list_pods.return_value = []
        client.recent_system_logs.return_value = []
        with self._clock():
            with self.assertRaisesRegex(PodMissingError, "no longer visible"):
                self.controller._wait_for_public_ssh(client, "pod1")
        self.assertEqual(client.pod_snapshot.call_count, 6)


class SshBootstrapTests(unittest.TestCase):
    def test_wait_reports_progress_then_ready(self):
        transfer = ScpClient(Path("/tmp/test-key"), "example", 1234)
        transfer._ssh = Mock(side_effect=[
            Mock(returncode=255, stderr="Connection refused"), Mock(returncode=0, stderr=""),
        ])
        now = [0.0]
        logs = []
        with patch.multiple(
            "scp_client.time",
            monotonic=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        ):
            transfer.wait_ready(log=logs.append)
        self.assertTrue(any("waiting for the image" in line for line in logs))
        self.assertTrue(any("SSH is ready" in line for line in logs))

    def test_wait_checks_for_console_stop(self):
        transfer = ScpClient(Path("/tmp/test-key"), "example", 1234)
        transfer._ssh = Mock(return_value=Mock(returncode=255, stderr="Connection refused"))
        check = Mock(side_effect=PodStoppedError("stopped in console"))
        with self.assertRaisesRegex(PodStoppedError, "stopped in console"):
            transfer.wait_ready(status_check=check)
        check.assert_called_once()


class SystemLogTests(unittest.TestCase):
    def test_sse_system_lines_are_deduplicated(self):
        client = RunPodClient("test-key", "https://example.invalid/graphql")
        response = Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            "id: event-1", 'data: {"source":"system","line":"Pulling image"}', "",
            "id: event-2", 'data: {"source":"container","line":"ignore"}', "",
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.session.get = Mock(return_value=response)
        seen = set()
        self.assertEqual(client.recent_system_logs("pod1", seen), ["Pulling image"])
        self.assertEqual(client.recent_system_logs("pod1", seen), [])


if __name__ == "__main__":
    unittest.main()
