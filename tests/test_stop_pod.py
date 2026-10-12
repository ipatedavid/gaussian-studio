from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from config import AppConfig
from job_controller import JobController


class StopPodTests(unittest.TestCase):
    def setUp(self):
        self.controller = JobController(AppConfig())
        self.controller.active_pod_id = "pod1"
        self.controller._active_client = Mock(
            api_key="test-key", graphql_url="https://example.invalid/graphql"
        )
        self.logs = []
        self.errors = []
        self.active_pods = []
        self.controller.log_line.connect(self.logs.append)
        self.controller.error.connect(self.errors.append)
        self.controller.active_pod_changed.connect(self.active_pods.append)

    def _request_and_run(self, stop_client):
        with patch("job_controller.threading.Thread") as thread, patch(
            "job_controller.RunPodClient", return_value=stop_client
        ):
            self.controller.stop_pod()
            thread.return_value.start.assert_called_once()
            target = thread.call_args.kwargs["target"]
            target(*thread.call_args.kwargs["args"])

    def test_stop_terminates_only_active_pod_and_clears_it(self):
        stop_client = Mock()
        self._request_and_run(stop_client)
        stop_client.terminate_pod.assert_called_once_with("pod1")
        self.assertIsNone(self.controller.active_pod_id)
        self.assertTrue(self.controller.stop_succeeded)
        self.assertTrue(self.controller._stop_done.is_set())
        self.assertIn(None, self.active_pods)
        self.assertFalse(self.errors)

    def test_failed_stop_keeps_pod_available_for_retry(self):
        stop_client = Mock()
        stop_client.terminate_pod.side_effect = RuntimeError("API unavailable")
        stop_client.pod_snapshot.return_value = {"id": "pod1"}
        self._request_and_run(stop_client)
        self.assertEqual(self.controller.active_pod_id, "pod1")
        self.assertFalse(self.controller.stop_succeeded)
        self.assertFalse(self.controller._stop_requested.is_set())
        self.assertEqual(self.active_pods[-1], "pod1")
        self.assertIn("may still be billing", self.errors[-1])

    def test_failed_mutation_with_absent_pod_counts_as_stopped(self):
        stop_client = Mock()
        stop_client.terminate_pod.side_effect = RuntimeError("API timeout")
        stop_client.pod_snapshot.return_value = None
        stop_client.list_pods.return_value = []
        self._request_and_run(stop_client)
        self.assertTrue(self.controller.stop_succeeded)
        self.assertIsNone(self.controller.active_pod_id)
        self.assertFalse(self.errors)

    def test_transient_status_404_does_not_claim_termination(self):
        stop_client = Mock()
        stop_client.terminate_pod.side_effect = RuntimeError("API timeout")
        stop_client.pod_snapshot.return_value = None
        stop_client.list_pods.return_value = [{"id": "pod1"}]
        self._request_and_run(stop_client)
        self.assertFalse(self.controller.stop_succeeded)
        self.assertEqual(self.controller.active_pod_id, "pod1")
        self.assertIn("may still be billing", self.errors[-1])


if __name__ == "__main__":
    unittest.main()
