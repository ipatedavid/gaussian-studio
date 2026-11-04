from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))

from PySide6 import QtCore

from config import AppConfig
from job_controller import JobController, power_cap_warning
from scp_client import ScpClient

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


def smi(stdout: str, code: int = 0):
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


class GpuPowerTests(unittest.TestCase):
    def client(self):
        return ScpClient(Path("/nonexistent"), "80.15.7.37", 51282)

    def test_reads_limits(self):
        with patch.object(ScpClient, "_ssh", return_value=smi(
                "NVIDIA GeForce RTX 5080, 250.00, 360.00, 3120\n")):
            gpus = self.client().gpu_power()
        self.assertEqual(gpus, [{"name": "NVIDIA GeForce RTX 5080", "limit_w": 250.0,
                                 "stock_w": 360.0, "max_sm_mhz": "3120"}])

    def test_cards_without_power_figures_report_nothing(self):
        with patch.object(ScpClient, "_ssh", return_value=smi("Tesla T4, [N/A], [N/A], 1590\n")):
            self.assertIsNone(self.client().gpu_power())
        with patch.object(ScpClient, "_ssh", return_value=smi("", 9)):
            self.assertIsNone(self.client().gpu_power())

    def test_only_a_real_cap_warns(self):
        capped = {"name": "RTX 5080", "limit_w": 250.0, "stock_w": 360.0}
        stock = {"name": "RTX 3090", "limit_w": 350.0, "stock_w": 350.0}
        trimmed = {"name": "RTX 3090", "limit_w": 320.0, "stock_w": 350.0}  # 91%: normal
        self.assertIn("250 W of its stock 360 W", power_cap_warning(capped, "80.15.7.37"))
        self.assertIsNone(power_cap_warning(stock, "h"))
        self.assertIsNone(power_cap_warning(trimmed, "h"))

    def test_a_capped_pod_warns_and_the_job_carries_on(self):
        job = JobController(AppConfig())
        logs, warnings = [], []
        job.log_line.connect(logs.append, QtCore.Qt.DirectConnection)
        job.warning.connect(warnings.append, QtCore.Qt.DirectConnection)
        with patch.object(ScpClient, "_ssh", return_value=smi(
                "NVIDIA GeForce RTX 5080, 250.00, 360.00, 3120\n")):
            job._check_gpu_power(self.client())
        self.assertEqual(len(warnings), 1)
        self.assertTrue(any("power limit 250 W (stock 360 W)" in line for line in logs))

    def test_a_failed_check_never_stops_the_job(self):
        job = JobController(AppConfig())
        warnings = []
        job.warning.connect(warnings.append, QtCore.Qt.DirectConnection)
        with patch.object(ScpClient, "_ssh", side_effect=subprocess.TimeoutExpired("ssh", 30)):
            job._check_gpu_power(self.client())
        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
