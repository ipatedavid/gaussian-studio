"""A vulkaninfo that hangs counts as no device, so the other driver setups still run."""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

import pipeline


def hang(*args, **kwargs):
    raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 90))


class VulkanProbeTests(unittest.TestCase):
    def test_hung_probe_reports_no_device(self):
        with patch.object(pipeline.subprocess, "run", side_effect=hang):
            hardware, text = pipeline._vulkan_probe()
        self.assertEqual(hardware, [])
        self.assertIn("timed out", text)

    def test_hung_default_setup_falls_back_to_egl(self):
        with patch.object(pipeline.subprocess, "run", side_effect=hang), \
             patch.object(pipeline.shutil, "which", return_value="/usr/bin/vulkaninfo"), \
             patch.object(pipeline, "_ldconfig_text", return_value="libEGL_nvidia.so.0"), \
             patch.object(pipeline, "_try_vulkan_icd",
                          return_value=(["NVIDIA GeForce RTX 5090"], "")) as egl:
            pipeline._ensure_vulkan_gpu()
        egl.assert_called_once_with("libEGL_nvidia.so.0")


if __name__ == "__main__":
    unittest.main()
