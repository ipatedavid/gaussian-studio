from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "tests"))

import pipeline
import test_spirula_options as options
from config import QUALITY_LEVELS, AppConfig, scaled_steps
from storage import InputPlan, VideoInfo
from scp_client import ScpClient, TransferError


class QualityCommandTests(options.SpirulaOptionTests):
    """Runs the real _run_spirula with a fake binary and checks the quality flags."""

    def run_spirula(self, **kwargs):
        quality = kwargs.pop("quality", "medium")
        override = kwargs.pop("override", False)
        vignetting = kwargs.pop("vignetting", False)
        splats = kwargs.pop("splats", 0)
        with patch.multiple(pipeline, SPIRULA_QUALITY=quality, STEPS_OVERRIDE=override,
                            MAX_STEPS=12_345, SPIRULA_VIGNETTING=vignetting,
                            SPIRULA_MAX_SPLATS=splats):
            return super().run_spirula(**kwargs)

    def test_preset_sets_budget_and_steps(self):
        train, _, _, _ = self.run_spirula(quality="high")
        self.assertEqual(self.flag(train, "--quality"), "high")
        # Five images: the level's own steps.
        self.assertEqual(self.flag(train, "--num-iterations"), "50000")

    def test_big_captures_train_longer(self):
        with patch.object(pipeline, "_registered_images", return_value=450):
            train, _, _, _ = self.run_spirula(quality="medium")
        self.assertEqual(self.flag(train, "--num-iterations"), "45000")
        with patch.object(pipeline, "_registered_images", return_value=2000):
            train, _, _, _ = self.run_spirula(quality="high")
        self.assertEqual(self.flag(train, "--num-iterations"), "125000")

    def test_steps_set_by_hand_are_not_scaled(self):
        with patch.object(pipeline, "_registered_images", return_value=2000):
            train, _, _, _ = self.run_spirula(quality="medium", override=True)
        self.assertEqual(self.flag(train, "--num-iterations"), "12345")

    def test_vignetting_switches_the_camera_model(self):
        train, _, _, _ = self.run_spirula(vignetting=True)
        self.assertEqual(self.flag(train, "--ppisp-param-type"), "no_crf")
        train, _, _, _ = self.run_spirula(vignetting=False)
        self.assertNotIn("--ppisp-param-type", train)

    def test_splat_limit_set_by_hand(self):
        output = io.StringIO()
        with patch.object(pipeline, "_registered_images", return_value=850), \
                contextlib.redirect_stdout(output):
            train, _, _, _ = self.run_spirula(quality="high", splats=6_000_000)
        self.assertEqual(self.flag(train, "--quality"), "high")
        self.assertEqual(self.flag(train, "--cap-max"), "6000000")
        # Steps still follow the photos only.
        self.assertEqual(self.flag(train, "--num-iterations"), "125000")
        self.assertIn("up to 6,000,000 splats", output.getvalue())

    def test_preset_limit_adds_no_flag(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            train, _, _, _ = self.run_spirula(quality="high")
        self.assertNotIn("--cap-max", train)
        self.assertIn("up to 3,000,000 splats", output.getvalue())

    def test_override_keeps_budget_but_sets_steps(self):
        train, _, _, _ = self.run_spirula(quality="ultra", override=True)
        self.assertEqual(self.flag(train, "--quality"), "ultra")
        self.assertEqual(self.flag(train, "--num-iterations"), "12345")


for _name in [n for n in dir(options.SpirulaOptionTests) if n.startswith("test_")]:
    setattr(QualityCommandTests, _name, None)


class StepScalingTests(unittest.TestCase):
    def test_rule(self):
        cases = {None: 30_000, 0: 30_000, 120: 30_000, 300: 30_000, 301: 31_000,
                 450: 45_000, 600: 60_000, 750: 75_000, 751: 75_000, 5000: 75_000}
        for images, steps in cases.items():
            self.assertEqual(scaled_steps(30_000, images), steps, images)
        self.assertEqual(scaled_steps(15_000, 1000), 37_500)
        self.assertEqual(scaled_steps(80_000, 10_000), 200_000)

    def test_app_and_pod_agree(self):
        for base in (15_000, 30_000, 50_000, 80_000):
            for images in (None, 10, 299, 300, 301, 333, 517, 749, 750, 1234, 99_999):
                self.assertEqual(scaled_steps(base, images),
                                 pipeline.scaled_steps(base, images), (base, images))

    def test_plan_image_counts(self):
        photos = InputPlan("images", Path("/p"), tuple(Path(f"{i}.jpg") for i in range(7)), 1, 1, 30)
        self.assertEqual(photos.image_count, 7)
        video = InputPlan("video", Path("/v.mp4"), (), 1, 1, 30, 450,
                          VideoInfo(150.0, 1920, 1080, 30.0))
        self.assertEqual(video.image_count, 450)
        unknown = InputPlan("drive_video", Path("/r"), (), 1, 1, 30, 600,
                            VideoInfo(0.0, 0, 0, 0.0))
        self.assertIsNone(unknown.image_count)


class PresetTableTests(unittest.TestCase):
    def test_app_and_pod_agree_on_presets(self):
        for key, (_label, steps, splats) in QUALITY_LEVELS.items():
            self.assertEqual(pipeline.SPIRULA_PRESETS[key], (steps, splats))

    def test_settings_reach_the_pod(self):
        client = ScpClient(Path("/tmp/test-key"), "example", 1234)
        captured = []

        class FakeProcess:
            returncode = 0
            stdout = None

            def __init__(self, args, **kwargs):
                captured.append(args[-1])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def wait(self):
                return 0

        with patch("scp_client.subprocess.Popen", FakeProcess), \
                patch("scp_client._stream_output", lambda process, log: None):
            client.run_pipeline(10, 1, lambda line: None, engine="spirula",
                                spirula_quality="high", steps_override=False,
                                spirula_vignetting=True)
            client.run_pipeline(10, 1, lambda line: None, engine="spirula",
                                spirula_quality="high", spirula_max_splats=6_000_000)
        for setting in ("PIPELINE_SPIRULA_QUALITY=high", "PIPELINE_SPIRULA_VIGNETTING=1",
                        "PIPELINE_STEPS_OVERRIDE=0"):
            self.assertIn(setting, captured[0])
        self.assertNotIn("PIPELINE_SPIRULA_MAX_SPLATS", captured[0])
        self.assertIn("PIPELINE_SPIRULA_MAX_SPLATS=6000000 PIPELINE_SPIRULA_VIGNETTING=0",
                      captured[1])
        with self.assertRaises(TransferError):
            client.run_pipeline(10, 1, lambda line: None, engine="spirula",
                                spirula_quality="high", spirula_max_splats=50)
        with self.assertRaises(TransferError):
            client.run_pipeline(10, 1, lambda line: None, engine="spirula", spirula_quality="max")


class SplatBudgetTests(unittest.TestCase):
    def test_preset_or_set_by_hand(self):
        config = AppConfig()
        config.spirula_quality, config.spirula_max_splats = "high", 6_000_000
        self.assertEqual(config.splat_budget(), 3_000_000)
        config.splats_override = True
        self.assertEqual(config.splat_budget(), 6_000_000)


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
from PySide6 import QtWidgets
app = QtWidgets.QApplication([])
import app as studio
studio.JobManager = job_manager.JobManager

window = studio.MainWindow()
window.show()
studio.choose(window.quality, "high")
assert window._splat_budget() == 3_000_000 and not window.splats.isVisible()
assert window.splats.value() == 3.0, window.splats.value()
assert window.splats_mode.itemText(0) == "Auto · 3 million", window.splats_mode.itemText(0)
studio.choose(window.splats_mode, "custom")
assert window.splats.isVisible()
window.splats.setValue(6.0)
assert window._splat_budget() == 6_000_000, window._splat_budget()
assert "6M splats" in window._spirula_options_text(), window._spirula_options_text()
studio.choose(window.quality, "ultra")
assert window._splat_budget() == 6_000_000  # set by hand wins over the level
window._save_settings()
saved = config.AppConfig()
saved.load_state()
assert saved.splats_override and saved.spirula_max_splats == 6_000_000
assert saved.splat_budget() == 6_000_000
studio.choose(window.splats_mode, "auto")
assert window._splat_budget() == 10_000_000 and window.splats.value() == 10.0
# the subject dropdown stands for the three old switches
studio.choose(window.subject, "turntable")
window._save_settings()
saved.load_state()
assert saved.spirula_object and saved.spirula_turntable and not saved.spirula_main_scene
studio.choose(window.subject, "main")
window._save_settings()
saved.load_state()
assert saved.spirula_main_scene and not saved.spirula_object and not saved.spirula_turntable
# depth strength shows only with a depth guidance level
studio.choose(window.guidance, "off")
assert not window.depth_weight.isVisible()
studio.choose(window.guidance, "depth")
assert window.depth_weight.isVisible()
assert not window.one_surface.isEnabled()
# people masking: on by default, sits out object scans, saved with the settings
assert window.people_masks.isChecked() and window.people_masks.isEnabled()
assert "people masked" in window._spirula_options_text(), window._spirula_options_text()
studio.choose(window.subject, "object")
assert not window.people_masks.isEnabled()
assert "people masked" not in window._spirula_options_text()
studio.choose(window.subject, "scene")
assert window.people_masks.isEnabled()
window.people_masks.setChecked(False)
window._save_settings()
saved.load_state()
assert not saved.spirula_people_masks
print("OK")
"""


class SplatLimitWindowTests(unittest.TestCase):
    def test_splat_limit_control(self):
        result = subprocess.run(
            [sys.executable, "-c", WINDOW_SCRIPT, str(ROOT / "desktop")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        self.assertIn("OK", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
