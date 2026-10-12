from __future__ import annotations

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
from config import QUALITY_LEVELS, scaled_steps
from storage import InputPlan, VideoInfo
from scp_client import ScpClient, TransferError


class QualityCommandTests(options.SpirulaOptionTests):
    """Runs the real _run_spirula with a fake binary and checks the quality flags."""

    def run_spirula(self, **kwargs):
        quality = kwargs.pop("quality", "medium")
        override = kwargs.pop("override", False)
        vignetting = kwargs.pop("vignetting", False)
        with patch.multiple(pipeline, SPIRULA_QUALITY=quality, STEPS_OVERRIDE=override,
                            MAX_STEPS=12_345, SPIRULA_VIGNETTING=vignetting):
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
        for setting in ("PIPELINE_SPIRULA_QUALITY=high", "PIPELINE_SPIRULA_VIGNETTING=1",
                        "PIPELINE_STEPS_OVERRIDE=0"):
            self.assertIn(setting, captured[0])
        with self.assertRaises(TransferError):
            client.run_pipeline(10, 1, lambda line: None, engine="spirula", spirula_quality="max")


if __name__ == "__main__":
    unittest.main()
