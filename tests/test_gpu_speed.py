from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from gpu_speed import REFERENCE, Run, SpeedModel, duration_text, gpu_id_for, parse_text

A5000 = "NVIDIA RTX A5000"
LOG = """\
[15:46:20] Selected GPU: RTX A5000 (SECURE, Low stock, $0.270/hr quoted)
[15:48:04] [pipeline] extracting about 147 frames at 3 fps (every 10 source frames)
[15:49:20] [match]   Camera 1: images 147, 3840x2160, opencv-fisheye, focal 1225.5 px (given)
[15:52:00] * [0] NVIDIA RTX A5000 (discrete, 24564 MB)
[15:52:03] step      1/50000 (  0%)  splats 116482  [elapsed 00:00:01.946 | ETA --:--]
[17:38:16] step  50000/50000 (100%)  splats 3000000  [elapsed 01:45:29.943 | ETA 00:00:00.000]
[17:38:19] Training complete. Steps: 50000   Time: 01:45:29.943
"""


def run(gpu, ms, splats=3_000_000, kind="video", resolution="3840x2160", setup=600.0):
    return Run(gpu, splats, kind, resolution, ms, ms * 50, setup)


class GpuSpeedTests(unittest.TestCase):
    def test_reads_a_finished_training_from_its_log(self):
        parsed = parse_text(LOG)
        self.assertEqual(parsed.gpu_id, A5000)
        self.assertEqual(parsed.splats, 3_000_000)
        self.assertEqual(parsed.kind, "video")
        self.assertEqual(parsed.resolution, "3840x2160")
        self.assertAlmostEqual(parsed.ms_per_step, 126.6, places=1)
        self.assertIsNone(parse_text(LOG.replace("Training complete", "Training stopped")))

    def test_names_map_to_runpod_ids(self):
        self.assertEqual(gpu_id_for("RTX A5000 (SECURE)"), A5000)
        self.assertEqual(gpu_id_for("NVIDIA GeForce RTX 3090"), REFERENCE)

    def test_specs_until_a_job_measures_the_gpu(self):
        model = SpeedModel([])
        self.assertFalse(model.speed(A5000).measured)
        self.assertIsNone(model.speed("AMD Instinct MI300X OAM"))

        model = SpeedModel([run(REFERENCE, 100), run(A5000, 125)])
        self.assertTrue(model.speed(A5000).measured)
        self.assertAlmostEqual(model.speed(A5000).factor, 0.8)
        self.assertTrue(model.speed(REFERENCE).measured)

    def test_different_workloads_are_not_compared(self):
        model = SpeedModel([run(REFERENCE, 70, splats=1_000_000), run(A5000, 240, kind="photos")])
        self.assertFalse(model.speed(A5000).measured)

    def test_estimate_scales_past_jobs_by_speed(self):
        model = SpeedModel([run(REFERENCE, 100, setup=600)])
        on_3090 = model.estimate(REFERENCE, 30_000, 3_000_000, "video")
        self.assertTrue(on_3090.from_history)
        self.assertAlmostEqual(on_3090.train_s, 3000)
        self.assertAlmostEqual(on_3090.total_s, 3600)
        faster = model.estimate("NVIDIA GeForce RTX 4090", 30_000, 3_000_000, "video")
        self.assertAlmostEqual(faster.train_s, 3000 / 1.6)
        # Photos with no photo jobs yet: the video time, doubled.
        self.assertAlmostEqual(model.estimate(REFERENCE, 30_000, 3_000_000, "photos").train_s, 6000)

    def test_estimate_without_history_uses_defaults(self):
        estimate = SpeedModel([]).estimate(REFERENCE, 30_000, 1_000_000, "video")
        self.assertFalse(estimate.from_history)
        self.assertAlmostEqual(estimate.train_s, 30_000 * 0.070)

    def test_duration_text(self):
        self.assertEqual(duration_text(20), "1 min")
        self.assertEqual(duration_text(45 * 60), "45 min")
        self.assertEqual(duration_text(125 * 60), "2 h 05 min")


if __name__ == "__main__":
    unittest.main()
