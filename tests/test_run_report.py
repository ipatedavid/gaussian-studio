from __future__ import annotations

import csv
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "worker"))

from PySide6 import QtCore

import pipeline
import run_report
from config import AppConfig
from job_manager import JobManager

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])

# A High run that crosses midnight: pod requested 23:00:00, terminated 01:00:30.
LOG = """[22:59:58] Selected GPU: RTX A5000 (SECURE, High stock, $0.270/hr quoted)
[23:00:00] Sending one RunPod pod creation request...
[23:01:00] Workspace has 56.8 GiB free; uploading worker scripts.
[23:03:00] [pipeline] verifying runtime
[23:04:00] [pipeline] processing images with Spirula SfM
[23:04:01] [pipeline] 100 images prepared for training
[23:05:00] [run]     Reprojection error: mean 1.247 px, median 0.953 px, over 9 observations
[23:10:00] [pipeline] Spirula SfM registered 98 of 100 images
[23:10:00] [stats] capture {"images": 100, "registered": 98, "width": 6000, "height": 4000, "reprojection_error_px": 0.8}
[23:20:00] [pipeline] starting Spirula training for 1000 iterations (quality: high, up to 3,000 splats; floater suppression: strong)
[23:20:05] Cameras: 98 (98 after splitting), seed points: 1234 (train_frame_scale=1.0)
[23:20:10] Warning: training may run out of GPU memory. Projected peak: 22.43 ± 4.12 GiB   free
[23:21:00] step      1/1000 (  0%)  splats 1234  [elapsed 00:00:01] rgb_loss=0.1  ssim=0  psnr=0
[23:30:00] step    201/1000 ( 20%)  splats 3000  [elapsed] rgb_loss=0.1  ssim=0.7  psnr=20.0
[23:40:00] step    501/1000 ( 50%)  splats 3000  [elapsed] rgb_loss=0.1  ssim=0.8  psnr=25.0
[23:50:00] step    951/1000 ( 95%)  splats 3000  [elapsed] rgb_loss=0.1  ssim=0.9  psnr=27.0
[00:50:00] [vram] training peak: 19.5 of 24.0 GiB (81%); GPU busy 96% on average
[00:50:00] [pipeline] exporting from /workspace/job/run
[00:51:00] Processing finished; downloading the result files...
[00:55:00] Results saved to /somewhere
[01:00:30] Pod abc terminated.
"""


def write_ply(path: Path, opacities: list[float]) -> None:
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
              "property float x\nproperty float opacity\nend_header\n" % len(opacities))
    body = b"".join(struct.pack("<2f", 0.0, value) for value in opacities)
    path.write_bytes(header.encode() + body)


class LogParsingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.log = self.root / "job.log"
        self.log.write_text(LOG)

    def tearDown(self):
        self.tmp.cleanup()

    def test_phases_cost_and_curve(self):
        row = run_report.build_row({"job_id": "j", "log_path": str(self.log),
                                    "cost_per_hour": 0.27, "started_at": 0})
        self.assertEqual(row["billed_min"], 120.5)            # across midnight
        self.assertAlmostEqual(row["cost_usd"], 120.5 / 60 * 0.27, places=3)
        self.assertEqual(row["pod_start_min"], 1.0)
        self.assertEqual(row["inputs_min"], 2.0)
        self.assertEqual(row["sfm_min"], 6.0)
        self.assertEqual(row["prep_min"], 10.0)
        self.assertEqual(row["training_min"], 90.0)
        self.assertEqual(row["download_min"], 4.0)
        self.assertEqual(row["quality"], "high")
        self.assertEqual(row["splat_cap"], 3000)
        self.assertEqual(row["cap_reached_step"], 201)
        self.assertEqual(row["cap_reached_pct"], 20)
        self.assertEqual(row["psnr_mid"], 25.0)
        self.assertEqual(row["psnr_end"], 27.0)
        self.assertEqual(row["late_gain_db"], 2.0)
        self.assertEqual(row["seed_points"], 1234)
        self.assertEqual(row["vram_peak_gib"], 19.5)
        self.assertEqual(row["gpu_busy_pct"], 96)
        self.assertEqual(row["vram_projected_gib"], 22.43)
        # the worker's [stats] line wins over Spirula's own SfM printout
        self.assertEqual(row["reprojection_error_px"], 0.8)
        self.assertEqual((row["width"], row["registered"]), (6000, 98))

    def test_zero_error_from_stats_keeps_spirulas_mean(self):
        self.log.write_text(LOG.replace('"reprojection_error_px": 0.8', '"reprojection_error_px": 0.0'))
        row = run_report.build_row({"job_id": "j", "log_path": str(self.log)})
        self.assertEqual(row["reprojection_error_px"], 1.247)

    def test_price_falls_back_to_the_quote(self):
        row = run_report.build_row({"job_id": "j", "log_path": str(self.log)})
        self.assertEqual(row["price_per_hour"], 0.27)

    def test_job_without_pod_costs_nothing(self):
        self.log.write_text("[10:00:00] Checking the selected GPU's current capacity...\n")
        row = run_report.build_row({"job_id": "j", "log_path": str(self.log),
                                    "cost_per_hour": 0.3})
        self.assertIsNone(row["cost_usd"])

    def test_splat_stats_are_cached_next_to_the_ply(self):
        logit = lambda p: math.log(p / (1 - p))
        write_ply(self.root / "gaussian_splat.ply",
                  [logit(0.5), logit(0.03), logit(0.005), logit(0.9)])
        stats = run_report.cached_splat_stats(self.root)
        self.assertEqual(stats["splats_in_file"], 4)
        self.assertEqual(stats["faint_share"], 0.5)
        self.assertEqual(stats["invisible_share"], 0.25)
        self.assertTrue((self.root / run_report.SPLAT_STATS).is_file())
        (self.root / "gaussian_splat.ply").unlink()
        self.assertEqual(run_report.cached_splat_stats(self.root)["faint_share"], 0.5)

    def test_csv_has_every_column(self):
        path = run_report.write_csv([run_report.build_row(
            {"job_id": "j", "log_path": str(self.log), "rating": 4, "rating_note": "sharp"})],
            self.root / "runs.csv")
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0]), list(run_report.COLUMNS))
        self.assertEqual((rows[0]["rating"], rows[0]["note"]), ("4", "sharp"))


class RatingTests(unittest.TestCase):
    def test_rating_is_kept_in_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = JobManager(AppConfig(), ledger_path=root / "ledger.json",
                                 log_dir=root, history_path=root / "jobs.db")
            manager.history.save({"job_id": "a", "title": "t", "outcome": "finished",
                                  "started_at": 1})
            job = manager.load_history()[0]
            manager.rate_job(job, 5, "crisp")
            again = manager._historic_job(manager.history.recent()[0])
            self.assertEqual((again.rating, again.rating_note), (5, "crisp"))
            manager.write_runs_table().join()
            with (root / "runs.csv").open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["rating"], "5")
            manager.history.close()


class CaptureStatsTests(unittest.TestCase):
    def test_reads_errors_and_tracks_from_the_sparse_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            model = dataset / "sparse" / "0"
            model.mkdir(parents=True)
            (model / "images.bin").write_bytes(struct.pack("<Q", 2))
            points = struct.pack("<Q", 2)
            for index, (error, track) in enumerate(((1.0, 2), (0.5, 4))):
                points += struct.pack("<Q3d3BdQ", index, 0, 0, 0, 0, 0, 0, error, track)
                points += b"\0" * 8 * track
            (model / "points3D.bin").write_bytes(points)
            images = dataset / "images"
            images.mkdir()
            png = b"\x89PNG\r\n\x1a\n" + b"\0" * 8 + struct.pack(">II", 640, 480)
            (images / "a.png").write_bytes(png)
            stats = pipeline._capture_stats(dataset, images, 3, 2)
        self.assertEqual(stats, {"images": 3, "registered": 2, "width": 640, "height": 480,
                                 "sparse_points": 2, "reprojection_error_px": 0.75,
                                 "mean_track_length": 3.0})


if __name__ == "__main__":
    unittest.main()
