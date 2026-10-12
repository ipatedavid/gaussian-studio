from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from PySide6 import QtCore

from config import AppConfig
from drive_client import DriveFile, DriveListing
from job_controller import JobController
from models import PodInfo
from pod_ledger import PodLedger
from storage import VideoInfo

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


class FakeRunPod:
    terminated: list[str] = []

    def __init__(self, *args):
        pass

    def select_gpu_types(self, *args, **kwargs):
        return [{"id": "A40", "_cloud_type": "SECURE", "_available": True,
                 "_display_name": "A40", "_stock_status": "High", "_hourly_price": 0.4}]

    def create_pod(self, **kwargs):
        FakeRunPod.last_create = kwargs
        return PodInfo(pod_id="pod1", name="gaussian-" + kwargs["job_id"], status="RUNNING",
                       cost_per_hour=0.41)

    def pod_snapshot(self, pod_id):
        return {"id": pod_id, "status": "RUNNING"}

    @staticmethod
    def direct_ssh_endpoint(snapshot):
        return ("203.0.113.5", 22022)

    @staticmethod
    def is_public_address(host):
        return True

    def recent_system_logs(self, pod_id, seen):
        return []

    def list_pods(self):
        return []

    def terminate_pod(self, pod_id):
        FakeRunPod.terminated.append(pod_id)


class FakeScp:
    calls: list[tuple] = []

    def __init__(self, *args):
        pass

    def wait_ready(self, **kwargs):
        pass

    def free_bytes(self):
        return 10 ** 13

    def upload_scripts(self, *args):
        pass

    def verify_video_tools(self):
        FakeScp.calls.append(("verify_video_tools",))

    def upload(self, images, reserve, log):
        FakeScp.calls.append(("upload", len(images)))

    def upload_video(self, video, reserve, log):
        FakeScp.calls.append(("upload_video", video.name))

    def upload_videos(self, videos, reserve, log):
        FakeScp.calls.append(("upload_videos", tuple(video.name for video in videos)))

    def fetch_drive_inputs(self, manifest, log):
        FakeScp.calls.append(("fetch_drive_inputs", manifest["kind"],
                              *(item["name"] for item in manifest["files"])))

    def run_pipeline(self, steps, downscales, log, **kwargs):
        FakeScp.calls.append(("run_pipeline", kwargs))

    def download(self, destination, log):
        destination.mkdir(parents=True)
        (destination / "gaussian_splat.ply").write_bytes(b"ply\n")


class FakeDrive:
    def __init__(self, *args):
        self.video = DriveFile("v1", "walk.MOV", "source_video.mov", 10 ** 9, "m", 1920, 1080, 60000)
        self.low = DriveFile("v2", "low.mp4", "source_video.mp4", 10 ** 9, "n", 1920, 1080, 30000)

    def list_images(self, folder_id, folder_name=""):
        return DriveListing(folder_id, folder_name, (), 0, (self.video, self.low))

    def probe_video(self, video):
        FakeScp.calls.append(("probe_video", video.id))
        return replace(video, fps=30.0)

    def build_video_manifest(self, folder_name, videos):
        # Same naming as the real client.
        names = ([videos[0].dest] if len(videos) == 1 else
                 [f"source_video_{i}{Path(v.name).suffix.lower()}" for i, v in enumerate(videos, 1)])
        return {"kind": "video", "files": [{"name": name} for name in names]}


class JobFlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        key = root / "id_ed25519"
        key.write_text("private")
        Path(str(key) + ".pub").write_text("ssh-ed25519 AAAA test")
        (root / "sa.json").write_text("{}")
        self.photos = root / "photos"
        self.photos.mkdir()
        for index in range(4):
            (self.photos / f"p{index}.jpg").write_bytes(b"x" * 100)
        config = AppConfig()
        config.runpod_api_key = "key"
        config.ssh_private_key = str(key)
        config.google_service_account_file = str(root / "sa.json")
        config.max_gpu_price = 1.0
        self.config = config
        self.ledger = PodLedger(root / "ledger.json")
        self.root = root
        FakeScp.calls = []
        FakeRunPod.terminated = []
        self.patches = [
            patch("job_controller.RunPodClient", FakeRunPod),
            patch("job_controller.ScpClient", FakeScp),
            patch("job_controller.DriveClient", FakeDrive),
            patch("job_controller.time.sleep", lambda seconds: None),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.tmp.cleanup()

    def make_job(self):
        job = JobController(self.config, ledger=self.ledger, log_dir=self.root / "logs")
        job.title = "test"
        job.running = True
        return job

    def test_photo_job_records_and_terminates_its_pod(self):
        job = self.make_job()
        job._run_job(self.photos, ("A40", "SECURE"))
        self.assertEqual(job.outcome, "finished")
        self.assertEqual(FakeRunPod.last_create["job_id"], job.job_id)
        self.assertEqual(FakeRunPod.terminated, ["pod1"])
        entry = self.ledger.entries()[0]
        self.assertEqual((entry["state"], entry["pod_id"]), ("terminated", "pod1"))
        self.assertIn(("upload", 4), FakeScp.calls)
        self.assertTrue(job.results_dir.name.endswith(job.job_id[:6]))
        self.assertTrue((job.results_dir / "job.log").is_file())

    def test_drive_video_job_fetches_on_the_pod(self):
        job = self.make_job()
        job._run_job(
            Path("."), ("A40", "SECURE"), source_type="drive_video", engine="spirula",
            drive_folder=("f1", "captures"), drive_video_id="v1", results_root=self.root / "res",
        )
        self.assertEqual(job.outcome, "finished")
        self.assertIn(("fetch_drive_inputs", "video", "source_video.mov"), FakeScp.calls)
        # Spirula installs ffmpeg itself, so the pod's tools are not checked first.
        self.assertNotIn(("verify_video_tools",), FakeScp.calls)
        pipeline_call = next(call for call in FakeScp.calls if call[0] == "run_pipeline")
        self.assertEqual(pipeline_call[1]["source_type"], "video")
        self.assertEqual(pipeline_call[1]["video_fps"], 3.0)
        self.assertEqual(job.results_dir.parent, self.root / "res" / "walk")

    def test_two_drive_videos_are_fetched_as_one_job(self):
        job = self.make_job()
        job._run_job(
            Path("."), ("A40", "SECURE"), source_type="drive_video", engine="spirula",
            drive_folder=("f1", "captures"), drive_video_id="v1,v2", results_root=self.root / "res",
        )
        self.assertEqual(job.outcome, "finished")
        self.assertIn(("fetch_drive_inputs", "video", "source_video_1.mov", "source_video_2.mp4"),
                      FakeScp.calls)
        self.assertIn(("probe_video", "v2"), FakeScp.calls)
        pipeline_call = next(call for call in FakeScp.calls if call[0] == "run_pipeline")
        self.assertEqual(pipeline_call[1]["video_fps"], 3.0)
        # 60 s and 30 s of Drive video at 3 fps.
        self.assertIn("about 270 frames (3 per second of video)", (job.results_dir / "job.log").read_text())

    def test_two_local_videos_upload_in_order(self):
        high, low = self.root / "high.MOV", self.root / "low.MOV"
        high.write_bytes(b"x" * 1000)
        low.write_bytes(b"y" * 500)
        info = VideoInfo(40.0, 3840, 2160, 30.0)
        with patch("storage.probe_video", return_value=info):
            job = self.make_job()
            job._run_job(high, ("A40", "SECURE"), source_type="video", engine="spirula",
                         more_videos=(low,))
        self.assertEqual(job.outcome, "finished")
        self.assertIn(("upload_videos", ("high.MOV", "low.MOV")), FakeScp.calls)
        pipeline_call = next(call for call in FakeScp.calls if call[0] == "run_pipeline")
        self.assertEqual(pipeline_call[1]["video_fps"], 3.0)
        self.assertEqual(job.results_dir.parent, self.root / "results" / "high")

    def test_two_videos_with_nerfstudio_are_refused_before_a_pod(self):
        high, low = self.root / "high.MOV", self.root / "low.MOV"
        high.write_bytes(b"x")
        low.write_bytes(b"y")
        with patch("storage.probe_video", return_value=VideoInfo(40.0, 1920, 1080, 30.0)):
            job = self.make_job()
            job._run_job(high, ("A40", "SECURE"), source_type="video", engine="nerfstudio",
                         more_videos=(low,))
        self.assertEqual(job.outcome, "failed")
        self.assertFalse(hasattr(FakeRunPod, "last_create") and
                         FakeRunPod.last_create.get("job_id") == job.job_id)

    def test_cancel_before_creation_creates_no_pod(self):
        job = self.make_job()
        job.stop_pod()
        job._run_job(self.photos, ("A40", "SECURE"))
        self.assertEqual(job.outcome, "stopped")
        self.assertFalse(hasattr(FakeRunPod, "last_create") and
                         FakeRunPod.last_create.get("job_id") == job.job_id)
        self.assertEqual(FakeRunPod.terminated, [])


if __name__ == "__main__":
    unittest.main()
