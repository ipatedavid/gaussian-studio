"""Photos and videos together in one job, from a local or a Google Drive folder."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))

import pipeline
from drive_client import DriveClient, DriveFile, DriveListing
from scp_client import ScpClient
from storage import GIB, VideoInfo, plan_drive_mixed, plan_mixed
import test_job_flow as flow
from test_job_flow import FakeDrive, FakeRunPod, FakeScp


def drive_photo(index: int) -> DriveFile:
    return DriveFile(f"p{index}", f"IMG_{index}.JPG", f"IMG_{index}.JPG", 8 * 1024 ** 2, "m")


class PlanTests(unittest.TestCase):
    def test_local_photos_and_videos_plan_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            photos = []
            for index in range(5):
                photo = folder / f"p{index}.jpg"
                photo.write_bytes(b"x" * 1000)
                photos.append(photo)
            walk = folder / "walk.mp4"
            walk.write_bytes(b"v" * 5000)
            with patch("storage.probe_video", return_value=VideoInfo(60, 3840, 2160, 30)):
                plan = plan_mixed(folder, tuple(photos), (walk,), 0, 1, 3.0)
        self.assertEqual((plan.kind, plan.pipeline_source_type), ("mixed", "mixed"))
        self.assertEqual((plan.photo_count, plan.video_count, plan.frame_target), (5, 1, 180))
        self.assertEqual(plan.image_count, 185)
        self.assertEqual(plan.input_bytes, 10000)
        self.assertEqual(plan.results_parent, folder / "results")
        self.assertTrue(plan.has_photos and plan.has_videos and not plan.is_video)

    def test_drive_photos_and_videos_need_room_for_both(self):
        photos = tuple(drive_photo(index) for index in range(100))
        walk = DriveFile("v1", "walk.MOV", "source_video.mov", 2 * GIB, "m", 3840, 2160, 60000)
        plan = plan_drive_mixed("f1", "Room", photos, (walk,), Path("/r"), 0, 1, 3.0)
        alone = plan_drive_mixed("f1", "Room", photos[:1], (walk,), Path("/r"), 0, 1, 3.0)
        self.assertEqual((plan.kind, plan.photo_count, plan.video_count), ("drive_mixed", 100, 1))
        self.assertEqual(plan.job_title, "Room · photos + 1 video")
        self.assertGreater(plan.working_reserve, alone.working_reserve + 5 * 99 * 8 * 1024 ** 2 - 1)

    def test_needs_photos_and_videos(self):
        with self.assertRaises(ValueError):
            plan_drive_mixed("f1", "Room", (), (), Path("/r"), 0, 1, 3.0)


class PodStagingTests(unittest.TestCase):
    def run_spirula(self, photos: int, videos: list[str], *, fisheye=False):
        """Run the real _run_spirula on a mixed input with a fake binary."""
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed, train_dir, input_dir = root / "processed", root / "train", root / "input"
            input_dir.mkdir()
            for index in range(photos):
                (input_dir / f"IMG_{index}.JPG").write_bytes(b"jpg")
            paths = [input_dir / name for name in videos]
            for path in paths:
                path.write_bytes(b"video")

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None):
                commands.append(list(args))
                if args[1] == "sam":
                    out = Path(args[args.index("-o") + 1])
                    for index in range(4):
                        (out / f"{index:05d}.png").write_bytes(b"png")
                elif args[1] == "sfm":
                    model = processed / "sparse" / "0"
                    model.mkdir(parents=True)
                    (model / "images.bin").write_bytes(struct.pack("<Q", photos + 4 * len(videos)))
                elif args[1] == "train":
                    (train_dir / "run").mkdir(parents=True)
                    (train_dir / "run" / "splat.ply").write_bytes(b"ply\n")

            sizes = {".JPG": (6000, 4000), ".png": (3840, 2160)}
            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=input_dir, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=train_dir, SOURCE_TYPE="mixed", VIDEO_FPS=3.0,
                SPIRULA_GUIDANCE="off", SPIRULA_RESULT={}, PREVIEW=False, FISHEYE=fisheye,
                FISHEYE_FOCAL_MM=7.5, SENSOR_WIDTH_MM=23.5,
                run_command=fake_run,
                _spirula_install_packages=lambda: None,
                _install_spirula=lambda: Path("/opt/spirula"),
                _ensure_vulkan_gpu=lambda: None,
                _video_timing=lambda video: (10.0, 30.0),
                _image_size=lambda path: sizes[path.suffix],
            ), patch.object(pipeline.shutil, "which", return_value=None):
                image_count, *_ = pipeline._run_spirula(paths)
                staged = sorted(str(p.relative_to(processed / "images"))
                                for p in (processed / "images").rglob("*") if p.is_file())
        return commands, image_count, staged

    def test_photos_and_each_video_get_their_own_folder(self):
        commands, image_count, staged = self.run_spirula(3, ["source_video_1.mov",
                                                             "source_video_2.mp4"])
        self.assertEqual(image_count, 11)
        self.assertEqual([s for s in staged if s.startswith("photos/")],
                         ["photos/IMG_0.JPG", "photos/IMG_1.JPG", "photos/IMG_2.JPG"])
        self.assertEqual(sum(s.startswith("v1/") for s in staged), 4)
        self.assertEqual(sum(s.startswith("v2/") for s in staged), 4)
        sfm = next(c for c in commands if c[1] == "sfm")
        # Photos are not a sequence; each video is. Cameras follow the folders.
        self.assertEqual(sfm[sfm.index("--data-type") + 1], "individual")
        self.assertEqual([sfm[i + 1] for i, flag in enumerate(sfm) if flag == "--sequence"],
                         ["v1", "v2"])
        self.assertNotIn("--camera-mode", sfm)

    def test_one_video_still_gets_a_folder(self):
        _, _, staged = self.run_spirula(3, ["source_video_1.mov"])
        self.assertEqual(sum(s.startswith("v1/") for s in staged), 4)

    def test_fisheye_focal_is_set_per_folder(self):
        commands, _, _ = self.run_spirula(3, ["source_video_1.mov"], fisheye=True)
        sfm = next(c for c in commands if c[1] == "sfm")
        self.assertEqual(sfm[sfm.index("--camera-mode") + 1], "folder")
        self.assertEqual(sfm[sfm.index("--camera-model") + 1], "opencv-fisheye")
        focals = [sfm[i + 1] for i, flag in enumerate(sfm) if flag == "--focal"]
        # 7.5 mm over a 23.5 mm sensor: 6000 px photos and 3840 px frames.
        self.assertEqual(focals, ["photos=1914.9", "v1=1225.5"])


class FileDrive(http.server.BaseHTTPRequestHandler):
    bodies: dict[str, bytes] = {}

    def do_GET(self):
        data = type(self).bodies[self.path.split("/files/")[1].split("?")[0]]
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            chunk = data[first:last + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(data)}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, *args):
        pass


class DriveTests(unittest.TestCase):
    def test_manifest_numbers_the_videos(self):
        client = DriveClient.__new__(DriveClient)
        client.api_base = "https://example"
        client.token_for_pod = lambda: ("tok", 0)
        walk = DriveFile("v1", "walk.MOV", "source_video.mov", 10, "m")
        manifest = client.build_mixed_manifest("Room", (drive_photo(1),), (walk,))
        self.assertEqual(manifest["kind"], "mixed")
        self.assertEqual([f["name"] for f in manifest["files"]], ["IMG_1.JPG", "source_video_1.mov"])

    def test_manifest_needs_photos_and_numbered_videos(self):
        photo = {"id": "p1", "name": "IMG_1.JPG", "size": 1}
        video = {"id": "v1", "name": "source_video_1.mov", "size": 1}
        pipeline._validate_drive_manifest("mixed", [photo, video])
        for files in ([photo], [video], [photo, dict(video, name="source_video.mov")]):
            with self.assertRaises(RuntimeError):
                pipeline._validate_drive_manifest("mixed", files)

    def test_pod_fetches_photos_and_a_ranged_video(self):
        FileDrive.bodies = {"p1": os.urandom(4000), "p2": os.urandom(3000),
                            "v1": os.urandom(3 * 1024 * 1024 + 9)}
        names = {"p1": "IMG_1.JPG", "p2": "IMG_2.JPG", "v1": "source_video_1.mov"}
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FileDrive)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "m.json").write_text(json.dumps({
                    "kind": "mixed", "api_base": f"http://127.0.0.1:{server.server_port}",
                    "token": "t",
                    "files": [{"id": key, "name": names[key], "size": len(body),
                               "md5": hashlib.md5(body).hexdigest()}
                              for key, body in FileDrive.bodies.items()],
                }))
                with patch.multiple(pipeline, JOB_ROOT=root, INPUT_DIR=root / "input",
                                    DRIVE_MANIFEST=root / "m.json", DRIVE_RANGE_BYTES=1024 * 1024):
                    pipeline.fetch_drive_inputs()
                for key, name in names.items():
                    self.assertEqual((root / "input" / name).read_bytes(), FileDrive.bodies[key])
        finally:
            server.shutdown()
            server.server_close()


class UploadTests(unittest.TestCase):
    def test_photos_then_numbered_videos_verified_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo, walk = Path(tmp) / "IMG_1.JPG", Path(tmp) / "walk.MOV"
            photo.write_bytes(b"p" * 10)
            walk.write_bytes(b"v" * 20)
            client = ScpClient.__new__(ScpClient)
            client.host = "h"
            client.free_bytes = lambda: 10 ** 12
            client._upload_batch = Mock()
            client._verify_uploaded_images = Mock()
            client.upload_mixed([photo], [walk], 100)
        destinations = [c.kwargs.get("destination", "") for c in client._upload_batch.call_args_list]
        self.assertTrue(destinations[1].endswith("/input/source_video_1.mov"), destinations)
        self.assertEqual(client._verify_uploaded_images.call_args_list[-1].args[0],
                         {"IMG_1.JPG": 10, "source_video_1.mov": 20})


class MixedScp(FakeScp):
    def upload_mixed(self, images, videos, reserve, log):
        FakeScp.calls.append(("upload_mixed", len(images), tuple(v.name for v in videos)))


class MixedDrive(FakeDrive):
    def list_images(self, folder_id, folder_name=""):
        photos = tuple(drive_photo(index) for index in range(3))
        return DriveListing(folder_id, folder_name, photos, 0, (self.video, self.low))

    def build_mixed_manifest(self, folder_name, photos, videos):
        return {"kind": "mixed", "files": [{"name": p.dest} for p in photos]
                + [{"name": f"source_video_{i}"} for i, _ in enumerate(videos, 1)]}


class MixedJobFlowTests(unittest.TestCase):
    """The real controller, with a fake RunPod, pod and Drive (set up as in test_job_flow)."""

    tearDown = flow.JobFlowTests.tearDown
    make_job = flow.JobFlowTests.make_job

    def setUp(self):
        flow.JobFlowTests.setUp(self)
        FakeRunPod.last_create = None
        for item in (patch("job_controller.ScpClient", MixedScp),
                     patch("job_controller.DriveClient", MixedDrive)):
            item.start()
            self.patches.append(item)

    def test_local_folder_uploads_photos_then_videos(self):
        walk = self.photos / "walk.mp4"
        walk.write_bytes(b"v" * 100)
        job = self.make_job()
        with patch("storage.probe_video", return_value=VideoInfo(20, 1920, 1080, 30)):
            job._run_job(self.photos, ("A40", "SECURE"), source_type="mixed", engine="spirula",
                         more_videos=(walk,))
        self.assertEqual(job.outcome, "finished")
        self.assertIn(("upload_mixed", 4, ("walk.mp4",)), FakeScp.calls)
        pipeline_call = next(call for call in FakeScp.calls if call[0] == "run_pipeline")
        self.assertEqual(pipeline_call[1]["source_type"], "mixed")
        self.assertEqual(job.results_dir.parent, self.photos / "results")

    def test_drive_folder_fetches_photos_and_videos(self):
        job = self.make_job()
        job._run_job(Path("."), ("A40", "SECURE"), source_type="drive_mixed", engine="spirula",
                     drive_folder=("f1", "Room"), drive_video_id="v2",
                     results_root=self.root / "res")
        self.assertEqual(job.outcome, "finished")
        self.assertIn(("fetch_drive_inputs", "mixed", "IMG_0.JPG", "IMG_1.JPG", "IMG_2.JPG",
                       "source_video_1"), FakeScp.calls)
        self.assertIn("3 photos and 1 video (low.mp4", (job.results_dir / "job.log").read_text())

    def test_nerfstudio_refuses_before_creating_a_pod(self):
        walk = self.photos / "walk.mp4"
        walk.write_bytes(b"v" * 100)
        job = self.make_job()
        with patch("storage.probe_video", return_value=VideoInfo(20, 1920, 1080, 30)):
            job._run_job(self.photos, ("A40", "SECURE"), source_type="mixed",
                         engine="nerfstudio", more_videos=(walk,))
        self.assertEqual(job.outcome, "failed")
        self.assertIsNone(FakeRunPod.last_create)
        self.assertNotIn("upload_mixed", [call[0] for call in FakeScp.calls])


WINDOW_SCRIPT = r"""
import os, sys, tempfile
from pathlib import Path
from unittest.mock import patch
os.environ["RUNPOD_API_KEY"] = ""  # no RunPod calls
sys.path.insert(0, sys.argv[1])
tmp = Path(tempfile.mkdtemp())
import config
config.STATE_FILE = tmp / "app_state.json"
import job_manager
real = job_manager.JobManager
job_manager.JobManager = lambda cfg: real(cfg, ledger_path=tmp / "ledger.json",
                                          log_dir=tmp / "logs", history_path=tmp / "jobs.db")
from PySide6 import QtCore, QtWidgets
app = QtWidgets.QApplication([])
import app as studio
studio.JobManager = job_manager.JobManager
from storage import VideoInfo

capture = tmp / "capture"
capture.mkdir()
for index in range(4):
    (capture / f"p{index}.jpg").write_bytes(b"x" * 100)
for name in ("high.mp4", "low.mov"):
    (capture / name).write_bytes(b"v" * 100)

with patch("app.probe_video", return_value=VideoInfo(30, 1920, 1080, 30)), \
     patch("storage.probe_video", return_value=VideoInfo(30, 1920, 1080, 30)):
    window = studio.MainWindow()
    assert window.local_mode.isChecked()
    window.input_folder.setText(str(capture))
    window._local_folder_changed()
    rows = [window.capture_items.item(i).text() for i in range(window.capture_items.count())]
    assert len(rows) == 3 and rows[0].startswith("4 photos"), rows
    assert "high.mp4" in rows[1] and "0:30" in rows[1], rows
    plan = window._source_plan()
    assert (plan.kind, plan.photo_count, plan.video_count) == ("mixed", 4, 2), plan.kind
    assert window.frames_box.isVisibleTo(window)

    window.capture_items.item(2).setCheckState(QtCore.Qt.Unchecked)  # low.mov
    assert window.config.local_unticked == ["low.mov"], window.config.local_unticked
    assert window._source_plan().video_count == 1

    window.capture_items.item(0).setCheckState(QtCore.Qt.Unchecked)  # the photos
    plan = window._source_plan()
    assert (plan.kind, plan.source.name) == ("video", "high.mp4"), plan.kind

    window.capture_items.item(1).setCheckState(QtCore.Qt.Unchecked)
    try:
        window._source_plan()
        raise AssertionError("nothing ticked should not plan")
    except ValueError as exc:
        assert "Tick" in str(exc), exc
    assert not window.frames_box.isVisibleTo(window)

    window.capture_items.item(0).setCheckState(QtCore.Qt.Checked)
    assert window._source_plan().kind == "images"
    window._save_settings()
    saved = config.AppConfig()
    saved.load_state()
    assert saved.input_kind == "local" and saved.local_unticked == ["high.mp4", "low.mov"]

    # A different folder starts with everything ticked again.
    other = tmp / "other"
    other.mkdir()
    for index in range(3):
        (other / f"q{index}.jpg").write_bytes(b"x")
    window.input_folder.setText(str(other))
    window._local_folder_changed()
    assert window.config.local_unticked == [] and window.capture_items.count() == 1
print("OK")
"""


class WindowTests(unittest.TestCase):
    def test_checklist_drives_the_plan(self):
        # Widgets need their own QApplication; other modules create a plain one.
        result = subprocess.run(
            [sys.executable, "-c", WINDOW_SCRIPT, str(ROOT / "desktop")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        self.assertIn("OK", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
