"""Several videos of one scene in a single job (Spirula Studio only)."""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import shutil
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


import pipeline
from drive_client import DriveClient, DriveFile
from scp_client import ScpClient
from storage import VideoInfo, plan_drive_video, plan_input


class PodPipelineTests(unittest.TestCase):
    def run_spirula(self, videos, durations, *, fisheye=False):
        """Run the real _run_spirula with a fake binary and record its commands."""
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "processed"
            train_dir = root / "train"
            paths = [root / name for name in videos]

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None):
                commands.append(list(args))
                if args[1] == "sam":
                    out = Path(args[args.index("-o") + 1])
                    for index in range(4):
                        (out / f"{index:05d}.png").write_bytes(b"png")
                elif args[1] == "sfm":
                    model = processed / "sparse" / "0"
                    model.mkdir(parents=True)
                    (model / "images.bin").write_bytes(struct.pack("<Q", 4 * len(videos)))
                elif args[1] == "train":
                    (train_dir / "run").mkdir(parents=True)
                    (train_dir / "run" / "splat.ply").write_bytes(b"ply\n")

            timing = dict(zip(paths, durations))
            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=root, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=train_dir, SOURCE_TYPE="video", VIDEO_FPS=3.0,
                SPIRULA_GUIDANCE="off", SPIRULA_RESULT={}, PREVIEW=False, FISHEYE=fisheye,
                run_command=fake_run,
                _spirula_install_packages=lambda: None,
                _install_spirula=lambda: Path("/opt/spirula"),
                _ensure_vulkan_gpu=lambda: None,
                _video_timing=lambda video: (timing[video], 30.0),
                _fisheye_sfm_args=lambda d: ["--camera-mode", "single"],
            ), patch.object(pipeline.shutil, "which", return_value=None):
                image_count, *_ = pipeline._run_spirula(paths)
        return commands, image_count

    def test_two_videos_become_two_sequences(self):
        commands, image_count = self.run_spirula(["source_video_1.mov", "source_video_2.mov"],
                                                 [60.0, 30.0])
        extracts = [c for c in commands if c[1] == "sam"]
        self.assertEqual([Path(c[3]).name for c in extracts],
                         ["source_video_1.mov", "source_video_2.mov"])
        self.assertEqual([Path(c[c.index("-o") + 1]).name for c in extracts], ["v1", "v2"])
        # 3 fps from 30 fps footage: every 10th frame of each whole video.
        self.assertEqual([c[c.index("-s") + 1] for c in extracts], ["10", "10"])
        sfm = next(c for c in commands if c[1] == "sfm")
        self.assertEqual(sfm[sfm.index("--data-type") + 1], "video")
        self.assertEqual([sfm[i + 1] for i, flag in enumerate(sfm) if flag == "--sequence"],
                         ["v1", "v2"])
        self.assertEqual(image_count, 8)

    def test_one_video_keeps_the_old_layout(self):
        commands, _ = self.run_spirula(["source_video.mp4"], [60.0])
        extract = next(c for c in commands if c[1] == "sam")
        self.assertEqual(Path(extract[extract.index("-o") + 1]).name, "images")
        self.assertEqual(extract[extract.index("-s") + 1], "10")
        self.assertNotIn("--sequence", next(c for c in commands if c[1] == "sfm"))

    def test_fisheye_still_uses_one_camera(self):
        commands, _ = self.run_spirula(["source_video_1.mov", "source_video_2.mov"],
                                       [10.0, 10.0], fisheye=True)
        sfm = next(c for c in commands if c[1] == "sfm")
        self.assertEqual(sfm[sfm.index("--camera-mode") + 1], "single")
        self.assertEqual(sfm.count("--sequence"), 2)

    def test_every_video_is_sampled_at_the_same_rate(self):
        timing = {Path("a"): (600.0, 30.0), Path("b"): (60.0, 29.97), Path("c"): (10.0, 24.0)}
        with patch.object(pipeline, "_video_timing", lambda v: timing[v]):
            # Ten minutes at 3 fps: 1,800 frames, no cap.
            self.assertEqual(pipeline._video_frame_plan(Path("a"), 3.0), (10, 1800))
            self.assertEqual(pipeline._video_frame_plan(Path("b"), 3.0), (10, 180))
            self.assertEqual(pipeline._video_frame_plan(Path("c"), 30.0), (1, 240))

    def test_input_videos_are_found_in_pick_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("source_video_10.mp4", "source_video_2.MOV", "source_video_1.mp4",
                         "notes.txt", "source_video_x.mp4", "source_video_3.txt"):
                (root / name).write_bytes(b"v")
            with patch.object(pipeline, "INPUT_DIR", root):
                found = [p.name for p in pipeline._input_videos()]
        self.assertEqual(found, ["source_video_1.mp4", "source_video_2.MOV", "source_video_10.mp4"])

    def test_nerfstudio_refuses_several_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "source_video_1.mp4").write_bytes(b"v")
            (root / "source_video_2.mp4").write_bytes(b"v")
            with patch.multiple(pipeline, INPUT_DIR=root, ENGINE="nerfstudio", SOURCE_TYPE="video",
                                VIDEO_FPS=3.0, FISHEYE=False):
                with self.assertRaisesRegex(RuntimeError, "only supported with Spirula"):
                    pipeline.run_pipeline()

    def test_manifest_names(self):
        def entry(name):
            return {"id": "abc", "name": name, "size": 10}
        pipeline._validate_drive_manifest("video", [entry("source_video.mp4")])
        pipeline._validate_drive_manifest("video", [entry("source_video_1.mp4"),
                                                    entry("source_video_2.mov")])
        bad = (
            [entry("source_video.mp4"), entry("source_video_2.mp4")],
            [entry("source_video_1.mp4"), entry("source_video_1.mp4")],
            [entry("source_video_0.mp4")],
            [entry("source_video_1.txt")],
            [],
        )
        for files in bad:
            with self.assertRaises(RuntimeError, msg=files):
                pipeline._validate_drive_manifest("video", files)


class TwoFileDrive(http.server.BaseHTTPRequestHandler):
    """Serves /files/<id>?alt=media for several ids, honouring Range headers."""

    bodies: dict[str, bytes] = {}

    def do_GET(self):
        file_id = self.path.split("/files/")[1].split("?")[0]
        data = type(self).bodies[file_id]
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


class PodDriveFetchTests(unittest.TestCase):
    def test_two_videos_are_fetched_and_verified(self):
        TwoFileDrive.bodies = {"big": os.urandom(3 * 1024 * 1024 + 7), "small": os.urandom(5000)}
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TwoFileDrive)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                names = {"big": "source_video_1.mov", "small": "source_video_2.mp4"}
                manifest = {
                    "kind": "video", "api_base": f"http://127.0.0.1:{server.server_port}",
                    "token": "t",
                    "files": [{"id": key, "name": names[key], "size": len(body),
                               "md5": hashlib.md5(body).hexdigest()}
                              for key, body in TwoFileDrive.bodies.items()],
                }
                (root / "m.json").write_text(json.dumps(manifest))
                with patch.multiple(pipeline, JOB_ROOT=root, INPUT_DIR=root / "input",
                                    DRIVE_MANIFEST=root / "m.json", DRIVE_RANGE_BYTES=1024 * 1024):
                    pipeline.fetch_drive_inputs()
                for key, name in names.items():
                    self.assertEqual((root / "input" / name).read_bytes(), TwoFileDrive.bodies[key])
                self.assertEqual(sorted(p.name for p in (root / "input").iterdir()),
                                 sorted(names.values()))
        finally:
            server.shutdown()
            server.server_close()


class PlanTests(unittest.TestCase):
    def test_local_videos_add_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            high, low = Path(tmp) / "high.MOV", Path(tmp) / "low.MOV"
            high.write_bytes(b"x" * 300)
            low.write_bytes(b"y" * 200)
            infos = {high: VideoInfo(80.0, 3840, 2160, 30.0), low: VideoInfo(30.0, 1920, 1080, 30.0)}
            with patch("storage.probe_video", lambda v: infos[v]):
                plan = plan_input(high, "video", 0, 0, more_videos=(low,))
                self.assertEqual(plan.videos, (high, low))
                self.assertEqual(plan.video_count, 2)
                self.assertEqual(plan.display_name, "high.MOV + 1 more")
                self.assertEqual(plan.input_bytes, 500)
                # 80 s and 30 s at 3 fps.
                self.assertEqual(plan.frame_target, 330)
                self.assertEqual(plan.video_info, VideoInfo(110.0, 3840, 2160, 30.0))
                self.assertEqual(plan.results_parent, Path(tmp) / "results" / "high")
                single = plan_input(high, "video", 0, 0)
                self.assertEqual((single.videos, single.video_count, single.display_name),
                                 ((), 1, "high.MOV"))
                with self.assertRaisesRegex(ValueError, "picked twice"):
                    plan_input(high, "video", 0, 0, more_videos=(high,))
            with patch("storage.probe_video", lambda v: infos[v]):
                with self.assertRaisesRegex(ValueError, "between 0.2 and 30"):
                    plan_input(high, "video", 0, 0, 60.0, more_videos=(low,))

    def test_jobs_are_named_after_the_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            lobby = Path(tmp) / "Elevator lobby"
            lobby.mkdir()
            high, low = lobby / "DSCF0001.MOV", lobby / "DSCF0002.MOV"
            high.write_bytes(b"x")
            low.write_bytes(b"y")
            for index in range(3):
                (lobby / f"p{index}.jpg").write_bytes(b"j")
            with patch("storage.probe_video", return_value=VideoInfo(30.0, 1920, 1080, 30.0)):
                self.assertEqual(plan_input(high, "video", 0, 0).job_title, "Elevator lobby")
                both = plan_input(high, "video", 0, 0, more_videos=(low,))
            self.assertEqual(both.job_title, "Elevator lobby · 2 videos")
            self.assertEqual(both.display_name, "DSCF0001.MOV + 1 more")
            self.assertEqual(plan_input(lobby, "images", 0, 0).job_title, "Elevator lobby")
        drive = plan_drive_video("f", "Garden", [
            DriveFile("a", "high.MOV", "source_video.mov", 10, "", 1, 1, 60000),
        ], Path("/r"), 0, 0)
        self.assertEqual(drive.job_title, "Garden")

    def test_drive_videos_add_up(self):
        videos = [
            DriveFile("a", "high.MOV", "source_video.mov", 10, "", 3840, 2160, 60000),
            DriveFile("b", "low.mp4", "source_video.mp4", 5, "", 1920, 1080, 20000),
        ]
        plan = plan_drive_video("f", "Garden", videos, Path("/r"), 0, 0)
        self.assertEqual((plan.video_count, plan.frame_target, plan.input_bytes), (2, 240, 15))
        self.assertEqual(plan.display_name, "high.MOV + 1 more")
        self.assertEqual(plan.results_parent, Path("/r/high"))


class DesktopTransferTests(unittest.TestCase):
    def test_upload_videos_names_each_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            high, low = Path(tmp) / "high.MOV", Path(tmp) / "low.mp4"
            high.write_bytes(b"abc")
            low.write_bytes(b"de")
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            client.free_bytes = Mock(return_value=10 ** 9)
            logs = []
            with patch.object(client, "_upload_batch") as batch, \
                    patch.object(client, "_verify_uploaded_images") as verify:
                client.upload_videos([high, low], 1000, logs.append)
            self.assertEqual(
                [call.kwargs["destination"] for call in batch.call_args_list],
                ["root@example:/workspace/job/input/source_video_1.mov",
                 "root@example:/workspace/job/input/source_video_2.mp4"],
            )
            verify.assert_called_once_with({"source_video_1.mov": 3, "source_video_2.mp4": 2})
            self.assertTrue(any("Verified uploaded 2 videos" in line for line in logs))

    def test_drive_manifest_renames_several_videos(self):
        client = DriveClient.__new__(DriveClient)
        client.api_base = "https://www.googleapis.com/drive/v3"
        client.token_for_pod = lambda: ("tok", 0)
        one = DriveFile("a", "high.MOV", "source_video.mov", 10, "m1")
        two = DriveFile("b", "low.mp4", "source_video.mp4", 5, "m2")
        names = [f["name"] for f in client.build_video_manifest("G", [one, two])["files"]]
        self.assertEqual(names, ["source_video_1.mov", "source_video_2.mp4"])
        self.assertEqual([f["name"] for f in client.build_video_manifest("G", one)["files"]],
                         ["source_video.mov"])
        pipeline._validate_drive_manifest(
            "video", client.build_video_manifest("G", [one, two])["files"])


class DriveProbeTests(unittest.TestCase):
    def test_header_read_with_token_and_ranges(self):
        """ffprobe reads length and frame rate from Drive without the whole file."""
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "clip.mp4"
            if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
                self.skipTest("ffmpeg/ffprobe are not installed")
            made = subprocess.run(
                ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=25",
                 "-t", "12", str(video)], capture_output=True)
            if made.returncode:
                self.skipTest("ffmpeg could not make the test video")
            TwoFileDrive.bodies = {"vid1": video.read_bytes()}
            seen_auth = []

            class AuthDrive(TwoFileDrive):
                def do_GET(self):
                    seen_auth.append(self.headers.get("Authorization"))
                    super().do_GET()

            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), AuthDrive)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                client = DriveClient.__new__(DriveClient)
                client.api_base = f"http://127.0.0.1:{server.server_port}"
                client._token = lambda: "tok"
                probed = client.probe_video(DriveFile("vid1", "clip.mp4", "source_video.mp4", 1, ""))
            finally:
                server.shutdown()
                server.server_close()
        self.assertEqual((probed.width, probed.height, probed.fps), (640, 360, 25.0))
        self.assertAlmostEqual(probed.duration_ms, 12000, delta=50)
        self.assertTrue(seen_auth and all(value == "Bearer tok" for value in seen_auth))
        plan = plan_drive_video("f", "G", [probed], Path("/r"), 0, 0, 2.5)
        self.assertEqual(plan.frame_target, 30)  # every 10th of 300 frames

    def test_failed_probe_keeps_drive_metadata(self):
        client = DriveClient.__new__(DriveClient)
        client.api_base = "http://127.0.0.1:9"  # nothing listens here
        client._token = lambda: "tok"
        video = DriveFile("vid1", "clip.mp4", "source_video.mp4", 1, "", 1920, 1080, 60000)
        self.assertEqual(client.probe_video(video, timeout=20), video)


DIALOG_SCRIPT = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from PySide6 import QtWidgets
from drive_client import DriveFile, DriveListing
from drive_dialog import DriveFolderDialog

app = QtWidgets.QApplication([])
VIDEOS = (
    DriveFile("a", "high.MOV", "source_video.mov", 10, ""),
    DriveFile("b", "low.mp4", "source_video.mp4", 5, ""),
    DriveFile("c", "other.mp4", "source_video.mp4", 5, ""),
)

class Client:
    email = "sa@example.iam.gserviceaccount.com"
    def list_folders(self, parent):
        return []
    def list_images(self, folder_id, folder_name):
        return DriveListing(folder_id, folder_name, (), 0, VIDEOS)

dialog = DriveFolderDialog(Client(), None, "f1", "Garden")
deadline = time.time() + 5
while not dialog.choose_button.isEnabled() and time.time() < deadline:
    app.processEvents()
    time.sleep(0.01)
# A folder of only videos can be chosen; its videos are ticked on the New job page.
assert dialog.choose_button.isEnabled()
assert "3 videos" in dialog.status.text(), dialog.status.text()
dialog._accept()
assert [v.id for v in dialog.selected.videos] == ["a", "b", "c"], dialog.selected
print("OK")
"""


class DriveDialogTests(unittest.TestCase):
    def test_folder_of_videos_can_be_chosen(self):
        # Widgets need a QApplication; other test modules create a plain
        # QCoreApplication first, so this runs in its own process.
        result = subprocess.run(
            [sys.executable, "-c", DIALOG_SCRIPT, str(ROOT / "desktop")],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        if result.returncode and "cannot open shared object file" in result.stderr:
            self.skipTest("Qt's window libraries are missing here (libEGL/libGL)")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
