"""Photos in sub-folders: each sub-folder is fitted as its own camera (lens).

Close-ups shot at another focus distance have another focal length, so they
must not share a camera with the rest of the capture.
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "worker"))

import pipeline
from drive_client import FOLDER_MIME, DriveClient, DriveFile
from storage import pick_photos, photo_subfolders, plan_drive_input, plan_input


def write_photos(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"x" * 100)


class LocalFolderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        write_photos(self.folder, "A1.JPG", "A2.JPG", "A3.JPG")
        write_photos(self.folder / "close-up", "B1.JPG", "B2.JPG")
        write_photos(self.folder / "results", "R1.JPG")
        write_photos(self.folder / ".cache", "C1.JPG")
        (self.folder / "empty").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_sub_folders_with_photos_are_found(self):
        groups = photo_subfolders(self.folder)
        self.assertEqual([(name, [p.name for p in photos]) for name, photos in groups],
                         [("close-up", ["B1.JPG", "B2.JPG"])])

    def test_no_photo_folders_means_the_folders_own_photos(self):
        self.assertEqual([p.name for p in pick_photos(self.folder)],
                         ["A1.JPG", "A2.JPG", "A3.JPG"])

    def test_plan_maps_sub_folder_photos_to_their_folder(self):
        plan = plan_input(self.folder, "images", 0, 1, photo_folders=("", "close-up"))
        self.assertEqual(plan.photo_count, 5)
        self.assertEqual(plan.camera_groups, {"B1.JPG": "close-up", "B2.JPG": "close-up"})
        self.assertEqual(plan.photo_folders, ("", "close-up"))

    def test_sub_folder_alone(self):
        write_photos(self.folder / "close-up", "B3.JPG")
        plan = plan_input(self.folder, "images", 0, 1, photo_folders=("close-up",))
        self.assertEqual(plan.photo_count, 3)
        self.assertEqual(plan.photo_folders, ("close-up",))

    def test_plain_folder_has_no_groups(self):
        plan = plan_input(self.folder, "images", 0, 1)
        self.assertEqual((plan.camera_groups, plan.photo_folders), ({}, ()))

    def test_same_name_in_two_folders_is_refused(self):
        write_photos(self.folder / "close-up", "a1.jpg")
        with self.assertRaisesRegex(ValueError, "two folders"):
            pick_photos(self.folder, ("", "close-up"))

    def test_missing_sub_folder_is_refused(self):
        with self.assertRaisesRegex(ValueError, "gone"):
            pick_photos(self.folder, ("", "gone"))


class DriveListingTests(unittest.TestCase):
    def test_sub_folder_photos_are_listed_with_their_group(self):
        client = object.__new__(DriveClient)
        contents = {
            "root": [
                {"id": "a1", "name": "DSCF1.JPG", "size": "10"},
                {"id": "v1", "name": "walk.mp4", "size": "50"},
            ],
            "sub": [
                {"id": "b1", "name": "DSCF1.JPG", "size": "20"},
                {"id": "b2", "name": "clip.mp4", "size": "30"},
            ],
        }

        def fake_list(params, key="files"):
            parent = params["q"].split("'")[1]
            if f"mimeType='{FOLDER_MIME}'" in params["q"]:
                return [{"id": "sub", "name": "close-up"}] if parent == "root" else []
            return contents[parent]

        with patch.object(client, "_list", side_effect=fake_list, create=True):
            listing = client.list_images("root", "Museum")
        self.assertEqual([(f.id, f.dest, f.group) for f in listing.files],
                         [("a1", "DSCF1.JPG", ""), ("b1", "DSCF1_2.JPG", "close-up")])
        # Sub-folder videos are not part of the capture.
        self.assertEqual([v.id for v in listing.videos], ["v1"])

    def test_drive_plan_maps_pod_names(self):
        files = (DriveFile("a", "A.JPG", "A.JPG", 1, "m"), DriveFile("b", "B.JPG", "B.JPG", 1, "m"),
                 DriveFile("c", "A.JPG", "A_2.JPG", 1, "m", group="close-up"))
        plan = plan_drive_input("f", "Museum", files, Path("/r"), 0)
        self.assertEqual(plan.camera_groups, {"A_2.JPG": "close-up"})
        self.assertEqual(plan.photo_folders, ("", "close-up"))


def colmap_model(model: Path, cameras: dict[int, float], images: list[tuple[str, int]]) -> None:
    """A minimal binary COLMAP model: OPENCV_FISHEYE cameras and posed images."""
    model.mkdir(parents=True)
    data = struct.pack("<Q", len(cameras))
    for camera_id, focal in cameras.items():
        data += struct.pack("<iiQQ", camera_id, 5, 6000, 4000)
        data += struct.pack("<8d", focal, focal, 3000, 2000, 0, 0, 0, 0)
    (model / "cameras.bin").write_bytes(data)
    data = struct.pack("<Q", len(images))
    for index, (name, camera_id) in enumerate(images, 1):
        data += struct.pack("<i7di", index, 1, 0, 0, 0, 0, 0, 0, camera_id)
        data += name.encode() + b"\0" + struct.pack("<Q", 1) + struct.pack("<ddq", 1, 2, -1)
    (model / "images.bin").write_bytes(data)


@unittest.skipIf(os.name == "nt", "pod-side code: pods run Linux")
class PodStagingTests(unittest.TestCase):
    def run_spirula(self, groups: dict[str, str] | None, *, fisheye=True):
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed, train_dir, input_dir = root / "processed", root / "train", root / "input"
            write_photos(input_dir, "A1.JPG", "A2.JPG", "B1.JPG", "B2.JPG")
            groups_file = root / "camera_groups.json"
            if groups is not None:
                groups_file.write_text(json.dumps({"groups": groups}))

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None, **_options):
                commands.append(list(args))
                if args[1] == "sfm":
                    colmap_model(processed / "sparse" / "0", {1: 1915.0, 2: 1953.3}, [
                        ("photos/A1.JPG", 1), ("photos/A2.JPG", 1),
                        ("photos_2/B1.JPG", 2), ("photos_2/B2.JPG", 2),
                    ])
                elif args[1] == "train":
                    (train_dir / "run").mkdir(parents=True)
                    (train_dir / "run" / "splat.ply").write_bytes(b"ply\n")

            result: dict = {}
            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=input_dir, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=train_dir, CAMERA_GROUPS_FILE=groups_file,
                SOURCE_TYPE="images", SPIRULA_GUIDANCE="off", SPIRULA_RESULT=result,
                PREVIEW=False, FISHEYE=fisheye, FISHEYE_FOCAL_MM=7.5, SENSOR_WIDTH_MM=23.5,
                run_command=fake_run,
                _spirula_install_packages=lambda: None,
                _install_spirula=lambda: Path("/opt/spirula"),
                _ensure_vulkan_gpu=lambda: None,
                _image_size=lambda path: (6000, 4000),
            ), patch.object(pipeline.shutil, "which", return_value=None):
                pipeline._run_spirula([])
                staged = sorted(str(p.relative_to(processed / "images"))
                                for p in (processed / "images").rglob("*") if p.is_file())
        sfm = next(c for c in commands if c[1] == "sfm")
        return sfm, staged, result

    def test_each_sub_folder_is_its_own_fisheye_camera(self):
        sfm, staged, result = self.run_spirula({"B1.JPG": "close-up", "B2.JPG": "close-up"})
        self.assertEqual(staged, ["photos/A1.JPG", "photos/A2.JPG",
                                  "photos_2/B1.JPG", "photos_2/B2.JPG"])
        self.assertEqual(sfm[sfm.index("--camera-mode") + 1], "folder")
        self.assertEqual([sfm[i + 1] for i, flag in enumerate(sfm) if flag == "--focal"],
                         ["photos=1914.9", "photos_2=1914.9"])
        self.assertNotIn("--sequence", sfm)
        self.assertEqual(result["photo_folders"], {"photos": "", "photos_2": "close-up"})
        self.assertEqual(result["fitted_focal_px"], {"photos": 1915.0, "photos_2": 1953.3})

    def test_regular_lens_also_gets_one_camera_per_folder(self):
        sfm, _, _ = self.run_spirula({"B1.JPG": "close-up", "B2.JPG": "close-up"}, fisheye=False)
        self.assertEqual(sfm[sfm.index("--camera-mode") + 1], "folder")

    def test_without_sub_folders_nothing_changes(self):
        sfm, staged, result = self.run_spirula(None)
        self.assertEqual(staged, ["A1.JPG", "A2.JPG", "B1.JPG", "B2.JPG"])
        self.assertEqual(sfm[sfm.index("--camera-mode") + 1], "single")
        self.assertNotIn("photo_folders", result)

    def test_unreadable_group_file_stops_the_job(self):
        with self.assertRaisesRegex(RuntimeError, "camera_groups.json"):
            self.run_spirula({"B1.JPG": ""})


WINDOW_SCRIPT = r"""
import os, sys, tempfile
from pathlib import Path
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

capture = tmp / "museum"
(capture / "close-up").mkdir(parents=True)
for index in range(4):
    (capture / f"p{index}.jpg").write_bytes(b"x" * 100)
for index in range(3):
    (capture / "close-up" / f"c{index}.jpg").write_bytes(b"x" * 100)

window = studio.MainWindow()
window.input_folder.setText(str(capture))
window._local_folder_changed()
rows = [window.capture_items.item(i).text() for i in range(window.capture_items.count())]
assert len(rows) == 2 and rows[0].startswith("4 photos"), rows
assert rows[1].startswith("close-up/") and "own lens fit" in rows[1], rows
plan = window._source_plan()
assert (plan.kind, plan.photo_count, plan.photo_folders) == ("images", 7, ("", "close-up")), plan

window.capture_items.item(1).setCheckState(QtCore.Qt.Unchecked)
plan = window._source_plan()
assert (plan.photo_count, plan.photo_folders) == (4, ()), plan.photo_folders

window.capture_items.item(1).setCheckState(QtCore.Qt.Checked)
window.capture_items.item(0).setCheckState(QtCore.Qt.Unchecked)
plan = window._source_plan()
assert (plan.photo_count, plan.photo_folders) == (3, ("close-up",)), plan.photo_folders
print("OK")
"""


class WindowTests(unittest.TestCase):
    def test_each_sub_folder_is_a_checklist_row(self):
        result = subprocess.run(
            [sys.executable, "-c", WINDOW_SCRIPT, str(ROOT / "desktop")],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        )
        self.assertIn("OK", result.stdout, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
