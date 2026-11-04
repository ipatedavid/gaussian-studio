"""People masking (people_masks.py) and the shared PyTorch environment (torch_env.py)."""
from __future__ import annotations

import math
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

import people_masks
import torch_env


def looking(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World-to-camera rotation (OpenCV axes) for a camera in a +Z-up world, turned
    left by yaw from +Y and tilted up by pitch."""
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([-math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch),
                        math.sin(pitch)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward])


PINHOLE = {"model": "PINHOLE", "width": 1000, "height": 600,
           "params": (500.0, 500.0, 500.0, 300.0)}
FISHEYE = {"model": "OPENCV_FISHEYE", "width": 1500, "height": 1000,
           "params": (617.0, 617.0, 750.0, 500.0, -0.01, 0.0, 0.0, 0.0)}
UP = np.array([0.0, 0.0, 1.0])


def blob(rows: slice, cols: slice, shape=(600, 1000)) -> np.ndarray:
    mask = np.zeros(shape, bool)
    mask[rows, cols] = True
    return mask


class EyeLevelRuleTests(unittest.TestCase):
    def test_person_standing_in_front_is_kept(self):
        # Lower half of a level camera's view is below the horizon.
        self.assertTrue(people_masks.reaches_below_eye_level(
            blob(slice(200, 550), slice(400, 450)), PINHOLE, looking(0, 0), UP))

    def test_figure_on_the_ceiling_is_dropped(self):
        self.assertFalse(people_masks.reaches_below_eye_level(
            blob(slice(20, 200), slice(400, 450)), PINHOLE, looking(0, 0), UP))

    def test_camera_tilted_up_sees_feet_low_in_the_frame(self):
        # Tilted up 40 degrees, only the very bottom of a 62-degree-tall view is below
        # the horizon; a person there still counts, one mid-frame does not.
        tilted = looking(30, 40)
        self.assertFalse(people_masks.reaches_below_eye_level(
            blob(slice(250, 350), slice(450, 550)), PINHOLE, tilted, UP))
        fisheye_tilted = looking(0, 40)
        self.assertTrue(people_masks.reaches_below_eye_level(
            blob(slice(700, 1000), slice(700, 800), (1000, 1500)), FISHEYE, fisheye_tilted, UP))

    def test_masks_at_a_smaller_size_than_the_camera(self):
        small = blob(slice(100, 280), slice(200, 230), (300, 500))
        self.assertTrue(people_masks.reaches_below_eye_level(small, PINHOLE, looking(0, 0), UP))

    def test_world_up_is_spirulas_z_unless_the_cameras_disagree(self):
        upright = [{"rotation": looking(yaw, 0)} for yaw in (0, 90, 200)]
        np.testing.assert_allclose(people_masks.world_up(upright), UP)
        # A model whose up is -Y (not levelled by Spirula): trust the cameras.
        swap = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # maps +Z to -Y
        sideways = [{"rotation": looking(yaw, 0) @ swap.T} for yaw in (0, 90, 200)]
        np.testing.assert_allclose(people_masks.world_up(sideways), swap @ UP, atol=1e-9)


class ColmapReadingTests(unittest.TestCase):
    def test_reads_cameras_and_placed_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "sparse" / "0"
            model.mkdir(parents=True)
            (model / "cameras.bin").write_bytes(
                struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 5, 1500, 1000)
                + struct.pack("<8d", *FISHEYE["params"]))
            images = struct.pack("<Q", 2)
            for image_id, name in ((2, "v1/b.png"), (1, "a.jpg")):
                images += struct.pack("<i7di", image_id, 1, 0, 0, 0, 0, 0, 0, 1)
                images += name.encode() + b"\0" + struct.pack("<Q", 1)
                images += struct.pack("<ddq", 10.0, 20.0, -1)
            (model / "images.bin").write_bytes(images)
            self.assertEqual(people_masks.largest_model(Path(tmp)), model)
            cameras = people_masks.read_cameras(model / "cameras.bin")
            placed = people_masks.read_images(model / "images.bin")
        self.assertEqual(cameras[1]["model"], "OPENCV_FISHEYE")
        self.assertEqual([image["name"] for image in placed], ["a.jpg", "v1/b.png"])
        np.testing.assert_allclose(placed[0]["rotation"], np.eye(3))


class TorchEnvTests(unittest.TestCase):
    def test_uv_unpacks_without_restoring_owners(self):
        # Some pod volumes refuse chown even for root; tar then fails the whole step.
        commands = []
        worker = types.SimpleNamespace(
            run_command=lambda args, **kw: commands.append(list(args)),
            _download_file=lambda url, path, label: None)
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / "env"
            env.mkdir()
            with patch.object(torch_env, "UV_DIR", Path(tmp) / "uv"), \
                 patch.object(torch_env.shutil, "which", return_value=None), \
                 patch.object(torch_env, "torch_index", return_value="index"):
                python = torch_env.ensure(worker, env)
            self.assertTrue((env / "torch-ready").is_file())
        tar = next(c for c in commands if c[0] == "tar")
        self.assertIn("--no-same-owner", tar)
        self.assertEqual(python, env / "bin" / "python")
        install = next(c for c in commands if "torch" in c)
        self.assertEqual(install[install.index("--index-url") + 1], "index")

    def test_ready_environment_is_reused(self):
        commands = []
        worker = types.SimpleNamespace(run_command=lambda args, **kw: commands.append(args))
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp)
            (env / "torch-ready").write_text("ok")
            torch_env.ensure(worker, env)
        self.assertEqual(commands, [])

    def test_blackwell_cards_get_cuda_12_8_torch(self):
        for cap, index in (("8.6", torch_env.TORCH_INDEX),
                           ("12.0", torch_env.TORCH_INDEX_BLACKWELL)):
            done = types.SimpleNamespace(stdout=cap + "\n")
            with patch.object(torch_env.subprocess, "run", return_value=done):
                self.assertEqual(torch_env.torch_index(), index)


if __name__ == "__main__":
    unittest.main()
