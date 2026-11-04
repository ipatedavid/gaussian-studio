"""The camera solve is saved beside the splat (cameras.zip), so poses can be checked later."""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

import pipeline  # noqa: E402


def _model(folder: Path, cameras: int) -> None:
    folder.mkdir(parents=True)
    (folder / "images.bin").write_bytes(struct.pack("<Q", cameras))
    (folder / "cameras.bin").write_bytes(b"cam")
    (folder / "points3D.bin").write_bytes(b"pts")


class CameraSolveTests(unittest.TestCase):
    def test_largest_model_is_zipped_as_sparse_0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _model(root / "processed" / "sparse" / "0", 12)
            _model(root / "processed" / "sparse" / "1", 900)
            (root / "output").mkdir()
            with patch.object(pipeline, "OUTPUT_DIR", root / "output"):
                name = pipeline._save_camera_solve(root / "processed")
            self.assertEqual(name, "cameras.zip")
            with zipfile.ZipFile(root / "output" / "cameras.zip") as archive:
                self.assertEqual(sorted(archive.namelist()),
                                 ["sparse/0/cameras.bin", "sparse/0/images.bin", "sparse/0/points3D.bin"])
                self.assertEqual(struct.unpack("<Q", archive.read("sparse/0/images.bin"))[0], 900)

    def test_nerfstudio_layout_and_no_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "output").mkdir()
            with patch.object(pipeline, "OUTPUT_DIR", root / "output"):
                self.assertIsNone(pipeline._save_camera_solve(root / "processed"))
                _model(root / "processed" / "colmap" / "sparse" / "0", 30)
                self.assertEqual(pipeline._save_camera_solve(root / "processed"), "cameras.zip")


if __name__ == "__main__":
    unittest.main()
