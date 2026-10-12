"""Object scans: BiRefNet subject masks and Spirula's centered-object preset."""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "tests"))

import pipeline
import test_spirula_options as options
from scp_client import ScpClient, TransferError


class MaskTests(unittest.TestCase):
    def make_dataset(self, root: Path, layout: dict[str, list[str]]) -> Path:
        dataset = root / "processed"
        for folder, names in layout.items():
            target = dataset / "images" / folder
            target.mkdir(parents=True, exist_ok=True)
            for name in names:
                (target / name).write_bytes(b"img")
        return dataset

    def run_masks(self, layout, *, fail=False, produce=None):
        commands = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self.make_dataset(root, layout)

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None):
                commands.append(list(args))
                if fail:
                    raise RuntimeError("Command failed with exit code 1")
                frames = Path(args[args.index("--frames") + 1])
                out = Path(args[args.index("--out") + 1])
                out.mkdir(parents=True)
                count = len([p for p in frames.iterdir()
                             if p.suffix.lower() in pipeline.SAM_IMAGE_EXTENSIONS])
                for index in range(produce if produce is not None else count):
                    # Content records which frame it was, to check the renaming.
                    (out / f"frame_{index:05d}.png").write_text(str(index))

            with patch.multiple(pipeline, JOB_ROOT=root, run_command=fake_run,
                                _ensure_birefnet=lambda: None):
                written = pipeline._spirula_object_masks(Path("/opt/spirula"), dataset)
            masks = {str(p.relative_to(dataset / "masks")): p.read_text()
                     for p in sorted((dataset / "masks").rglob("*.png"))} \
                if (dataset / "masks").exists() else {}
        return written, masks, commands

    def test_masks_are_named_after_their_images(self):
        written, masks, commands = self.run_masks(
            {".": ["IMG_0010.JPG", "IMG_0002.jpg", "notes.txt", "IMG_0003.webp"]})
        self.assertEqual(written, 2)
        # Sorted like Spirula sorts them; .webp and .txt are not read by `track`.
        self.assertEqual(masks, {"IMG_0002.png": "0", "IMG_0010.png": "1"})
        self.assertEqual(commands[0][1:5], ["sam", "track", "--model", str(pipeline.BIREFNET_CACHE)])

    def test_video_subfolders_get_their_own_masks(self):
        written, masks, commands = self.run_masks(
            {"v1": ["00000.png", "00001.png"], "v2": ["00000.png"]})
        self.assertEqual(written, 3)
        self.assertEqual(masks, {"v1/00000.png": "0", "v1/00001.png": "1", "v2/00000.png": "0"})
        self.assertEqual(len(commands), 2)

    def test_failure_trains_without_masks(self):
        written, masks, _ = self.run_masks({".": ["a.jpg", "b.jpg"]}, fail=True)
        self.assertEqual((written, masks), (0, {}))

    def test_wrong_mask_count_is_a_failure(self):
        written, masks, _ = self.run_masks({".": ["a.jpg", "b.jpg"]}, produce=1)
        self.assertEqual((written, masks), (0, {}))


class ObjectTrainTests(options.SpirulaOptionTests):
    def run_spirula(self, **kwargs):
        on = kwargs.pop("object_scan", False)
        turntable = kwargs.pop("turntable", False)
        masks = kwargs.pop("masks", 5)
        self.masked_before_sfm = []

        def fake_masks(spirula, dataset):
            # The fake SfM writes processed/sparse; masking first means it is not there yet.
            self.masked_before_sfm.append(not (dataset / "sparse").exists())
            return masks

        with patch.multiple(pipeline, SPIRULA_OBJECT=on, SPIRULA_TURNTABLE=turntable,
                            _spirula_object_masks=fake_masks):
            return super().run_spirula(**kwargs)

    def test_turntable_masks_before_the_cameras_are_solved(self):
        train, _, _, _ = self.run_spirula(object_scan=True, turntable=True)
        self.assertEqual(self.masked_before_sfm, [True])
        self.assertEqual(train[1:3], ["train", "centered-object"])

    def test_walk_around_object_scans_mask_after(self):
        self.run_spirula(object_scan=True)
        self.assertEqual(self.masked_before_sfm, [False])

    def test_turntable_without_masks_stops_before_training(self):
        with self.assertRaisesRegex(RuntimeError, "Stopped before training"):
            self.run_spirula(object_scan=True, turntable=True, masks=0)

    def test_object_scan_uses_the_preset(self):
        train, _, _, _ = self.run_spirula(object_scan=True)
        self.assertEqual(train[1:3], ["train", "centered-object"])

    def test_no_preset_without_masks(self):
        train, _, _, _ = self.run_spirula(object_scan=True, masks=0)
        self.assertNotIn("centered-object", train)

    def test_ordinary_scans_unchanged(self):
        train, _, _, _ = self.run_spirula()
        self.assertEqual(train[1:3], ["train", "--data"])


for _name in [n for n in dir(options.SpirulaOptionTests) if n.startswith("test_")]:
    setattr(ObjectTrainTests, _name, None)


def write_sparse(model: Path, cameras, points):
    """A COLMAP binary model: cameras as (centre, look_at) with world up +y, points as xyz."""
    import math
    model.mkdir(parents=True)
    with (model / "images.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for i, (c, target) in enumerate(cameras):
            fwd = [t - x for t, x in zip(target, c)]
            n = math.sqrt(sum(v * v for v in fwd))
            fwd = [v / n for v in fwd]
            down = [0.0, -1.0, 0.0]
            down = [d - fwd[k] * sum(a * b for a, b in zip(down, fwd)) for k, d in enumerate(down)]
            n = math.sqrt(sum(v * v for v in down))
            down = [v / n for v in down]
            right = [down[1] * fwd[2] - down[2] * fwd[1], down[2] * fwd[0] - down[0] * fwd[2],
                     down[0] * fwd[1] - down[1] * fwd[0]]
            r = [right, down, fwd]  # world-to-camera rows
            tr = r[0][0] + r[1][1] + r[2][2]
            qw = math.sqrt(max(0.0, 1 + tr)) / 2
            qx = math.copysign(math.sqrt(max(0.0, 1 + r[0][0] - r[1][1] - r[2][2])) / 2, r[2][1] - r[1][2])
            qy = math.copysign(math.sqrt(max(0.0, 1 - r[0][0] + r[1][1] - r[2][2])) / 2, r[0][2] - r[2][0])
            qz = math.copysign(math.sqrt(max(0.0, 1 - r[0][0] - r[1][1] + r[2][2])) / 2, r[1][0] - r[0][1])
            t = [-sum(r[k][j] * c[j] for j in range(3)) for k in range(3)]
            f.write(struct.pack("<i7di", i + 1, qw, qx, qy, qz, *t, 1) + f"im{i}.jpg".encode() + b"\0")
            f.write(struct.pack("<Q", 0))
    with (model / "points3D.bin").open("wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for i, p in enumerate(points):
            f.write(struct.pack("<Q3d3BdQ", i + 1, *p, 0, 0, 0, 0.5, 3) + b"\0" * 24)


class MainSceneRegionTests(unittest.TestCase):
    def region(self, cameras, points):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp)
            write_sparse(dataset / "sparse" / "0", cameras, points)
            description = pipeline._write_main_scene_region(dataset)
            region = json.loads((dataset / "roi" / "main-scene.json").read_text()) if description else None
        return description, region

    def test_orbit_capture_gets_a_cylinder_around_the_subject(self):
        import math
        cams = [((3 * math.cos(a), 1.2, 3 * math.sin(a)), (0, 0.4, 0))
                for a in (i * 2 * math.pi / 24 for i in range(24))]
        pts = [(0.3 * math.cos(i), 0.1 + 0.8 * (i % 10) / 10, 0.3 * math.sin(i)) for i in range(200)]
        pts += [(40.0 + i, 5.0, 0.0) for i in range(3)]  # far stray points
        description, region = self.region(cams, pts)
        self.assertIn("cylinder", description)
        self.assertEqual(region["type"], "cylinder")
        centre = region["center"]
        self.assertLess(math.hypot(centre[0], centre[2]), 0.05)
        self.assertAlmostEqual(region["half"][0], 0.7 * 3, places=2)
        self.assertAlmostEqual(abs(region["rotation"][7]), 1.0, places=3)  # axis is world up

    def test_walk_through_gets_a_box_without_the_far_tail(self):
        import random
        rng = random.Random(1)
        cams = [((x, 1.6, 0.0), (x, 1.6, 5.0)) for x in range(10)]  # walking, all facing one way
        pts = [(rng.uniform(-2, 12), rng.uniform(0, 3), rng.uniform(2, 8)) for _ in range(2000)]
        pts += [(rng.uniform(-500, 500), 50.0, 400.0) for _ in range(10)]  # far background, < 1%
        description, region = self.region(cams, pts)
        self.assertEqual(region["type"], "box")
        self.assertLess(max(region["half"]), 10)  # the far points did not stretch it

    def test_too_little_reconstruction_means_no_region(self):
        description, region = self.region([((0, 0, 0), (0, 0, 1))] * 2, [(0, 0, 1)] * 10)
        self.assertIsNone(description)


class MainSceneTrainTests(options.SpirulaOptionTests):
    def run_spirula(self, **kwargs):
        on = kwargs.pop("main_scene", False)
        with patch.multiple(pipeline, SPIRULA_MAIN_SCENE=on,
                            _write_main_scene_region=lambda dataset: "box around the scene"):
            return super().run_spirula(**kwargs)

    def test_region_is_named_to_the_trainer(self):
        train, _, _, _ = self.run_spirula(main_scene=True)
        self.assertTrue(self.flag(train, "--roi-region").endswith("roi/main-scene.json"))

    def test_off_by_default(self):
        train, _, _, _ = self.run_spirula()
        self.assertNotIn("--roi-region", train)

    def test_draft_is_never_raised(self):
        with patch.object(pipeline, "_registered_images", return_value=2000), \
                patch.multiple(pipeline, SPIRULA_QUALITY="low", STEPS_OVERRIDE=False):
            train, _, _, _ = self.run_spirula()
        self.assertEqual(self.flag(train, "--num-iterations"), "15000")


for _name in [n for n in dir(options.SpirulaOptionTests) if n.startswith("test_")]:
    setattr(MainSceneTrainTests, _name, None)


class MatcherTests(unittest.TestCase):
    def run_spirula(self, matcher, registered=5):
        commands = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            for index in range(5):
                (inputs / f"p{index}.jpg").write_bytes(b"jpg")
            processed, train_dir = root / "processed", root / "train"

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None):
                commands.append(list(args))
                if args[1] == "sfm":
                    model = processed / "sparse" / "0"
                    model.mkdir(parents=True)
                    (model / "images.bin").write_bytes(struct.pack("<Q", registered))
                elif args[1] == "train":
                    (train_dir / "run").mkdir(parents=True)
                    (train_dir / "run" / "splat.ply").write_bytes(b"ply\n")

            fetched = []
            with patch.multiple(
                pipeline, JOB_ROOT=root, INPUT_DIR=inputs, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=train_dir, SOURCE_TYPE="images", SPIRULA_MATCHER=matcher,
                SPIRULA_GUIDANCE="off", SPIRULA_RESULT={}, PREVIEW=False,
                run_command=fake_run, _spirula_install_packages=lambda: None,
                _install_spirula=lambda: Path("/opt/spirula"), _ensure_vulkan_gpu=lambda: None,
                _ensure_sfm_models=fetched.append,
            ), patch.object(pipeline.shutil, "which", return_value=None):
                pipeline._run_spirula([])
        sfm = next(c for c in commands if c[1] == "sfm")
        return sfm, fetched

    def test_standard_is_spirulas_default(self):
        sfm, fetched = self.run_spirula("sift")
        self.assertNotIn("--features", sfm)
        self.assertEqual(fetched, [])

    def test_learned_and_loma_pick_their_frontends(self):
        sfm, fetched = self.run_spirula("aliked")
        self.assertEqual(sfm[sfm.index("--features") + 1:sfm.index("--features") + 4],
                         ["aliked-n16rot", "--matcher", "lightglue"])
        self.assertEqual(fetched, ["aliked"])
        sfm, fetched = self.run_spirula("loma")
        self.assertEqual(sfm[sfm.index("--features") + 1], "loma-b")
        self.assertEqual(sfm[sfm.index("--matcher") + 1], "loma-b")
        self.assertEqual(fetched, ["loma"])

    def test_a_split_solve_suggests_a_stronger_matcher(self):
        with self.assertRaisesRegex(RuntimeError, "try Feature matching: Learned, or LoMa"):
            self.run_spirula("sift", registered=1)
        with self.assertRaisesRegex(RuntimeError, "Try Feature matching: LoMa"):
            self.run_spirula("aliked", registered=1)

    def test_models_are_verified_and_reused(self):
        import hashlib
        body = b"model-bytes" * 100
        files = [("tiny.onnx", "https://example/tiny.onnx", hashlib.sha256(body).hexdigest(), len(body))]
        downloads = []

        def fake_download(url, destination, label):
            downloads.append(url)
            destination.write_bytes(body)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(pipeline.os.environ, {"XDG_CACHE_HOME": tmp}), \
                patch.dict(pipeline.SFM_MATCHERS, {"aliked": ([], files)}), \
                patch.object(pipeline, "_download_file", fake_download):
            pipeline._ensure_sfm_models("aliked")
            pipeline._ensure_sfm_models("aliked")  # cached: no second download
            cached = Path(tmp) / "spirula-studio" / "models" / "tiny.onnx"
            self.assertEqual(cached.read_bytes(), body)
        self.assertEqual(len(downloads), 1)

    def test_a_corrupt_download_is_refused(self):
        files = [("tiny.onnx", "https://example/tiny.onnx", "0" * 64, 5)]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(pipeline.os.environ, {"XDG_CACHE_HOME": tmp}), \
                patch.dict(pipeline.SFM_MATCHERS, {"aliked": ([], files)}), \
                patch.object(pipeline, "_download_file", lambda u, d, l: d.write_bytes(b"12345")):
            with self.assertRaisesRegex(RuntimeError, "did not download correctly"):
                pipeline._ensure_sfm_models("aliked")
            self.assertFalse((Path(tmp) / "spirula-studio" / "models" / "tiny.onnx").exists())


class PodSettingTests(unittest.TestCase):
    def capture(self, **kwargs):
        client = ScpClient(Path("/tmp/test-key"), "example", 1234)
        captured = []

        class FakeProcess:
            returncode = 0
            stdout = None

            def __init__(self, args, **kw):
                captured.append(args[-1])

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def wait(self):
                return 0

        with patch("scp_client.subprocess.Popen", FakeProcess), \
                patch("scp_client._stream_output", lambda process, log: None):
            client.run_pipeline(10, 1, lambda line: None, **kwargs)
        return captured[0]

    def test_feature_matching_reaches_the_pod(self):
        self.assertIn("PIPELINE_SPIRULA_MATCHER=loma", self.capture(engine="spirula", spirula_matcher="loma"))
        self.assertIn("PIPELINE_SPIRULA_MATCHER=sift", self.capture(engine="spirula"))
        with self.assertRaisesRegex(TransferError, "feature matching"):
            self.capture(engine="spirula", spirula_matcher="orb")

    def test_turntable_only_with_object_scan(self):
        self.assertIn("PIPELINE_SPIRULA_TURNTABLE=1",
                      self.capture(engine="spirula", spirula_object=True, spirula_turntable=True))
        self.assertIn("PIPELINE_SPIRULA_TURNTABLE=0",
                      self.capture(engine="spirula", spirula_turntable=True))

    def test_object_setting_reaches_the_pod(self):
        self.assertIn("PIPELINE_SPIRULA_OBJECT=1", self.capture(engine="spirula", spirula_object=True))
        self.assertIn("PIPELINE_SPIRULA_OBJECT=0", self.capture(engine="spirula"))
        with self.assertRaisesRegex(TransferError, "Object scans"):
            self.capture(engine="nerfstudio", spirula_object=True)

    def test_birefnet_is_the_size_spirula_expects(self):
        # Spirula's model catalog lists birefnet-general.safetensors at this size.
        self.assertEqual(pipeline.BIREFNET_BYTES, 444_473_596)
        self.assertEqual(pipeline.BIREFNET_CACHE.name, "birefnet-general.safetensors")


if __name__ == "__main__":
    unittest.main()
