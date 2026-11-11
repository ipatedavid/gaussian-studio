from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

import pipeline
from scp_client import ScpClient, TransferError


class SpirulaOptionTests(unittest.TestCase):
    def run_spirula(self, *, floaters="off", guidance="off", geometry_fails=False,
                    write_depths=True, preview=True, distraction="off",
                    crash_after_writing=False, train_fails_with_normals=False,
                    sfm_crash=None, one_surface=False, people_masks=False,
                    masks_fail=False):
        commands: list[list[str]] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "input"
            inputs.mkdir()
            for index in range(5):
                (inputs / f"p{index}.jpg").write_bytes(b"jpg")
            processed = root / "processed"
            train_dir = root / "train"

            def fake_run(args, *, cwd=None, ok_exit_codes=(0,), env=None, **_options):
                commands.append(list(args))
                verb = args[1]
                if verb.endswith("people_masks.py"):
                    if masks_fail:
                        (processed / "masks").mkdir()
                        raise RuntimeError("Command failed with exit code 1")
                    (processed / "masks").mkdir()
                    (processed / "masks" / "p0.png").write_bytes(b"m")
                    (processed / "people_masks.json").write_text(
                        '{"photos": 5, "with_people": 3, "masked_share": 0.03}')
                    return 0
                if verb == "sfm":
                    if sfm_crash == "before_writing":
                        return pipeline.SEGFAULT_EXIT
                    model = processed / "sparse" / "0"
                    model.mkdir(parents=True)
                    (model / "images.bin").write_bytes(struct.pack("<Q", 5))
                    if sfm_crash == "after_writing":
                        return pipeline.SEGFAULT_EXIT
                elif verb == "geometry":
                    if geometry_fails:
                        (processed / "normals").mkdir()
                        raise RuntimeError("Command failed with exit code 1")
                    (processed / "normals").mkdir()
                    for index in range(5):
                        (processed / "normals" / f"p{index}.png").write_bytes(b"n")
                    if "--depth" in args and write_depths:
                        (processed / "depths").mkdir()
                        (processed / "depths" / "p0.png").write_bytes(b"d")
                    if crash_after_writing:
                        raise RuntimeError("Command failed with exit code -11")
                elif verb == "train":
                    if train_fails_with_normals and "--load-normals" not in args:
                        raise RuntimeError("Command failed with exit code 1")
                    (train_dir / "run").mkdir(parents=True)
                    (train_dir / "run" / "splat.ply").write_bytes(b"ply\n")

            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=inputs, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=train_dir, SOURCE_TYPE="images",
                SPIRULA_FLOATERS=floaters, SPIRULA_GUIDANCE=guidance,
                SPIRULA_DISTRACTION=distraction, SPIRULA_DEPTH_WEIGHT=0.07, SPIRULA_RESULT={}, PREVIEW=preview,
                SPIRULA_ONE_SURFACE=one_surface, SPIRULA_PEOPLE_MASKS=people_masks,
                OUTPUT_DIR=root / "output",
                run_command=fake_run,
                _spirula_install_packages=lambda: None,
                _install_spirula=lambda: Path("/opt/spirula"),
                _ensure_vulkan_gpu=lambda: None,
            ), patch.object(pipeline.shutil, "which", return_value=None), \
                    patch.object(pipeline.torch_env, "ensure", return_value=Path("/env/bin/python")):
                pipeline._run_spirula(None)
                used = pipeline.SPIRULA_RESULT.get("guidance_used")
                leftovers = (processed / "normals").exists()
                self.people = pipeline.SPIRULA_RESULT.get("people_masks", "not run")
                self.masks_left = (processed / "masks").exists()
        self.trains = [command for command in commands if command[1] == "train"]
        self.masking = [command for command in commands if command[1].endswith("people_masks.py")]
        train = self.trains[-1]
        geometry = [command for command in commands if command[1] == "geometry"]
        return train, geometry, used, leftovers

    @staticmethod
    def flag(command, name):
        return command[command.index(name) + 1] if name in command else None

    def test_defaults_match_the_old_behaviour(self):
        train, geometry, used, _ = self.run_spirula()
        self.assertEqual(geometry, [])
        self.assertEqual(used, "off")
        self.assertEqual(self.flag(train, "--floater-suppression"), "off")
        self.assertEqual(self.flag(train, "--distraction-robustness"), "off")
        self.assertEqual(self.flag(train, "--load-normals"), "0")
        self.assertIsNone(self.flag(train, "--depth-supervision-weight"))
        self.assertEqual(self.flag(train, "--viewer-port"), "7007")
        self.assertEqual(self.flag(train, "--keep-viewer-alive"), "0")
        self.assertNotIn("--disable-viewer", train)

    def test_ignore_moving_things(self):
        train, _, _, _ = self.run_spirula(distraction="strong", floaters="strong")
        self.assertEqual(self.flag(train, "--distraction-robustness"), "strong")
        self.assertEqual(self.flag(train, "--floater-suppression"), "strong")

    def test_one_surface_without_guidance(self):
        train, geometry, used, _ = self.run_spirula(one_surface=True)
        self.assertEqual(geometry, [])
        self.assertEqual(used, "off")
        self.assertEqual(self.flag(train, "--depth-distortion-reg"), "0.01")
        self.assertEqual(self.flag(train, "--normal-distortion-reg"), "0.01")
        self.assertEqual(self.flag(train, "--load-depths"), "0")
        self.assertEqual(self.flag(train, "--load-normals"), "0")
        plain, _, _, _ = self.run_spirula()
        self.assertNotIn("--depth-distortion-reg", plain)

    def test_one_surface_with_guidance_is_added_once(self):
        train, _, _, _ = self.run_spirula(guidance="surfaces", one_surface=True)
        self.assertEqual(train.count("--depth-distortion-reg"), 1)

    def test_preview_off_disables_the_viewer(self):
        train, _, _, _ = self.run_spirula(preview=False)
        self.assertEqual(self.flag(train, "--disable-viewer"), "1")
        self.assertNotIn("--viewer-port", train)

    def test_surfaces_guidance(self):
        train, geometry, used, _ = self.run_spirula(floaters="mild", guidance="surfaces")
        self.assertEqual(len(geometry), 1)
        self.assertNotIn("--depth", geometry[0])
        self.assertEqual(used, "surfaces")
        self.assertEqual(self.flag(train, "--floater-suppression"), "mild")
        self.assertEqual(self.flag(train, "--depth-distortion-reg"), "0.01")
        self.assertEqual(self.flag(train, "--load-depths"), "0")
        self.assertIsNone(self.flag(train, "--load-normals"))

    def test_depth_guidance(self):
        train, geometry, used, _ = self.run_spirula(floaters="strong", guidance="depth")
        self.assertIn("--depth", geometry[0])
        self.assertEqual(used, "depth")
        self.assertEqual(self.flag(train, "--depth-supervision-weight"), "0.07")

    def test_depth_maps_missing_falls_back_to_surfaces(self):
        train, _, used, _ = self.run_spirula(guidance="depth", write_depths=False)
        self.assertEqual(used, "surfaces")
        self.assertIsNone(self.flag(train, "--depth-supervision-weight"))

    def test_failed_geometry_trains_without_guidance_and_cleans_up(self):
        train, _, used, leftovers = self.run_spirula(guidance="surfaces", geometry_fails=True)
        self.assertEqual(used, "off")
        self.assertFalse(leftovers)
        self.assertEqual(self.flag(train, "--load-normals"), "0")


    def test_crash_after_every_map_is_written_keeps_the_guidance(self):
        train, _, used, leftovers = self.run_spirula(guidance="surfaces",
                                                     crash_after_writing=True)
        self.assertEqual(used, "surfaces")
        self.assertTrue(leftovers)
        self.assertIsNone(self.flag(train, "--load-normals"))
        self.assertEqual(len(self.trains), 1)

    def test_crash_with_depth_maps_missing_keeps_surfaces_only(self):
        train, _, used, _ = self.run_spirula(guidance="depth", crash_after_writing=True,
                                             write_depths=False)
        self.assertEqual(used, "surfaces")
        self.assertEqual(self.flag(train, "--load-depths"), "0")

    def test_training_that_rejects_kept_maps_reruns_without_them(self):
        train, _, used, leftovers = self.run_spirula(
            guidance="surfaces", crash_after_writing=True, train_fails_with_normals=True)
        self.assertEqual(len(self.trains), 2)
        self.assertEqual(used, "off")
        self.assertFalse(leftovers)
        self.assertEqual(self.flag(train, "--load-normals"), "0")


    @unittest.skipIf(os.name == "nt", "pod-side code: pods run Linux")
    def test_people_masks_run_after_the_solve(self):
        train, _, _, _ = self.run_spirula(people_masks=True)
        self.assertEqual(len(self.masking), 1)
        self.assertEqual(self.masking[0][0], "/env/bin/python")
        self.assertIn("--check-sheet", self.masking[0])
        self.assertTrue(self.flag(self.masking[0], "--check-sheet").endswith("people_masks.jpg"))
        self.assertEqual(self.people["with_people"], 3)
        self.assertTrue(self.masks_left)
        self.assertEqual(train[1], "train")

    def test_failed_people_masks_train_on_whole_photos(self):
        train, _, _, _ = self.run_spirula(people_masks=True, masks_fail=True)
        self.assertIsNone(self.people)
        self.assertFalse(self.masks_left)
        self.assertEqual(train[1], "train")

    def test_people_masks_skip_object_scans(self):
        with patch.multiple(pipeline, SPIRULA_OBJECT=True,
                            _spirula_object_masks=lambda spirula, dataset: 5):
            self.run_spirula(people_masks=True)
        self.assertEqual(self.masking, [])
        self.assertEqual(self.people, "not run")

    def test_people_masks_off_by_default(self):
        self.run_spirula()
        self.assertEqual(self.masking, [])

    def test_sfm_crash_after_writing_the_model_keeps_it(self):
        # Spirula 2026.9.30 segfaults on exit after AI matchers (ALIKED + LightGlue, LoMa).
        train, _, _, _ = self.run_spirula(sfm_crash="after_writing")
        self.assertEqual(train[1], "train")

    def test_sfm_crash_before_writing_a_model_stops(self):
        with self.assertRaisesRegex(RuntimeError, "before writing a camera model"):
            self.run_spirula(sfm_crash="before_writing")


class RunPipelineCommandTests(unittest.TestCase):
    def test_options_reach_the_pod_only_for_spirula(self):
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
                                spirula_floaters="mild", spirula_guidance="depth",
                                spirula_depth_weight=0.1, spirula_people_masks=True)
            client.run_pipeline(10, 1, lambda line: None, engine="nerfstudio",
                                spirula_floaters="mild")
        self.assertIn("PIPELINE_SPIRULA_FLOATERS=mild", captured[0])
        self.assertIn("PIPELINE_SPIRULA_GUIDANCE=depth", captured[0])
        self.assertIn("PIPELINE_SPIRULA_DEPTH_WEIGHT=0.1 ", captured[0])
        self.assertIn("PIPELINE_SPIRULA_ONE_SURFACE=0 ", captured[0])
        self.assertIn("PIPELINE_SPIRULA_PEOPLE_MASKS=1 ", captured[0])
        self.assertNotIn("SPIRULA_FLOATERS", captured[1])
        with self.assertRaises(TransferError):
            client.run_pipeline(10, 1, lambda line: None, engine="spirula",
                                spirula_guidance="lots")


if __name__ == "__main__":
    unittest.main()
