from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

import pipeline
from pipeline import _forward_output


@unittest.skipIf(os.name == "nt", "pod-side code: pods run Linux")
class WorkerOutputTests(unittest.TestCase):
    def test_forwards_carriage_return_progress(self):
        script = (
            "import sys; "
            "sys.stdout.write('features 1/3\\r'); sys.stdout.flush(); "
            "sys.stdout.write('features 2/3\\r'); sys.stdout.flush(); "
            "sys.stdout.write('features 3/3\\n'); sys.stdout.flush()"
        )
        output = io.StringIO()
        with subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process, redirect_stdout(output):
            _forward_output(process, "test")
            self.assertEqual(process.wait(), 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            ["features 1/3", "features 2/3", "features 3/3"],
        )


@unittest.skipIf(os.name == "nt", "pod-side code: pods run Linux")
class FinishedButHungTests(unittest.TestCase):
    """A trainer that prints that it is done but never exits is stopped, not waited on."""

    def setUp(self):
        space = patch.object(pipeline, "_low_space", return_value=None)  # no /workspace here
        space.start()
        self.addCleanup(space.stop)

    def run_child(self, script: str, grace: float) -> tuple[int, str, float]:
        output = io.StringIO()
        started = time.monotonic()
        with redirect_stdout(output):
            code = pipeline.run_command([sys.executable, "-c", script],
                                        finished_marker="Training complete", exit_grace=grace)
        return code, output.getvalue(), time.monotonic() - started

    def test_silent_after_finishing_is_stopped_and_counts_as_done(self):
        code, output, took = self.run_child(
            "import time; print('Training complete. Steps: 10', flush=True); time.sleep(600)", 1)
        self.assertEqual(code, 0)
        self.assertIn("has not exited in 1 s; stopping it", output)
        self.assertLess(took, 30)

    def test_output_after_finishing_restarts_the_wait(self):
        code, output, _ = self.run_child(
            "import time\nprint('Training complete', flush=True)\n"
            "for i in range(4):\n    time.sleep(0.8); print('eval', i, flush=True)\n", 1.5)
        self.assertEqual(code, 0)
        self.assertNotIn("stopping it", output)
        self.assertIn("eval 3", output)

    def test_without_the_marker_failures_still_fail(self):
        with self.assertRaises(RuntimeError), redirect_stdout(io.StringIO()):
            pipeline.run_command([sys.executable, "-c", "print('step 1'); raise SystemExit(3)"],
                                 finished_marker="Training complete", exit_grace=1)


class VideoPipelineTests(unittest.TestCase):
    def test_video_is_extracted_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "source_video.mp4").write_bytes(b"video")
            commands = []
            processed = root / "processed"
            training = root / "training"
            exports = root / "exports"
            outputs = root / "output"

            def run(args, **_kwargs):
                commands.append(args)
                if args[0] == "ns-process-data":
                    (processed / "images").mkdir(parents=True)
                    for number in range(3):
                        (processed / "images" / f"frame_{number:05}.png").write_bytes(b"frame")
                    (processed / "transforms.json").write_text(json.dumps({
                        "frames": [
                            {"file_path": f"images/frame_{number:05}.png"}
                            for number in range(3)
                        ]
                    }))
                elif args[0] == "ns-train":
                    (training / "config.yml").write_text("config")
                elif args[0] == "ns-export":
                    (exports / "result.ply").write_bytes(b"ply\n")

            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=inputs, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=training, EXPORT_DIR=exports, OUTPUT_DIR=outputs,
                SOURCE_TYPE="video", VIDEO_FPS=3.0, _video_timing=lambda video: (30.0, 30.0),
            ), patch.object(pipeline, "run_command", side_effect=run), \
                patch.object(pipeline.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                redirect_stdout(io.StringIO()):
                pipeline.run_pipeline()

            self.assertEqual(commands[1][0:2], ["ns-process-data", "video"])
            self.assertEqual(
                commands[1][commands[1].index("--num-frames-target") + 1], "90"
            )
            self.assertEqual(commands[2][0:2], ["ns-train", "splatfacto"])
            summary = json.loads((outputs / "pipeline_summary.json").read_text())
            self.assertEqual(summary["source_type"], "video")
            self.assertEqual(summary["image_count"], 3)
            self.assertEqual(summary["registered_images"], 3)
            self.assertEqual(summary["video_fps"], 3.0)


class PoseRegistrationTests(unittest.TestCase):
    def test_aborts_before_training_when_too_few_images_registered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "input"
            inputs.mkdir()
            (inputs / "source_video.mp4").write_bytes(b"video")
            commands = []
            processed = root / "processed"

            def run(args, **_kwargs):
                commands.append(args)
                if args[0] == "ns-process-data":
                    (processed / "images").mkdir(parents=True)
                    for number in range(10):
                        (processed / "images" / f"frame_{number:05}.png").write_bytes(b"frame")
                    # COLMAP only found poses for 2 of the 10 images.
                    (processed / "transforms.json").write_text(json.dumps({
                        "frames": [
                            {"file_path": f"images/frame_{number:05}.png"}
                            for number in range(2)
                        ]
                    }))

            with patch.multiple(
                pipeline,
                JOB_ROOT=root, INPUT_DIR=inputs, PROCESSED_DIR=processed,
                TRAIN_OUTPUT_DIR=root / "training", EXPORT_DIR=root / "exports",
                OUTPUT_DIR=root / "output",
                SOURCE_TYPE="video", VIDEO_FPS=3.0, _video_timing=lambda video: (30.0, 30.0),
            ), patch.object(pipeline, "run_command", side_effect=run), \
                patch.object(pipeline.shutil, "which", return_value="/usr/bin/ffmpeg"), \
                redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "only registered 2 of 10"):
                    pipeline.run_pipeline()

            self.assertFalse(any(args[0] == "ns-train" for args in commands))


if __name__ == "__main__":
    unittest.main()