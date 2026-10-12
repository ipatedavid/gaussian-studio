from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    from PIL import Image
except ImportError:  # Pillow is a test-only dependency (tests/requirements.txt)
    Image = None

import pipeline
import test_job_flow as flow
import test_spirula_options as options
from scp_client import ScpClient, TransferError


@unittest.skipIf(Image is None, "Pillow is not installed (pip install -r tests/requirements.txt)")
class ImageSizeTests(unittest.TestCase):
    def test_reads_jpeg_and_png_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpeg, png = Path(tmp) / "a.jpg", Path(tmp) / "b.png"
            Image.new("RGB", (6000, 4000)).save(jpeg, quality=50)
            Image.new("RGB", (3840, 2160)).save(png)
            (Path(tmp) / "c.jpg").write_bytes(b"not an image")
            self.assertEqual(pipeline._image_size(jpeg), (6000, 4000))
            self.assertEqual(pipeline._image_size(png), (3840, 2160))
            self.assertIsNone(pipeline._image_size(Path(tmp) / "c.jpg"))

    def test_focal_length_in_pixels_for_the_xt20(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (6000, 4000)).save(Path(tmp) / "0001.jpg", quality=50)
            with patch.multiple(pipeline, FISHEYE_FOCAL_MM=7.5, SENSOR_WIDTH_MM=23.5, SPIRULA_RESULT={}):
                args = pipeline._fisheye_sfm_args(Path(tmp))
                self.assertEqual(pipeline.SPIRULA_RESULT["fisheye_focal_px"], 1914.9)
        self.assertEqual(args, ["--camera-model", "opencv-fisheye", "--camera-mode", "single",
                                "--focal", "1914.9"])

    def test_portrait_photos_use_the_long_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            Image.new("RGB", (4000, 6000)).save(Path(tmp) / "0001.jpg", quality=50)
            with patch.multiple(pipeline, FISHEYE_FOCAL_MM=7.5, SENSOR_WIDTH_MM=23.5, SPIRULA_RESULT={}):
                args = pipeline._fisheye_sfm_args(Path(tmp))
        self.assertEqual(args[-1], "1914.9")


class SpirulaCommandTests(options.SpirulaOptionTests):
    """Runs the real _run_spirula with a fake binary; checks the fisheye and splat-type flags."""

    def run_spirula(self, **kwargs):
        fisheye = kwargs.pop("fisheye", False)
        primitive = kwargs.pop("primitive", "3dgs")
        sfm_calls = []
        real_sfm_args = pipeline._fisheye_sfm_args
        with patch.multiple(pipeline, FISHEYE=fisheye, SPIRULA_PRIMITIVE=primitive), \
                patch.object(pipeline, "_fisheye_sfm_args",
                             lambda d: sfm_calls.append(d) or ["--camera-model", "opencv-fisheye",
                                                               "--camera-mode", "single", "--focal", "99.0"]):
            self._sfm_seen = sfm_calls
            return super().run_spirula(**kwargs)

    def test_standard_splats_by_default(self):
        train, _, _, _ = self.run_spirula()
        self.assertEqual(self.flag(train, "--primitive"), "3dgs")
        self.assertEqual(self._sfm_seen, [])

    def test_3dgut_and_fisheye(self):
        train, _, _, _ = self.run_spirula(fisheye=True, primitive="3dgut")
        self.assertEqual(self.flag(train, "--primitive"), "3dgut")
        self.assertEqual(len(self._sfm_seen), 1)


# The inherited option tests already ran in their own module.
for _name in [n for n in dir(options.SpirulaOptionTests) if n.startswith("test_")]:
    setattr(SpirulaCommandTests, _name, None)


class PodSettingsTests(unittest.TestCase):
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

    def test_fisheye_and_primitive_reach_the_pod(self):
        command = self.capture(engine="spirula", spirula_primitive="3dgut", fisheye=True,
                               fisheye_focal_mm=7.5, sensor_width_mm=23.5)
        self.assertIn("PIPELINE_SPIRULA_PRIMITIVE=3dgut", command)
        self.assertIn("PIPELINE_FISHEYE=1 PIPELINE_FISHEYE_FOCAL_MM=7.5 PIPELINE_SENSOR_WIDTH_MM=23.5", command)

    def test_ignore_moving_things_reaches_the_pod(self):
        self.assertIn("PIPELINE_SPIRULA_DISTRACTION=mild",
                      self.capture(engine="spirula", spirula_distraction="mild"))
        self.assertIn("PIPELINE_SPIRULA_DISTRACTION=off", self.capture(engine="spirula"))
        with self.assertRaisesRegex(TransferError, "distraction"):
            self.capture(engine="spirula", spirula_distraction="max")

    def test_no_fisheye_settings_by_default(self):
        self.assertNotIn("PIPELINE_FISHEYE", self.capture(engine="spirula"))

    def test_fisheye_with_nerfstudio_is_refused(self):
        with self.assertRaisesRegex(TransferError, "only supported with Spirula"):
            self.capture(engine="nerfstudio", fisheye=True)


class FisheyeJobTests(flow.JobFlowTests):
    def test_nerfstudio_fisheye_job_creates_no_pod(self):
        self.config.fisheye_lens = True
        had = "last_create" in vars(flow.FakeRunPod)
        previous = vars(flow.FakeRunPod).get("last_create")
        flow.FakeRunPod.last_create = None
        self.addCleanup(lambda: setattr(flow.FakeRunPod, "last_create", previous) if had
                        else delattr(flow.FakeRunPod, "last_create"))
        job = self.make_job()
        errors = []
        job.error.connect(errors.append)
        job._run_job(self.photos, ("A40", "SECURE"), engine="nerfstudio")
        self.assertEqual(job.outcome, "failed")
        self.assertIsNone(flow.FakeRunPod.last_create)
        self.assertTrue(any("only supported with Spirula" in error for error in errors))


for _name in ("test_photo_job_records_and_terminates_its_pod", "test_drive_video_job_fetches_on_the_pod",
              "test_cancel_before_creation_creates_no_pod"):
    setattr(FisheyeJobTests, _name, None)


if __name__ == "__main__":
    unittest.main()
