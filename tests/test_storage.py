from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))
from storage import GIB, estimate_video_frames, list_images, plan_input, storage_plan, VideoInfo


class StoragePlanTests(unittest.TestCase):
    def test_plan_grows_with_photo_bytes_and_respects_minimum(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index in range(3):
                with (folder / f"photo_{index}.jpg").open("wb") as file:
                    file.truncate(GIB)
            images = list_images(folder)
            total, reserve, volume_gb = storage_plan(images, 40)
            self.assertEqual(total, 3 * GIB)
            self.assertGreater(reserve, total)
            self.assertGreaterEqual(volume_gb * GIB, 1.1 * (total + reserve))
            self.assertGreaterEqual(volume_gb, 40)

    def test_video_plan_targets_three_frames_per_second_and_reserves_extraction_space(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "capture.mp4"
            with video.open("wb") as file:
                file.truncate(200 * 1024**2)
            info = VideoInfo(60.0, 3840, 2160, 30.0)
            with patch("storage.probe_video", return_value=info):
                plan = plan_input(video, "video", 0, 1)
            self.assertEqual(estimate_video_frames(info, 3.0), 180)
            self.assertEqual((plan.frame_target, plan.extract_fps), (180, 3.0))
            self.assertGreater(plan.estimated_frame_bytes, 6 * GIB)
            self.assertGreaterEqual(
                plan.volume_gb * GIB,
                1.1 * (plan.input_bytes + plan.working_reserve),
            )
            self.assertEqual(plan.results_parent, video.parent / "results" / "capture")

    def test_long_video_is_sampled_end_to_end_without_a_cap(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "long_capture.mp4"
            video.write_bytes(b"placeholder")
            info = VideoInfo(240.0, 1920, 1080, 30.0)
            with patch("storage.probe_video", return_value=info):
                default = plan_input(video, "video", 0, 1)
                faster = plan_input(video, "video", 0, 1, 5.0)
            # Every 10th frame of 7,200, then every 6th.
            self.assertEqual(default.frame_target, 720)
            self.assertEqual(faster.frame_target, 1200)
            self.assertGreater(faster.estimated_frame_bytes, default.estimated_frame_bytes)

    def test_frame_step_matches_the_pod(self):
        # 29.97 fps at 2 fps: every 15th frame (round(14.985)).
        self.assertEqual(estimate_video_frames(VideoInfo(60.0, 1, 1, 29.97), 2.0), 120)
        # Faster than the source: every frame.
        self.assertEqual(estimate_video_frames(VideoInfo(10.0, 1, 1, 24.0), 30.0), 240)
        # Drive does not report a frame rate: length x fps.
        self.assertEqual(estimate_video_frames(VideoInfo(10.5, 1, 1, 0.0), 3.0), 32)

    def test_ffprobe_reads_video_metadata_without_decoding_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "capture.mov"
            video.write_bytes(b"placeholder")
            output = '{"streams":[{"width":1920,"height":1080,"avg_frame_rate":"30000/1001"}],"format":{"duration":"60.0"}}'
            with patch("storage.subprocess.run", return_value=Mock(stdout=output)) as run:
                from storage import probe_video
                info = probe_video(video)
            self.assertAlmostEqual(info.fps, 29.97, places=2)
            self.assertEqual(info.duration_seconds, 60.0)
            self.assertIn("-show_entries", run.call_args.args[0])

    def test_ffprobe_uses_stream_fallback_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "capture.mkv"
            video.write_bytes(b"placeholder")
            output = (
                '{"streams":[{"width":1280,"height":720,'
                '"avg_frame_rate":"0/0","r_frame_rate":"30/1","duration":"10.0"}],'
                '"format":{"duration":"N/A"}}'
            )
            with patch("storage.subprocess.run", return_value=Mock(stdout=output)):
                from storage import probe_video
                info = probe_video(video)
            self.assertEqual(info.duration_seconds, 10.0)
            self.assertEqual(info.fps, 30.0)

    def test_short_video_is_rejected_before_pod_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "blink.mp4"
            video.write_bytes(b"x")
            with patch("storage.probe_video", return_value=VideoInfo(0.5, 1920, 1080, 30.0)):
                with self.assertRaisesRegex(ValueError, "raise the frames per second"):
                    plan_input(video, "video", 0, 1)


if __name__ == "__main__":
    unittest.main()
