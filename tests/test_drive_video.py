from __future__ import annotations

import hashlib
import http.server
import json
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "worker"))

import pipeline
from drive_client import DriveClient, DriveFile
from storage import GIB, plan_drive_video


class FakeDrive(http.server.BaseHTTPRequestHandler):
    """Serves one file at /files/<id>?alt=media, honouring Range headers."""

    body = b""
    range_requests = 0

    def do_GET(self):
        data = type(self).body
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if match:
            type(self).range_requests += 1
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


class WorkerDriveVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        FakeDrive.body = os.urandom(5 * 1024 * 1024 + 123)
        FakeDrive.range_requests = 0
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeDrive)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.patches = [
            patch.object(pipeline, "JOB_ROOT", root),
            patch.object(pipeline, "INPUT_DIR", root / "input"),
            patch.object(pipeline, "DRIVE_MANIFEST", root / "drive_manifest.json"),
            patch.object(pipeline, "DRIVE_RANGE_BYTES", 1024 * 1024),
        ]
        for item in self.patches:
            item.start()
        self.root = root

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def write_manifest(self, name="source_video.mov", md5=None, kind="video"):
        body = FakeDrive.body
        manifest = {
            "kind": kind,
            "api_base": f"http://127.0.0.1:{self.server.server_port}",
            "token": "t",
            "files": [{
                "id": "vid1", "name": name, "size": len(body),
                "md5": md5 if md5 is not None else hashlib.md5(body).hexdigest(),
            }],
        }
        (self.root / "drive_manifest.json").write_text(json.dumps(manifest))

    def test_video_is_fetched_in_parallel_ranges_and_verified(self):
        self.write_manifest()
        pipeline.fetch_drive_inputs()
        target = self.root / "input" / "source_video.mov"
        self.assertEqual(target.read_bytes(), FakeDrive.body)
        self.assertEqual(FakeDrive.range_requests, 6)
        self.assertEqual([p.name for p in (self.root / "input").iterdir()], ["source_video.mov"])
        self.assertFalse((self.root / "drive_manifest.json").exists())

    def test_checksum_mismatch_fails(self):
        self.write_manifest(md5="0" * 32)
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            pipeline.fetch_drive_inputs()

    def test_video_manifest_must_use_the_source_video_name(self):
        self.write_manifest(name="../evil.mov")
        with self.assertRaisesRegex(RuntimeError, "invalid entry"):
            pipeline.fetch_drive_inputs()


class DriveVideoPlanTests(unittest.TestCase):
    def video(self, **extra):
        values = dict(id="v1", name="Walkaround.MOV", dest="source_video.mov",
                      size=2 * GIB, md5="", width=3840, height=2160, duration_ms=95_000)
        values.update(extra)
        return DriveFile(**values)

    def test_auto_frames_use_drive_duration(self):
        plan = plan_drive_video("f1", "Captures", self.video(), Path("/r"), 0, 1)
        self.assertEqual(plan.kind, "drive_video")
        self.assertEqual(plan.frame_target, 285)
        self.assertEqual(plan.pipeline_source_type, "video")
        self.assertEqual(plan.results_parent, Path("/r/Walkaround"))
        self.assertEqual(plan.display_name, "Walkaround.MOV")

    def test_unknown_metadata_still_plans(self):
        plan = plan_drive_video(
            "f1", "Captures", self.video(width=0, height=0, duration_ms=0), Path("/r"), 0, 1,
        )
        # Length unknown: plan storage for 600 frames; the pod still samples at 3 fps.
        self.assertEqual((plan.frame_target, plan.extract_fps), (600, 3.0))
        self.assertGreater(plan.volume_gb, 30)

    def test_drive_video_metadata_is_parsed(self):
        file = DriveClient._video_file({
            "id": "abc", "name": "clip.MP4", "size": "1000", "md5Checksum": "m",
            "videoMediaMetadata": {"width": 1920, "height": "1080", "durationMillis": "61000"},
        })
        self.assertEqual(file.dest, "source_video.mp4")
        self.assertEqual((file.width, file.height, file.duration_ms), (1920, 1080, 61000))


if __name__ == "__main__":
    unittest.main()
