from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from scp_client import ScpClient, TransferError, _stream_output


class UploadBatchTests(unittest.TestCase):
    def test_checks_capacity_sends_one_batch_and_verifies(self):
        with tempfile.TemporaryDirectory() as temporary:
            images = []
            for index in range(3):
                path = Path(temporary) / f"photo-{index}.jpg"
                path.write_bytes(b"photo")
                images.append(path)
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            client.free_bytes = Mock(return_value=10**9)
            logs = []
            with patch.object(client, "_upload_batch") as batch, \
                 patch.object(client, "_verify_uploaded_images") as verify:
                client.upload(images, reserve_bytes=1000, log=logs.append)
            client.free_bytes.assert_called_once_with()
            batch.assert_called_once()
            self.assertEqual(batch.call_args.args[0], images)
            verify.assert_called_once_with({image.name: 5 for image in images})
            self.assertTrue(any("one SCP connection" in line for line in logs))
            self.assertTrue(any("Verified 3/3 uploaded photos" in line for line in logs))

    def test_video_upload_uses_one_scp_file_and_verifies_remote_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "clip with spaces.MP4"
            video.write_bytes(b"video-data")
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            client.free_bytes = Mock(return_value=10**9)
            with patch.object(client, "_upload_batch") as batch, \
                 patch.object(client, "_verify_uploaded_images") as verify:
                client.upload_video(video, reserve_bytes=1000)
            batch.assert_called_once()
            self.assertEqual(batch.call_args.args[0], [video])
            self.assertEqual(
                batch.call_args.kwargs["destination"],
                "root@example:/workspace/job/input/source_video.mp4",
            )
            verify.assert_called_once_with({"source_video.mp4": 10})

    def test_rejects_insufficient_space_before_scp(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "photo.jpg"
            image.write_bytes(b"photo")
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            client.free_bytes = Mock(return_value=100)
            with patch.object(client, "_upload_batch") as batch:
                with self.assertRaisesRegex(TransferError, "insufficient free space"):
                    client.upload([image], reserve_bytes=1000)
            batch.assert_not_called()

    def test_scp_batch_uses_one_process_for_all_photos(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fake_scp = folder / "scp"
            fake_scp.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "with open(os.environ['SCP_ARGS_FILE'], 'w') as file:\n"
                "    json.dump(sys.argv[1:], file)\n"
                "sys.stderr.write('photo.jpg 100% 5KB 1MB/s 00:00\\r')\n"
                "sys.stderr.flush()\n"
            )
            fake_scp.chmod(0o755)
            photos = folder / "photos with spaces"
            photos.mkdir()
            images = [photos / f"photo {index}.jpg" for index in range(3)]
            for image in images:
                image.write_bytes(b"photo")
            args_file = folder / "scp-args.json"
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            logs = []
            with patch.dict(os.environ, {
                "PATH": f"{folder}:{os.environ['PATH']}",
                "SCP_ARGS_FILE": str(args_file),
            }):
                client._upload_batch(images, logs.append)
            arguments = json.loads(args_file.read_text())
            self.assertEqual(arguments[-4:-1], [str(image) for image in images])
            self.assertEqual(arguments[-1], "root@example:/workspace/job/input/")
            self.assertTrue(any("100%" in line for line in logs))

    def test_download_logs_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fake_scp = folder / "scp"
            fake_scp.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "source, target = sys.argv[-2], sys.argv[-1]\n"
                "name = source.rsplit('/', 1)[-1]\n"
                "for percent in (40, 100):\n"
                "    sys.stderr.write(f'{name} {percent}% 12MB 6.0MB/s 00:01 ETA\\r')\n"
                "    sys.stderr.flush()\n"
                "data = b'ply\\nbody' if name.endswith('.ply') else json.dumps({}).encode()\n"
                "open(target, 'wb').write(data)\n"
            )
            fake_scp.chmod(0o755)
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            logs = []
            results = folder / "results"
            with patch.dict(os.environ, {"PATH": f"{folder}:{os.environ['PATH']}"}), \
                    patch.object(client, "_remote_file_info", return_value=None):
                client.download(results, logs.append)
            self.assertTrue(any(
                line.startswith("[SCP] gaussian_splat.ply") and "100%" in line for line in logs
            ))
            self.assertEqual((results / "gaussian_splat.ply").read_bytes(), b"ply\nbody")
            self.assertFalse(any(path.suffix == ".part" for path in results.iterdir()))

    def _fake_pod(self, folder: Path, *, flaky: bool = False) -> Path:
        """A fake `ssh` that runs the pod-side command locally against `folder/output`."""
        output = folder / "output"
        output.mkdir()
        fake_ssh = folder / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib, os, subprocess, sys\n"
            "command = sys.argv[-1].replace('/workspace/job/output', os.environ['FAKE_OUTPUT'])\n"
            "key = hashlib.md5(command.encode()).hexdigest()\n"
            "marker = os.path.join(os.environ['FAKE_OUTPUT'], '..', 'dropped-' + key)\n"
            "if os.environ.get('FAKE_FLAKY') and 'dd ' in command and not os.path.exists(marker):\n"
            "    open(marker, 'w').close()\n"
            "    data = subprocess.run(['bash', '-c', command], capture_output=True).stdout\n"
            "    sys.stdout.buffer.write(data[: len(data) // 3])\n"
            "    sys.exit(255)\n"
            "sys.exit(subprocess.run(['bash', '-c', command]).returncode)\n"
        )
        fake_ssh.chmod(0o755)
        fake_scp = folder / "scp"
        fake_scp.write_text(
            "#!/usr/bin/env python3\n"
            "import os, shutil, sys\n"
            "source = sys.argv[-2].split(':', 1)[1].replace('/workspace/job/output', os.environ['FAKE_OUTPUT'])\n"
            "open(os.path.join(os.environ['FAKE_OUTPUT'], '..', 'scp-used'), 'w').close()\n"
            "shutil.copyfile(source, sys.argv[-1])\n"
        )
        fake_scp.chmod(0o755)
        (output / "gaussian_splat.ply").write_bytes(b"ply\n" + os.urandom(40 * 1024 * 1024))
        (output / "pipeline_summary.json").write_text("{}")
        return output

    def _download(self, folder: Path, **env):
        client = ScpClient(Path("/tmp/test-key"), "example", 1234)
        logs: list[str] = []
        with patch.dict(os.environ, {
            "PATH": f"{folder}:{os.environ['PATH']}",
            "FAKE_OUTPUT": str(folder / "output"), **env,
        }):
            client.download(folder / "results", logs.append)
        return logs

    def test_large_result_downloads_over_parallel_connections(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = self._fake_pod(folder)
            logs = self._download(folder)
            result = folder / "results" / "gaussian_splat.ply"
            self.assertEqual(result.read_bytes(), (output / "gaussian_splat.ply").read_bytes())
            self.assertTrue(any("over 8 parallel connections" in line for line in logs))
            self.assertTrue(any(line.startswith("[download] gaussian_splat.ply") for line in logs))
            self.assertFalse((folder / "scp-used").exists() and not
                             (folder / "results" / "pipeline_summary.json").exists())

    def test_parallel_parts_resume_after_dropped_connections(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = self._fake_pod(folder)
            with patch("scp_client.time.sleep", lambda seconds: None):
                logs = self._download(folder, FAKE_FLAKY="1")
            result = folder / "results" / "gaussian_splat.ply"
            self.assertEqual(result.read_bytes(), (output / "gaussian_splat.ply").read_bytes())
            self.assertTrue(any("Downloaded and verified" in line for line in logs))

    def test_checksum_mismatch_falls_back_to_scp(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = self._fake_pod(folder)
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            size = (output / "gaussian_splat.ply").stat().st_size
            logs: list[str] = []
            with patch.dict(os.environ, {
                "PATH": f"{folder}:{os.environ['PATH']}", "FAKE_OUTPUT": str(output),
            }), patch.object(client, "_remote_file_info", return_value=(size, "0" * 32)):
                client.download(folder / "results", logs.append)
            self.assertTrue(any("retrying with a single connection" in line for line in logs))
            self.assertEqual(
                (folder / "results" / "gaussian_splat.ply").read_bytes(),
                (output / "gaussian_splat.ply").read_bytes(),
            )

    def test_scp_batch_reports_transfer_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fake_scp = folder / "scp"
            fake_scp.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "print('disk full', file=sys.stderr)\n"
                "sys.exit(7)\n"
            )
            fake_scp.chmod(0o755)
            image = folder / "photo.jpg"
            image.write_bytes(b"photo")
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            with patch.dict(os.environ, {"PATH": f"{folder}:{os.environ['PATH']}"}):
                with self.assertRaisesRegex(TransferError, "disk full"):
                    client._upload_batch([image], None)

    def test_verification_rejects_missing_extra_and_wrong_sized_files(self):
        client = ScpClient(Path("/tmp/test-key"), "example", 1234)
        client._ssh = Mock(return_value=Mock(
            returncode=0,
            stdout=json.dumps({"photo-a.jpg": 2, "unexpected.jpg": 5}),
            stderr="",
        ))
        with self.assertRaises(TransferError) as raised:
            client._verify_uploaded_images({"photo-a.jpg": 3, "photo-b.jpg": 4})
        for detail in ("missing", "unexpected", "wrong size"):
            self.assertIn(detail, str(raised.exception))


class StreamingOutputTests(unittest.TestCase):
    def test_carriage_returns_and_unterminated_output_arrive_before_exit(self):
        logs = []
        script = (
            "import sys,time; "
            "sys.stdout.write('step 1\\r'); sys.stdout.flush(); "
            "time.sleep(.2); "
            "sys.stdout.write('step 2\\r'); sys.stdout.flush(); "
            "time.sleep(.2); "
            "sys.stdout.write('done'); sys.stdout.flush(); "
            "time.sleep(.2)"
        )
        with subprocess.Popen(
            [sys.executable, "-u", "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            _stream_output(process, lambda line: logs.append((line, time.monotonic())))
            exit_time = time.monotonic()
            self.assertEqual(process.wait(), 0)
        self.assertEqual([line for line, _ in logs], ["step 1", "step 2", "done"])
        self.assertLess(logs[0][1], exit_time - .2)

    def test_reports_activity_when_child_is_quiet(self):
        logs = []
        with subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(.2)"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            _stream_output(process, logs.append, heartbeat_seconds=.05)
            self.assertEqual(process.wait(), 0)
        self.assertTrue(any("still running" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
