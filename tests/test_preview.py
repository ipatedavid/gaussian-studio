from __future__ import annotations

import http.server
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "desktop"))
sys.path.insert(0, str(ROOT / "tests"))

from PySide6 import QtCore

import job_controller
import scp_client
from scp_client import ScpClient
import test_job_flow as flow

APP = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


class Viewer(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html>viewer</html>"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class FakeTunnel:
    def __init__(self, url):
        self.url = url
        self.closed = False

    def alive(self):
        return not self.closed

    def close(self):
        self.closed = True


class PreviewScp(flow.FakeScp):
    tunnels: list = []
    viewer_url = ""
    pipeline_kwargs: dict = {}
    seen_ready = threading.Event()

    def open_tunnel(self, remote_port=7007):
        tunnel = FakeTunnel(PreviewScp.viewer_url)
        PreviewScp.tunnels.append(tunnel)
        return tunnel

    def run_pipeline(self, steps, downscales, log, **kwargs):
        PreviewScp.pipeline_kwargs = kwargs
        log("[pipeline] starting Spirula training for 30000 iterations (floater suppression: off; guidance: off)")
        if kwargs.get("preview", True):
            PreviewScp.seen_ready.wait(10)


class PreviewFlowTests(flow.JobFlowTests):
    def setUp(self):
        super().setUp()
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Viewer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        PreviewScp.viewer_url = f"http://127.0.0.1:{self.server.server_port}/"
        PreviewScp.tunnels = []
        PreviewScp.seen_ready = threading.Event()
        self.extra = patch("job_controller.ScpClient", PreviewScp)
        self.extra.start()

    def tearDown(self):
        self.extra.stop()
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def run_job(self, preview=True):
        self.config.live_preview = preview
        job = self.make_job()
        ready, closed = [], []
        job.preview_ready.connect(lambda url: (ready.append(url), PreviewScp.seen_ready.set()),
                                  QtCore.Qt.DirectConnection)
        job.preview_closed.connect(closed.append, QtCore.Qt.DirectConnection)
        job._run_job(self.photos, ("A40", "SECURE"), engine="spirula")
        return job, ready, closed

    def test_preview_opens_when_training_starts_and_closes_after(self):
        job, ready, closed = self.run_job()
        self.assertEqual(job.outcome, "finished")
        self.assertEqual(ready, [PreviewScp.viewer_url])
        self.assertEqual(len(closed), 1)
        self.assertIn("Training finished", closed[0])
        self.assertTrue(all(tunnel.closed for tunnel in PreviewScp.tunnels))
        self.assertTrue(PreviewScp.pipeline_kwargs["preview"])

    def test_preview_off_opens_no_tunnel(self):
        job, ready, closed = self.run_job(preview=False)
        self.assertEqual(job.outcome, "finished")
        self.assertEqual((ready, closed, PreviewScp.tunnels), ([], [], []))
        self.assertFalse(PreviewScp.pipeline_kwargs["preview"])


# The base class's flow tests already ran in their own module.
for _name in [n for n in dir(flow.JobFlowTests) if n.startswith("test_")]:
    setattr(PreviewFlowTests, _name, None)


class TunnelCommandTests(unittest.TestCase):
    def test_open_tunnel_forwards_a_local_port(self):
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Viewer)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as tmp:
            fake_ssh = Path(tmp) / "ssh"
            # Parses the -L argument and relays TCP like `ssh -N -L` would.
            fake_ssh.write_text(
                "#!/usr/bin/env python3\n"
                "import os, socket, sys, threading\n"
                "spec = sys.argv[sys.argv.index('-L') + 1]\n"
                "bind, lport, rhost, rport = spec.split(':')\n"
                "assert '-N' in sys.argv and bind == '127.0.0.1'\n"
                "rport = int(os.environ['FAKE_REMOTE_PORT'])\n"
                "listener = socket.socket(); listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                "listener.bind((bind, int(lport))); listener.listen()\n"
                "def pipe(a, b):\n"
                "    try:\n"
                "        while (data := a.recv(65536)):\n"
                "            b.sendall(data)\n"
                "    finally:\n"
                "        b.close()\n"
                "while True:\n"
                "    client, _ = listener.accept()\n"
                "    upstream = socket.create_connection((rhost, rport))\n"
                "    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()\n"
                "    threading.Thread(target=pipe, args=(upstream, client), daemon=True).start()\n"
            )
            fake_ssh.chmod(0o755)
            client = ScpClient(Path("/tmp/test-key"), "example", 1234)
            with patch.dict(os.environ, {
                "PATH": f"{tmp}:{os.environ['PATH']}",
                "FAKE_REMOTE_PORT": str(server.server_port),
            }):
                tunnel = client.open_tunnel()
                try:
                    deadline = 50
                    while deadline and not job_controller.viewer_responds(tunnel.url):
                        deadline -= 1
                        threading.Event().wait(0.1)
                    with urllib.request.urlopen(tunnel.url, timeout=5) as response:
                        self.assertEqual(response.read(), b"<html>viewer</html>")
                    self.assertTrue(tunnel.alive())
                finally:
                    tunnel.close()
                self.assertFalse(tunnel.alive())
        server.shutdown()
        server.server_close()

    def test_preview_settings_reach_the_pod(self):
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
            client.run_pipeline(10, 1, lambda line: None, engine="spirula")
            client.run_pipeline(10, 1, lambda line: None, engine="nerfstudio", preview=False)
        self.assertIn("PIPELINE_PREVIEW=1 PIPELINE_PREVIEW_PORT=7007", captured[0])
        self.assertIn("PIPELINE_PREVIEW=0", captured[1])


if __name__ == "__main__":
    unittest.main()
