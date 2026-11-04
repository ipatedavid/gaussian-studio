"""The Windows transfers through the real ssh program, against a small local SSH server.

The server (paramiko) runs each command in bash on this computer, as the pod would: Git
Bash on Windows, /bin/bash elsewhere. This checks what the fake-ssh tests can't: the real
client's binary pipes, its options (a known_hosts file in a folder with a space, the null
device) and -tt without a console.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

from scp_client import ScpClient, TransferError, _stream_output  # noqa: E402

try:
    import paramiko
except ImportError:
    paramiko = None

GIT_BASH = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Git" / "bin" / "bash.exe"
BASH = str(GIT_BASH) if os.name == "nt" else shutil.which("bash")
SSH = shutil.which("ssh") and shutil.which("ssh-keygen")


def bash_path(path: Path) -> str:
    """A local folder as bash on the 'pod' names it (Git Bash: /c/Users/...)."""
    text = path.resolve().as_posix()
    if os.name == "nt" and text[1:2] == ":":
        text = f"/{text[0].lower()}{text[2:]}"
    return text


class _Server(paramiko.ServerInterface if paramiko else object):
    def __init__(self, allowed):
        self.allowed = allowed
        self.command = None
        self.ready = threading.Event()

    def check_auth_publickey(self, username, key):
        return paramiko.AUTH_SUCCESSFUL if key == self.allowed else paramiko.AUTH_FAILED

    def get_allowed_auths(self, username):
        return "publickey"

    def check_channel_request(self, kind, chanid):
        return paramiko.OPEN_SUCCEEDED if kind == "session" else paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_channel_pty_request(self, *args):
        return True

    def check_channel_exec_request(self, channel, command):
        self.command = command.decode()
        self.ready.set()
        return True


def _run(channel, command: str) -> None:
    """Run one command in bash, wiring the channel to its stdin, stdout and stderr."""
    process = subprocess.Popen([BASH, "-c", command], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def feed():
        try:
            for chunk in iter(lambda: channel.recv(65536), b""):
                process.stdin.write(chunk)
        except OSError:
            pass
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    def pump(stream, send):
        for chunk in iter(lambda: stream.read1(65536), b""):
            send(chunk)

    threads = [threading.Thread(target=feed, daemon=True),
               threading.Thread(target=pump, args=(process.stdout, channel.sendall)),
               threading.Thread(target=pump, args=(process.stderr, channel.sendall_stderr))]
    for thread in threads:
        thread.start()
    for thread in threads[1:]:
        thread.join()
    channel.send_exit_status(process.wait())
    channel.close()


# CI sets GS_REQUIRE_REAL_SSH so that a missing piece fails there instead of skipping.
@unittest.skipUnless((paramiko and SSH and BASH and Path(BASH).exists())
                     or os.environ.get("GS_REQUIRE_REAL_SSH"),
                     "needs paramiko, the ssh program and bash")
class RealSshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A folder with a space, like a Windows user name with one, for the known_hosts file.
        cls.temporary = tempfile.TemporaryDirectory(prefix="gs ssh ")
        cls.folder = Path(cls.temporary.name)
        key = cls.folder / "client key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
        cls.key = key
        allowed = paramiko.Ed25519Key(filename=str(key))
        host_key = paramiko.RSAKey.generate(2048)
        cls.listener = socket.socket()
        cls.listener.bind(("127.0.0.1", 0))
        cls.listener.listen(32)
        cls.port = cls.listener.getsockname()[1]

        def serve():
            while True:
                try:
                    sock, _ = cls.listener.accept()
                except OSError:
                    return
                threading.Thread(target=cls._session, args=(sock, host_key, allowed),
                                 daemon=True).start()

        threading.Thread(target=serve, daemon=True).start()
        cls.tempdir = patch.object(tempfile, "tempdir", str(cls.folder))
        cls.tempdir.start()
        cls.windows = patch("scp_client.WINDOWS", True)
        cls.windows.start()

    @staticmethod
    def _session(sock, host_key, allowed):
        transport = paramiko.Transport(sock)
        transport.add_server_key(host_key)
        server = _Server(allowed)
        try:
            transport.start_server(server=server)
            channel = transport.accept(30)
            if channel is not None and server.ready.wait(30):
                _run(channel, server.command)
                # Let the client hang up, so nothing still queued is cut off.
                deadline = time.monotonic() + 30
                while transport.is_active() and time.monotonic() < deadline:
                    time.sleep(0.05)
        except Exception:  # a client that hangs up early
            pass
        finally:
            transport.close()

    @classmethod
    def tearDownClass(cls):
        cls.windows.stop()
        cls.tempdir.stop()
        cls.listener.close()
        cls.temporary.cleanup()

    def setUp(self):
        self.client = ScpClient(self.key, "127.0.0.1", self.port)
        self.pod = Path(tempfile.mkdtemp(prefix="pod ", dir=self.folder))
        self.remote = bash_path(self.pod)
        self.logs: list[str] = []

    def test_commands_run_and_answer(self):
        result = self.client._ssh("echo hello; echo oops >&2; exit 3")
        self.assertEqual((result.returncode, result.stdout.strip()), (3, "hello"))
        self.assertTrue(result.stderr.strip().endswith("oops"), result.stderr)

    def test_photos_go_up_and_the_result_comes_back(self):
        photos = []
        for index, size in enumerate((5, 3 * 1024 ** 2 + 7)):
            photo = self.folder / f"photo {index}.jpg"
            photo.write_bytes(os.urandom(size))
            photos.append(photo)
        self.client._upload_batch(photos, self.logs.append,
                                  destination=f"root@127.0.0.1:{self.remote}/")
        for photo in photos:
            self.assertEqual((self.pod / photo.name).read_bytes(), photo.read_bytes())
        self.client._copy_file(str(photos[0]), f"root@127.0.0.1:{self.remote}/renamed.jpg")
        self.assertEqual((self.pod / "renamed.jpg").read_bytes(), photos[0].read_bytes())

        local = self.folder / "back.part"
        self.client._scp_with_progress([f"root@127.0.0.1:{self.remote}/photo 1.jpg"], str(local),
                                       self.logs.append, "photo download")
        self.assertEqual(local.read_bytes(), photos[1].read_bytes())
        self.assertTrue(any("100%" in line for line in self.logs))

    def test_a_large_result_comes_back_over_parallel_connections(self):
        data = os.urandom(20 * 1024 ** 2)
        (self.pod / "splat.ply").write_bytes(data)
        info = self.client._remote_file_info(f"{self.remote}/splat.ply")
        self.assertIsNotNone(info)
        local = self.folder / "splat.part"
        self.client._download_parallel(f"{self.remote}/splat.ply", local, *info, self.logs.append)
        self.assertEqual(local.read_bytes(), data)

    def test_missing_files_fail_with_the_pods_reason(self):
        with self.assertRaisesRegex(TransferError, "No such file"):
            self.client._scp_with_progress([f"root@127.0.0.1:{self.remote}/none.ply"],
                                           str(self.folder / "none"), None, "none download")

    def test_a_live_log_streams_with_a_terminal(self):
        # As run_pipeline runs the job: -tt, output streamed while it runs.
        with subprocess.Popen(
            ["ssh", *self.client.options, "-tt", "-n", "-p", str(self.port), "root@127.0.0.1",
             "for i in 1 2 3; do echo step $i; sleep 0.2; done"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            _stream_output(process, self.logs.append)
            self.assertEqual(process.wait(), 0)
        self.assertEqual([line for line in self.logs if line.startswith("step")],
                         ["step 1", "step 2", "step 3"])


if __name__ == "__main__":
    unittest.main()
