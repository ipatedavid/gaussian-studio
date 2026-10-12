from __future__ import annotations

import codecs
import concurrent.futures
import errno
import fcntl
import hashlib
import json
import os
import pty
import re
import select
import selectors
import shlex
import socket
import struct
import subprocess
import tempfile
import termios
import threading
import time
from pathlib import Path
from typing import Callable


class TransferError(RuntimeError):
    pass


REMOTE_OUTPUT_DIR = "/workspace/job/output"
# One SCP stream over a long route (pod abroad -> this computer) is limited by
# latency, not bandwidth: about 1 Mbit/s was measured. Several connections each
# reading their own byte range multiply that.
DOWNLOAD_STREAMS = 8
PARALLEL_DOWNLOAD_MIN_BYTES = 16 * 1024 ** 2
DOWNLOAD_PART_ATTEMPTS = 4
PREVIEW_PORT = 7007   # the training viewer's port on the pod
REMOTE_JOB_LOG = "/workspace/job/job.log"  # the worker's output, kept for reconnecting


class Tunnel:
    """An `ssh -N -L` forward from 127.0.0.1:<local_port> to a port on the pod."""

    def __init__(self, process: subprocess.Popen, local_port: int):
        self.process = process
        self.local_port = local_port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.local_port}/"

    def alive(self) -> bool:
        return self.process.poll() is None

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _stream_output(
    process: subprocess.Popen[bytes],
    log: Callable[[str], None],
    *,
    heartbeat_seconds: float = 15,
) -> None:
    """Forward SSH output as it arrives, including carriage-return progress."""
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    last_log = time.monotonic()
    last_data = last_log

    def emit(value: str) -> None:
        nonlocal last_log
        line = _ANSI_ESCAPE.sub("", value).replace("\x08", "").strip()
        if line:
            log(line)
            last_log = time.monotonic()

    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map():
            for _key, _events in selector.select(
                timeout=min(1, max(0.01, heartbeat_seconds / 2))
            ):
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    pending += decoder.decode(b"", final=True)
                    emit(pending)
                    selector.unregister(process.stdout)
                    break
                last_data = time.monotonic()
                pending += decoder.decode(chunk)
                # CLIs often redraw progress with CR instead of printing LF.
                parts = re.split(r"[\r\n]", pending)
                pending = parts.pop()
                for part in parts:
                    emit(part)
                if len(pending) >= 8192:
                    emit(pending)
                    pending = ""
            now = time.monotonic()
            if pending and now - last_data >= 2:
                emit(pending)
                pending = ""
            if selector.get_map() and now - last_log >= heartbeat_seconds:
                log("Processing is still running; waiting for the next tool update.")
                last_log = now


def _describe_ssh_failure(returncode: int, stderr: str) -> str:
    """Turn an ssh failure into a short, actionable sentence."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    text = " ".join(lines)
    lowered = text.lower()
    if returncode == 1 and not lines:
        return "connected; the pod's bootstrap is still running"
    if "host identification has changed" in lowered or "host key verification failed" in lowered:
        return "host key mismatch (" + (lines[-1] if lines else "no detail") + ")"
    if "permission denied" in lowered:
        return (
            "the pod rejected the SSH key; check that the key in Settings matches "
            "the .pub file next to it (" + lines[-1] + ")"
        )
    if "connection refused" in lowered:
        return "connection refused; the pod's SSH daemon is not listening yet"
    if "timed out" in lowered:
        return "connection timed out; the public port is not reachable yet"
    if "connection reset" in lowered or "kex_exchange_identification" in lowered:
        return "connection reset by the pod; SSH is still starting (" + lines[-1] + ")"
    if lines:
        return f"ssh exit {returncode}: {lines[-1][:200]}"
    return f"ssh exit {returncode} with no message"


class ScpClient:
    def __init__(self, private_key: Path, host: str, port: int):
        self.private_key = private_key
        self.host = host
        self.port = port
        # RunPod reuses IP:port pairs across pods, but every new container has
        # new host keys. A shared ~/.ssh/known_hosts would then reject the new
        # pod ("host identification has changed") and BatchMode hides the
        # message. Each job gets its own empty known_hosts file instead; the
        # key is still pinned for the rest of the job.
        self._known_hosts_dir = tempfile.TemporaryDirectory(prefix="gaussian-known-hosts-")
        known_hosts = Path(self._known_hosts_dir.name) / "known_hosts"
        self.options = [
            "-i", str(private_key), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", "GlobalKnownHostsFile=/dev/null",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
        ]

    def _ssh(self, command: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", *self.options, "-p", str(self.port), f"root@{self.host}", command],
            capture_output=True, text=True, timeout=timeout,
        )

    def _verbose_probe(self) -> str:
        """One verbose ssh attempt, boiled down to how far the handshake got."""
        try:
            result = subprocess.run(
                ["ssh", *self.options, "-vv", "-n", "-p", str(self.port),
                 f"root@{self.host}", "true"],
                capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired:
            return "the verbose probe timed out"
        except OSError as exc:
            return f"could not run ssh: {exc}"
        markers = (
            "Connection established", "Remote protocol version", "kex_exchange_identification",
            "Connection closed", "Connection reset", "Connection refused", "timed out",
            "Offering public key", "Server accepts key", "Authentication succeeded",
            "Permission denied", "no matching", "Host key",
        )
        seen: list[str] = []
        for line in result.stderr.splitlines():
            line = line.replace("debug1: ", "").replace("debug2: ", "").strip()
            if any(marker in line for marker in markers) and line not in seen:
                seen.append(line[:160])
        return " | ".join(seen[-6:]) or "no handshake details were printed"

    def wait_ready(
        self, timeout: int = 600,
        log: Callable[[str], None] | None = None,
        status_check: Callable[[], None] | None = None,
        endpoint_provider: Callable[[], tuple[str, int] | None] | None = None,
        network_failure_limit: int = 360,
    ) -> None:
        started = time.monotonic()
        deadline = started + timeout
        next_progress = started
        next_probe = started
        next_endpoint_check = started + 15
        network_failing_since: float | None = None
        last_detail = "no connection attempt has finished yet"
        while time.monotonic() < deadline:
            # RunPod can change a pod's public address or port; follow it.
            if endpoint_provider and time.monotonic() >= next_endpoint_check:
                next_endpoint_check = time.monotonic() + 15
                try:
                    current = endpoint_provider()
                except Exception:
                    current = None
                if current and current != (self.host, self.port):
                    if log:
                        log(f"RunPod now reports SSH at {current[0]}:{current[1]} "
                            f"(was {self.host}:{self.port}); switching to it.")
                    self.host, self.port = current
                    network_failing_since = None
            try:
                result = self._ssh("test -d /workspace/worker", timeout=60)
                if result.returncode == 0:
                    if log:
                        log("SSH is ready and the pod bootstrap completed.")
                    return
                last_detail = _describe_ssh_failure(result.returncode, result.stderr)
                lowered = (result.stderr or "").lower()
                if any(word in lowered for word in (
                    "refused", "closed by", "timed out", "reset", "no route", "unreachable",
                )):
                    network_failing_since = network_failing_since or time.monotonic()
                    if log and time.monotonic() >= next_probe:
                        next_probe = time.monotonic() + 60
                        log(f"SSH handshake detail for {self.host}:{self.port}: "
                            f"{self._verbose_probe()}")
                else:
                    network_failing_since = None
            except subprocess.TimeoutExpired:
                last_detail = (
                    "connected, but the login did not finish within 60 s "
                    "(the pod's SSH login is hanging)"
                )
            except OSError as exc:
                last_detail = f"could not run ssh on this computer: {exc}"
            if status_check:
                status_check()
            now = time.monotonic()
            if log and now >= next_progress:
                log(
                    f"Public SSH port exists; waiting for the image, SSH daemon, "
                    f"and worker bootstrap ({int(now - started)}s elapsed). "
                    f"Last SSH result ({self.host}:{self.port}): {last_detail}"
                )
                next_progress = now + 15
            if network_failing_since and now - network_failing_since >= network_failure_limit:
                raise TransferError(
                    f"The pod's public SSH port ({self.host}:{self.port}) has not accepted a "
                    f"working connection for {int(now - network_failing_since)} s. "
                    f"Last result: {last_detail}. This usually means the host's port "
                    "forwarding is broken; stop the pod and try again, ideally on Secure Cloud "
                    "(set CLOUD_TYPE=SECURE in .env)."
                )
            time.sleep(5)
        raise TransferError(
            "SSH or worker bootstrap did not become ready within 10 minutes. "
            f"Last SSH result: {last_detail}"
        )

    def open_tunnel(self, remote_port: int = PREVIEW_PORT) -> Tunnel:
        """Forward a free local port to `remote_port` on the pod (localhost only)."""
        local_port = _free_local_port()
        process = subprocess.Popen(
            ["ssh", *self.options, "-N",
             "-o", "ExitOnForwardFailure=yes",
             "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
             "-p", str(self.port), f"root@{self.host}"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return Tunnel(process, local_port)

    # ---- reconnecting to a job whose app session was lost -----------------
    # Both helpers only send shell/Python one-liners; the worker scripts already
    # on the pod are never changed.

    _STATE_SCRIPT = r"""
import glob, json, os, re
me = {os.getpid(), os.getppid()}
argvs = []
for pid in os.listdir('/proc'):
    if pid.isdigit() and int(pid) not in me:
        try:
            raw = open('/proc/%s/cmdline' % pid, 'rb').read()
        except OSError:
            continue
        argvs.append([part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part])
# Match the program actually running, not any text in a command line (an ssh
# command that merely mentions job.py must not count).
def is_job(argv):
    return (len(argv) > 1 and os.path.basename(argv[0]).startswith('python')
            and '/workspace/worker/job.py' in argv[1:3])
def is_trainer(argv):
    if argv and os.path.basename(argv[0]) == 'spirula':
        return len(argv) > 1 and argv[1] in ('train', 'geometry', 'sfm', 'sam')
    return any(os.path.basename(part) in ('ns-train', 'ns-export', 'ns-process-data', 'colmap')
               for part in argv[:2])
root = '/workspace/job/nerfstudio_outputs'
ckpts = [(int(m.group(1)), os.path.getmtime(path))
         for path in glob.glob(root + '/**/step-*.ckpt', recursive=True)
         for m in [re.search(r'step-0*(\d+)', path)] if m]
steps = [step for step, _ in ckpts]
# Training writes its config when it starts: with the newest checkpoint's
# time, that gives the speed so far.
configs = glob.glob(root + '/*/config.json') + glob.glob(root + '/**/config.yml', recursive=True)
total = None
for cfg in glob.glob(root + '/*/config.json'):
    try:
        total = json.load(open(cfg)).get('num_iterations') or total
    except (OSError, ValueError):
        pass
for cfg in glob.glob(root + '/**/config.yml', recursive=True):
    try:
        found = re.search(r'max_num_iterations: (\d+)', open(cfg).read())
        total = int(found.group(1)) if found else total
    except OSError:
        pass
out = '/workspace/job/output/'
print(json.dumps({
    'job_running': any(is_job(argv) for argv in argvs),
    'trainer_running': any(is_trainer(argv) for argv in argvs),
    'ply_ready': os.path.isfile(out + 'gaussian_splat.ply') and os.path.getsize(out + 'gaussian_splat.ply') > 0,
    'summary_ready': os.path.isfile(out + 'pipeline_summary.json'),
    'step': max(steps) if steps else None,
    'step_time': max(ckpts)[1] if ckpts else None,
    'train_started': min(map(os.path.getmtime, configs)) if configs else None,
    'total_steps': total,
    'spirula': os.path.isdir('/workspace/spirula'),
}))
"""

    def has_job_log(self) -> bool:
        """Whether the pod keeps the worker's output (jobs started by older workers don't)."""
        return self._ssh(f"test -f {REMOTE_JOB_LOG}", timeout=60).returncode == 0

    def follow_job_log(self, log: Callable[[str], None], stop: threading.Event,
                       lines: int = 40) -> None:
        """Stream the worker's output log from the pod, its last lines first, until stop is set."""
        with subprocess.Popen(
            ["ssh", *self.options, "-n", "-p", str(self.port), f"root@{self.host}",
             f"tail -n {int(lines)} -F {REMOTE_JOB_LOG}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ) as process:
            def end_on_stop():
                stop.wait()
                if process.poll() is None:
                    process.terminate()

            threading.Thread(target=end_on_stop, daemon=True).start()
            _stream_output(process, log, heartbeat_seconds=float("inf"))

    def remote_job_state(self) -> dict:
        """What is running on the pod and whether the result files exist yet."""
        result = self._ssh("python3 -c " + shlex.quote(self._STATE_SCRIPT), timeout=60)
        if result.returncode:
            raise TransferError(f"Could not read the pod's job state: {result.stderr.strip()[-200:]}")
        try:
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise TransferError("The pod returned an unreadable job state.") from exc

    _FINISH_SCRIPT = r"""
import glob, json, os, shutil, subprocess, sys
engine = sys.argv[1]
root, out = '/workspace/job/nerfstudio_outputs', '/workspace/job/output'
os.makedirs(out, exist_ok=True)
newest = lambda paths: max(paths, key=os.path.getmtime) if paths else None
if engine == 'spirula':
    ply = newest(glob.glob(root + '/**/splat.ply', recursive=True))
else:
    config = newest(glob.glob(root + '/**/config.yml', recursive=True))
    if config is None:
        sys.exit('no Nerfstudio training config was found')
    subprocess.run(['ns-export', 'gaussian-splat', '--load-config', config,
                    '--output-dir', '/workspace/job/exports'], check=True)
    ply = newest(glob.glob('/workspace/job/exports/**/*.ply', recursive=True))
if ply is None:
    sys.exit('training left no splat file')
shutil.copy2(ply, out + '/gaussian_splat.ply')
if not os.path.isfile(out + '/pipeline_summary.json'):
    json.dump({'engine': engine, 'reattached': True, 'source_ply': ply,
               'note': 'Finished by the app after it reconnected; the pod job script had stopped.'},
              open(out + '/pipeline_summary.json', 'w'), indent=2)
print('finished from', ply)
"""

    def finish_remote(self, engine: str) -> str:
        """Put the trained splat where download() expects it, as job.py would have."""
        if engine not in {"nerfstudio", "spirula"}:
            raise TransferError(f"Unknown processing engine: {engine}")
        result = self._ssh(
            "python3 -c " + shlex.quote(self._FINISH_SCRIPT) + f" {engine}", timeout=1800,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise TransferError(
                "Could not finish the job on the pod: " + (detail[-1] if detail else "unknown error")
            )
        return result.stdout.strip()

    def free_bytes(self) -> int:
        result = self._ssh("df -B1 --output=avail /workspace | tail -n 1")
        if result.returncode:
            raise TransferError(f"Could not verify pod disk space: {result.stderr.strip()}")
        try:
            return int(result.stdout.strip())
        except ValueError as exc:
            raise TransferError("Pod returned an invalid free-space reading.") from exc

    def _copy_file(
        self, source: str, destination: str,
        log: Callable[[str], None] | None = None,
        progress: str = "Transfer",
    ) -> None:
        with subprocess.Popen(
            ["scp", *self.options, "-P", str(self.port), source, destination],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        ) as process:
            while True:
                try:
                    _, stderr = process.communicate(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if log:
                        log(f"{progress} is still in progress...")
            if process.returncode:
                raise TransferError(f"{progress} failed: {stderr.strip()}")

    def upload_scripts(
        self, worker_dir: Path, log: Callable[[str], None] | None = None,
    ) -> None:
        for filename in ("job.py", "pipeline.py"):
            source = worker_dir / filename
            if not source.is_file():
                raise TransferError(f"Worker script is missing: {source}")
            if log:
                log(f"Uploading worker script {filename}...")
            self._copy_file(
                str(source), f"root@{self.host}:/workspace/worker/{filename}",
                log, f"Worker script {filename} upload",
            )
            if log:
                log(f"Uploaded worker script {filename}.")

    def _upload_batch(
        self, images: list[Path], log: Callable[[str], None] | None,
        *, destination: str | None = None, description: str = "Photo batch",
    ) -> None:
        """Send input files through one SCP connection, streaming its progress meter."""
        self._scp_with_progress(
            [str(image) for image in images],
            destination or f"root@{self.host}:/workspace/job/input/",
            log, f"{description} upload",
        )

    def _scp_with_progress(
        self, sources: list[str], destination: str,
        log: Callable[[str], None] | None, label: str,
    ) -> None:
        """Run one SCP transfer (either direction) and log its progress meter."""
        master, slave = pty.openpty()
        process = None
        try:
            # SCP emits its progress meter only when stderr is a terminal.
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 160, 0, 0))
            process = subprocess.Popen(
                ["scp", *self.options, "-P", str(self.port), *sources, destination],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=slave,
            )
            os.close(slave)
            slave = -1
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            pending = ""
            error_lines: list[str] = []
            last_log = time.monotonic()
            last_progress = 0.0

            def report(value: str) -> None:
                nonlocal last_log, last_progress
                line = _ANSI_ESCAPE.sub("", value).replace("\x08", "").strip()
                if not line:
                    return
                now = time.monotonic()
                if "%" in line:
                    if log and (now - last_progress >= 3 or re.search(r"\b100%", line)):
                        log(f"[SCP] {line}")
                        last_log = last_progress = now
                else:
                    error_lines.append(line)
                    if log:
                        log(f"[SCP] {line}")
                        last_log = now

            while True:
                ready, _, _ = select.select([master], [], [], 1)
                if ready:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError as exc:
                        if exc.errno != errno.EIO:
                            raise
                        chunk = b""
                    if not chunk:
                        pending += decoder.decode(b"", final=True)
                        report(pending)
                        break
                    pending += decoder.decode(chunk)
                    parts = re.split(r"[\r\n]", pending)
                    pending = parts.pop()
                    for part in parts:
                        report(part)
                    if len(pending) >= 8192:
                        report(pending)
                        pending = ""
                if log and time.monotonic() - last_log >= 15:
                    log(f"{label} is still running...")
                    last_log = time.monotonic()
            if process.wait():
                detail = "; ".join(error_lines[-3:]) or f"exit code {process.returncode}"
                raise TransferError(f"{label} failed: {detail}")
        finally:
            if slave != -1:
                os.close(slave)
            os.close(master)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def _verify_uploaded_images(self, expected: dict[str, int]) -> None:
        script = (
            "import json; from pathlib import Path; "
            "root=Path('/workspace/job/input'); "
            "print(json.dumps({p.name: p.stat().st_size for p in root.iterdir() "
            "if p.is_file() and not p.is_symlink()}))"
        )
        result = self._ssh("python3 -c " + shlex.quote(script), timeout=120)
        if result.returncode:
            raise TransferError(
                f"Could not verify uploaded photos: {result.stderr.strip()}"
            )
        try:
            actual = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TransferError("Pod returned invalid photo verification data.") from exc
        if not isinstance(actual, dict):
            raise TransferError("Pod returned invalid photo verification data.")
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        wrong_size = {name for name in expected.keys() & actual.keys()
                      if expected[name] != actual[name]}
        if missing or extra or wrong_size:
            details = []
            for label, names in (("missing", missing), ("unexpected", extra),
                                 ("wrong size", wrong_size)):
                if names:
                    details.append(f"{label}: {', '.join(sorted(names)[:3])}")
            raise TransferError(
                "Uploaded files failed verification (" + "; ".join(details) + "). "
                "Processing was not started."
            )

    def upload(
        self, images: list[Path], reserve_bytes: int,
        log: Callable[[str], None] | None = None,
    ) -> None:
        source_stats = {image: image.stat() for image in images}
        expected = {image.name: source_stats[image].st_size for image in images}
        if len(expected) != len(images):
            raise TransferError("Photo filenames are not unique; upload was not started.")
        total_bytes = sum(expected.values())
        if self.free_bytes() < total_bytes + reserve_bytes:
            raise TransferError(
                "Pod has insufficient free space for the entire photo batch and "
                "processing reserve. No photos were uploaded."
            )
        if log:
            log(f"Sending {len(images)} photos ({total_bytes / 1024**2:.1f} MiB) "
                "in one SCP connection...")
        started = time.monotonic()
        self._upload_batch(images, log)
        for image, before in source_stats.items():
            after = image.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise TransferError(
                    f"Source photo changed during upload: {image.name}. "
                    "Processing was not started."
                )
        self._verify_uploaded_images(expected)
        if log:
            elapsed = max(time.monotonic() - started, 0.001)
            log(f"Verified {len(images)}/{len(images)} uploaded photos "
                f"({total_bytes / 1024**2:.1f} MiB) in {elapsed:.0f}s "
                f"({total_bytes * 8 / elapsed / 1_000_000:.1f} Mbps effective).")

    def verify_video_tools(self) -> None:
        result = self._ssh("command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null")
        if result.returncode:
            raise TransferError(
                "The RunPod image needs ffmpeg and ffprobe to process video."
            )

    def upload_video(
        self, video: Path, reserve_bytes: int,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.upload_videos([video], reserve_bytes, log)

    @staticmethod
    def remote_video_name(video: Path, index: int, count: int) -> str:
        """source_video.mp4 for a single video, source_video_<n>.mp4 when there are several."""
        stem = "source_video" if count == 1 else f"source_video_{index}"
        return stem + video.suffix.lower()

    def upload_mixed(
        self, images: list[Path], videos: list[Path], reserve_bytes: int,
        log: Callable[[str], None] | None = None,
    ) -> None:
        """Photos, then videos named source_video_<n>; each step verifies everything so far."""
        video_bytes = sum(video.stat().st_size for video in videos)
        self.upload(images, reserve_bytes + video_bytes, log)
        photos = {image.name: image.stat().st_size for image in images}
        self.upload_videos(videos, reserve_bytes, log, already_uploaded=photos, numbered=True)

    def upload_videos(
        self, videos: list[Path], reserve_bytes: int,
        log: Callable[[str], None] | None = None,
        already_uploaded: dict[str, int] | None = None, numbered: bool = False,
    ) -> None:
        before = [video.stat() for video in videos]
        total = sum(stat.st_size for stat in before)
        if self.free_bytes() < total + reserve_bytes:
            raise TransferError(
                "Pod has insufficient free space for the video and extracted frames. "
                "No video was uploaded."
            )
        started = time.monotonic()
        expected = {}
        for index, (video, stat) in enumerate(zip(videos, before), start=1):
            remote_name = self.remote_video_name(video, index, 2 if numbered else len(videos))
            if log and len(videos) > 1:
                log(f"Uploading video {index} of {len(videos)}: {video.name}")
            self._upload_batch(
                [video], log,
                destination=f"root@{self.host}:/workspace/job/input/{remote_name}",
                description="Video",
            )
            after = video.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise TransferError(
                    f"{video.name} changed during upload. Processing was not started."
                )
            expected[remote_name] = stat.st_size
        self._verify_uploaded_images({**(already_uploaded or {}), **expected})
        if log:
            elapsed = max(time.monotonic() - started, 0.001)
            noun = "video" if len(videos) == 1 else f"{len(videos)} videos"
            log(
                f"Verified uploaded {noun} ({total / 1024**2:.1f} MiB) "
                f"in {elapsed:.0f}s ({total * 8 / elapsed / 1_000_000:.1f} Mbps effective)."
            )

    def fetch_drive_inputs(self, manifest: dict, log: Callable[[str], None]) -> None:
        """Give the pod a Drive file list plus a short-lived token; the pod fetches the photos."""
        descriptor, name = tempfile.mkstemp(prefix="drive-manifest-", suffix=".json")
        local = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            os.chmod(local, 0o600)
            log("Sending the Google Drive file list and a one-hour read-only token to the pod...")
            self._copy_file(
                str(local), f"root@{self.host}:/workspace/job/drive_manifest.json",
                log, "Drive manifest upload",
            )
        finally:
            local.unlink(missing_ok=True)
        self._ssh("chmod 600 /workspace/job/drive_manifest.json")
        command = "cd /workspace/worker && python3 -u pipeline.py fetch-drive"
        with subprocess.Popen(
            ["ssh", *self.options, "-tt", "-n", "-p", str(self.port),
             f"root@{self.host}", command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            _stream_output(process, log)
            if process.wait():
                raise TransferError(
                    f"Fetching photos from Google Drive failed (exit code {process.returncode}). "
                    "See the log above for the reason."
                )

    def run_pipeline(
        self, steps: int, downscales: int, log: Callable[[str], None], *,
        source_type: str = "images", video_fps: float = 3.0,
        engine: str = "nerfstudio", spirula_version: str = "",
        spirula_floaters: str = "off", spirula_guidance: str = "off",
        spirula_distraction: str = "off",
        spirula_depth_weight: float = 0.05, preview: bool = True,
        spirula_primitive: str = "3dgs", spirula_quality: str = "medium",
        spirula_vignetting: bool = False, spirula_object: bool = False,
        spirula_turntable: bool = False, spirula_main_scene: bool = False,
        spirula_matcher: str = "sift",
        steps_override: bool = True, fisheye: bool = False,
        fisheye_focal_mm: float = 7.5, sensor_width_mm: float = 23.5,
    ) -> None:
        if spirula_quality not in {"low", "medium", "high", "ultra"}:
            raise TransferError(f"Unknown quality level: {spirula_quality}")
        if spirula_primitive not in {"3dgs", "3dgut"}:
            raise TransferError(f"Unknown splat type: {spirula_primitive}")
        if fisheye and engine != "spirula":
            raise TransferError("Fisheye photos are only supported with Spirula Studio.")
        if spirula_object and engine != "spirula":
            raise TransferError("Object scans are only supported with Spirula Studio.")
        if fisheye and not (1 <= float(fisheye_focal_mm) <= 50 and 5 <= float(sensor_width_mm) <= 60):
            raise TransferError("The fisheye focal length or sensor width is out of range.")
        if spirula_floaters not in {"off", "mild", "strong"}:
            raise TransferError(f"Unknown floater suppression level: {spirula_floaters}")
        if spirula_matcher not in {"sift", "aliked", "loma"}:
            raise TransferError(f"Unknown feature matching: {spirula_matcher}")
        if spirula_distraction not in {"off", "mild", "strong"}:
            raise TransferError(f"Unknown distraction robustness level: {spirula_distraction}")
        if spirula_guidance not in {"off", "surfaces", "depth"}:
            raise TransferError(f"Unknown geometry guidance level: {spirula_guidance}")
        if not 0 <= float(spirula_depth_weight) <= 1:
            raise TransferError("Depth guidance strength must be between 0 and 1.")
        if source_type not in {"images", "video", "mixed"}:
            raise TransferError(f"Unknown source type: {source_type}")
        if engine not in {"nerfstudio", "spirula"}:
            raise TransferError(f"Unknown processing engine: {engine}")
        if spirula_version and not re.fullmatch(r"[0-9][0-9A-Za-z._-]*", spirula_version):
            raise TransferError(f"Invalid Spirula version: {spirula_version!r}")
        if source_type in {"video", "mixed"} and not 0.2 <= float(video_fps) <= 30:
            raise TransferError("Frames per second must be between 0.2 and 30.")
        command = (
            f"PIPELINE_MAX_STEPS={steps} PIPELINE_NUM_DOWNSCALES={downscales} "
            f"PIPELINE_SOURCE_TYPE={source_type} "
            f"PIPELINE_VIDEO_FPS={float(video_fps):g} "
            f"PIPELINE_ENGINE={engine} "
            f"PIPELINE_PREVIEW={1 if preview else 0} PIPELINE_PREVIEW_PORT={PREVIEW_PORT} "
            + (f"PIPELINE_SPIRULA_VERSION={spirula_version} " if spirula_version else "")
            + (
                f"PIPELINE_SPIRULA_FLOATERS={spirula_floaters} "
                f"PIPELINE_SPIRULA_DISTRACTION={spirula_distraction} "
                f"PIPELINE_SPIRULA_GUIDANCE={spirula_guidance} "
                f"PIPELINE_SPIRULA_DEPTH_WEIGHT={float(spirula_depth_weight):g} "
                f"PIPELINE_SPIRULA_PRIMITIVE={spirula_primitive} "
                f"PIPELINE_SPIRULA_QUALITY={spirula_quality} "
                f"PIPELINE_SPIRULA_VIGNETTING={1 if spirula_vignetting else 0} "
                f"PIPELINE_SPIRULA_OBJECT={1 if spirula_object else 0} "
                f"PIPELINE_SPIRULA_TURNTABLE={1 if spirula_object and spirula_turntable else 0} "
                f"PIPELINE_SPIRULA_MAIN_SCENE={1 if spirula_main_scene else 0} "
                f"PIPELINE_SPIRULA_MATCHER={spirula_matcher} "
                f"PIPELINE_STEPS_OVERRIDE={1 if steps_override else 0} "
                if engine == "spirula" else ""
            )
            + (
                f"PIPELINE_FISHEYE=1 PIPELINE_FISHEYE_FOCAL_MM={float(fisheye_focal_mm):g} "
                f"PIPELINE_SENSOR_WIDTH_MM={float(sensor_width_mm):g} "
                if fisheye else ""
            )
            + "python3 -u /workspace/worker/job.py"
        )
        log(f"Starting remote image processing ({engine})...")
        with subprocess.Popen(
            ["ssh", *self.options, "-tt", "-n", "-p", str(self.port),
             f"root@{self.host}", command],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ) as process:
            _stream_output(process, log)
            if process.wait():
                raise TransferError(f"Remote pipeline failed with exit code {process.returncode}.")
        log("Remote image processing finished.")

    def download(
        self, destination: Path, log: Callable[[str], None] | None = None,
    ) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        pending = []
        for filename in ("gaussian_splat.ply", "pipeline_summary.json"):
            target = destination / filename
            temporary = destination / (filename + ".part")
            temporary.unlink(missing_ok=True)
            if log:
                log(f"Downloading {filename}...")
            remote = f"{REMOTE_OUTPUT_DIR}/{filename}"
            info = self._remote_file_info(remote) if filename.endswith(".ply") else None
            parallel_done = False
            if info and info[0] >= PARALLEL_DOWNLOAD_MIN_BYTES:
                try:
                    self._download_parallel(remote, temporary, *info, log)
                    parallel_done = True
                except TransferError as exc:
                    if log:
                        log(f"Parallel download failed ({exc}); retrying with a single connection.")
                    temporary.unlink(missing_ok=True)
            if not parallel_done:
                # Same progress meter as uploads: percent, size, speed and ETA.
                self._scp_with_progress(
                    [f"root@{self.host}:{remote}"], str(temporary), log, f"{filename} download",
                )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise TransferError(f"Download failed for {filename}: empty or missing file.")
            pending.append((temporary, target))
            if log:
                log(f"Downloaded {filename} ({temporary.stat().st_size / 1024**2:.1f} MiB).")
        with pending[0][0].open("rb") as ply_file:
            if ply_file.read(4) != b"ply\n":
                raise TransferError("Downloaded Gaussian splat is not a valid PLY file.")
        try:
            json.loads(pending[1][0].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise TransferError("Downloaded pipeline summary is invalid JSON.") from exc
        for temporary, target in pending:
            temporary.replace(target)

    def _remote_file_info(self, remote: str) -> tuple[int, str] | None:
        """Size and MD5 of a file on the pod, or None if they cannot be read."""
        quoted = shlex.quote(remote)
        try:
            result = self._ssh(f"stat -c %s {quoted} && md5sum {quoted}", timeout=300)
        except (OSError, subprocess.TimeoutExpired):
            return None
        lines = result.stdout.split() if result.returncode == 0 else []
        if len(lines) < 2 or not lines[0].isdigit() or not re.fullmatch(r"[0-9a-f]{32}", lines[1]):
            return None
        return int(lines[0]), lines[1]

    def _download_parallel(
        self, remote: str, local: Path, size: int, md5: str,
        log: Callable[[str], None] | None,
    ) -> None:
        """Fetch one pod file over several SSH connections, then verify its MD5."""
        streams = max(1, min(DOWNLOAD_STREAMS, size // (4 * 1024 ** 2)))
        part_size = -(-size // streams)
        with local.open("wb") as handle:
            handle.truncate(size)
        lock = threading.Lock()
        abort = threading.Event()
        received = 0

        def add(count: int) -> None:
            nonlocal received
            with lock:
                received += count

        def fetch(first: int, length: int) -> None:
            done = 0
            failures = 0
            while done < length:
                if abort.is_set():
                    raise TransferError("cancelled")
                command = (
                    f"dd if={shlex.quote(remote)} bs=4M iflag=skip_bytes,count_bytes "
                    f"skip={first + done} count={length - done} status=none"
                )
                process = subprocess.Popen(
                    ["ssh", *self.options, "-p", str(self.port), f"root@{self.host}", command],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                before = done
                try:
                    with local.open("r+b") as out:
                        out.seek(first + done)
                        while done < length:
                            if abort.is_set():
                                break
                            chunk = process.stdout.read(min(1024 ** 2, length - done))
                            if not chunk:
                                break
                            out.write(chunk)
                            done += len(chunk)
                            add(len(chunk))
                finally:
                    if process.poll() is None and (abort.is_set() or done >= length):
                        process.terminate()
                    stderr = process.communicate()[1].decode("utf-8", "replace").strip()
                if done >= length:
                    return
                failures = 0 if done > before else failures + 1
                if failures >= DOWNLOAD_PART_ATTEMPTS:
                    raise TransferError(
                        f"part at byte {first} stopped at {done} of {length} bytes"
                        + (f": {stderr.splitlines()[-1][:160]}" if stderr else "")
                    )
                time.sleep(2 * failures)

        started = last_log = time.monotonic()
        if log:
            log(f"Downloading {size / 1024**2:.1f} MiB over {streams} parallel connections...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
            pending = {
                pool.submit(fetch, first, min(part_size, size - first))
                for first in range(0, size, part_size)
            }
            try:
                while pending:
                    finished, pending = concurrent.futures.wait(
                        pending, timeout=1, return_when=concurrent.futures.FIRST_EXCEPTION,
                    )
                    for future in finished:
                        future.result()
                    now = time.monotonic()
                    if log and (now - last_log >= 3 or not pending):
                        with lock:
                            got = received
                        rate = got / max(now - started, 0.001)
                        eta = (size - got) / rate if rate > 0 else 0
                        log(
                            f"[download] {local.name.removesuffix('.part')}  {got * 100 // size}%  "
                            f"{got / 1024**2:.0f} / {size / 1024**2:.0f} MiB  "
                            f"{rate / 1024**2:.2f} MiB/s  ETA {int(eta) // 60:02d}:{int(eta) % 60:02d}"
                        )
                        last_log = now
            except BaseException:
                abort.set()
                raise
        if local.stat().st_size != size:
            raise TransferError(f"got {local.stat().st_size} bytes, expected {size}")
        digest = hashlib.md5()
        with local.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
                digest.update(chunk)
        if digest.hexdigest() != md5:
            raise TransferError("checksum mismatch after the parallel download")
        if log:
            elapsed = max(time.monotonic() - started, 0.001)
            log(
                f"Downloaded and verified {size / 1024**2:.1f} MiB in {elapsed:.0f}s "
                f"({size / elapsed / 1024**2:.2f} MiB/s)."
            )