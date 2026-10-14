from __future__ import annotations

import shutil
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from PySide6 import QtCore

from config import (
    DEPTH_GUIDANCE, ENGINE_LABELS, ENGINES, GUIDANCE_LABELS, MATCHER_LABELS, QUALITY_LEVELS,
    AppConfig, quality_steps,
)
from runpod_client import PodCreationUncertainError, RunPodCapacityError, RunPodClient, RunPodError
from drive_client import DriveClient
from gpu_speed import duration_text
from scp_client import ConnectionLostError, ScpClient
from pod_ledger import PodLedger
from storage import (
    DEFAULT_VIDEO_FPS, folder_photos, plan_drive_input, plan_drive_mixed, plan_drive_video,
    plan_input, plan_mixed, safe_dir_name,
)


# Log lines the pod prints right before training (and its viewer) starts.
PREVIEW_MARKERS = ("[pipeline] starting spirula training", "[pipeline] starting splatfacto")
PREVIEW_START_TIMEOUT = 900
# A job script that is alive but leaves the result missing this long after the
# trainer stopped is treated as stuck; the app then finishes the job itself.
STUCK_JOB_SECONDS = 120
REJOIN_ATTEMPTS = 8  # with REJOIN_WAIT_SECONDS: about four minutes to get the pod back
REJOIN_WAIT_SECONDS = 30


def viewer_responds(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return 200 <= response.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


class PodStoppedError(RuntimeError):
    pass


class PodMissingError(RuntimeError):
    pass


class JobCancelledError(RuntimeError):
    pass


def _time_left(state: dict) -> str:
    """", about 3 h 10 min left", from the speed between training start and the newest checkpoint."""
    step, total = state.get("step"), state.get("total_steps")
    started, saved = state.get("train_started"), state.get("step_time")
    if not (step and total and started and saved) or saved <= started or step >= total:
        return ""
    return f", about {duration_text((total - step) * (saved - started) / step)} left"


class JobController(QtCore.QObject):
    """One job: one pod, from creation to termination. Several can run at once."""

    log_line = QtCore.Signal(str)
    status_changed = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(bool)
    active_pod_changed = QtCore.Signal(object)
    preview_ready = QtCore.Signal(str)    # local URL of the live training viewer
    preview_closed = QtCore.Signal(str)   # why the preview ended, shown in its tab
    stop_started = QtCore.Signal()

    def __init__(
        self, config: AppConfig, *, ledger: PodLedger | None = None,
        upload_lock: threading.Lock | None = None, log_dir: Path | None = None,
    ):
        super().__init__()
        # The caller passes a copy: settings changed later never reach a running job.
        self.config = config
        self.running = False
        self._lock = threading.Lock()
        self.active_pod_id: str | None = None
        self._active_client: RunPodClient | None = None
        self._stop_requested = threading.Event()
        self._stop_done = threading.Event()
        self._stop_succeeded = False
        self._auto_terminating = False
        # A random ID per job keeps pod names unique, so the same capture can run
        # on both engines at once and a pod is never confused with another job's.
        self.job_id = uuid.uuid4().hex[:12]
        self.pod_name = f"gaussian-{self.job_id}"
        self.pod_requested = False
        self.ledger = ledger
        self.upload_lock = upload_lock
        # Shown in the job list.
        self.title = ""
        self.engine = ""
        self.gpu_label = ""
        self.cost_per_hour: float | None = None
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.outcome: str | None = None   # "finished", "failed" or "stopped"
        # Your own 1-5 score of the result and a short note, for comparing runs.
        self.rating: int | None = None
        self.rating_note = ""
        self.status_text = "Waiting to start"
        self.results_dir: Path | None = None
        # Shown in the job list and kept in the job history.
        self.context = ""
        self.rate_text = ""
        self.last_pod_id: str | None = None
        self.historic = False   # loaded from history; never runs
        self.preview_url: str | None = None
        self._preview_stop = threading.Event()
        self._preview_lock = threading.Lock()
        self._preview_tunnel = None
        self._preview_started = False
        # True while the pod runs (or has finished) the job on its own: the app
        # may close then, and reconnects to the pod when it opens again.
        self._on_pod = False
        self._stop_log = threading.Event()
        self._log_lock = threading.Lock()
        self.log_path: Path | None = None
        if log_dir is not None:
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                self.log_path = log_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{self.job_id}.log"
            except OSError:
                self.log_path = None
        # Direct connections run in the emitting (job) thread, so the log file
        # is complete even if the window is busy or the app crashes.
        self.log_line.connect(self._write_log, QtCore.Qt.DirectConnection)
        self.error.connect(lambda message: self._write_log(f"[ERROR] {message}"),
                           QtCore.Qt.DirectConnection)
        self.status_changed.connect(self._remember_status, QtCore.Qt.DirectConnection)

    def _write_log(self, message: str) -> None:
        if self.log_path is None:
            return
        with self._log_lock:
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(f"[{datetime.now():%H:%M:%S}] {message}\n")
            except OSError:
                pass

    def _remember_status(self, message: str) -> None:
        self.status_text = message

    def _ledger(self, **fields) -> None:
        if self.ledger is not None:
            self.ledger.update(self.job_id, pod_name=self.pod_name, title=self.title, **fields)

    @property
    def busy(self) -> bool:
        """True while this job may own a pod or is changing one."""
        return self.running or self.stop_in_progress or self.active_pod_id is not None

    @property
    def detachable(self) -> bool:
        """True when closing the app leaves this job safe on its pod."""
        return self._on_pod and not self.stop_in_progress

    def detach(self) -> None:
        """The app is closing: drop the preview and log connections, leave the pod alone.

        The SSH session running the job script is left open on purpose: jobs
        started by older workers end when it closes.
        """
        self._stop_log.set()
        self._stop_preview("The app closed.")

    @property
    def stop_succeeded(self) -> bool:
        with self._lock:
            return self._stop_succeeded

    @property
    def stop_in_progress(self) -> bool:
        with self._lock:
            return self._stop_requested.is_set() and not self._stop_done.is_set()

    def _check_stop_requested(self) -> None:
        if self._stop_requested.is_set():
            raise JobCancelledError("Stop requested from the app.")

    @staticmethod
    def _pod_absent(client: RunPodClient, pod_id: str) -> bool:
        # The REST status endpoint can briefly return 404 while GraphQL still
        # lists a live pod. Do not treat one 404 as confirmed termination.
        try:
            if client.pod_snapshot(pod_id) is not None:
                return False
            return not any(pod.get("id") == pod_id for pod in client.list_pods())
        except Exception:
            return False

    def _clear_active_pod(self, pod_id: str) -> None:
        with self._lock:
            if self.active_pod_id != pod_id:
                return
            self.active_pod_id = None
            self._active_client = None
        self.active_pod_changed.emit(None)

    def stop_pod(self) -> None:
        with self._lock:
            pod_id = self.active_pod_id
            client = self._active_client
            if self._stop_requested.is_set() or self._auto_terminating:
                return
            cancel = self.running and not pod_id
            if not cancel and (not pod_id or client is None):
                return
            self._stop_requested.set()
            self._stop_succeeded = False
            if cancel:
                # No pod yet. If a creation request is in flight, the job thread
                # sees the cancellation and terminates whatever pod it returns.
                self._stop_done.set()
            else:
                self._stop_done.clear()
        if cancel:
            self.stop_started.emit()
            self.status_changed.emit("Cancelling...")
            self.log_line.emit("Cancel requested before the pod was ready.")
            return
        self.stop_started.emit()
        self.status_changed.emit("Terminating RunPod pod...")
        self.log_line.emit(f"Stop requested. Terminating pod {pod_id}...")
        threading.Thread(target=self._stop_pod, args=(client, pod_id), daemon=True).start()

    def _stop_pod(self, client: RunPodClient, pod_id: str) -> None:
        # The job thread may be polling RunPod at the same time. Use a separate
        # HTTP session for this request.
        try:
            stop_client = RunPodClient(client.api_key, client.graphql_url)
            try:
                stop_client.terminate_pod(pod_id)
            except Exception as exc:
                if not self._pod_absent(stop_client, pod_id):
                    self.error.emit(
                        f"Could not confirm termination of pod {pod_id}: {exc}. "
                        "Check RunPod; it may still be billing."
                    )
                    return
            with self._lock:
                self._stop_succeeded = True
            self._ledger(state="terminated", pod_id=pod_id)
            self.log_line.emit(f"Pod {pod_id} terminated from the app.")
            self.status_changed.emit("Pod terminated")
            self._clear_active_pod(pod_id)
        except Exception as exc:
            self.error.emit(
                f"Could not terminate pod {pod_id}: {exc}. "
                "Check RunPod; it may still be billing."
            )
        finally:
            with self._lock:
                # Keep cancellation set after a confirmed stop until the job
                # thread exits. On failure, allow the user to retry.
                if not self._stop_succeeded or not self.running:
                    self._stop_requested.clear()
                active_pod_id = self.active_pod_id if not self._stop_succeeded else None
            self._stop_done.set()
            if active_pod_id:
                self.active_pod_changed.emit(active_pod_id)

    def _wait_for_public_ssh(self, client: RunPodClient, pod_id: str) -> tuple[str, int]:
        deadline = time.monotonic() + 600
        next_progress = 0.0
        next_logs = 0.0
        last_status = None
        missing_checks = 0
        seen_log_ids: set[str] = set()
        self.status_changed.emit("Waiting for pod startup...")
        self.log_line.emit(
            f"Pod {pod_id} is starting. Waiting for RunPod to expose its public SSH port."
        )
        while time.monotonic() < deadline:
            self._check_stop_requested()
            now = time.monotonic()
            try:
                snapshot = client.pod_snapshot(pod_id)
            except RunPodError as exc:
                snapshot = None
                if now >= next_progress:
                    self.log_line.emit(f"Pod status check delayed: {exc}")
                    next_progress = now + 15
                # A failed API read is not evidence that the pod was removed.
                missing_checks = 0
            else:
                if snapshot is None:
                    try:
                        listed = next(
                            (pod for pod in client.list_pods() if pod.get("id") == pod_id),
                            None,
                        )
                    except RunPodError:
                        listed = "unknown"
                    if isinstance(listed, dict):
                        listed_status = str(listed.get("desiredStatus") or "").upper()
                        if listed_status in {"EXITED", "STOPPED", "TERMINATED"}:
                            raise PodStoppedError(
                                f"Pod {pod_id} is {listed_status.lower()} in RunPod. "
                                "Processing stopped after it was changed in the console."
                            )
                        missing_checks = 0
                        detail = "still listed, but status details are pending"
                    elif listed == "unknown":
                        missing_checks = 0
                        detail = "status checks are temporarily unavailable"
                    else:
                        missing_checks += 1
                        detail = "not visible in either status API"
                    if now >= next_progress:
                        self.log_line.emit(f"Pod {pod_id}: {detail}; checking again.")
                        next_progress = now + 15
                    if missing_checks >= 6:
                        raise PodMissingError(
                            f"Pod {pod_id} is no longer visible in RunPod after repeated "
                            "checks. It may have been stopped or deleted in the console. "
                            "Check RunPod before starting another job."
                        )
                else:
                    missing_checks = 0
                    status = str(snapshot.get("status") or "UNKNOWN").upper()
                    if status in {"EXITED", "STOPPED", "TERMINATED"}:
                        raise PodStoppedError(
                            f"Pod {pod_id} is {status.lower()} in RunPod. "
                            "Processing stopped; restart the job only after checking the console."
                        )
                    if status == "ERROR":
                        raise RuntimeError(
                            f"Pod {pod_id} entered an error state during startup. "
                            "Check its system logs in RunPod."
                        )
                    endpoint = client.direct_ssh_endpoint(snapshot)
                    if endpoint and not client.is_public_address(endpoint[0]):
                        raise RuntimeError(
                            f"RunPod gave pod {pod_id} a private address ({endpoint[0]}), not a "
                            "public IP, so direct SSH (needed for SCP) cannot work on this host. "
                            "Its Connect tab will only offer basic SSH. Try again on Secure Cloud "
                            "(Settings > Cloud type)."
                        )
                    if endpoint:
                        self._emit_system_lines(
                            pod_id, client.recent_system_logs(pod_id, seen_log_ids)
                        )
                        self.log_line.emit(
                            f"Public SSH port assigned to pod {pod_id}: "
                            f"{endpoint[0]}:{endpoint[1]}."
                        )
                        return endpoint
                    if status != last_status or now >= next_progress:
                        self.log_line.emit(
                            f"Pod {pod_id}: {status.lower()}; container or public SSH "
                            "port is still starting."
                        )
                        self.status_changed.emit(f"Pod {status.lower()} — waiting for SSH")
                        next_progress = now + 15
                    last_status = status
            if now >= next_logs:
                self._emit_system_lines(
                    pod_id, client.recent_system_logs(pod_id, seen_log_ids)
                )
                next_logs = now + 5
            time.sleep(5)
        raise RuntimeError(
            f"Pod {pod_id} did not expose a public SSH port within 10 minutes. "
            "Check RunPod system logs."
        )

    CONTAINER_START_LIMIT = 3
    _container_starts = 0

    def _emit_system_lines(self, pod_id: str, lines) -> None:
        """Show RunPod system events and stop if the container is crash-looping."""
        for line in lines:
            self.log_line.emit(f"[RunPod system] {line}")
            lowered = line.lower()
            if "start container" in lowered and "begin" in lowered:
                self._container_starts += 1
        if self._container_starts >= self.CONTAINER_START_LIMIT:
            raise RuntimeError(
                f"Pod {pod_id}'s container has been started {self._container_starts} times: "
                "it keeps exiting and RunPod keeps restarting it. Stopping the pod so you "
                "are not billed for a crash loop. Check the pod's container logs in RunPod."
            )

    def _bootstrap_status_check(self, client: RunPodClient, pod_id: str):
        missing_checks = 0
        seen_log_ids: set[str] = set()
        next_logs = 0.0

        def check() -> None:
            nonlocal missing_checks, next_logs
            self._check_stop_requested()
            # RunPod's own events (container started / restarted / exited) show whether the
            # container is crash-looping, which looks like a flapping SSH port from outside.
            if time.monotonic() >= next_logs:
                next_logs = time.monotonic() + 20
                try:
                    lines = client.recent_system_logs(pod_id, seen_log_ids)
                except Exception:
                    lines = []
                self._emit_system_lines(pod_id, lines)
            try:
                snapshot = client.pod_snapshot(pod_id)
            except RunPodError:
                return
            if snapshot is None:
                try:
                    listed = next(
                        (pod for pod in client.list_pods() if pod.get("id") == pod_id),
                        None,
                    )
                except RunPodError:
                    return
                if listed is None:
                    missing_checks += 1
                    if missing_checks >= 6:
                        raise PodMissingError(
                            f"Pod {pod_id} disappeared while SSH was starting. "
                            "Check RunPod before starting another job."
                        )
                    return
                status = str(listed.get("desiredStatus") or "").upper()
            else:
                status = str(snapshot.get("status") or "").upper()
            missing_checks = 0
            if status in {"EXITED", "STOPPED", "TERMINATED"}:
                raise PodStoppedError(
                    f"Pod {pod_id} was {status.lower()} while SSH was starting. "
                    "Processing stopped after it was changed in RunPod."
                )

        return check

    def run_job(
        self, source: Path, gpu_selection: tuple[str, str], *,
        source_type: str = "images", video_fps: float = DEFAULT_VIDEO_FPS,
        engine: str | None = None,
        drive_folder: tuple[str, str] | None = None, results_root: Path | None = None,
        drive_video_id: str = "", more_videos: tuple[Path, ...] = (),
        start: bool = True,
    ) -> None:
        """Mark the job running and start its thread (or leave that to start_thread())."""
        engine = engine or self.config.engine
        if engine not in ENGINES:
            self.error.emit(f"Unknown processing engine: {engine}")
            return
        with self._lock:
            if self.running:
                self.error.emit("A pipeline is already running.")
                return
            if self.active_pod_id:
                self.error.emit(
                    f"Pod {self.active_pod_id} is still associated with the last job. "
                    "Terminate it before starting another."
                )
                return
            self._stop_requested.clear()
            self._stop_succeeded = False
            self._auto_terminating = False
            self.running = True
            self.engine = engine
            self.started_at = time.time()
            self.finished_at = None
            self.outcome = None
        options = {"source_type": source_type, "video_fps": video_fps}
        if engine != "nerfstudio":
            options["engine"] = engine
        if self.config.spirula_version and engine == "spirula":
            options["spirula_version"] = self.config.spirula_version
        if drive_folder:
            options["drive_folder"] = drive_folder
        if results_root:
            options["results_root"] = results_root
        if drive_video_id:
            options["drive_video_id"] = drive_video_id
        if more_videos:
            options["more_videos"] = tuple(more_videos)
        self._thread = threading.Thread(
            target=self._run_job, args=(source, gpu_selection), kwargs=options, daemon=True,
        )
        if start:
            self.start_thread()

    def start_thread(self) -> None:
        thread, self._thread = getattr(self, "_thread", None), None
        if thread is not None:
            thread.start()

    def _plan(
        self, source: Path, source_type: str, video_fps: float,
        drive_folder: tuple[str, str] | None, drive_video_id: str, results_root: Path | None,
        more_videos: tuple[Path, ...] = (),
    ):
        """Re-check the input from scratch; returns (plan, drive client, drive listing).

        "mixed" is every photo in the source folder plus more_videos; "drive_mixed"
        is every photo in the Drive folder plus the videos in drive_video_id.
        """
        if source_type == "mixed":
            source = source.expanduser().resolve(strict=True)
            plan = plan_mixed(
                source, tuple(folder_photos(source)),
                tuple(path.expanduser().resolve(strict=True) for path in more_videos),
                self.config.volume_gb, self.config.pipeline_num_downscales, video_fps,
            )
            return plan, None, None
        if source_type not in {"drive", "drive_video", "drive_mixed"}:
            source = source.expanduser().resolve(strict=True)
            plan = plan_input(
                source, source_type, self.config.volume_gb,
                self.config.pipeline_num_downscales, video_fps,
                tuple(path.expanduser().resolve(strict=True) for path in more_videos),
            )
            return plan, None, None
        if not drive_folder:
            raise RuntimeError("Choose a Google Drive folder first.")
        credentials = self.config.drive_credentials_path()
        if credentials is None:
            raise RuntimeError(
                "Google Drive is not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE in .env."
            )
        self.status_changed.emit("Reading the Google Drive folder...")
        drive = DriveClient(credentials)
        listing = drive.list_images(*drive_folder)
        root = results_root or self.config.results_root()
        if source_type == "drive":
            plan = plan_drive_input(
                listing.folder_id, listing.folder_name, listing.files, root,
                self.config.volume_gb,
            )
        else:
            by_id = {item.id: item for item in listing.videos}
            wanted = [video_id for video_id in drive_video_id.split(",") if video_id]
            missing = [video_id for video_id in wanted if video_id not in by_id]
            if missing or not wanted:
                raise RuntimeError(
                    "A chosen video is no longer in the Drive folder "
                    f"'{listing.folder_name}'."
                )
            self.status_changed.emit("Reading the video details from Google Drive...")
            videos = [drive.probe_video(by_id[video_id]) for video_id in wanted]
            if source_type == "drive_mixed":
                plan = plan_drive_mixed(
                    listing.folder_id, listing.folder_name, listing.files, tuple(videos),
                    root, self.config.volume_gb,
                    self.config.pipeline_num_downscales, video_fps,
                )
            else:
                plan = plan_drive_video(
                    listing.folder_id, listing.folder_name, videos,
                    root, self.config.volume_gb,
                    self.config.pipeline_num_downscales, video_fps,
                )
        return plan, drive, listing

    # ------------------------------------------------------------------
    # Live training preview
    # ------------------------------------------------------------------

    def _pipeline_log(self, transfer):
        """Forward pod output; start the preview when training begins."""
        def log(line: str) -> None:
            self.log_line.emit(line)
            if (self.config.live_preview and not self._preview_started
                    and line.strip().lower().startswith(PREVIEW_MARKERS)):
                self._preview_started = True
                threading.Thread(target=self._run_preview, args=(transfer,), daemon=True).start()
        return log

    def _open_preview_tunnel(self, transfer):
        tunnel = transfer.open_tunnel()
        with self._preview_lock:
            if self._preview_stop.is_set():
                tunnel.close()
                return None
            self._preview_tunnel = tunnel
        return tunnel

    def _run_preview(self, transfer) -> None:
        """Tunnel to the pod's training viewer and report its URL once it answers."""
        try:
            tunnel = None
            deadline = time.monotonic() + PREVIEW_START_TIMEOUT
            while not self._preview_stop.is_set():
                if tunnel is None or not tunnel.alive():
                    if tunnel is not None:
                        tunnel.close()
                    tunnel = self._open_preview_tunnel(transfer)
                    if tunnel is None:
                        return
                    self._preview_stop.wait(1.5)
                if viewer_responds(tunnel.url):
                    break
                if time.monotonic() > deadline:
                    self.log_line.emit(
                        "The live preview did not start; training continues normally."
                    )
                    self.preview_closed.emit("The training viewer did not answer, so there is no live preview for this job.")
                    return
                self._preview_stop.wait(3)
            else:
                return
            self.preview_url = tunnel.url
            self.log_line.emit(f"Live preview connected through an SSH tunnel ({tunnel.url}).")
            self.preview_ready.emit(tunnel.url)
            # Keep the tunnel up for as long as training runs.
            while not self._preview_stop.wait(5):
                if tunnel.alive():
                    continue
                tunnel = self._open_preview_tunnel(transfer)
                if tunnel is None:
                    return
                self._preview_stop.wait(1.5)
                if viewer_responds(tunnel.url):
                    self.preview_url = tunnel.url
                    self.log_line.emit("Live preview reconnected.")
                    self.preview_ready.emit(tunnel.url)
        except Exception as exc:  # the preview is optional; never fail the job
            self.log_line.emit(f"Live preview unavailable: {exc}")

    def _stop_preview(self, reason: str) -> None:
        """Close the tunnel (if any). Safe to call more than once."""
        with self._preview_lock:
            already = self._preview_stop.is_set()
            self._preview_stop.set()
            tunnel, self._preview_tunnel = self._preview_tunnel, None
        if tunnel is not None:
            tunnel.close()
        if not already and self._preview_started:
            self.preview_url = None
            self.preview_closed.emit(reason)

    def _hold_upload_slot(self) -> bool:
        """Local uploads share this computer's connection: run them one at a time."""
        if self.upload_lock is None:
            return False
        if not self.upload_lock.acquire(blocking=False):
            self.status_changed.emit("Waiting for another job's upload to finish...")
            self.log_line.emit(
                "Another job is uploading from this computer; this upload starts when it ends."
            )
            while not self.upload_lock.acquire(timeout=1):
                self._check_stop_requested()
        return True

    def _run_job(
        self, source: Path, gpu_selection: tuple[str, str], *,
        source_type: str = "images", video_fps: float = DEFAULT_VIDEO_FPS,
        engine: str = "nerfstudio", spirula_version: str = "",
        drive_folder: tuple[str, str] | None = None, results_root: Path | None = None,
        drive_video_id: str = "", more_videos: tuple[Path, ...] = (),
    ) -> None:
        pod_id = None
        drive = None
        drive_listing = None
        self._container_starts = 0
        client = None
        preserve_pod = False
        skip_termination = False
        success = False
        try:
            plan, drive, drive_listing = self._plan(
                source, source_type, video_fps, drive_folder, drive_video_id,
                results_root, more_videos,
            )
            source = plan.source
            self.log_line.emit(f"Job {self.job_id}; its pod will be named {self.pod_name}.")
            self.log_line.emit(f"Processing engine: {ENGINE_LABELS[engine]}.")
            if self.config.spirula_object and engine != "spirula":
                raise RuntimeError(
                    "Object scans are only supported with Spirula Studio. Switch the engine "
                    "or turn off the object scan option. No pod was created."
                )
            if self.config.fisheye_lens and engine != "spirula":
                raise RuntimeError(
                    "Fisheye photos are only supported with Spirula Studio. Switch the engine "
                    "or turn off the fisheye lens option. No pod was created."
                )
            if plan.is_mixed and engine != "spirula":
                raise RuntimeError(
                    "Photos and videos in one job are only supported with Spirula Studio. "
                    "Switch the engine or untick the photos or the videos. No pod was created."
                )
            if plan.video_count > 1 and engine != "spirula":
                raise RuntimeError(
                    "Several videos in one job are only supported with Spirula Studio. "
                    "Switch the engine or pick one video. No pod was created."
                )
            if self.config.fisheye_lens:
                self.log_line.emit(
                    f"Fisheye lens: {self.config.fisheye_focal_mm:g} mm on a "
                    f"{self.config.sensor_width_mm:g} mm-wide sensor."
                )
            if engine == "spirula":
                self.log_line.emit(
                    f"Spirula splat type: {self.config.spirula_primitive}; quality "
                    f"{QUALITY_LEVELS[self.config.spirula_quality][0]} (up to "
                    f"{QUALITY_LEVELS[self.config.spirula_quality][2]:,} splats, "
                    + (f"{self.config.pipeline_max_steps:,} steps set by hand)."
                       if self.config.steps_override else
                       f"about {quality_steps(self.config.spirula_quality, plan.image_count):,} steps)."
                       )
                    + (" Lens darkening at the edges is corrected."
                       if self.config.spirula_vignetting else "")
                    + (f" Feature matching: {MATCHER_LABELS[self.config.spirula_matcher]}."
                       if self.config.spirula_matcher != "sift" else "")
                    + (" Training keeps to the main scene; far background is left out."
                       if self.config.spirula_main_scene else "")
                    + (" Object scan: the background is cut away with AI masks"
                       + (", before the cameras are solved (turntable)." if self.config.spirula_turntable
                          else ".")
                       if self.config.spirula_object else "")
                )
                self.log_line.emit(
                    f"Spirula options: floater suppression {self.config.spirula_floaters}; "
                    f"AI geometry guidance {GUIDANCE_LABELS[self.config.spirula_guidance]}"
                    + (f" (depth strength {self.config.spirula_depth_weight:g})"
                       if self.config.spirula_guidance in DEPTH_GUIDANCE else "")
                    + (f"; ignore moving things {self.config.spirula_distraction}"
                       if self.config.spirula_distraction != "off" else "")
                    + "."
                )
            image_name = self.config.image_for(engine)
            if not image_name:
                raise RuntimeError(f"No RunPod image is configured for {ENGINE_LABELS[engine]}.")
            input_bytes = plan.input_bytes
            working_reserve = plan.working_reserve
            volume_gb = plan.volume_gb
            if plan.is_mixed:
                videos = (", ".join(path.name for path in plan.videos) if plan.kind == "mixed"
                          else ", ".join(item.name for item in plan.drive_videos))
                where = (f"Google Drive folder '{plan.drive_folder_name}'"
                         if plan.is_drive else f"folder {source.name}")
                self.log_line.emit(
                    f"{where}: {plan.photo_count} photos and {plan.video_count} "
                    f"video{'s' if plan.video_count != 1 else ''} ({videos}; "
                    f"{input_bytes / 1024**2:.1f} MiB). Extracting about {plan.frame_target} "
                    f"frames on the pod ({plan.extract_fps:g} per second of video); photos and "
                    "frames are solved together, each with its own camera."
                    + (" The pod fetches everything straight from Google." if plan.is_drive else "")
                )
            elif plan.kind == "video" and plan.video_count > 1:
                self.log_line.emit(
                    f"Preparing {plan.video_count} videos of one scene "
                    f"({', '.join(path.name for path in plan.videos)}; "
                    f"{input_bytes / 1024**2:.1f} MiB); extracting about {plan.frame_target} "
                    f"frames on the pod ({plan.extract_fps:g} per second of video)."
                )
            elif plan.kind == "video":
                self.log_line.emit(
                    f"Preparing video {source.name} ({input_bytes / 1024**2:.1f} MiB); "
                    f"extracting about {plan.frame_target} frames on the pod "
                    f"({plan.extract_fps:g} per second of video)."
                )
            elif plan.kind == "drive_video":
                self.log_line.emit(
                    f"Google Drive video '{plan.display_name}' ({input_bytes / 1024**2:.1f} MiB); "
                    f"the pod fetches it straight from Google and extracts about "
                    f"{plan.frame_target} frames ({plan.extract_fps:g} per second of video)."
                )
            elif plan.kind == "drive":
                skipped = (
                    f" ({drive_listing.skipped} other files ignored)" if drive_listing.skipped else ""
                )
                self.log_line.emit(
                    f"Google Drive folder '{plan.drive_folder_name}': {len(plan.drive_files)} "
                    f"photos ({input_bytes / 1024**2:.1f} MiB){skipped}. The pod will fetch "
                    "them straight from Google; nothing is uploaded from this computer."
                )
            else:
                self.log_line.emit(
                    f"Preparing {len(plan.images)} photos ({input_bytes / 1024**2:.1f} MiB)."
                )
            if not self.config.has_runpod():
                raise RuntimeError("Configure a RunPod API key first.")
            key = Path(self.config.ssh_private_key).expanduser()
            public_key_file = Path(str(key) + ".pub")
            if not key.is_file() or not public_key_file.is_file():
                raise RuntimeError("SSH private key and matching .pub file are required.")
            public_key = public_key_file.read_text(encoding="utf-8").strip()
            if not public_key.startswith(("ssh-ed25519 ", "ssh-rsa ", "ecdsa-sha2-")):
                raise RuntimeError("SSH public key is invalid.")
            # The job ID keeps folders of jobs started in the same second apart.
            results_dir = plan.results_parent / (
                f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{self.job_id[:6]}"
            )
            client = RunPodClient(self.config.runpod_api_key, self.config.runpod_graphql_url)
            self.status_changed.emit("Checking GPU availability...")
            self.log_line.emit("Checking the selected GPU's current capacity...")
            candidates = client.select_gpu_types(
                self.config.max_gpu_price, self.config.min_gpu_vram_gb,
                cloud_type=self.config.cloud_type,
                container_disk_gb=self.config.container_disk_gb,
                volume_gb=volume_gb,
                min_vcpu=self.config.min_vcpu,
                max_vram_gb=self.config.max_gpu_vram_gb,
                min_memory_gb=self.config.min_memory_gb,
            )
            candidate = next((item for item in candidates
                              if (item["id"], item["_cloud_type"]) == gpu_selection), None)
            if candidate is None or not candidate["_available"]:
                raise RunPodCapacityError(
                    "Selected GPU no longer has matching public-SSH capacity. "
                    "Refresh the GPU list and choose another."
                )
            self.log_line.emit(
                f"Selected GPU: {candidate['_display_name']} ({candidate['_cloud_type']}, "
                f"{candidate['_stock_status']} stock, ${candidate['_hourly_price']:.3f}/hr quoted)"
            )
            self.gpu_label = f"{candidate['_display_name']} ({candidate['_cloud_type']})"
            self.cost_per_hour = candidate["_hourly_price"]
            self._check_stop_requested()
            self.status_changed.emit("Creating one RunPod pod...")
            self.log_line.emit("Sending one RunPod pod creation request...")
            # Recorded before the request: if anything goes wrong from here on,
            # the pod name is on disk and the untracked-pod check can find it.
            self.pod_requested = True
            self._ledger(
                state="creating", engine=engine, gpu=self.gpu_label,
                results_dir=str(results_dir), steps=self.config.pipeline_max_steps,
            )
            try:
                pod = client.create_pod(
                    job_id=self.job_id, image_name=image_name, engine=engine,
                    ssh_public_key=public_key, gpu_type_id=candidate["id"],
                    cloud_type=candidate["_cloud_type"],
                    container_disk_gb=self.config.container_disk_gb,
                    volume_gb=volume_gb, min_vcpu=self.config.min_vcpu,
                    min_memory_gb=self.config.min_memory_gb,
                )
            except PodCreationUncertainError:
                self._ledger(state="uncertain")
                raise
            except Exception:
                self._ledger(state="not_created")
                raise
            pod_id = pod.pod_id
            if pod.cost_per_hour is not None:
                self.cost_per_hour = pod.cost_per_hour
            with self._lock:
                self.active_pod_id = pod_id
                self._active_client = client
            self._ledger(state="running", pod_id=pod_id)
            self.active_pod_changed.emit(pod_id)
            if pod.recovered_from_error:
                self.log_line.emit(
                    f"RunPod returned an error, but pod {pod_id} was found with the "
                    "requested settings. Continuing this job without creating another pod."
                )
            else:
                self.log_line.emit(f"Pod created: {pod_id}")
            if pod.cost_per_hour is not None and pod.cost_per_hour > self.config.max_gpu_price:
                raise RuntimeError(
                    f"Pod price ${pod.cost_per_hour:.3f}/hr exceeds the configured "
                    f"${self.config.max_gpu_price:.3f}/hr limit."
                )
            self._check_stop_requested()
            endpoint = self._wait_for_public_ssh(client, pod_id)
            self._check_stop_requested()
            transfer = ScpClient(key, *endpoint)
            def current_endpoint():
                snapshot = client.pod_snapshot(pod_id)
                return client.direct_ssh_endpoint(snapshot) if snapshot else None

            transfer.wait_ready(
                log=self.log_line.emit,
                status_check=self._bootstrap_status_check(client, pod_id),
                endpoint_provider=current_endpoint,
            )
            self._check_stop_requested()
            free_bytes = transfer.free_bytes()
            required_bytes = input_bytes + working_reserve
            if free_bytes < required_bytes:
                raise RuntimeError(
                    f"Pod has only {free_bytes / 1024**3:.1f} GiB free in /workspace; "
                    f"this job requires at least {required_bytes / 1024**3:.1f} GiB. "
                    "No photos were uploaded."
                )
            self.log_line.emit(
                f"Workspace has {free_bytes / 1024**3:.1f} GiB free; "
                "uploading worker scripts."
            )
            transfer.upload_scripts(Path(__file__).resolve().parent.parent / "worker", self.log_line.emit)
            self._check_stop_requested()
            # Nerfstudio's image ships ffmpeg; Spirula's pipeline installs it itself.
            if plan.is_video and engine == "nerfstudio":
                transfer.verify_video_tools()
            if plan.kind == "drive_mixed":
                self.status_changed.emit("Fetching photos and videos from Google Drive...")
                transfer.fetch_drive_inputs(
                    drive.build_mixed_manifest(drive_listing.folder_name, plan.drive_files,
                                               plan.drive_videos),
                    self.log_line.emit,
                )
            elif plan.kind == "drive_video":
                self.status_changed.emit(
                    "Fetching the video from Google Drive..." if plan.video_count == 1
                    else f"Fetching {plan.video_count} videos from Google Drive...")
                transfer.fetch_drive_inputs(
                    drive.build_video_manifest(drive_listing.folder_name, plan.drive_files),
                    self.log_line.emit,
                )
            elif plan.kind == "drive":
                self.status_changed.emit("Fetching photos from Google Drive...")
                transfer.fetch_drive_inputs(drive.build_manifest(drive_listing), self.log_line.emit)
            else:
                holding = self._hold_upload_slot()
                try:
                    self._check_stop_requested()
                    if plan.kind == "mixed":
                        self.status_changed.emit(
                            f"Uploading {plan.photo_count} photos and {plan.video_count} videos...")
                        self.log_line.emit(
                            f"Uploading {plan.photo_count} photos, then {plan.video_count} "
                            f"video{'s' if plan.video_count != 1 else ''} "
                            f"({input_bytes / 1024**2:.1f} MiB)..."
                        )
                        transfer.upload_mixed(list(plan.images), list(plan.videos),
                                              working_reserve, self.log_line.emit)
                    elif plan.kind == "video" and plan.video_count > 1:
                        self.status_changed.emit(f"Uploading {plan.video_count} videos...")
                        self.log_line.emit(
                            f"Uploading {plan.video_count} videos "
                            f"({input_bytes / 1024**2:.1f} MiB), one after another..."
                        )
                        transfer.upload_videos(list(plan.videos), working_reserve,
                                               self.log_line.emit)
                    elif plan.kind == "video":
                        self.status_changed.emit("Uploading video...")
                        self.log_line.emit(
                            f"Uploading video ({input_bytes / 1024**2:.1f} MiB) in one SCP connection..."
                        )
                        transfer.upload_video(source, working_reserve, self.log_line.emit)
                    else:
                        self.status_changed.emit(f"Uploading {len(plan.images)} photos...")
                        self.log_line.emit(
                            f"Uploading {len(plan.images)} photos ({input_bytes / 1024**2:.1f} MiB)..."
                        )
                        transfer.upload(list(plan.images), working_reserve, self.log_line.emit)
                finally:
                    if holding:
                        self.upload_lock.release()
            self._check_stop_requested()
            free_bytes = transfer.free_bytes()
            if free_bytes < working_reserve:
                raise RuntimeError(
                    f"Only {free_bytes / 1024**3:.1f} GiB remains after upload; "
                    f"at least {working_reserve / 1024**3:.1f} GiB was planned. "
                    "Processing was not started."
                )
            self._check_stop_requested()
            self._on_pod = True
            self.status_changed.emit("Processing images...")
            try:
                try:
                    transfer.run_pipeline(
                        self.config.pipeline_max_steps, self.config.pipeline_num_downscales,
                        self._pipeline_log(transfer),
                        source_type=plan.pipeline_source_type,
                        video_fps=plan.extract_fps or DEFAULT_VIDEO_FPS,
                        engine=engine, spirula_version=spirula_version,
                        spirula_floaters=self.config.spirula_floaters,
                        spirula_distraction=self.config.spirula_distraction,
                        spirula_guidance=self.config.spirula_guidance,
                        spirula_depth_weight=self.config.spirula_depth_weight,
                        preview=self.config.live_preview,
                        spirula_primitive=self.config.spirula_primitive,
                        spirula_quality=self.config.spirula_quality,
                        spirula_vignetting=self.config.spirula_vignetting,
                        spirula_object=self.config.spirula_object,
                        spirula_turntable=self.config.spirula_turntable,
                        spirula_main_scene=self.config.spirula_main_scene,
                        spirula_matcher=self.config.spirula_matcher,
                        steps_override=self.config.steps_override,
                        fisheye=self.config.fisheye_lens,
                        fisheye_focal_mm=self.config.fisheye_focal_mm,
                        sensor_width_mm=self.config.sensor_width_mm,
                    )
                except ConnectionLostError as exc:
                    if self._stop_requested.is_set():
                        raise
                    self._rejoin_job(transfer, engine, exc)
            except BaseException:
                self._stop_preview("Processing stopped, so the live preview closed.")
                raise
            self._stop_preview("Training finished. The result is downloading to this computer.")
            self._check_stop_requested()
            self.status_changed.emit("Downloading results...")
            self.log_line.emit("Processing finished; downloading the result files...")
            try:
                transfer.download(results_dir, self.log_line.emit)
            except Exception:
                preserve_pod = not self._stop_requested.is_set()
                raise
            self._check_stop_requested()
            success = True
            self.results_dir = results_dir
            self.log_line.emit(f"Results saved to {results_dir}")
        except Exception as exc:
            skip_termination = self._report_failure(exc)
        finally:
            self._wrap_up(pod_id, client, preserve_pod, skip_termination, success)

    # ------------------------------------------------------------------
    # Reconnecting to a pod whose app session was lost
    # ------------------------------------------------------------------

    def reattach(self, entry: dict, results_root: Path, *, start: bool = True) -> None:
        """Take over a pod that is still running a job from an earlier app session."""
        with self._lock:
            if self.running:
                return
            self._stop_requested.clear()
            self._stop_succeeded = False
            self.running = True
            self.job_id = entry.get("job_id") or self.job_id
            self.pod_name = entry.get("pod_name") or self.pod_name
            self.pod_requested = True
            self.title = entry.get("title") or self.pod_name
            self.engine = entry.get("engine") or ""
            self.gpu_label = entry.get("gpu") or ""
            self.cost_per_hour = entry.get("cost")
            self.started_at = entry.get("created_at") or time.time()
        if self.log_path is not None:
            # Continue the job's original log file when it exists.
            earlier = sorted(self.log_path.parent.glob(f"*_{self.job_id}.log"))
            if earlier:
                self.log_path = earlier[-1]
        self._thread = threading.Thread(
            target=self._attach_job, args=(dict(entry), Path(results_root)), daemon=True,
        )
        if start:
            self.start_thread()

    def _attach_job(self, entry: dict, results_root: Path) -> None:
        pod_id = entry.get("pod_id")
        client = None
        preserve_pod = False
        skip_termination = False
        success = False
        self._container_starts = 0
        try:
            if not pod_id:
                raise RuntimeError(
                    f"The pod ID for {self.pod_name} was never recorded; check RunPod's console."
                )
            client = RunPodClient(self.config.runpod_api_key, self.config.runpod_graphql_url)
            with self._lock:
                self.active_pod_id = pod_id
                self._active_client = client
            self._ledger(state="running", pod_id=pod_id)
            self.active_pod_changed.emit(pod_id)
            self.log_line.emit(f"Reconnecting to pod {pod_id} ({self.pod_name}).")
            self.status_changed.emit("Reconnecting to the pod...")
            endpoint = self._wait_for_public_ssh(client, pod_id)
            key = Path(self.config.ssh_private_key).expanduser()
            transfer = ScpClient(key, *endpoint)

            def current_endpoint():
                snapshot = client.pod_snapshot(pod_id)
                return client.direct_ssh_endpoint(snapshot) if snapshot else None

            transfer.wait_ready(
                log=self.log_line.emit,
                status_check=self._bootstrap_status_check(client, pod_id),
                endpoint_provider=current_endpoint,
            )
            state = transfer.remote_job_state()
            engine = self.engine or ("spirula" if state.get("spirula") else "nerfstudio")
            self.engine = engine
            if self.config.live_preview and state.get("trainer_running"):
                self._preview_started = True
                threading.Thread(target=self._run_preview, args=(transfer,), daemon=True).start()
            self.status_changed.emit("Processing images (reconnected)...")
            self._on_pod = True
            self._follow_with_log(transfer, engine)
            self._stop_preview("Training finished. The result is downloading to this computer.")
            results_dir = Path(entry["results_dir"]) if entry.get("results_dir") else (
                results_root / safe_dir_name(Path(self.title).stem)
                / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{self.job_id[:6]}"
            )
            self.status_changed.emit("Downloading results...")
            self.log_line.emit("Processing finished; downloading the result files...")
            try:
                transfer.download(results_dir, self.log_line.emit)
            except Exception:
                preserve_pod = not self._stop_requested.is_set()
                raise
            self._check_stop_requested()
            success = True
            self.results_dir = results_dir
            self.log_line.emit(f"Results saved to {results_dir}")
        except Exception as exc:
            skip_termination = self._report_failure(exc)
        finally:
            self._wrap_up(pod_id, client, preserve_pod, skip_termination, success)

    def _follow_with_log(self, transfer, engine: str) -> None:
        """Follow a job the app is no longer attached to, showing the pod's own output."""
        live_log, stop_log = threading.Event(), self._stop_log
        if transfer.has_job_log():
            self.log_line.emit(
                "Following the pod's output again, starting with the last lines it printed."
            )
            live_log.set()

            def follow_log():
                try:
                    transfer.follow_job_log(self.log_line.emit, stop_log)
                except Exception:
                    pass  # the checkpoint lines take over
                finally:
                    live_log.clear()

            threading.Thread(target=follow_log, daemon=True).start()
        else:
            self.log_line.emit(
                "This job's pod keeps no copy of what it printed while the app was closed; "
                "progress below comes from its training checkpoints."
            )
        try:
            self._follow_remote_job(transfer, engine, live_log)
        finally:
            stop_log.set()

    def _rejoin_job(self, transfer, engine: str, lost: ConnectionLostError) -> None:
        """The SSH session running the job dropped; the job outlives it, so follow it again.

        Raises the original error when the pod can't be reached or the job is gone.
        """
        self.log_line.emit(
            "The connection to the pod dropped while the job was running. "
            "The job keeps running on the pod; reconnecting..."
        )
        self.status_changed.emit("Reconnecting to the pod...")
        state = None
        for attempt in range(REJOIN_ATTEMPTS):
            try:
                state = transfer.remote_job_state()
                break
            except Exception as exc:
                self.log_line.emit(
                    f"Pod not reachable yet ({exc}); attempt {attempt + 1} of {REJOIN_ATTEMPTS}."
                )
                if self._stop_requested.wait(REJOIN_WAIT_SECONDS):
                    raise JobCancelledError("Stop requested from the app.")
        if state is None:
            raise lost
        done = state.get("ply_ready") and state.get("summary_ready")
        if not (state.get("job_running") or state.get("trainer_running") or done):
            self.log_line.emit("The pod's job is no longer running and left no result.")
            raise lost
        self.log_line.emit("Reconnected; the job is still going.")
        self.status_changed.emit("Processing images (reconnected)...")
        self._follow_with_log(transfer, engine)

    def _follow_remote_job(self, transfer, engine: str,
                           live_log: threading.Event | None = None) -> None:
        """Wait until the pod's result files exist, finishing the job ourselves if needed.

        While live_log is set the pod's own output is streaming, so checkpoint
        lines would only repeat it.
        """
        last_step = None
        stuck_since = None
        failures = 0
        while True:
            self._check_stop_requested()
            try:
                state = transfer.remote_job_state()
                failures = 0
            except Exception as exc:
                failures += 1
                if failures >= 4:
                    raise
                self.log_line.emit(f"Pod check failed ({exc}); trying again.")
                if self._stop_requested.wait(15):
                    raise JobCancelledError("Stop requested from the app.")
                continue
            step, total = state.get("step"), state.get("total_steps")
            if step is not None and step != last_step:
                last_step = step
                if live_log is None or not live_log.is_set():
                    self.log_line.emit(
                        f"[pipeline] training checkpoint: step {step:,}"
                        + (f" of {total:,} ({100 * step // total}%)" if total else "")
                        + _time_left(state)
                    )
            done = state.get("ply_ready") and state.get("summary_ready")
            if done and not state.get("job_running"):
                self.log_line.emit("The pod finished the job on its own.")
                return
            if not state.get("job_running") and not state.get("trainer_running"):
                self.log_line.emit(
                    "The pod's job script is no longer running; finishing the last step from the app."
                )
                self.log_line.emit(transfer.finish_remote(engine))
                return
            if state.get("job_running") and not state.get("trainer_running") and not done:
                if stuck_since is None:
                    self.log_line.emit(
                        "Training has ended. Giving the pod's job script up to "
                        f"{STUCK_JOB_SECONDS // 60} minutes to save the result before the app "
                        "does it instead."
                    )
                stuck_since = stuck_since or time.monotonic()
                if time.monotonic() - stuck_since > STUCK_JOB_SECONDS:
                    self.log_line.emit(
                        "Training ended but the pod's job script has not written the result; "
                        "finishing the last step from the app."
                    )
                    self.log_line.emit(transfer.finish_remote(engine))
                    return
            else:
                stuck_since = None
            if self._stop_requested.wait(15):
                raise JobCancelledError("Stop requested from the app.")

    def _report_failure(self, exc: Exception) -> bool:
        """Log why the job stopped; True when the pod must not be terminated by us."""
        if isinstance(exc, (PodStoppedError, PodMissingError)):
            if self._stop_requested.is_set():
                self.log_line.emit(f"Job interrupted after stop request: {exc}")
            else:
                self.error.emit(str(exc))
            return True
        if isinstance(exc, (RunPodCapacityError, PodCreationUncertainError)):
            self.error.emit(str(exc))
        elif self._stop_requested.is_set():
            self.log_line.emit(f"Job interrupted after stop request: {exc}")
        else:
            self.error.emit(str(exc))
            self.log_line.emit(traceback.format_exc())
        return False

    def _wrap_up(self, pod_id, client, preserve_pod: bool, skip_termination: bool,
                 success: bool) -> None:
        """Terminate (or deliberately keep) the pod and record how the job ended."""
        self._on_pod = False
        self._stop_preview("The job ended, so the live preview closed.")
        with self._lock:
            stop_requested = self._stop_requested.is_set()
            if pod_id and not stop_requested and not skip_termination and not preserve_pod:
                self._auto_terminating = True
        if stop_requested:
            self._stop_done.wait()
        stopped = self.stop_succeeded
        if pod_id and stopped:
            pass
        elif pod_id and skip_termination:
            # The untracked-pod check settles whether it is really gone.
            self._ledger(state="unverified")
            self.log_line.emit(
                f"No termination request sent for pod {pod_id} because it was "
                "stopped or no longer visible. Verify its state in RunPod."
            )
        elif pod_id and client and not preserve_pod:
            self.status_changed.emit("Terminating RunPod pod...")
            self.log_line.emit(f"Terminating RunPod pod {pod_id}...")
            try:
                client.terminate_pod(pod_id)
                self.log_line.emit(f"Pod {pod_id} terminated.")
                self._ledger(state="terminated")
                self._clear_active_pod(pod_id)
            except Exception as exc:
                if self._pod_absent(client, pod_id):
                    self.log_line.emit(f"Pod {pod_id} is no longer present in RunPod.")
                    self._ledger(state="terminated")
                    self._clear_active_pod(pod_id)
                else:
                    self.error.emit(
                        f"Pod {pod_id} may still be billing: termination failed: {exc}"
                    )
                success = False
        elif pod_id and preserve_pod:
            self._ledger(state="kept")
            self.error.emit(f"Download failed. Pod {pod_id} was kept so you can recover results; it may still be billing.")
        with self._lock:
            cancelled = self._stop_requested.is_set()
            self.running = False
            self._auto_terminating = False
            self._stop_requested.clear()
            remaining_pod_id = self.active_pod_id
            self.finished_at = time.time()
            self.outcome = (
                "stopped" if stopped or (cancelled and not success)
                else "finished" if success else "failed"
            )
        if remaining_pod_id:
            self.active_pod_changed.emit(remaining_pod_id)
        if success and self.log_path is not None and self.results_dir is not None:
            try:
                shutil.copy2(self.log_path, self.results_dir / "job.log")
            except OSError:
                pass
        self.status_changed.emit(
            "Stopped" if self.outcome == "stopped" else "Finished" if success else "Failed"
        )
        self.finished.emit(success)