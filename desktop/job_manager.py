"""Runs several jobs at once without losing track of any pod.

Safety rules:
- Every pod name (gaussian-<job id>) is written to the pod ledger before the
  creation request is sent.
- A new job is refused while the number of pods (this session's plus any
  untracked ones) is at the configured limit. The check and the job start
  happen under one lock, so two clicks cannot both take the last slot.
- RunPod's own pod list is compared with the jobs of this session at startup
  and every minute. Any gaussian-* pod no running job owns is reported as
  untracked, with a button to terminate it. Nothing is terminated
  automatically: an untracked pod may hold results someone wants to recover.
"""
from __future__ import annotations

import copy
import threading
import time
from pathlib import Path

from PySide6 import QtCore

from config import APP_DIR, AppConfig
from job_controller import JobController
from job_history import JobHistory
from pod_ledger import PodLedger
from runpod_client import RunPodClient

POD_NAME_PREFIX = "gaussian-"
UNTRACKED_CHECK_SECONDS = 60
LEDGER_FILE = APP_DIR / "pod_ledger.json"
LOG_DIR = APP_DIR / "logs"
HISTORY_FILE = APP_DIR / "jobs.db"
HISTORY_LIMIT = 100


class JobManager(QtCore.QObject):
    job_added = QtCore.Signal(object)
    gpu_candidates_changed = QtCore.Signal(object)
    gpu_lookup_finished = QtCore.Signal()
    status_changed = QtCore.Signal(str)
    error = QtCore.Signal(str)
    # list of dicts: id, name, status, cost, title, created_at
    untracked_pods_changed = QtCore.Signal(object)
    untracked_check_failed = QtCore.Signal(str)
    notice = QtCore.Signal(str)

    def __init__(self, config: AppConfig, *, ledger_path: Path = LEDGER_FILE,
                 log_dir: Path = LOG_DIR, history_path: Path = HISTORY_FILE):
        super().__init__()
        self.config = config
        self.ledger = PodLedger(ledger_path)
        self.log_dir = log_dir
        self.history = JobHistory(history_path)
        self.jobs: list[JobController] = []
        self.untracked: list[dict] = []
        self._lock = threading.Lock()
        self._upload_lock = threading.Lock()
        self._checking = False
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(UNTRACKED_CHECK_SECONDS * 1000)
        self._timer.timeout.connect(self.check_untracked_pods)

    def start_monitoring(self) -> None:
        self.check_untracked_pods()
        self._timer.start()

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _pods_in_use(self) -> int:
        return sum(1 for job in self.jobs if job.busy) + len(self.untracked)

    def slots_text(self) -> str:
        with self._lock:
            used = self._pods_in_use()
        return f"{used} of {self.config.max_concurrent_pods} pod slots in use"

    def start_job(self, *, title: str, gpu_label: str, run_args: tuple, run_kwargs: dict,
                  context: str = "", rate: str = "") -> JobController:
        """Create and start a job, or raise ValueError when the pod limit is reached."""
        with self._lock:
            limit = max(1, int(self.config.max_concurrent_pods))
            in_use = self._pods_in_use()
            if in_use >= limit:
                untracked = (
                    f" ({len(self.untracked)} of them untracked; review them in the sidebar)"
                    if self.untracked else ""
                )
                raise ValueError(
                    f"{in_use} pod{'s are' if in_use != 1 else ' is'} already in use{untracked}, "
                    f"and the limit is {limit}. Wait for a job to finish or raise the limit "
                    "in Settings."
                )
            job = JobController(
                copy.deepcopy(self.config), ledger=self.ledger,
                upload_lock=self._upload_lock, log_dir=self.log_dir,
            )
            job.title = title
            job.gpu_label = gpu_label
            job.context = context
            job.rate_text = rate
            # run_job marks the job running before returning, so it counts
            # toward the limit before the lock is released.
            job.run_job(*run_args, start=False, **run_kwargs)
            if not job.running:
                raise ValueError("The job could not start; check the input and engine.")
            self.jobs.append(job)
        self._track(job)
        # Listeners connect to the job's signals first, so no log line is lost.
        self.job_added.emit(job)
        job.start_thread()
        return job

    def reattach(self, pod: dict) -> JobController:
        """Reconnect to an untracked pod and finish its job (download, then terminate)."""
        entry = dict(self.ledger.by_pod_name().get(pod["name"], {}))
        entry.setdefault("job_id", pod["name"].removeprefix(POD_NAME_PREFIX))
        entry["pod_name"] = pod["name"]
        entry["pod_id"] = pod["id"]
        entry.setdefault("title", pod.get("title") or pod["name"])
        entry["cost"] = pod.get("cost")
        with self._lock:
            job = JobController(
                copy.deepcopy(self.config), ledger=self.ledger,
                upload_lock=self._upload_lock, log_dir=self.log_dir,
            )
            job.reattach(entry, self.config.results_root(), start=False)
            job.context = "Reconnected to a running pod"
            job.rate_text = f"${pod['cost']:.3f}/hr" if pod.get("cost") is not None else ""
            # The same job may be in the list from history; it is replaced.
            self.jobs = [item for item in self.jobs
                         if not (item.historic and item.job_id == job.job_id)]
            self.jobs.append(job)
            self.untracked = [item for item in self.untracked if item["id"] != pod["id"]]
            remaining = list(self.untracked)
        self._track(job)
        self.job_added.emit(job)
        self.untracked_pods_changed.emit(remaining)
        job.start_thread()
        return job

    def remove_job(self, job: JobController) -> bool:
        """Take a finished job off the list and out of history (its files stay)."""
        with self._lock:
            if job.busy or job not in self.jobs:
                return False
            self.jobs.remove(job)
        try:
            self.history.delete(job.job_id)
        except Exception:
            pass
        return True

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    @staticmethod
    def snapshot(job: JobController) -> dict:
        return {
            "job_id": job.job_id,
            "title": job.title,
            "context": job.context,
            "engine": job.engine,
            "gpu_label": job.gpu_label,
            "cost_per_hour": job.cost_per_hour,
            "pod_id": job.active_pod_id or job.last_pod_id,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "outcome": job.outcome,
            "status_text": job.status_text,
            "results_dir": str(job.results_dir) if job.results_dir else None,
            "log_path": str(job.log_path) if job.log_path else None,
            "running": job.running,
        }

    def _save(self, job: JobController) -> None:
        try:
            self.history.save(self.snapshot(job))
        except Exception:
            pass  # history is a convenience; never let it break a job

    def _track(self, job: JobController) -> None:
        """Save the job now and whenever its state changes (from the job's own thread)."""
        def pod_changed(pod_id):
            if pod_id:
                job.last_pod_id = pod_id
            self._save(job)

        job.status_changed.connect(lambda _text: self._save(job), QtCore.Qt.DirectConnection)
        job.active_pod_changed.connect(pod_changed, QtCore.Qt.DirectConnection)
        job.finished.connect(lambda _ok: self._save(job), QtCore.Qt.DirectConnection)
        self._save(job)

    def load_history(self) -> list[JobController]:
        """Past jobs (newest first) as read-only entries for the job list."""
        if self.history.get_meta("imported_logs") is None:
            titles = {entry["job_id"]: entry.get("title") or "" for entry in self.ledger.entries()
                      if entry.get("job_id")}
            try:
                self.history.import_logs(self.log_dir, titles)
            except Exception:
                pass
            self.history.set_meta("imported_logs", "1")
        loaded = []
        with self._lock:
            live = {job.job_id for job in self.jobs}
            for record in self.history.recent(HISTORY_LIMIT):
                if record.get("job_id") in live:
                    continue
                job = self._historic_job(record)
                self.jobs.append(job)
                loaded.append(job)
        return loaded

    def _historic_job(self, record: dict) -> JobController:
        job = JobController(copy.deepcopy(self.config))
        job.historic = True
        job.job_id = record["job_id"]
        job.pod_name = f"{POD_NAME_PREFIX}{job.job_id}"
        job.title = record.get("title") or f"Job {job.job_id}"
        job.context = record.get("context") or ""
        job.engine = record.get("engine") or ""
        job.gpu_label = record.get("gpu_label") or ""
        job.cost_per_hour = record.get("cost_per_hour")
        job.rate_text = f"${job.cost_per_hour:.3f}/hr" if job.cost_per_hour else "unknown"
        job.last_pod_id = record.get("pod_id")
        job.started_at = record.get("started_at")
        outcome = record.get("outcome")
        finished = record.get("finished_at")
        if record.get("running") or not outcome:
            # The app closed while it was running.
            outcome = "interrupted"
            finished = finished or record.get("updated_at")
        job.outcome = outcome
        job.finished_at = finished
        job.status_text = record.get("status_text") or outcome.capitalize()
        if record.get("results_dir"):
            job.results_dir = Path(record["results_dir"])
        job.log_path = Path(record["log_path"]) if record.get("log_path") else None
        return job

    def busy_jobs(self) -> list[JobController]:
        with self._lock:
            return [job for job in self.jobs if job.busy]

    def hourly_cost(self) -> float:
        with self._lock:
            jobs = sum(job.cost_per_hour or 0 for job in self.jobs if job.active_pod_id)
            return jobs + sum(pod.get("cost") or 0 for pod in self.untracked)

    # ------------------------------------------------------------------
    # GPU lookup
    # ------------------------------------------------------------------

    def refresh_gpus(self, volume_gb: int) -> None:
        config = copy.deepcopy(self.config)

        def work() -> None:
            try:
                if not config.has_runpod():
                    raise RuntimeError("Configure a RunPod API key first.")
                self.status_changed.emit("Checking RunPod GPUs...")
                client = RunPodClient(config.runpod_api_key, config.runpod_graphql_url)
                candidates = client.select_gpu_types(
                    config.max_gpu_price, config.min_gpu_vram_gb,
                    cloud_type=config.cloud_type,
                    container_disk_gb=config.container_disk_gb,
                    volume_gb=volume_gb,
                    min_vcpu=config.min_vcpu,
                    min_memory_gb=config.min_memory_gb,
                )
                self.gpu_candidates_changed.emit(candidates)
                available = sum(item["_available"] for item in candidates)
                self.status_changed.emit(
                    f"{available} GPU options match public SSH and storage; "
                    f"{len(candidates) - available} have no matching capacity"
                )
            except Exception as exc:
                self.gpu_candidates_changed.emit([])
                self.error.emit(f"GPU lookup failed: {exc}")
            finally:
                self.gpu_lookup_finished.emit()

        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------------------
    # Untracked pods
    # ------------------------------------------------------------------

    def _owned_pod_names(self) -> set[str]:
        # A job owns its pod name from the moment it sends the creation request
        # until the pod is confirmed gone, so a pod that is still being created
        # is never reported as untracked.
        return {job.pod_name for job in self.jobs if job.pod_requested and job.busy}

    def check_untracked_pods(self) -> None:
        if not self.config.has_runpod():
            return
        with self._lock:
            if self._checking:
                return
            self._checking = True
        config = copy.deepcopy(self.config)

        def work() -> None:
            try:
                pods = RunPodClient(config.runpod_api_key, config.runpod_graphql_url).list_pods()
                self._reconcile(pods)
            except Exception as exc:
                self.untracked_check_failed.emit(f"Could not check RunPod for untracked pods: {exc}")
            finally:
                with self._lock:
                    self._checking = False

        threading.Thread(target=work, daemon=True).start()

    def _reconcile(self, pods: list[dict]) -> None:
        live = [
            pod for pod in pods
            if str(pod.get("name") or "").startswith(POD_NAME_PREFIX)
            and str(pod.get("desiredStatus") or "").upper() != "TERMINATED"
        ]
        listed_names = {pod.get("name") for pod in live}
        ledger = self.ledger.by_pod_name()
        with self._lock:
            owned = self._owned_pod_names()
            untracked = []
            for pod in live:
                if pod.get("name") in owned:
                    continue
                entry = ledger.get(pod.get("name"), {})
                cost = pod.get("adjustedCostPerHr")
                if cost is None:
                    cost = pod.get("costPerHr")
                try:
                    cost = float(cost) if cost is not None else None
                except (TypeError, ValueError):
                    cost = None
                untracked.append({
                    "id": str(pod.get("id") or ""),
                    "name": str(pod.get("name") or ""),
                    "status": str(pod.get("desiredStatus") or "unknown").lower(),
                    "cost": cost,
                    "title": entry.get("title") or "",
                    "created_at": entry.get("created_at"),
                })
            self.untracked = untracked
        # Open ledger entries whose pod RunPod no longer lists are settled.
        for entry in self.ledger.open_entries():
            name = entry.get("pod_name")
            if name and name not in listed_names and name not in owned:
                # Give a creation request in flight time to show up first.
                if time.time() - float(entry.get("updated_at") or 0) > 300:
                    self.ledger.update(entry["job_id"], state="gone")
        self.untracked_pods_changed.emit(list(untracked))

    def terminate_untracked(self, pod_id: str, pod_name: str = "") -> None:
        config = copy.deepcopy(self.config)

        def work() -> None:
            client = RunPodClient(config.runpod_api_key, config.runpod_graphql_url)
            try:
                client.terminate_pod(pod_id)
                gone = True
            except Exception as exc:
                gone = JobController._pod_absent(client, pod_id)
                if not gone:
                    self.error.emit(
                        f"Could not terminate pod {pod_id}: {exc}. Check RunPod; it may still be billing."
                    )
            if gone:
                self.notice.emit(f"Untracked pod {pod_id} terminated.")
                for entry in self.ledger.open_entries():
                    if entry.get("pod_id") == pod_id or (pod_name and entry.get("pod_name") == pod_name):
                        self.ledger.update(entry["job_id"], state="terminated", pod_id=pod_id)
                with self._lock:
                    self.untracked = [pod for pod in self.untracked if pod["id"] != pod_id]
                    remaining = list(self.untracked)
                self.untracked_pods_changed.emit(remaining)
            self.check_untracked_pods()

        threading.Thread(target=work, daemon=True).start()

