"""How fast each RunPod GPU trains splats, and what a job would take on it.

Speeds are relative to an RTX 3090. Every GPU starts from an estimate built
from its FP32 compute and memory bandwidth. Once one of your jobs has trained
on it with the same splat count, source kind and image size as a job on
another GPU, the measured ratio replaces the estimate.

Time and cost use your own finished Spirula jobs: the time per training step
of past jobs with the same splat count and source kind (photos or video
frames), converted to RTX 3090 time, then divided by the chosen GPU's speed.
Setup (pod start, download, camera solving) is added from your past jobs too,
since it costs the same per hour but barely depends on the GPU.
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, replace
from pathlib import Path

REFERENCE = "NVIDIA GeForce RTX 3090"

# RunPod GPU id -> (display name, speed relative to an RTX 3090, estimated
# from FP32 compute and memory bandwidth). GPUs missing here, MIG slices and
# AMD cards among them, show no speed.
SPEC_SPEED: dict[str, tuple[str, float]] = {
    "NVIDIA GeForce RTX 3070": ("RTX 3070", 0.52),
    "NVIDIA GeForce RTX 3080": ("RTX 3080", 0.82),
    "NVIDIA GeForce RTX 3080 Ti": ("RTX 3080 Ti", 0.96),
    "NVIDIA GeForce RTX 3090": ("RTX 3090", 1.0),
    "NVIDIA GeForce RTX 3090 Ti": ("RTX 3090 Ti", 1.10),
    "NVIDIA GeForce RTX 4070 Ti": ("RTX 4070 Ti", 0.80),
    "NVIDIA GeForce RTX 4080": ("RTX 4080", 1.15),
    "NVIDIA GeForce RTX 4080 SUPER": ("RTX 4080 SUPER", 1.20),
    "NVIDIA GeForce RTX 4090": ("RTX 4090", 1.60),
    "NVIDIA GeForce RTX 5080": ("RTX 5080", 1.30),
    "NVIDIA GeForce RTX 5090": ("RTX 5090", 2.30),
    "NVIDIA RTX A2000": ("RTX A2000", 0.27),
    "NVIDIA RTX A4000": ("RTX A4000", 0.51),
    "NVIDIA RTX A4500": ("RTX A4500", 0.67),
    "NVIDIA RTX A5000": ("RTX A5000", 0.80),
    "NVIDIA RTX A6000": ("RTX A6000", 0.92),
    "NVIDIA A40": ("A40", 0.88),
    "NVIDIA L4": ("L4", 0.52),
    "NVIDIA L40": ("L40", 1.50),
    "NVIDIA L40S": ("L40S", 1.55),
    "NVIDIA RTX 2000 Ada Generation": ("RTX 2000 Ada", 0.29),
    "NVIDIA RTX 4000 Ada Generation": ("RTX 4000 Ada", 0.54),
    "NVIDIA RTX 4000 SFF Ada Generation": ("RTX 4000 Ada SFF", 0.42),
    "NVIDIA RTX 5000 Ada Generation": ("RTX 5000 Ada", 1.06),
    "NVIDIA RTX 6000 Ada Generation": ("RTX 6000 Ada", 1.60),
    "NVIDIA RTX PRO 4000 Blackwell": ("RTX PRO 4000", 0.95),
    "NVIDIA RTX PRO 4500 Blackwell": ("RTX PRO 4500", 1.20),
    "NVIDIA RTX PRO 4500 Blackwell Server Edition": ("RTX PRO 4500 SE", 1.20),
    "NVIDIA RTX PRO 5000 Blackwell": ("RTX PRO 5000", 1.70),
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": ("RTX PRO 6000", 2.50),
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition": ("RTX PRO 6000 WK", 2.60),
    "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": ("RTX PRO 6000 MaxQ", 2.10),
    "Tesla V100-PCIE-16GB": ("Tesla V100", 0.65),
    "Tesla V100-SXM2-16GB": ("V100 SXM2", 0.65),
    "NVIDIA A100-SXM4-40GB": ("A100 SXM 40GB", 0.96),
    "NVIDIA A100 80GB PCIe": ("A100 PCIe", 1.07),
    "NVIDIA A100-SXM4-80GB": ("A100 SXM", 1.10),
    "NVIDIA H100 PCIe": ("H100 PCIe", 1.60),
    "NVIDIA H100 NVL": ("H100 NVL", 2.10),
    "NVIDIA H100 80GB HBM3": ("H100 SXM", 2.20),
    "NVIDIA H200": ("H200 SXM", 2.40),
    "NVIDIA H200 NVL": ("H200 NVL", 2.30),
    "NVIDIA B200": ("B200", 3.00),
}
_BY_DISPLAY = {display.lower(): gpu_id for gpu_id, (display, _) in SPEC_SPEED.items()}

# RTX 3090 milliseconds per step when there are no past jobs to go on
# (video frames around 4K; full-size camera photos take about twice as long).
DEFAULT_MS = {200_000: 45.0, 1_000_000: 70.0, 3_000_000: 100.0, 10_000_000: 200.0}
PHOTO_FACTOR = 2.0
DEFAULT_SETUP_S = 15 * 60


@dataclass(frozen=True)
class Run:
    """One finished Spirula training, read back from its log."""
    gpu_id: str
    splats: int
    kind: str           # "photos" or "video"
    resolution: str     # "3840x2160", or "" when the log doesn't say
    ms_per_step: float
    train_s: float
    setup_s: float | None = None  # the rest of the job's time, from job history


_TIMESTAMP = r"^\[\d\d:\d\d:\d\d\]\s*"
_DEVICE = re.compile(_TIMESTAMP + r"\* \[0\] (.+?) \((?:discrete|integrated)", re.M)
_SELECTED = re.compile(r"Selected GPU: (.+?) \(")
_DONE = re.compile(r"Training complete\. Steps: (\d+)\s+Time: (\d+):(\d+):([\d.]+)")
_SPLATS = re.compile(r"\bsplats (\d+)")
_CAMERA = re.compile(r"Camera 1: images \d+, (\d+x\d+)")
_VIDEO = re.compile(r"\[pipeline\] extracting .*frames")
_cache: dict[tuple[str, int, int], Run | None] = {}


def gpu_id_for(name: str) -> str:
    """RunPod GPU id for a device or display name ("RTX A5000 (SECURE)" works too)."""
    name = re.sub(r"\s*\(.*$", "", name or "").strip()
    if name in SPEC_SPEED:
        return name
    return _BY_DISPLAY.get(name.lower(), name)


def display_name(gpu_id: str) -> str:
    return SPEC_SPEED.get(gpu_id, (gpu_id,))[0]


def parse_log(path: Path) -> Run | None:
    """The last finished training in a job log, or None. Cached per file version."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        _cache[key] = parse_text(path.read_text(encoding="utf-8", errors="replace"))
    return _cache[key]


def parse_text(text: str) -> Run | None:
    done = list(_DONE.finditer(text))
    if not done:
        return None
    last = done[-1]
    steps = int(last.group(1))
    train_s = int(last.group(2)) * 3600 + int(last.group(3)) * 60 + float(last.group(4))
    before = text[:last.start()]
    devices = _DEVICE.findall(before)
    selected = _SELECTED.findall(before)
    gpu = gpu_id_for(devices[-1] if devices else selected[-1] if selected else "")
    splats = [int(value) for value in _SPLATS.findall(before)]
    camera = _CAMERA.findall(before)
    if not gpu or not splats or steps <= 0 or train_s <= 0:
        return None
    return Run(gpu, max(splats), "video" if _VIDEO.search(before) else "photos",
               camera[-1] if camera else "", 1000 * train_s / steps, train_s)


def runs_from_history(records: list[dict]) -> list[Run]:
    """Training runs of the finished Spirula jobs among job history records."""
    runs = []
    for record in records:
        if record.get("outcome") != "finished" or not record.get("log_path"):
            continue
        run = parse_log(Path(record["log_path"]))
        if run is None:
            continue
        started, finished = record.get("started_at"), record.get("finished_at")
        if started and finished and finished - started > run.train_s:
            run = replace(run, setup_s=finished - started - run.train_s)
        runs.append(run)
    return runs


@dataclass(frozen=True)
class Speed:
    factor: float       # 1.0 = an RTX 3090
    measured: bool      # from your own jobs rather than from the specs


@dataclass(frozen=True)
class Estimate:
    train_s: float
    total_s: float
    from_history: bool  # based on your past jobs of this kind, or on defaults


class SpeedModel:
    def __init__(self, runs: list[Run]):
        self.runs = runs
        self._speeds = self._measure()

    def _measure(self) -> dict[str, Speed]:
        speeds = {gpu_id: Speed(factor, False) for gpu_id, (_, factor) in SPEC_SPEED.items()}
        # Median time per step for each GPU on each kind of workload.
        groups: dict[tuple, dict[str, list[float]]] = {}
        for run in self.runs:
            key = (run.splats, run.kind, run.resolution)
            groups.setdefault(key, {}).setdefault(run.gpu_id, []).append(run.ms_per_step)
        ratios: dict[str, list[float]] = {}
        for by_gpu in groups.values():
            medians = {gpu: statistics.median(times) for gpu, times in by_gpu.items()}
            for gpu, ms in medians.items():
                for other, other_ms in medians.items():
                    if other != gpu and other in SPEC_SPEED:
                        ratios.setdefault(gpu, []).append(SPEC_SPEED[other][1] * other_ms / ms)
        if any(run.gpu_id == REFERENCE for run in self.runs):
            speeds[REFERENCE] = Speed(1.0, True)
        for gpu, values in ratios.items():
            if gpu == REFERENCE:
                continue
            factor = math.exp(sum(math.log(v) for v in values) / len(values))
            speeds[gpu] = Speed(round(factor, 2), True)
        return speeds

    def speed(self, gpu_id: str) -> Speed | None:
        return self._speeds.get(gpu_id)

    def runs_on(self, gpu_id: str) -> list[Run]:
        return [run for run in self.runs if run.gpu_id == gpu_id]

    def _reference_ms(self, splats: int, kind: str) -> tuple[float, bool]:
        """RTX 3090 milliseconds per step for this splat count and source kind."""
        def converted(runs):
            return [run.ms_per_step * self._speeds[run.gpu_id].factor
                    for run in runs if run.gpu_id in self._speeds]

        same = converted(r for r in self.runs if r.splats == splats and r.kind == kind)
        if same:
            return statistics.median(same), True
        other = converted(r for r in self.runs if r.splats == splats)
        if other:
            scale = PHOTO_FACTOR if kind == "photos" else 1 / PHOTO_FACTOR
            return statistics.median(other) * scale, True
        nearest = min(DEFAULT_MS, key=lambda cap: abs(math.log(cap / splats)))
        return DEFAULT_MS[nearest] * (PHOTO_FACTOR if kind == "photos" else 1.0), False

    def setup_s(self) -> float:
        setups = [run.setup_s for run in self.runs if run.setup_s]
        return statistics.median(setups) if setups else DEFAULT_SETUP_S

    def estimate(self, gpu_id: str, steps: int, splats: int, kind: str) -> Estimate | None:
        speed = self.speed(gpu_id)
        if speed is None or steps <= 0 or splats <= 0:
            return None
        ms, from_history = self._reference_ms(splats, kind)
        train = steps * ms / 1000 / speed.factor
        return Estimate(train, train + self.setup_s(), from_history)


def duration_text(seconds: float) -> str:
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"
