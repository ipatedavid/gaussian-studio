"""One row of numbers per job, for comparing runs: cost, time per phase, capture,
training curve, GPU memory, splat file and your own rating.

Everything comes from the job's log (every line is time-stamped), so jobs that ran
before this existed get rows too. The splat file is read when it is still on disk.

The app rewrites runs.csv next to jobs.db whenever a job ends or is rated.
By hand:  python3 run_report.py          (writes runs.csv and prints a summary)
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
RUNS_CSV = APP_DIR / "runs.csv"
SPLAT_STATS = "splat_stats.json"

# Phase starts, in order. A phase lasts until the next one found in the log.
PHASES = (
    ("pod_start", r"Sending one RunPod pod creation request"),
    ("inputs", r"uploading worker scripts"),
    ("setup", r"\[pipeline\] verifying runtime"),
    ("frames", r"\[pipeline\] extracting about"),
    ("sfm", r"\[pipeline\] (processing images with|reconstructing the)"),
    ("prep", r"\[pipeline\] Spirula SfM registered"),
    ("training", r"\[pipeline\] starting (Spirula|Splatfacto)"),
    ("export", r"\[pipeline\] exporting from"),
    ("download", r"Processing finished; downloading"),
    ("wrap_up", r"Results saved to"),
)
POD_GONE = re.compile(r"Pod \S+ terminated\.|is no longer present in RunPod")
LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (.*)$")
STEP = re.compile(r"step +(\d+)/(\d+) .*?splats (\d+).*?psnr=([\d.]+)")
TRAINING = re.compile(
    r"starting \w+ training for (\d+) iterations \(quality: (\w+), up to ([\d,]+) splats")
OLD_QUALITY = re.compile(r"quality (\w+) \(up to ([\d,]+) splats")
VRAM_PEAK = re.compile(r"\[vram\] training peak: ([\d.]+) of ([\d.]+) GiB.*?busy (\d+)%")
VRAM_PROJECTED = re.compile(r"Projected peak: ([\d.]+)")
QUALITY_NAMES = {"draft": "low", "standard": "medium", "high": "high", "ultra": "ultra"}

COLUMNS = (
    "date", "title", "job_id", "outcome", "engine", "quality", "gpu", "price_per_hour",
    "cost_usd", "billed_min",
    *(f"{name}_min" for name, _ in PHASES[:-1]),
    "images", "registered", "width", "height", "reprojection_error_px", "mean_track_length",
    "sparse_points", "seed_points",
    "steps", "splat_cap", "cap_reached_step", "cap_reached_pct", "final_splats",
    "psnr_mid", "psnr_end", "late_gain_db",
    "vram_peak_gib", "vram_total_gib", "vram_projected_gib", "gpu_busy_pct",
    "guidance", "guidance_failed", "guidance_salvaged", "depth_seconds", "depth_coverage",
    "depth_sfm_error", "ply_mb", "splats_in_file", "faint_share", "invisible_share",
    "rating", "note", "settings",
)


def _timed_lines(text: str):
    """(seconds since the first line, message), across midnight."""
    day = 0
    previous = None
    start = None
    for raw in text.splitlines():
        match = LINE.match(raw)
        if not match:
            continue
        clock = int(match[1]) * 3600 + int(match[2]) * 60 + int(match[3])
        if previous is not None and clock < previous - 60:
            day += 1
        previous = clock
        at = clock + day * 86400
        if start is None:
            start = at
        yield at - start, match[4]


def _window(curve: list[tuple[int, int, int, float]], low: float, high: float) -> float | None:
    values = [psnr for step, total, _, psnr in curve if low <= step / total <= high and psnr > 0]
    return round(sum(values) / len(values), 2) if values else None


def parse_log(path: Path) -> dict:
    """The numbers a job's log holds."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    row: dict = {}
    marks: dict[str, float] = {}
    end = last = None
    curve: list[tuple[int, int, int, float]] = []
    for at, message in _timed_lines(text):
        last = at
        for name, pattern in PHASES:
            if name not in marks and re.search(pattern, message):
                marks[name] = at
        if POD_GONE.search(message):
            end = at
        if match := STEP.search(message):
            curve.append((int(match[1]), int(match[2]), int(match[3]), float(match[4])))
        elif message.startswith("[stats] depth "):
            try:
                depth = json.loads(message[len("[stats] depth "):])
            except ValueError:
                depth = {}
            row["depth_seconds"] = depth.get("seconds")
            row["depth_coverage"] = depth.get("coverage")
            row["depth_sfm_error"] = depth.get("sfm_error_after")
        elif message.startswith("[stats] capture "):
            try:
                capture = json.loads(message[len("[stats] capture "):])
            except ValueError:
                capture = {}
            if not capture.get("reprojection_error_px"):
                capture.pop("reprojection_error_px", None)   # Spirula's printed mean stays
            row.update(capture)
        elif match := TRAINING.search(message):
            row["steps"] = int(match[1])
            row["quality"] = match[2]
            row["splat_cap"] = int(match[3].replace(",", ""))
            if used := re.search(r"guidance: ([\w-]+)\)", message):
                row["guidance"] = used[1]   # the last start wins after a retry
        elif "quality" not in row and (match := OLD_QUALITY.search(message)):
            row["quality"] = QUALITY_NAMES.get(match[1].lower(), match[1].lower())
            row["splat_cap"] = int(match[2].replace(",", ""))
        elif match := re.search(r"\[pipeline\] (\d+) images prepared", message):
            row.setdefault("images", int(match[1]))
        elif match := re.search(r"SfM registered (\d+) of (\d+)", message):
            row.setdefault("registered", int(match[1]))
        elif match := re.search(r"seed points: (\d+)", message):
            row["seed_points"] = int(match[1])
        elif match := VRAM_PEAK.search(message):
            row["vram_peak_gib"] = float(match[1])
            row["vram_total_gib"] = float(match[2])
            row["gpu_busy_pct"] = int(match[3])
        elif match := VRAM_PROJECTED.search(message):
            row["vram_projected_gib"] = float(match[1])
        elif "geometry guidance failed" in message:
            row["guidance_failed"] = True
        elif "geometry step crashed after writing all" in message:
            row["guidance_salvaged"] = True
        elif ("training failed with the kept guidance maps" in message
              or "depth guidance failed" in message):
            row["guidance_failed"] = True
        elif match := re.search(r"Reprojection error: mean ([\d.]+) px", message):
            row.setdefault("reprojection_error_px", float(match[1]))
        elif match := re.search(r"\$([\d.]+)/hr quoted", message):
            row["quoted_price"] = float(match[1])

    if "pod_start" in marks:
        end = end if end is not None else last
        row["billed_seconds"] = end - marks["pod_start"]
        found = sorted(marks.items(), key=lambda item: item[1])
        for (name, at), (_, following) in zip(found, [*found[1:], ("end", end)]):
            if name != "wrap_up":
                row[f"{name}_min"] = round((following - at) / 60, 1)

    if curve:
        total = curve[-1][1]
        cap = row.get("splat_cap") or max(splats for _, _, splats, _ in curve)
        reached = next((step for step, _, splats, _ in curve if splats >= cap * 0.999), None)
        row.setdefault("steps", total)
        row["final_splats"] = curve[-1][2]
        if reached is not None:
            row["cap_reached_step"] = reached
            row["cap_reached_pct"] = round(100 * reached / total)
        row["psnr_mid"] = _window(curve, 0.45, 0.55)
        row["psnr_end"] = _window(curve, 0.9, 1.0)
        if row["psnr_mid"] is not None and row["psnr_end"] is not None:
            row["late_gain_db"] = round(row["psnr_end"] - row["psnr_mid"], 2)
    return row


def splat_stats(ply: Path) -> dict:
    """Size and how many splats are faint or invisible (they cost viewers for little)."""
    with ply.open("rb") as handle:
        header = []
        while True:
            line = handle.readline()
            if not line:
                return {}
            header.append(line.decode("ascii", "replace").strip())
            if header[-1] == "end_header":
                break
        if "format binary_little_endian 1.0" not in header:
            return {}
        count = next(int(h.split()[2]) for h in header if h.startswith("element vertex"))
        props = [h.split() for h in header if h.startswith("property")]
        if any(p[1] != "float" for p in props):
            return {}
        names = [p[2] for p in props]
        if "opacity" not in names:
            return {}
        stride = len(names)
        column = names.index("opacity")
        faint = invisible = 0
        # logit thresholds: opacity below 5% and below 1%
        faint_at, invisible_at = math.log(0.05 / 0.95), math.log(0.01 / 0.99)
        left = count
        while left:
            rows = min(left, 200_000)
            data = memoryview(handle.read(rows * stride * 4)).cast("f")
            for value in data[column::stride]:
                if value < faint_at:
                    faint += 1
                    if value < invisible_at:
                        invisible += 1
            left -= rows
    return {
        "ply_mb": round(ply.stat().st_size / 1024**2, 1),
        "splats_in_file": count,
        "faint_share": round(faint / count, 3) if count else None,
        "invisible_share": round(invisible / count, 3) if count else None,
    }


def cached_splat_stats(results_dir: Path) -> dict:
    """splat_stats() of the job's PLY, kept in the results folder so it's read once."""
    ply = results_dir / "gaussian_splat.ply"
    cache = results_dir / SPLAT_STATS
    if not ply.is_file():
        try:
            return json.loads(cache.read_text())
        except (OSError, ValueError):
            return {}
    stamp = f"{ply.stat().st_size}:{ply.stat().st_mtime_ns}"
    try:
        cached = json.loads(cache.read_text())
        if cached.get("stamp") == stamp:
            return cached
    except (OSError, ValueError):
        pass
    try:
        stats = splat_stats(ply)
    except (OSError, ValueError, StopIteration):
        return {}
    stats["stamp"] = stamp
    try:
        cache.write_text(json.dumps(stats))
    except OSError:
        pass
    return stats


def build_row(record: dict) -> dict:
    row = parse_log(Path(record["log_path"])) if record.get("log_path") else {}
    if record.get("results_dir"):
        results = Path(record["results_dir"])
        row.update({k: v for k, v in cached_splat_stats(results).items() if k != "stamp"})
        try:
            summary = json.loads((results / "pipeline_summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
        for key, value in (summary.get("capture") or {}).items():
            row.setdefault(key, value)
        vram = summary.get("training_vram") or {}
        row.setdefault("vram_peak_gib", vram.get("peak_gib"))
        row.setdefault("vram_total_gib", vram.get("total_gib"))
    price = record.get("cost_per_hour") or row.pop("quoted_price", None)
    billed = row.pop("billed_seconds", None)
    started = record.get("started_at")
    row.update({
        "date": datetime.fromtimestamp(started).strftime("%Y-%m-%d %H:%M") if started else "",
        "title": record.get("title") or "",
        "job_id": record.get("job_id"),
        "outcome": record.get("outcome") or "",
        "engine": record.get("engine") or "",
        "gpu": record.get("gpu_label") or "",
        "price_per_hour": price,
        "billed_min": round(billed / 60, 1) if billed is not None else None,
        "cost_usd": round(billed / 3600 * price, 3) if billed is not None and price else None,
        "rating": record.get("rating"),
        "note": record.get("rating_note") or "",
        "settings": record.get("context") or "",
    })
    return row


def build_rows(records: list[dict]) -> list[dict]:
    return [build_row(record) for record in sorted(records, key=lambda r: r.get("started_at") or 0)]


def write_csv(rows: list[dict], path: Path = RUNS_CSV) -> Path:
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in COLUMNS})
    temp.replace(path)
    return path


def main() -> None:
    from job_history import JobHistory

    history = JobHistory(APP_DIR / "jobs.db")
    rows = build_rows(history.recent(1000))
    history.close()
    path = write_csv(rows, Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS_CSV)
    total = sum(row.get("cost_usd") or 0 for row in rows)
    print(f"{len(rows)} runs, ${total:.2f} in all -> {path}")
    for row in rows:
        print(f"{row['date']}  {row['title'][:26]:26} {row.get('quality') or '-':7}"
              f" ${row.get('cost_usd') or 0:5.2f}  train {row.get('training_min') or 0:6.1f} min"
              f"  cap@{row.get('cap_reached_pct') or '-':>3}%  late gain "
              f"{row.get('late_gain_db') if row.get('late_gain_db') is not None else '-'} dB"
              f"  rating {row.get('rating') or '-'}")


if __name__ == "__main__":
    main()
