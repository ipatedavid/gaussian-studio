from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

GIB = 1024 ** 3
MIB = 1024 ** 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi"}
MAX_VIDEOS = 8
DEFAULT_VIDEO_FPS = 3.0
MIN_VIDEO_FPS, MAX_VIDEO_FPS = 0.2, 30.0
# Frame count assumed for storage planning when Drive has not reported a length.
UNKNOWN_LENGTH_FRAMES = 600


@dataclass(frozen=True)
class VideoInfo:
    duration_seconds: float
    width: int
    height: int
    fps: float  # 0 when unknown (Google Drive does not report it)


# Plan kinds. "mixed" and "drive_mixed" hold photos and videos together.
VIDEO_KINDS = {"video", "drive_video", "mixed", "drive_mixed"}
PHOTO_KINDS = {"images", "drive", "mixed", "drive_mixed"}
DRIVE_KINDS = {"drive", "drive_video", "drive_mixed"}


@dataclass(frozen=True)
class InputPlan:
    kind: str
    source: Path
    images: tuple[Path, ...]
    input_bytes: int
    working_reserve: int
    volume_gb: int
    frame_target: int | None = None  # estimated frames from all videos together
    video_info: VideoInfo | None = None
    estimated_frame_bytes: int = 0
    # Google Drive input: photos stay in Drive and are fetched by the pod.
    drive_folder_id: str = ""
    drive_folder_name: str = ""
    drive_files: tuple = ()
    # Local video input: every picked video, in order (source is the first).
    videos: tuple[Path, ...] = ()
    extract_fps: float = 0.0  # frames taken per second of video
    # Drive photos + videos: the videos (drive_files holds the photos).
    drive_videos: tuple = ()

    @property
    def photo_count(self) -> int:
        if self.kind in {"images", "mixed"}:
            return len(self.images)
        if self.kind in {"drive", "drive_mixed"}:
            return len(self.drive_files)
        return 0

    @property
    def image_count(self) -> int | None:
        """Images training will see: photos plus the estimated video frames (None if unknown)."""
        if not self.has_videos:
            return self.photo_count
        if self.video_info is not None and self.video_info.duration_seconds > 0:
            return self.photo_count + (self.frame_target or 0)
        return None

    @property
    def video_count(self) -> int:
        if self.kind == "drive_video":
            return len(self.drive_files)
        if self.kind == "drive_mixed":
            return len(self.drive_videos)
        return len(self.videos) or (1 if self.kind == "video" else 0)

    @property
    def has_videos(self) -> bool:
        return self.kind in VIDEO_KINDS

    @property
    def has_photos(self) -> bool:
        return self.kind in PHOTO_KINDS

    @property
    def is_video(self) -> bool:
        """Only videos: no photos."""
        return self.kind in {"video", "drive_video"}

    @property
    def is_mixed(self) -> bool:
        return self.kind in {"mixed", "drive_mixed"}

    @property
    def is_drive(self) -> bool:
        return self.kind in DRIVE_KINDS

    @property
    def pipeline_source_type(self) -> str:
        """What the pod pipeline sees: Drive inputs are ordinary photos and videos."""
        if self.is_mixed:
            return "mixed"
        return "video" if self.is_video else "images"

    @property
    def display_name(self) -> str:
        if self.kind in {"drive", "drive_mixed"}:
            return self.drive_folder_name
        if self.kind == "mixed":
            return self.source.name
        if self.kind == "drive_video":
            name = self.drive_files[0].name
        else:
            name = self.source.name
        more = self.video_count - 1
        return f"{name} + {more} more" if more > 0 else name

    @property
    def job_title(self) -> str:
        """The folder the capture is in: what a job is called in the jobs list."""
        if self.kind in {"images", "mixed"}:
            folder = self.source.name
        elif self.is_drive:
            folder = self.drive_folder_name
        else:
            folder = self.source.parent.name
        if not folder:
            return self.display_name
        count = self.video_count
        if self.is_mixed:
            return f"{folder} · photos + {count} video{'s' if count != 1 else ''}"
        return f"{folder} · {count} videos" if count > 1 else folder

    @property
    def results_parent(self) -> Path:
        if self.is_drive:
            return self.source
        if self.kind == "video":
            return self.source.parent / "results" / self.source.stem
        return self.source / "results"


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError("Select an existing photo folder.")
    images = sorted(
        path for path in folder.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(images) < 3:
        raise ValueError(f"Found {len(images)} supported images; at least three are required.")
    return images


def list_videos(folder: Path) -> list[Path]:
    """Videos directly inside a folder, by name."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def folder_photos(folder: Path) -> list[Path]:
    """Photos directly inside a folder, by name (any number, unlike list_images)."""
    if not folder.is_dir():
        raise ValueError("Select an existing folder.")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def storage_from_sizes(sizes: list[int], minimum_volume_gb: int) -> tuple[int, int, int]:
    input_bytes = sum(sizes)
    working_reserve = 20 * GIB + 5 * input_bytes + 20 * MIB * len(sizes)
    requested = max(30, minimum_volume_gb, math.ceil(1.1 * (input_bytes + working_reserve) / GIB))
    return input_bytes, working_reserve, requested


def storage_plan(images: list[Path], minimum_volume_gb: int) -> tuple[int, int, int]:
    """Return input bytes, space to reserve after upload, and requested volume GiB."""
    return storage_from_sizes([image.stat().st_size for image in images], minimum_volume_gb)


def safe_dir_name(name: str) -> str:
    cleaned = "".join("_" if ch in '<>:"/\\|?*' or ord(ch) < 32 else ch for ch in name).strip(" .")
    return cleaned[:80] or "drive-folder"


def plan_drive_input(
    folder_id: str, folder_name: str, files: tuple, results_root: Path,
    minimum_volume_gb: int,
) -> InputPlan:
    """Plan a job whose photos live in a Google Drive folder (files have .size)."""
    if len(files) < 3:
        raise ValueError(
            f"Found {len(files)} supported photos in the Drive folder; at least three are required."
        )
    input_bytes, reserve, volume_gb = storage_from_sizes([f.size for f in files], minimum_volume_gb)
    return InputPlan(
        "drive", Path(results_root).expanduser() / safe_dir_name(folder_name), (),
        input_bytes, reserve, volume_gb,
        drive_folder_id=folder_id, drive_folder_name=folder_name, drive_files=tuple(files),
    )


def probe_video(video: Path) -> VideoInfo:
    if not video.is_file() or video.is_symlink():
        raise ValueError("Select an existing video file.")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Select an MP4, MOV, MKV, M4V, WebM, or AVI video.")
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
                "-of", "json", str(video),
            ],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except FileNotFoundError as exc:
        raise ValueError("ffprobe is required on this laptop to inspect video metadata.") from exc
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Could not inspect video with ffprobe: {exc}") from exc
    return probe_output(result.stdout)


def probe_output(text: str) -> VideoInfo:
    """VideoInfo from ffprobe's JSON (stream size/rate/duration, format duration)."""
    def first_positive(values, *, rational=False):
        for raw in values:
            try:
                value = float(Fraction(raw)) if rational else float(raw)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if math.isfinite(value) and value > 0:
                return value
        raise ValueError("No positive metadata value was found.")

    try:
        metadata = json.loads(text)
        stream = metadata["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        duration = first_positive(
            (metadata.get("format", {}).get("duration"), stream.get("duration"))
        )
        fps = first_positive(
            (stream.get("avg_frame_rate"), stream.get("r_frame_rate")),
            rational=True,
        )
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        raise ValueError("Video has missing or invalid duration, dimensions, or frame rate.") from exc
    if width <= 0 or height <= 0:
        raise ValueError("Video has invalid dimensions.")
    return VideoInfo(duration, width, height, fps)


def video_frame_step(source_fps: float, extract_fps: float) -> int:
    """Take every Nth source frame. Same rule as the pod pipeline."""
    return max(1, round(source_fps / extract_fps))


def estimate_video_frames(info: VideoInfo, extract_fps: float) -> int:
    """Frames the pod will take from the whole video at extract_fps (0 if unknown)."""
    if info.duration_seconds <= 0:
        return 0
    if info.fps > 0:
        available = math.floor(info.duration_seconds * info.fps)
        return math.ceil(available / video_frame_step(info.fps, extract_fps))
    return math.ceil(info.duration_seconds * extract_fps)


# Frame size assumed for storage planning when the resolution is unknown.
_FALLBACK_VIDEO_SIZE = (3840, 2160)


def combine_video_info(infos: list[VideoInfo]) -> VideoInfo:
    """One summary for several videos: total length, largest frame size."""
    if len(infos) == 1:
        return infos[0]
    known = all(info.duration_seconds > 0 for info in infos)
    return VideoInfo(
        sum(info.duration_seconds for info in infos) if known else 0.0,
        max(info.width for info in infos), max(info.height for info in infos),
        infos[0].fps if all(info.fps == infos[0].fps for info in infos) else 0.0,
    )


def _plan_video(
    kind: str, source: Path, infos: list[VideoInfo], input_bytes: int, minimum_volume_gb: int,
    num_downscales: int, video_fps: float, photo_sizes: tuple[int, ...] = (), **extra,
) -> InputPlan:
    video_fps = video_fps or DEFAULT_VIDEO_FPS
    if not MIN_VIDEO_FPS <= video_fps <= MAX_VIDEO_FPS:
        raise ValueError(
            f"Frames per second must be between {MIN_VIDEO_FPS:g} and {MAX_VIDEO_FPS:g}."
        )
    # Every video is sampled from start to end at the same rate.
    estimates = [estimate_video_frames(info, video_fps) for info in infos]
    for info, frames in zip(infos, estimates):
        if info.duration_seconds > 0 and frames < 3:
            raise ValueError(
                f"A {info.duration_seconds:.1f}s video gives only {frames} frame(s) at "
                f"{video_fps:g} fps; raise the frames per second."
            )
    frame_target = sum(frames or UNKNOWN_LENGTH_FRAMES for frames in estimates)
    info = combine_video_info(infos)
    if num_downscales < 0:
        raise ValueError("Image downscales cannot be negative.")
    width, height = (info.width, info.height) if info.width > 0 and info.height > 0 \
        else _FALLBACK_VIDEO_SIZE
    # Nerfstudio extracts PNG frames. Four bytes per pixel plus workspace for
    # COLMAP/training is a conservative estimate, not a hard maximum.
    scale_factor = sum(1 / (4 ** index) for index in range(num_downscales + 1))
    frame_bytes = math.ceil(width * height * 4 * frame_target * scale_factor)
    reserve = frame_bytes * 3 + 20 * GIB + 20 * MIB * frame_target
    if photo_sizes:
        # Photos travelling with the videos: their usual reserve, without a
        # second 20 GiB base (input_bytes already counts the photos once).
        reserve += 5 * sum(photo_sizes) + 20 * MIB * len(photo_sizes)
    volume_gb = max(
        30, minimum_volume_gb,
        math.ceil(1.1 * (input_bytes + reserve) / GIB),
    )
    return InputPlan(
        kind, source, extra.pop("images", ()), input_bytes, reserve, volume_gb,
        frame_target, info, frame_bytes, extract_fps=video_fps, **extra,
    )


def plan_input(
    source: Path, kind: str, minimum_volume_gb: int,
    num_downscales: int, video_fps: float = DEFAULT_VIDEO_FPS,
    more_videos: tuple[Path, ...] = (),
) -> InputPlan:
    if kind == "images":
        images = tuple(list_images(source))
        input_bytes, reserve, volume_gb = storage_plan(list(images), minimum_volume_gb)
        return InputPlan(kind, source, images, input_bytes, reserve, volume_gb)
    if kind != "video":
        raise ValueError(f"Unknown input type: {kind}")
    videos = (source, *more_videos)
    if len({video.resolve() for video in videos}) != len(videos):
        raise ValueError("The same video was picked twice.")
    if len(videos) > MAX_VIDEOS:
        raise ValueError(f"Pick at most {MAX_VIDEOS} videos for one job.")
    infos = [probe_video(video) for video in videos]
    return _plan_video(
        kind, source, infos, sum(video.stat().st_size for video in videos), minimum_volume_gb,
        num_downscales, video_fps, videos=videos if more_videos else (),
    )


def plan_drive_video(
    folder_id: str, folder_name: str, video, results_root: Path,
    minimum_volume_gb: int, num_downscales: int, video_fps: float = DEFAULT_VIDEO_FPS,
) -> InputPlan:
    """Plan a job for one or more videos stored in Google Drive (DriveFiles)."""
    videos = tuple(video) if isinstance(video, (list, tuple)) else (video,)
    if not videos:
        raise ValueError("Pick at least one video.")
    if len(videos) > MAX_VIDEOS:
        raise ValueError(f"Pick at most {MAX_VIDEOS} videos for one job.")
    infos = [
        VideoInfo((item.duration_ms or 0) / 1000, item.width or 0, item.height or 0,
                  getattr(item, "fps", 0.0) or 0.0)
        for item in videos
    ]
    return _plan_video(
        "drive_video",
        Path(results_root).expanduser() / safe_dir_name(Path(videos[0].name).stem),
        infos, sum(item.size for item in videos), minimum_volume_gb, num_downscales,
        video_fps,
        drive_folder_id=folder_id, drive_folder_name=folder_name, drive_files=videos,
    )


def _check_video_count(count: int) -> None:
    if count > MAX_VIDEOS:
        raise ValueError(f"Pick at most {MAX_VIDEOS} videos for one job.")


def plan_mixed(
    folder: Path, photos: tuple[Path, ...], videos: tuple[Path, ...], minimum_volume_gb: int,
    num_downscales: int, video_fps: float = DEFAULT_VIDEO_FPS,
) -> InputPlan:
    """Plan a job for local photos and videos together (both from one folder)."""
    if not photos or not videos:
        raise ValueError("A photos + videos job needs at least one of each.")
    _check_video_count(len(videos))
    if len({video.resolve() for video in videos}) != len(videos):
        raise ValueError("The same video was picked twice.")
    infos = [probe_video(video) for video in videos]
    photo_sizes = tuple(photo.stat().st_size for photo in photos)
    return _plan_video(
        "mixed", folder, infos, sum(photo_sizes) + sum(v.stat().st_size for v in videos),
        minimum_volume_gb, num_downscales, video_fps, photo_sizes,
        images=tuple(photos), videos=tuple(videos),
    )


def plan_drive_mixed(
    folder_id: str, folder_name: str, photos: tuple, videos: tuple, results_root: Path,
    minimum_volume_gb: int, num_downscales: int, video_fps: float = DEFAULT_VIDEO_FPS,
) -> InputPlan:
    """Plan a job for photos and videos from one Google Drive folder (DriveFiles)."""
    if not photos or not videos:
        raise ValueError("A photos + videos job needs at least one of each.")
    _check_video_count(len(videos))
    infos = [
        VideoInfo((item.duration_ms or 0) / 1000, item.width or 0, item.height or 0,
                  getattr(item, "fps", 0.0) or 0.0)
        for item in videos
    ]
    photo_sizes = tuple(item.size for item in photos)
    return _plan_video(
        "drive_mixed", Path(results_root).expanduser() / safe_dir_name(folder_name), infos,
        sum(photo_sizes) + sum(item.size for item in videos), minimum_volume_gb,
        num_downscales, video_fps, photo_sizes,
        drive_folder_id=folder_id, drive_folder_name=folder_name,
        drive_files=tuple(photos), drive_videos=tuple(videos),
    )
