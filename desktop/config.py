from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from storage import VIDEO_EXTENSIONS


# Values that come from .env and must never be frozen into app_state.json.
_NEVER_SAVED = frozenset({"runpod_api_key", "google_service_account_file"})
ENGINES = ("nerfstudio", "spirula")
ENGINE_LABELS = {"nerfstudio": "Nerfstudio", "spirula": "Spirula Studio"}
FLOATER_LEVELS = ("off", "mild", "strong")
GUIDANCE_LEVELS = ("off", "surfaces", "depth")
# Feature matching for Spirula's camera solver.
MATCHERS = ("sift", "aliked", "loma")
MATCHER_LABELS = {"sift": "Standard (SIFT)", "aliked": "Learned (ALIKED + LightGlue)", "loma": "LoMa"}
GUIDANCE_LABELS = {"off": "off", "surfaces": "surfaces", "depth": "surfaces + depth"}
PRIMITIVES = ("3dgs", "3dgut")
# Spirula quality presets: (label, training steps, splat budget).
QUALITY_LEVELS = {
    "low": ("Draft", 15_000, 200_000),
    "medium": ("Standard", 30_000, 1_000_000),
    "high": ("High", 50_000, 3_000_000),
    "ultra": ("Ultra", 80_000, 10_000_000),
}
# Each training step learns from one image. The quality levels' steps suit up
# to about 300 images; above that, steps rise in proportion so every image is
# still seen about as often, up to 2.5x the level's steps. Past that, a lower
# video fps helps more than longer training. The pod applies the same rule to
# the real image count.
STEP_REFERENCE_IMAGES = 300
MAX_STEP_SCALE = 2.5


def quality_steps(quality: str, images: int | None) -> int:
    """Training steps for a quality level and this many images. Draft is never raised."""
    base = QUALITY_LEVELS[quality][1]
    return base if quality == "low" else scaled_steps(base, images)


def scaled_steps(base: int, images: int | None) -> int:
    """Training steps for a quality level's base steps and this many images."""
    if not images or images <= STEP_REFERENCE_IMAGES:
        return base
    wanted = -(-base * images // STEP_REFERENCE_IMAGES // 1000) * 1000
    return min(int(base * MAX_STEP_SCALE), max(base, wanted))

# The Spirula Linux release is built on Ubuntu 24.04 (needs glibc >= 2.38), so
# it cannot run inside the Ubuntu 22.04 based nerfstudio image.
DEFAULT_SPIRULA_IMAGE = "nvidia/cuda:12.6.3-base-ubuntu24.04"
DEFAULT_SPIRULA_VERSION = "2026.9.30"

# Stands for "the photos" in local_unticked / drive_unticked (no video has this name).
PHOTOS_ITEM = "/photos"

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / "app_state.json"
ENV_FILE = APP_DIR / ".env"
# Where Settings copies a chosen Google service-account key.
DRIVE_KEY_FILE = APP_DIR / "service-account.json"
load_dotenv(ENV_FILE)


class DriveKeyError(ValueError):
    pass


def service_account_email(path: Path) -> str:
    """The client_email of a Google service-account key file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DriveKeyError(f"Could not read {Path(path).name} as a key file: {exc}") from exc
    if not isinstance(data, dict) or data.get("type") != "service_account" or not data.get("client_email"):
        raise DriveKeyError(
            f"{Path(path).name} is not a service-account key. Download it from the service "
            "account's Keys tab (Add key → Create new key → JSON)."
        )
    return str(data["client_email"])


def write_env_value(name: str, value: str) -> None:
    """Set one value in .env, creating the file readable by this user only."""
    from dotenv import set_key

    if not ENV_FILE.exists():
        ENV_FILE.touch(mode=0o600)
    set_key(str(ENV_FILE), name, value)
    ENV_FILE.chmod(0o600)


def install_drive_key(source: Path) -> Path:
    """Copy a service-account key into the app folder and point .env at it."""
    source = Path(source).expanduser().resolve()
    service_account_email(source)
    if source != DRIVE_KEY_FILE.resolve():
        data = source.read_bytes()
        DRIVE_KEY_FILE.touch(mode=0o600)
        DRIVE_KEY_FILE.write_bytes(data)
    DRIVE_KEY_FILE.chmod(0o600)
    write_env_value("GOOGLE_SERVICE_ACCOUNT_FILE", DRIVE_KEY_FILE.name)
    return DRIVE_KEY_FILE


@dataclass
class AppConfig:
    runpod_api_key: str = os.getenv("RUNPOD_API_KEY", "")
    runpod_graphql_url: str = os.getenv("RUNPOD_GRAPHQL_URL", "https://api.runpod.io/graphql")
    runpod_image: str = os.getenv("RUNPOD_IMAGE", "ghcr.io/nerfstudio-project/nerfstudio:latest")
    engine: str = os.getenv("ENGINE", "spirula")
    # Google Drive input (read-only service account). The key file stays on this
    # laptop; the pod only ever receives a short-lived access token.
    google_service_account_file: str = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    drive_folder_id: str = ""
    drive_folder_name: str = ""
    # Items left out of the chosen Drive folder: video names, or PHOTOS_ITEM.
    drive_unticked: list = field(default_factory=list)
    results_folder: str = os.getenv("RESULTS_FOLDER", "")
    spirula_image: str = os.getenv("SPIRULA_IMAGE", DEFAULT_SPIRULA_IMAGE)
    spirula_version: str = os.getenv("SPIRULA_VERSION", DEFAULT_SPIRULA_VERSION)
    # Spirula cleanup options; both off reproduces the earlier behaviour.
    spirula_floaters: str = os.getenv("SPIRULA_FLOATERS", "off")
    spirula_distraction: str = os.getenv("SPIRULA_DISTRACTION", "off")
    spirula_guidance: str = os.getenv("SPIRULA_GUIDANCE", "off")
    spirula_depth_weight: float = float(os.getenv("SPIRULA_DEPTH_WEIGHT", "0.05"))
    spirula_primitive: str = os.getenv("SPIRULA_PRIMITIVE", "3dgs")
    spirula_quality: str = os.getenv("SPIRULA_QUALITY", "medium")
    # Model lens darkening at the frame edges (turned on with the fisheye option).
    spirula_vignetting: bool = os.getenv("SPIRULA_VIGNETTING", "0") == "1"
    spirula_object: bool = os.getenv("SPIRULA_OBJECT", "0") == "1"
    spirula_turntable: bool = os.getenv("SPIRULA_TURNTABLE", "0") == "1"
    spirula_main_scene: bool = os.getenv("SPIRULA_MAIN_SCENE", "0") == "1"
    spirula_matcher: str = os.getenv("SPIRULA_MATCHER", "sift")
    # Off: Spirula trains for its quality preset's steps. On: pipeline_max_steps.
    steps_override: bool = os.getenv("STEPS_OVERRIDE", "0") == "1"
    # Fisheye lens on the capture (Spirula only). Defaults: TTArtisan 7.5mm on
    # a Fujifilm APS-C body (23.5 mm sensor width).
    fisheye_lens: bool = os.getenv("FISHEYE_LENS", "0") == "1"
    fisheye_focal_mm: float = float(os.getenv("FISHEYE_FOCAL_MM", "7.5"))
    sensor_width_mm: float = float(os.getenv("SENSOR_WIDTH_MM", "23.5"))
    ssh_private_key: str = os.getenv("SSH_PRIVATE_KEY", str(Path.home() / ".ssh/id_ed25519"))
    input_folder: str = os.getenv("INPUT_FOLDER", "")
    input_kind: str = os.getenv("INPUT_KIND", "local")  # "local" or "drive"
    # Items left out of the chosen local folder: video names, or PHOTOS_ITEM.
    local_unticked: list = field(default_factory=list)
    video_fps: float = float(os.getenv("VIDEO_FPS", "3"))  # frames taken per second of video
    max_gpu_price: float = float(os.getenv("MAX_GPU_PRICE_PER_HOUR", "0.55"))
    min_gpu_vram_gb: int = int(os.getenv("MIN_GPU_VRAM_GB", "24"))
    gpu_count: int = 1
    cloud_type: str = os.getenv("CLOUD_TYPE", "ALL")
    container_disk_gb: int = int(os.getenv("CONTAINER_DISK_GB", "30"))
    volume_gb: int = int(os.getenv("VOLUME_GB", "0"))
    min_vcpu: int = int(os.getenv("MIN_VCPU", "1"))
    min_memory_gb: int = int(os.getenv("MIN_MEMORY_GB", "8"))
    pipeline_max_steps: int = int(os.getenv("PIPELINE_MAX_STEPS", "30000"))
    pipeline_num_downscales: int = int(os.getenv("PIPELINE_NUM_DOWNSCALES", "1"))
    # Jobs are refused (not queued) while this many pods are in use.
    max_concurrent_pods: int = int(os.getenv("MAX_CONCURRENT_PODS", "3"))
    # Show the engine's own training viewer in the app through an SSH tunnel.
    live_preview: bool = os.getenv("LIVE_PREVIEW", "1") == "1"
    # Jobs panel expanded on wide windows (it always collapses when narrow).
    dock_expanded: bool = True

    def load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for key, value in data.items():
            if key in self.__dataclass_fields__ and key not in _NEVER_SAVED:
                setattr(self, key, value)
        if self.engine not in ENGINES:
            self.engine = "nerfstudio"
        self._migrate_input(data)
        if self.spirula_floaters not in FLOATER_LEVELS:
            self.spirula_floaters = "off"
        if self.spirula_distraction not in FLOATER_LEVELS:
            self.spirula_distraction = "off"
        if self.spirula_guidance not in GUIDANCE_LEVELS:
            self.spirula_guidance = "off"
        if self.spirula_primitive not in PRIMITIVES:
            self.spirula_primitive = "3dgs"
        if self.spirula_matcher not in MATCHERS:
            self.spirula_matcher = "sift"
        if self.spirula_quality not in QUALITY_LEVELS:
            self.spirula_quality = "medium"
        try:
            self.video_fps = min(30.0, max(0.2, float(self.video_fps)))
        except (TypeError, ValueError):
            self.video_fps = 3.0

    def _migrate_input(self, data: dict) -> None:
        """Settings saved before the Local / Google Drive picker."""
        kind = data.get("input_kind", self.input_kind)
        if kind == "video" and data.get("input_video"):
            picked = [Path(part.strip()).expanduser()
                      for part in str(data["input_video"]).split(";") if part.strip()]
            self.input_folder = str(picked[0].parent)
            names = {path.name for path in picked}
            try:
                others = [p.name for p in picked[0].parent.iterdir()
                          if p.suffix.lower() in VIDEO_EXTENSIONS and p.name not in names]
            except OSError:
                others = []
            self.local_unticked = [PHOTOS_ITEM, *sorted(others)]
        if kind not in {"local", "drive"}:
            self.input_kind = "local"
        for name in ("local_unticked", "drive_unticked"):
            value = getattr(self, name)
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                setattr(self, name, [])

    def image_for(self, engine: str) -> str:
        if engine == "spirula":
            return self.spirula_image.strip() or DEFAULT_SPIRULA_IMAGE
        if engine == "nerfstudio":
            return self.runpod_image.strip()
        raise ValueError(f"Unknown engine: {engine}")

    def save_state(self) -> None:
        data = asdict(self)
        for key in _NEVER_SAVED:
            data.pop(key, None)
        STATE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def has_runpod(self) -> bool:
        return bool(self.runpod_api_key.strip())

    def drive_credentials_path(self) -> Path | None:
        """Resolve GOOGLE_SERVICE_ACCOUNT_FILE (relative paths are relative to the app folder)."""
        text = self.google_service_account_file.strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            for base in (APP_DIR, Path.cwd()):
                if (base / path).is_file():
                    return (base / path).resolve()
            return (APP_DIR / path).resolve()
        return path

    def results_root(self) -> Path:
        text = self.results_folder.strip()
        return Path(text).expanduser() if text else APP_DIR / "results"