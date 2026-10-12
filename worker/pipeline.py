from __future__ import annotations

import codecs
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
import re
import selectors
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


JOB_ROOT = Path("/workspace/job")
INPUT_DIR = JOB_ROOT / "input"
OUTPUT_DIR = JOB_ROOT / "output"
PROCESSED_DIR = JOB_ROOT / "processed"
TRAIN_OUTPUT_DIR = JOB_ROOT / "nerfstudio_outputs"
EXPORT_DIR = JOB_ROOT / "exports"

MAX_STEPS = int(
    os.getenv("PIPELINE_MAX_STEPS", "30000")
)
NUM_DOWNSCALES = int(
    os.getenv("PIPELINE_NUM_DOWNSCALES", "1")
)
# "nerfstudio" (COLMAP + Splatfacto) or "spirula" (Spirula Studio's built-in
# SfM + trainer). Everything around the engine is shared.
ENGINE = os.getenv("PIPELINE_ENGINE", "nerfstudio")
SPIRULA_VERSION = os.getenv("PIPELINE_SPIRULA_VERSION", "2026.9.30")
# Optional full URL override for the Spirula Linux zip (mirror / pinned build).
SPIRULA_URL = os.getenv("PIPELINE_SPIRULA_URL", "")
SPIRULA_HOME = Path("/workspace/spirula")
# Spirula cleanup options (see the app's Engine & quality card).
# Floater suppression: off | mild | strong.
SPIRULA_FLOATERS = os.getenv("PIPELINE_SPIRULA_FLOATERS", "off")
SPIRULA_DISTRACTION = os.getenv("PIPELINE_SPIRULA_DISTRACTION", "off")
# Object scan: an AI model (BiRefNet, MIT licence) masks the main subject in
# every image, and Spirula trains with its centered-object preset, so the table
# and room are not reconstructed.
SPIRULA_OBJECT = os.getenv("PIPELINE_SPIRULA_OBJECT", "0") == "1"
# Turntable object scan: the object was turned while the camera stayed. The masks are
# made before SfM, which then ignores the (still) background, so the turning object
# reads as a camera circling it.
SPIRULA_TURNTABLE = os.getenv("PIPELINE_SPIRULA_TURNTABLE", "0") == "1"
# Keep to the main scene: a training region from the SfM result, so far background is not
# reconstructed (see _main_scene_region).
SPIRULA_MAIN_SCENE = os.getenv("PIPELINE_SPIRULA_MAIN_SCENE", "0") == "1"
# Feature matching for SfM: sift (Spirula's default), aliked (ALIKED + LightGlue) or loma
# (LoMa-B, DINOv2-based descriptors). The learned ones need checkpoints; they are fetched
# here (checked against Spirula's own sizes and SHA-256s) into the cache Spirula reads, so
# its own downloader, which floods the log with progress bars, never runs.
SPIRULA_MATCHER = os.getenv("PIPELINE_SPIRULA_MATCHER", "sift")
_GH = "https://github.com"
SFM_MATCHERS = {
    "sift": ([], []),
    "aliked": (
        ["--features", "aliked-n16rot", "--matcher", "lightglue"],
        [
            ("aliked-n16rot.onnx", f"{_GH}/colmap/colmap/releases/download/3.13.0/aliked-n16rot.onnx",
             "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547", 2997054),
            ("aliked-lightglue.onnx", f"{_GH}/colmap/colmap/releases/download/3.13.0/aliked-lightglue.onnx",
             "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d", 45804950),
        ],
    ),
    "loma": (
        ["--features", "loma-b", "--matcher", "loma-b"],
        [
            ("loma_detector.onnx", f"{_GH}/davnords/storage/releases/download/loma/loma_detector.onnx",
             "b6af99c5e730034ac9b675d1ebe05d0679af4569a3c26f10a6a50f91e02dc512", 26187830),
            ("loma_descriptor_dedode_g.onnx",
             f"{_GH}/davnords/storage/releases/download/loma/loma_descriptor_dedode_g.onnx",
             "5a7b9eaf7425d4513c5d7feae86080bae7ed3aceae7fb1b9f059d0752e2ad564", 1294905349),
            ("loma_matcher_B.onnx", f"{_GH}/davnords/storage/releases/download/loma/loma_matcher_B.onnx",
             "ba5a2773b29cace19f1240e14e5a080cca3eaf9f69a7adb829a1d470557001c7", 48361254),
        ],
    ),
}
SFM_MATCHER_LABELS = {"sift": "Standard (SIFT)", "aliked": "Learned (ALIKED + LightGlue)", "loma": "LoMa"}
BIREFNET_URL = "https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors"
BIREFNET_BYTES = 444_473_596
# Where Spirula looks for (and would otherwise download) `--model birefnet`.
BIREFNET_CACHE = Path.home() / ".cache" / "spirula-studio" / "models" / "birefnet-general.safetensors"
# The image types `spirula sam track` reads, in the order it numbers them.
SAM_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".exr", ".tif", ".tiff"}
# AI geometry guidance: off | surfaces (normal maps) | depth (normal + depth maps).
SPIRULA_GUIDANCE = os.getenv("PIPELINE_SPIRULA_GUIDANCE", "off")
SPIRULA_DEPTH_WEIGHT = float(os.getenv("PIPELINE_SPIRULA_DEPTH_WEIGHT", "0.05"))
# Live training preview: both engines serve a web viewer on this pod port. The
# app reaches it through an SSH tunnel; nothing is exposed publicly.
PREVIEW = os.getenv("PIPELINE_PREVIEW", "1") == "1"
PREVIEW_PORT = int(os.getenv("PIPELINE_PREVIEW_PORT", "7007"))
# Splat type: 3dgs (standard, opens in every viewer) or 3dgut (exact for
# fisheye/distorted lenses, slower, fewer viewers can open it).
SPIRULA_PRIMITIVE = os.getenv("PIPELINE_SPIRULA_PRIMITIVE", "3dgs")
# Spirula's quality preset sets the splat budget and the training length.
# PIPELINE_STEPS_OVERRIDE=1 keeps the preset's budget but trains for MAX_STEPS.
SPIRULA_QUALITY = os.getenv("PIPELINE_SPIRULA_QUALITY", "medium")
# Model lens vignetting in Spirula's per-photo camera model (PPISP). Spirula's
# default leaves vignetting and the response curve out ("no_crf_no_vig").
SPIRULA_VIGNETTING = os.getenv("PIPELINE_SPIRULA_VIGNETTING", "0") == "1"
STEPS_OVERRIDE = os.getenv("PIPELINE_STEPS_OVERRIDE", "1") == "1"
SPIRULA_PRESETS = {  # read from Spirula 2026.9.30's training configs
    "low": (15_000, 200_000),
    "medium": (30_000, 1_000_000),
    "high": (50_000, 3_000_000),
    "ultra": (80_000, 10_000_000),
}


# Each training step learns from one image. The quality levels' steps suit up
# to about 300 images; above that, steps rise in proportion so every image is
# still seen about as often, up to 2.5x the level's steps. Past that, a lower
# video fps helps more than longer training. The pod applies the same rule to
# the real image count. Kept in step with desktop/config.py.
STEP_REFERENCE_IMAGES = 300
MAX_STEP_SCALE = 2.5


def scaled_steps(base: int, images: int | None) -> int:
    """Training steps for a quality level's base steps and this many images."""
    if not images or images <= STEP_REFERENCE_IMAGES:
        return base
    wanted = -(-base * images // STEP_REFERENCE_IMAGES // 1000) * 1000
    return min(int(base * MAX_STEP_SCALE), max(base, wanted))


def _spirula_steps(images: int | None = None) -> int:
    if STEPS_OVERRIDE:
        return MAX_STEPS
    # Draft is for quick checks: never raised.
    if SPIRULA_QUALITY == "low":
        return SPIRULA_PRESETS["low"][0]
    return scaled_steps(SPIRULA_PRESETS[SPIRULA_QUALITY][0], images)
# Fisheye lens (Spirula only). Manual lenses write no EXIF focal length, so the
# app sends the lens and sensor size and the pod turns them into pixels.
FISHEYE = os.getenv("PIPELINE_FISHEYE", "0") == "1"
FISHEYE_FOCAL_MM = float(os.getenv("PIPELINE_FISHEYE_FOCAL_MM", "7.5"))
SENSOR_WIDTH_MM = float(os.getenv("PIPELINE_SENSOR_WIDTH_MM", "23.5"))
# What Spirula's own meshing and centered-object presets use for
# "one surface per pixel"; applied whenever guidance maps exist.
SPIRULA_DISTORTION_REG = "0.01"
SPIRULA_RUN_NAME = "run"
NVIDIA_VULKAN_HOME = Path("/workspace/nvidia-vulkan")
NVIDIA_SCRATCH_DIR = "/root"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".m4v", ".webm", ".avi"}
SOURCE_TYPE = os.getenv("PIPELINE_SOURCE_TYPE", "images")
VIDEO_FPS = float(os.getenv("PIPELINE_VIDEO_FPS", "0"))  # frames taken per second of video
# Matching method for COLMAP on image sets. "exhaustive" is the most robust for
# small hand-shot sets (under ~300 photos). Use "vocab_tree" for very large sets.
MATCHING_METHOD = os.getenv("PIPELINE_MATCHING_METHOD", "exhaustive")
# Abort before training if fewer than this fraction of images got camera poses.
MIN_REGISTERED_FRACTION = float(
    os.getenv("PIPELINE_MIN_REGISTERED_FRACTION", "0.6")
)
MIN_WORKSPACE_FREE = 5 * 1024 ** 3
MIN_CONTAINER_FREE = 2 * 1024 ** 3


def _low_space() -> str | None:
    for path, minimum in ((JOB_ROOT, MIN_WORKSPACE_FREE), (Path("/"), MIN_CONTAINER_FREE)):
        free = shutil.disk_usage(path).free
        if free < minimum:
            return f"{path} has only {free / 1024**3:.1f} GiB free; stopping before the disk fills."
    return None


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _forward_output(process: subprocess.Popen[bytes], command_name: str) -> None:
    """Forward child output and progress redraws without waiting for a newline."""
    assert process.stdout is not None
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    started = last_log = last_data = time.monotonic()

    def emit(value: str) -> None:
        nonlocal last_log
        line = _ANSI_ESCAPE.sub("", value).replace("\x08", "").strip()
        if line:
            print(line, flush=True)
            last_log = time.monotonic()

    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map():
            for _key, _events in selector.select(timeout=1):
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    pending += decoder.decode(b"", final=True)
                    emit(pending)
                    selector.unregister(process.stdout)
                    break
                last_data = time.monotonic()
                pending += decoder.decode(chunk)
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
            if selector.get_map() and now - last_log >= 15:
                print(
                    f"[worker] {command_name} is still running "
                    f"({int(now - started)}s elapsed).",
                    flush=True,
                )
                last_log = now


def run_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    ok_exit_codes: tuple[int, ...] = (0,),
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(_quote(arg) for arg in args)
    print(f"[cmd] {printable}", flush=True)

    space_error = _low_space()
    if space_error:
        raise RuntimeError(space_error)

    # Some native tools buffer stdout when it is piped through Python.
    command_args = (["stdbuf", "-oL", "-eL", *args]
                    if shutil.which("stdbuf") else args)
    process = subprocess.Popen(
        command_args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"},
        start_new_session=True,
    )
    stop = threading.Event()
    low_space_error: list[str] = []

    def watch_space() -> None:
        while not stop.wait(2):
            try:
                reason = _low_space()
            except OSError as exc:
                reason = f"Could not check pod disk space: {exc}"
            if reason and process.poll() is None:
                low_space_error.append(reason)
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                return

    watcher = threading.Thread(target=watch_space, daemon=True)
    watcher.start()
    try:
        if process.stdout is None:
            raise RuntimeError("Could not read child-process output.")
        _forward_output(process, args[0])
        exit_code = process.wait()
    finally:
        stop.set()
        watcher.join()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            process.wait()
    if low_space_error:
        raise RuntimeError(low_space_error[0])
    if exit_code not in ok_exit_codes:
        raise RuntimeError(
            f"Command failed with exit code {exit_code}: {printable}"
        )


def _quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"



# ---------------------------------------------------------------------------
# Spirula Studio engine
# ---------------------------------------------------------------------------

def _ldconfig_text() -> str:
    try:
        return subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=20,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _spirula_install_packages() -> None:
    libs = _ldconfig_text()
    packages = [
        package for library, package in (
            ("libOpenGL.so.0", "libopengl0"),
            ("libvulkan.so.1", "libvulkan1"),
            ("libgomp.so.1", "libgomp1"),
            # NVIDIA's Vulkan driver (libGLX_nvidia) links against these.
            ("libX11.so.6", "libx11-6"),
            ("libXext.so.6", "libxext6"),
            ("libEGL.so.1", "libegl1"),
        ) if library not in libs
    ]
    if not Path("/etc/ssl/certs/ca-certificates.crt").is_file():
        packages.append("ca-certificates")
    if not shutil.which("vulkaninfo"):
        packages.append("vulkan-tools")
    # `spirula geometry` downloads its AI model with curl.
    if SPIRULA_GUIDANCE != "off" and not shutil.which("curl"):
        packages.append("curl")
    if SOURCE_TYPE in {"video", "mixed"} and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        packages.append("ffmpeg")
    if not packages:
        print("[pipeline] Spirula system libraries already present", flush=True)
        return
    print(f"[pipeline] installing system packages: {' '.join(packages)}", flush=True)
    run_command(["apt-get", "update"])
    run_command([
        "env", "DEBIAN_FRONTEND=noninteractive",
        "apt-get", "install", "-y", "--no-install-recommends", *packages,
    ])


def _spirula_download_url() -> str:
    if SPIRULA_URL:
        return SPIRULA_URL
    if not re.fullmatch(r"[0-9][0-9A-Za-z._-]*", SPIRULA_VERSION):
        raise RuntimeError(f"Invalid Spirula version: {SPIRULA_VERSION!r}")
    return (
        "https://github.com/harry7557558/spirula-studio/releases/download/"
        f"v{SPIRULA_VERSION}/spirula-{SPIRULA_VERSION}-ubuntu-vulkan-x86_64.zip"
    )


def _find_spirula_binary(root: Path) -> Path | None:
    for path in sorted(root.rglob("spirula")):
        if path.is_file() and not path.is_symlink():
            return path
    return None


def _install_spirula() -> Path:
    target = SPIRULA_HOME / re.sub(r"[^0-9A-Za-z._-]", "_", SPIRULA_URL or SPIRULA_VERSION)[-80:]
    existing = _find_spirula_binary(target) if target.is_dir() else None
    if existing is None:
        url = _spirula_download_url()
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "spirula.zip.part"
        print(f"[pipeline] downloading Spirula Studio: {url}", flush=True)
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "gaussian-studio-worker"})
                with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as out:
                    total = int(response.headers.get("Content-Length") or 0)
                    done = 0
                    next_report = 25 * 1024 ** 2
                    while True:
                        chunk = response.read(1024 ** 2)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            print(
                                f"[pipeline] Spirula download {done / 1024**2:.0f}"
                                + (f" / {total / 1024**2:.0f}" if total else "") + " MiB",
                                flush=True,
                            )
                            next_report += 25 * 1024 ** 2
                if total and done != total:
                    raise RuntimeError(f"incomplete download ({done} of {total} bytes)")
                last_error = None
                break
            except Exception as exc:  # network errors are retried
                last_error = exc
                print(f"[pipeline] Spirula download attempt {attempt} failed: {exc}", flush=True)
                time.sleep(3 * attempt)
        if last_error is not None:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"Could not download Spirula Studio from {url}: {last_error}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(target)
        archive.unlink(missing_ok=True)
        existing = _find_spirula_binary(target)
        if existing is None:
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError("The downloaded Spirula archive does not contain a 'spirula' binary.")
    existing.chmod(0o755)
    # A missing shared library is the usual failure on a minimal image; say so
    # before spending time on SfM.
    try:
        linked = subprocess.run(
            ["ldd", str(existing)], capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        linked = ""
    missing = [line.split()[0] for line in linked.splitlines() if "not found" in line]
    if missing:
        raise RuntimeError(
            "Spirula cannot start because these libraries are missing in the pod image: "
            + ", ".join(missing)
            + ". The image must be Ubuntu 24.04 or newer."
        )
    print(f"[pipeline] Spirula binary ready: {existing}", flush=True)
    return existing


def _vulkan_probe() -> tuple[list[str], str]:
    """Return (hardware Vulkan device names, raw vulkaninfo text)."""
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=90,
            env={**os.environ, "VK_LOADER_DEBUG": "error"},
        )
    except subprocess.TimeoutExpired:
        # Some hosts' drivers hang here instead of failing; treat it as no
        # device so the other driver setups below still get their turn.
        print("[pipeline] vulkaninfo hung for 90 s with this driver setup", flush=True)
        return [], "vulkaninfo timed out after 90 seconds"
    text = f"{result.stdout}\n{result.stderr}"
    names = [match.strip() for match in re.findall(r"deviceName\s*=\s*(.+)", text)]
    hardware = [n for n in names if not re.search(r"llvmpipe|lavapipe|swiftshader|software", n, re.I)]
    return hardware, text


# Only the GL/Vulkan half of the driver. libcuda and friends stay the host's.
_NVIDIA_VULKAN_LIBS = (
    "libGLX_nvidia", "libEGL_nvidia", "libnvidia-glcore", "libnvidia-glsi",
    "libnvidia-eglcore", "libnvidia-glvkspirv", "libnvidia-rtcore", "libnvidia-gpucomp",
    "libnvidia-tls", "libnvidia-allocator",
)
_NVIDIA_SONAME_LINKS = {"libGLX_nvidia": "0", "libEGL_nvidia": "0", "libnvidia-allocator": "1"}
# Both libraries export the Vulkan ICD entry points. libGLX_nvidia (what the
# stock nvidia_icd.json names) initialises X11 first and fails on headless
# hosts with "Could not get 'vkCreateInstance' via 'vk_icdGetInstanceProcAddr'";
# libEGL_nvidia has no X11 dependency, so it is tried first.
_NVIDIA_ICD_LIBRARIES = ("libEGL_nvidia.so.0", "libGLX_nvidia.so.0")
VULKAN_ICD_DIR = Path("/root/gs-vulkan-icd")


def _try_vulkan_icd(library: str, api_version: str = "1.3.0") -> tuple[list[str], str]:
    """Point the Vulkan loader at one NVIDIA library and probe for a device.

    The loader settings stay in os.environ (so Spirula inherits them) only
    when a hardware device shows up.
    """
    VULKAN_ICD_DIR.mkdir(parents=True, exist_ok=True)
    icd = VULKAN_ICD_DIR / (re.sub(r"[^0-9A-Za-z]+", "_", library).strip("_") + ".json")
    icd.write_text(
        json.dumps({
            "file_format_version": "1.0.1",
            "ICD": {"library_path": library, "api_version": api_version},
        }),
        encoding="utf-8",
    )
    saved = {key: os.environ.get(key) for key in ("VK_ICD_FILENAMES", "VK_DRIVER_FILES")}
    os.environ["VK_ICD_FILENAMES"] = os.environ["VK_DRIVER_FILES"] = str(icd)
    hardware, text = _vulkan_probe()
    if hardware:
        print(f"[pipeline] Vulkan works through {library}", flush=True)
    else:
        print(f"[pipeline] no Vulkan device through {library}", flush=True)
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return hardware, text


def _nvidia_driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=30,
    )
    lines = result.stdout.strip().splitlines() if result.returncode == 0 else []
    version = lines[0].strip() if lines else ""
    if not re.fullmatch(r"\d+(\.\d+){1,2}", version):
        raise RuntimeError(f"Could not read the NVIDIA driver version (got {version!r}).")
    return version


def _download_file(url: str, destination: Path, label: str) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "gaussian-studio-worker"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        next_report = 100 * 1024 ** 2
        while True:
            chunk = response.read(1024 ** 2)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if done >= next_report:
                print(
                    f"[pipeline] {label} download {done / 1024**2:.0f}"
                    + (f" / {total / 1024**2:.0f}" if total else "") + " MiB",
                    flush=True,
                )
                next_report += 100 * 1024 ** 2
    if total and done != total:
        raise RuntimeError(f"incomplete download ({done} of {total} bytes)")


def _install_matching_nvidia_vulkan() -> tuple[Path, str]:
    """Fetch NVIDIA's own GL/Vulkan libraries at exactly the host driver version.

    Containers often get a partial or mismatched set of graphics libraries from
    the host. NVIDIA's .run installer can be unpacked without installing
    anything, which gives a complete, version-matched set. Returns the library
    directory (already on LD_LIBRARY_PATH) and the driver's Vulkan API version.
    """
    version = _nvidia_driver_version()
    libdir = NVIDIA_VULKAN_HOME / version
    icd = libdir / "nvidia_icd.json"
    if not icd.is_file():
        print(f"[pipeline] fetching NVIDIA {version} userspace libraries for Vulkan", flush=True)
        shutil.rmtree(libdir, ignore_errors=True)
        libdir.mkdir(parents=True)
        # Unpack on the container disk: it is much faster than the network
        # volume, and the 1.4 GiB of extracted files are thrown away anyway.
        scratch = Path(tempfile.mkdtemp(prefix="nvidia-", dir=NVIDIA_SCRATCH_DIR))
        installer = scratch / f"NVIDIA-Linux-x86_64-{version}.run"
        extract = scratch / "extract"
        errors = []
        for url in (
            f"https://download.nvidia.com/XFree86/Linux-x86_64/{version}/{installer.name}",
            f"https://us.download.nvidia.com/tesla/{version}/{installer.name}",
        ):
            try:
                _download_file(url, installer, "NVIDIA driver")
                break
            except Exception as exc:
                errors.append(f"{url}: {exc}")
                installer.unlink(missing_ok=True)
        else:
            shutil.rmtree(scratch, ignore_errors=True)
            raise RuntimeError("could not download the NVIDIA driver package: " + "; ".join(errors))
        api_version = "1.3.0"
        try:
            # The installer pipes into `tar xvf - 2>/dev/null`. As root, tar
            # tries to restore the archive's file owners; where that is refused
            # it fails, and the installer just prints "Extraction failed." and
            # kills itself with SIGTERM. Keep the extracting user as owner.
            run_command(
                ["sh", str(installer), "--extract-only", "--target", str(extract)],
                env={"TAR_OPTIONS": "--no-same-owner --no-same-permissions"},
            )
            for prefix in _NVIDIA_VULKAN_LIBS:
                for source in extract.glob(prefix + ".so*"):
                    if source.is_file() and not source.is_symlink():
                        shutil.copy2(source, libdir / source.name)
            try:
                shipped = json.loads((extract / "nvidia_icd.json").read_text(encoding="utf-8"))
                api_version = str(shipped["ICD"]["api_version"]) or api_version
            except (OSError, ValueError, KeyError, TypeError):
                pass
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        for prefix, soname in _NVIDIA_SONAME_LINKS.items():
            real = sorted(p for p in libdir.glob(prefix + ".so.*") if not p.is_symlink())
            link = libdir / f"{prefix}.so.{soname}"
            if real and not link.exists():
                link.symlink_to(real[0].name)
        if not (libdir / "libGLX_nvidia.so.0").exists():
            shutil.rmtree(libdir, ignore_errors=True)
            raise RuntimeError("the NVIDIA driver package did not contain libGLX_nvidia")
        icd.write_text(
            json.dumps({
                "file_format_version": "1.0.1",
                "ICD": {"library_path": str(libdir / "libGLX_nvidia.so.0"), "api_version": api_version},
            }),
            encoding="utf-8",
        )
    os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    for library in _NVIDIA_ICD_LIBRARIES:
        missing = _missing_libraries(libdir / library)
        if missing:
            print(f"[pipeline] warning: {library} still lacks: {', '.join(missing)}", flush=True)
    print(f"[pipeline] NVIDIA {version} Vulkan libraries ready in {libdir}", flush=True)
    try:
        api_version = json.loads(icd.read_text(encoding="utf-8"))["ICD"]["api_version"]
    except (OSError, ValueError, KeyError, TypeError):
        api_version = "1.3.0"
    return libdir, str(api_version)


def _missing_libraries(library: Path) -> list[str]:
    try:
        linked = subprocess.run(
            ["ldd", str(library)], capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.split()[0] for line in linked.splitlines() if "not found" in line]


def _ensure_vulkan_gpu() -> None:
    """Fail early (and cheaply) if the pod exposes no hardware Vulkan device."""
    icd_dirs = [Path("/etc/vulkan/icd.d"), Path("/usr/share/vulkan/icd.d"),
                Path("/usr/local/share/vulkan/icd.d")]
    has_icd = any(d.is_dir() and any(d.glob("nvidia_icd*.json")) for d in icd_dirs)
    if not has_icd and "libGLX_nvidia.so.0" in _ldconfig_text():
        icd_dirs[0].mkdir(parents=True, exist_ok=True)
        (icd_dirs[0] / "nvidia_icd.json").write_text(
            json.dumps({
                "file_format_version": "1.0.0",
                "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.0"},
            }),
            encoding="utf-8",
        )
        print("[pipeline] registered the NVIDIA Vulkan ICD", flush=True)
    if not shutil.which("vulkaninfo"):
        print("[pipeline] vulkaninfo unavailable; skipping the Vulkan device check", flush=True)
        return

    hardware, text = _vulkan_probe()
    matched = False
    if not hardware:
        # 1) The host's libraries are already mounted; only the ICD entry
        #    point may be wrong (GLX needs X11, EGL does not).
        print("[pipeline] the default Vulkan driver setup found no GPU; trying NVIDIA's headless (EGL) entry point", flush=True)
        if "libEGL_nvidia.so.0" in _ldconfig_text():
            hardware, text = _try_vulkan_icd("libEGL_nvidia.so.0")
    if not hardware:
        # 2) Version-matched copies of the whole GL/Vulkan half of the driver.
        print("[pipeline] switching to version-matched NVIDIA libraries", flush=True)
        try:
            libdir, api_version = _install_matching_nvidia_vulkan()
            matched = True
            for library in _NVIDIA_ICD_LIBRARIES:
                hardware, text = _try_vulkan_icd(str(libdir / library), api_version)
                if hardware:
                    break
        except Exception as exc:
            print(f"[pipeline] could not set up matching NVIDIA libraries: {exc}", flush=True)
    for name in hardware:
        print(f"[pipeline] Vulkan device: {name}", flush=True)
    if not hardware:
        listing = subprocess.run(
            "echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; "
            "ls /dev/nvidia* 2>&1 | head -12; "
            "ls /usr/lib/x86_64-linux-gnu | grep -i nvidia | grep -v '\\.so\\.[0-9]$'; "
            "for d in /etc/vulkan/icd.d /usr/share/vulkan/icd.d; do ls $d/* 2>/dev/null; done; "
            "ls -l /dev/dri 2>&1 | head -6; "
            "echo DISPLAY=$DISPLAY XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR; "
            f"for f in {NVIDIA_VULKAN_HOME}/*/libGLX_nvidia.so.0 {NVIDIA_VULKAN_HOME}/*/libEGL_nvidia.so.0; do "
            "[ -e $f ] && ldd $f 2>&1 | grep -i 'not found' | sed \"s#^#$f: #\"; done; "
            "VK_LOADER_DEBUG=error,warn timeout 60 vulkaninfo --summary 2>&1 | grep -iE 'error|warn|fail' | sort -u | head -15",
            shell=True, capture_output=True, text=True, timeout=120,
        ).stdout
        print(f"[pipeline] diagnostics:\n{listing}", flush=True)
        detail = " ".join(text.split())[:400]
        tried = ("even with version-matched NVIDIA libraries" if matched
                 else "and version-matched NVIDIA libraries could not be set up")
        raise RuntimeError(
            f"No hardware Vulkan GPU is visible inside this pod ({tried}), so Spirula "
            f"cannot train. vulkaninfo said: {detail or 'nothing'}. "
            "Switch the engine back to Nerfstudio or try a different GPU/host."
        )


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1 for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _registered_images(dataset: Path) -> int:
    """Registered camera count of the largest sparse model (what training picks)."""
    best = 0
    sparse = dataset / "sparse"
    if not sparse.is_dir():
        return 0
    for model in sparse.iterdir():
        if not model.is_dir():
            continue
        binary, text = model / "images.bin", model / "images.txt"
        if binary.is_file():
            with binary.open("rb") as handle:
                head = handle.read(8)
            if len(head) == 8:
                best = max(best, struct.unpack("<Q", head)[0])
        elif text.is_file():
            lines = [line for line in text.read_text(encoding="utf-8", errors="replace").splitlines()
                     if line.strip() and not line.startswith("#")]
            best = max(best, len(lines) // 2)
    return best


# ---- "Keep to the main scene": a training region from the SfM result -------------------
# Spirula trains only inside <dataset>/roi/<name>.json when one exists: splats outside barely
# grow and the photos are masked to what they show of the region, so nothing is spent on far
# background. The shapes follow Spirula's own region editor ("Box around the scene",
# "Cylinder around the subject"), computed here because they are GUI-only.

def _largest_model(dataset: Path) -> Path | None:
    best, best_count = None, -1
    for model in (dataset / "sparse").iterdir() if (dataset / "sparse").is_dir() else []:
        binary = model / "images.bin"
        if binary.is_file():
            with binary.open("rb") as handle:
                count = struct.unpack("<Q", handle.read(8))[0]
            if count > best_count:
                best, best_count = model, count
    return best


def _read_sparse(model: Path):
    """Camera centres, viewing and down directions, and well-seen 3D points (COLMAP binary)."""
    cameras = []
    with (model / "images.bin").open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<i7di", handle.read(64))
            while handle.read(1) not in (b"\0", b""):
                pass
            (points2d,) = struct.unpack("<Q", handle.read(8))
            handle.seek(points2d * 24, 1)
            # world-to-camera rotation from the quaternion; camera axes in world are its rows
            r = [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ]
            t = (tx, ty, tz)
            centre = tuple(-sum(r[j][i] * t[j] for j in range(3)) for i in range(3))
            cameras.append((centre, tuple(r[2]), tuple(r[1])))  # forward = +z, down = +y
    points = []
    with (model / "points3D.bin").open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            _, x, y, z, _, _, _, _, track = struct.unpack("<Q3d3BdQ", handle.read(51))
            handle.seek(track * 8, 1)
            if track >= 3:
                points.append((x, y, z))
    return cameras, points


def _norm(v):
    length = sum(c * c for c in v) ** 0.5
    return tuple(c / length for c in v) if length > 0 else v


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _solve3(m, v):
    """Solve a 3x3 system by Cramer's rule; None when it is singular."""
    def det(a):
        return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
                - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
                + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    d = det(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for col in range(3):
        a = [list(row) for row in m]
        for row in range(3):
            a[row][col] = v[row]
        out.append(det(a) / d)
    return tuple(out)


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q / 100 * (len(ordered) - 1)))))]


def _subject_centre(cameras, up):
    """Where the cameras' viewing directions meet, if the capture circles something there."""
    m = [[0.0] * 3 for _ in range(3)]
    v = [0.0] * 3
    for centre, forward, _ in cameras:
        for i in range(3):
            for j in range(3):
                proj = (1.0 if i == j else 0.0) - forward[i] * forward[j]
                m[i][j] += proj
                v[i] += proj * centre[j]
    focus = _solve3(m, v)
    if focus is None:
        return None
    facing = sum(1 for c, f, _ in cameras if _dot(_norm(_sub(focus, c)), f) > 0.5)
    if facing < 0.8 * len(cameras):
        return None
    # around it with no gap over 180 degrees (seen from above)
    import math
    e1 = _norm(_cross(up, (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)))
    e2 = _cross(up, e1)
    angles = sorted(math.atan2(_dot(_sub(c, focus), e2), _dot(_sub(c, focus), e1)) for c, _, _ in cameras)
    gaps = [b - a for a, b in zip(angles, angles[1:])] + [angles[0] + 2 * math.pi - angles[-1]]
    return focus if max(gaps) <= math.pi else None


def _main_scene_region(dataset: Path) -> tuple[dict, str] | None:
    model = _largest_model(dataset)
    if model is None:
        return None
    cameras, points = _read_sparse(model)
    if len(cameras) < 3 or len(points) < 50:
        return None
    # up: opposite of the cameras' average down direction (people hold cameras upright)
    up = _norm(tuple(-sum(d[i] for _, _, d in cameras) for i in range(3)))
    focus = _subject_centre(cameras, up)
    if focus is not None:
        # a capture that circles a subject: an upright cylinder around it
        import statistics
        radius = 0.7 * statistics.median(
            (_dot(_sub(c, focus), _sub(c, focus)) - _dot(_sub(c, focus), up) ** 2) ** 0.5 for c, _, _ in cameras)
        inside = [_dot(_sub(p, focus), up) for p in points
                  if (_dot(_sub(p, focus), _sub(p, focus)) - _dot(_sub(p, focus), up) ** 2) <= radius * radius]
        if len(inside) < 20:
            return None
        low, high = _percentile(inside, 1), _percentile(inside, 99)
        pad = 0.05 * (high - low)
        low, high = low - pad, high + pad
        e1 = _norm(_cross(up, (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)))
        e2 = _cross(up, e1)
        centre = tuple(focus[i] + up[i] * (low + high) / 2 for i in range(3))
        region = {"type": "cylinder", "center": list(centre), "half": [radius, radius, (high - low) / 2],
                  "rotation": [*e1, *e2, *up]}
        return region, f"cylinder around the subject, {2 * radius:.2f} wide and {high - low:.2f} tall"
    # otherwise a box: 1st..99th percentile along the main horizontal direction, across it and up
    mean = tuple(sum(p[i] for p in points) / len(points) for i in range(3))
    flat = [_sub(_sub(p, mean), tuple(up[i] * _dot(_sub(p, mean), up) for i in range(3))) for p in points]
    cov = [[sum(f[i] * f[j] for f in flat) / len(flat) for j in range(3)] for i in range(3)]
    e1 = _norm(_cross(up, (1.0, 0.0, 0.0) if abs(up[0]) < 0.9 else (0.0, 1.0, 0.0)))
    for _ in range(50):  # power iteration for the main horizontal direction
        e1 = _norm(tuple(sum(cov[i][j] * e1[j] for j in range(3)) for i in range(3)))
        e1 = _norm(_sub(e1, tuple(up[i] * _dot(e1, up) for i in range(3))))
    e2 = _cross(up, e1)
    sides, centre_local = [], []
    for axis in (e1, e2, up):
        coords = [_dot(p, axis) for p in points]
        low, high = _percentile(coords, 1), _percentile(coords, 99)
        pad = 0.05 * (high - low)
        sides.append((high - low) / 2 + pad)
        centre_local.append((low + high) / 2)
    centre = tuple(sum(centre_local[k] * (e1, e2, up)[k][i] for k in range(3)) for i in range(3))
    region = {"type": "box", "center": list(centre), "half": sides, "rotation": [*e1, *e2, *up]}
    return region, f"box around the scene, {2 * sides[0]:.2f} x {2 * sides[1]:.2f} x {2 * sides[2]:.2f}"


def _write_main_scene_region(dataset: Path) -> str | None:
    """Write <dataset>/roi/main-scene.json; returns its description, or None (no region)."""
    try:
        found = _main_scene_region(dataset)
    except (OSError, struct.error, ValueError, ZeroDivisionError) as exc:
        print(f"[pipeline] WARNING: could not work out the main scene ({exc}); training everything.",
              flush=True)
        return None
    if found is None:
        print("[pipeline] WARNING: not enough of a reconstruction to find the main scene; "
              "training everything.", flush=True)
        return None
    region, description = found
    (dataset / "roi").mkdir(exist_ok=True)
    (dataset / "roi" / "main-scene.json").write_text(json.dumps(region, indent=1), encoding="utf-8")
    print(f"[pipeline] keeping to the main scene: {description} (scene units)", flush=True)
    return description


def _video_frame_plan(video: Path, extract_fps: float) -> tuple[int, int]:
    """Take every Nth frame of the whole video: returns (N, frames that gives)."""
    duration, fps = _video_timing(video)
    step = max(1, round(fps / extract_fps))
    available = max(1, int(duration * fps))
    return step, -(-available // step)


def _video_timing(video: Path) -> tuple[float, float]:
    """Duration in seconds and frame rate of a video, from ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=avg_frame_rate,r_frame_rate,duration:format=duration",
            "-of", "json", str(video),
        ],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe could not read the video: {result.stderr.strip()}")
    info = json.loads(result.stdout)
    stream = info["streams"][0]

    def positive(values, rational=False):
        from fractions import Fraction
        for raw in values:
            try:
                value = float(Fraction(raw)) if rational else float(raw)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
            if value > 0:
                return value
        raise RuntimeError("Video has no usable duration or frame rate.")

    duration = positive((info.get("format", {}).get("duration"), stream.get("duration")))
    fps = positive((stream.get("avg_frame_rate"), stream.get("r_frame_rate")), rational=True)
    return duration, fps


SPIRULA_RESULT: dict = {}


def _image_size(path: Path) -> tuple[int, int] | None:
    """Width and height of a JPEG or PNG, read from its header (no imaging library)."""
    try:
        with path.open("rb") as handle:
            head = handle.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] != b"\xff\xd8":
                return None
            handle.seek(2)
            while True:
                marker = handle.read(2)
                if len(marker) < 2 or marker[0] != 0xFF:
                    return None
                if marker[1] in (0xD8, 0x01) or 0xD0 <= marker[1] <= 0xD7:
                    continue
                length = struct.unpack(">H", handle.read(2))[0]
                # SOF0..SOF15 except DHT (C4), JPG (C8) and DAC (CC) carry the size.
                if 0xC0 <= marker[1] <= 0xCF and marker[1] not in (0xC4, 0xC8, 0xCC):
                    height, width = struct.unpack(">HH", handle.read(5)[1:5])
                    return width, height
                handle.seek(length - 2, 1)
    except (OSError, struct.error):
        return None


def _fisheye_focal(images_dir: Path, label: str = "") -> float:
    """Starting focal length in pixels for the fisheye lens, from one image's size."""
    sample = next(
        (path for path in sorted(images_dir.rglob("*"))
         if path.suffix.lower() in IMAGE_EXTENSIONS and _image_size(path)),
        None,
    )
    if sample is None:
        raise RuntimeError("Could not read the size of any image for the fisheye lens setup.")
    width, height = _image_size(sample)
    # The long side of the photo spans the sensor width (also for 4K frames,
    # which the X-T20 reads from the full sensor width).
    focal_px = FISHEYE_FOCAL_MM / SENSOR_WIDTH_MM * max(width, height)
    print(
        f"[pipeline] fisheye lens: {FISHEYE_FOCAL_MM:g} mm on a {SENSOR_WIDTH_MM:g} mm-wide "
        f"sensor, {width}x{height} images{label}, starting focal length {focal_px:.0f} px",
        flush=True,
    )
    return focal_px


def _fisheye_sfm_args(images_dir: Path, groups: list[str] = ()) -> list[str]:
    """Spirula SfM flags for a fisheye lens, with the focal length in pixels.

    With groups (photos + videos), every sub-folder is its own camera and
    gets its own starting focal, since photos and video frames differ in size.
    """
    if not groups:
        focal_px = _fisheye_focal(images_dir)
        SPIRULA_RESULT["fisheye_focal_px"] = round(focal_px, 1)
        return [
            "--camera-model", "opencv-fisheye",
            "--camera-mode", "single",
            "--focal", f"{focal_px:.1f}",
        ]
    args = ["--camera-model", "opencv-fisheye", "--camera-mode", "folder"]
    focals = {}
    for group in groups:
        focals[group] = round(_fisheye_focal(images_dir / group, f" in {group}"), 1)
        args += ["--focal", f"{group}={focals[group]:.1f}"]
    SPIRULA_RESULT["fisheye_focal_px"] = focals
    return args


def _spirula_geometry(spirula: Path, dataset: Path) -> str:
    """Estimate per-frame surface directions (and depth) with Spirula's AI models.

    Returns the guidance level that is actually in place. A failure here is not
    fatal: training continues without guidance and the summary says so.
    """
    with_depth = SPIRULA_GUIDANCE == "depth"
    print(
        "[pipeline] estimating surface directions"
        + (" and depth" if with_depth else "")
        + " for every frame (AI geometry guidance)",
        flush=True,
    )
    started = time.time()
    try:
        run_command(
            [str(spirula), "geometry", str(dataset), *(["--depth"] if with_depth else [])],
            cwd=JOB_ROOT,
        )
    except RuntimeError as exc:
        for name in ("normals", "depths"):
            shutil.rmtree(dataset / name, ignore_errors=True)
        print(
            f"[pipeline] WARNING: geometry guidance failed ({exc}); training WITHOUT it. "
            "This run is not a fair comparison with a guided one.",
            flush=True,
        )
        return "off"
    normals = _count_images(dataset / "normals")
    depths = _count_images(dataset / "depths")
    print(
        f"[pipeline] geometry guidance ready in {time.time() - started:.0f}s: "
        f"{normals} normal maps" + (f", {depths} depth maps" if with_depth else ""),
        flush=True,
    )
    if normals == 0:
        print("[pipeline] WARNING: no normal maps were written; training WITHOUT guidance.",
              flush=True)
        return "off"
    if with_depth and depths == 0:
        print("[pipeline] WARNING: no depth maps were written; using surface guidance only.",
              flush=True)
        return "surfaces"
    return SPIRULA_GUIDANCE


def _spirula_model_dir() -> Path:
    """Where Spirula keeps fetched checkpoints ($XDG_CACHE_HOME or ~/.cache)."""
    root = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(root) / "spirula-studio" / "models"


def _ensure_sfm_models(matcher: str) -> None:
    """Fetch and verify the checkpoints a learned matcher needs, once per pod."""
    files = SFM_MATCHERS[matcher][1]
    folder = _spirula_model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    missing = [f for f in files if not ((folder / f[0]).is_file() and (folder / f[0]).stat().st_size == f[3])]
    if not missing:
        return
    started = time.time()
    total = sum(f[3] for f in missing)
    print(f"[pipeline] downloading the {SFM_MATCHER_LABELS[matcher]} models "
          f"({total / 1024**2:,.0f} MiB)", flush=True)
    for name, url, sha256, size in missing:
        partial = folder / (name + ".part")
        _download_file(url, partial, name)
        digest = hashlib.sha256()
        with partial.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
                digest.update(chunk)
        if partial.stat().st_size != size or digest.hexdigest() != sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"{name} did not download correctly (size or checksum differs).")
        partial.replace(folder / name)
    print(f"[pipeline] matching models ready in {time.time() - started:.0f}s", flush=True)


def _ensure_birefnet() -> None:
    """Download the subject-masking model once per pod (Spirula's cache path)."""
    if BIREFNET_CACHE.is_file() and BIREFNET_CACHE.stat().st_size == BIREFNET_BYTES:
        return
    BIREFNET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    partial = BIREFNET_CACHE.with_suffix(".part")
    started = time.time()
    print(f"[pipeline] downloading the subject-masking model (BiRefNet, "
          f"{BIREFNET_BYTES / 1024**2:.0f} MiB)", flush=True)
    _download_file(BIREFNET_URL, partial, "BiRefNet")
    if partial.stat().st_size != BIREFNET_BYTES:
        raise RuntimeError(f"BiRefNet download has the wrong size ({partial.stat().st_size} bytes)")
    partial.replace(BIREFNET_CACHE)
    print(f"[pipeline] BiRefNet ready in {time.time() - started:.0f}s", flush=True)


def _spirula_object_masks(spirula: Path, dataset: Path) -> int:
    """Mask the subject of every training image into <dataset>/masks/<rel>/<stem>.png.

    White is kept, black is cut away. Returns how many masks were written; a
    failure is not fatal (training then sees the whole image) and returns 0.
    """
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    folders = sorted({path.parent for path in images_dir.rglob("*")
                      if path.is_file() and path.suffix.lower() in SAM_IMAGE_EXTENSIONS})
    started = time.time()
    try:
        _ensure_birefnet()
        written = 0
        for folder in folders:
            images = sorted((path for path in folder.iterdir()
                             if path.is_file() and path.suffix.lower() in SAM_IMAGE_EXTENSIONS),
                            key=str)
            scratch = JOB_ROOT / "mask_scratch"
            shutil.rmtree(scratch, ignore_errors=True)
            print(f"[pipeline] cutting the subject out of {len(images)} images"
                  + (f" in {folder.relative_to(images_dir)}" if folder != images_dir else ""),
                  flush=True)
            run_command(
                [str(spirula), "sam", "track", "--model", str(BIREFNET_CACHE),
                 "--frames", str(folder), "--out", str(scratch)],
                cwd=JOB_ROOT,
            )
            # `track` numbers its masks frame_00000.png... in sorted path order.
            produced = sorted(scratch.glob("frame_*.png"))
            if len(produced) != len(images):
                raise RuntimeError(f"{len(produced)} masks for {len(images)} images")
            target = masks_dir / folder.relative_to(images_dir)
            target.mkdir(parents=True, exist_ok=True)
            for mask, image in zip(produced, images):
                mask.replace(target / (image.stem + ".png"))
            written += len(images)
        shutil.rmtree(JOB_ROOT / "mask_scratch", ignore_errors=True)
    except RuntimeError as exc:
        shutil.rmtree(masks_dir, ignore_errors=True)
        print(f"[pipeline] WARNING: subject masking failed ({exc}); training on the whole "
              "images, so the background will be reconstructed too.", flush=True)
        return 0
    print(f"[pipeline] {written} subject masks ready in {time.time() - started:.0f}s",
          flush=True)
    return written


def _link_photos(destination: Path) -> None:
    """Spirula expects <dataset>/images. Hard links cost no extra disk space."""
    for path in sorted(INPUT_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            target = destination / path.name
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def _input_photo_count() -> int:
    return sum(1 for path in INPUT_DIR.iterdir()
               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _run_spirula(videos: list[Path]) -> tuple[int, int, Path, float]:
    """Spirula frame extraction/SfM + training; returns the same tuple as Nerfstudio."""
    print("[pipeline] verifying runtime", flush=True)
    if shutil.which("nvidia-smi"):
        run_command(["nvidia-smi", "-L"])
    _spirula_install_packages()
    spirula = _install_spirula()
    _ensure_vulkan_gpu()

    dataset = PROCESSED_DIR
    images_dir = dataset / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sequences: list[str] = []
    camera_groups: list[str] = []

    if SOURCE_TYPE == "mixed":
        # Photos and video frames come from different cameras (or modes), so
        # each gets its own folder, and SfM fits one camera per folder.
        photos_dir = images_dir / "photos"
        photos_dir.mkdir()
        _link_photos(photos_dir)
        camera_groups.append("photos")
    if SOURCE_TYPE in {"video", "mixed"}:
        # Several videos: each gets its own sub-folder, and SfM is told every
        # folder is one continuous sequence so it only chains frames within it
        # (the videos are still matched against each other by appearance).
        for index, video in enumerate(videos, start=1):
            destination = images_dir
            if len(videos) > 1 or SOURCE_TYPE == "mixed":
                camera_groups.append(f"v{index}")
                sequences.append(f"v{index}")
                destination = images_dir / f"v{index}"
                destination.mkdir()
            skip, frames = _video_frame_plan(video, VIDEO_FPS)
            print(
                f"[pipeline] extracting about {frames} frames at {VIDEO_FPS:g} fps "
                + (f"from video {index} of {len(videos)} " if len(videos) > 1 else "")
                + f"(every {skip} source frames)",
                flush=True,
            )
            run_command(
                [
                    str(spirula), "sam", "extract", str(video),
                    "-o", str(destination), "-s", str(skip), "--decoder", "ffmpeg",
                ],
                cwd=JOB_ROOT,
            )
        if SOURCE_TYPE == "mixed":
            print("[pipeline] reconstructing the photos and frames together with Spirula SfM",
                  flush=True)
            data_type = "individual"
        else:
            print("[pipeline] reconstructing the frames with Spirula SfM", flush=True)
            data_type = "video"
    else:
        print("[pipeline] processing images with Spirula SfM", flush=True)
        _link_photos(images_dir)
        data_type = "individual"

    image_count = _count_images(images_dir)
    if image_count < 3:
        raise RuntimeError(
            f"Preprocessing produced only {image_count} images; "
            "at least three overlapping views are required."
        )
    print(f"[pipeline] {image_count} images prepared for training", flush=True)

    if SPIRULA_TURNTABLE:
        # SfM picks up <dataset>/masks beside the images by itself.
        print("[pipeline] turntable scan: masking the background before solving the cameras",
              flush=True)
        SPIRULA_RESULT["object_masks"] = _spirula_object_masks(spirula, dataset)
        if not SPIRULA_RESULT["object_masks"]:
            raise RuntimeError(
                "Turntable scan: the background could not be masked, and without masks the "
                "still background makes the camera positions impossible to solve. "
                "Stopped before training."
            )

    if SPIRULA_MATCHER != "sift":
        _ensure_sfm_models(SPIRULA_MATCHER)
    print(f"[pipeline] feature matching: {SFM_MATCHER_LABELS[SPIRULA_MATCHER]}", flush=True)

    # Exit code 3 means "reconstructed but partial"; the registered-fraction
    # check below decides whether that is good enough.
    run_command(
        [
            str(spirula), "sfm", "auto", str(images_dir), "-o", str(dataset),
            "--data-type", data_type,
            *SFM_MATCHERS[SPIRULA_MATCHER][0],
            *[flag for name in sequences for flag in ("--sequence", name)],
            *((_fisheye_sfm_args(images_dir, camera_groups) if SOURCE_TYPE == "mixed"
               else _fisheye_sfm_args(images_dir)) if FISHEYE else []),
        ],
        cwd=JOB_ROOT,
        ok_exit_codes=(0, 3),
    )
    registered = _registered_images(dataset)
    print(f"[pipeline] Spirula SfM registered {registered} of {image_count} images", flush=True)
    if registered < 3 or registered < MIN_REGISTERED_FRACTION * image_count:
        stronger = {
            "sift": " Repetitive or shiny subjects (identical rods, tiles, windows) often split "
                    "like this: try Feature matching: Learned, or LoMa.",
            "aliked": " Try Feature matching: LoMa, or reshoot with more of the surroundings "
                      "in frame.",
            "loma": " Reshoot with slower, complete loops and more of the surroundings in frame.",
        }[SPIRULA_MATCHER]
        raise RuntimeError(
            f"Spirula SfM only registered {registered} of {image_count} images; "
            "camera poses are unreliable, aborting before training." + stronger
        )

    if SPIRULA_MAIN_SCENE:
        SPIRULA_RESULT["main_scene"] = _write_main_scene_region(dataset)
    if SPIRULA_OBJECT and not SPIRULA_TURNTABLE:
        SPIRULA_RESULT["object_masks"] = _spirula_object_masks(spirula, dataset)
    steps = _spirula_steps(registered)
    base = SPIRULA_PRESETS[SPIRULA_QUALITY][0]
    if not STEPS_OVERRIDE and steps > base:
        print(
            f"[pipeline] {registered} images: training {steps:,} steps instead of {base:,} "
            "so each image is still seen about as often"
            + (" (the 2.5x limit)" if steps == int(base * MAX_STEP_SCALE) else ""),
            flush=True,
        )
    SPIRULA_RESULT["steps"] = steps
    guidance = _spirula_geometry(spirula, dataset) if SPIRULA_GUIDANCE != "off" else "off"
    SPIRULA_RESULT["guidance_used"] = guidance
    cleanup = [
        "--floater-suppression", SPIRULA_FLOATERS,
        "--distraction-robustness", SPIRULA_DISTRACTION,
    ]
    if guidance == "off":
        cleanup += ["--load-normals", "0", "--load-depths", "0"]
    else:
        cleanup += [
            "--depth-distortion-reg", SPIRULA_DISTORTION_REG,
            "--normal-distortion-reg", SPIRULA_DISTORTION_REG,
        ]
        if guidance == "depth":
            cleanup += ["--depth-supervision-weight", f"{SPIRULA_DEPTH_WEIGHT:g}"]
        else:
            cleanup += ["--load-depths", "0"]
    print(
        f"[pipeline] starting Spirula training for {steps} iterations "
        f"(quality: {SPIRULA_QUALITY}, up to {SPIRULA_PRESETS[SPIRULA_QUALITY][1]:,} splats; "
        f"floater suppression: {SPIRULA_FLOATERS}; ignore moving things: "
        f"{SPIRULA_DISTRACTION}; guidance: {guidance})",
        flush=True,
    )
    timestamp_before = time.time()
    run_command(
        [
            str(spirula), "train",
            # Positional preset: masked loss, centred on the object, mild cleanup.
            # Only with masks; without them it is an ordinary scene.
            *(["centered-object"] if SPIRULA_RESULT.get("object_masks") else []),
            # Spirula 2026.9.30 needs the region named; it does not look in <dataset>/roi itself.
            *(["--roi-region", str(dataset / "roi" / "main-scene.json")]
              if SPIRULA_RESULT.get("main_scene") else []),
            "--data", str(dataset),
            "--output-dir-prefix", str(TRAIN_OUTPUT_DIR),
            "--output-dir-name", SPIRULA_RUN_NAME,
            "--quality", SPIRULA_QUALITY,
            *(["--ppisp-param-type", "no_crf"] if SPIRULA_VIGNETTING else []),
            "--num-iterations", str(steps),
            "--primitive", SPIRULA_PRIMITIVE,
            *(["--viewer-port", str(PREVIEW_PORT)] if PREVIEW else ["--disable-viewer", "1"]),
            "--keep-viewer-alive", "0",
            *cleanup,
        ],
        cwd=JOB_ROOT,
    )
    plys = sorted(TRAIN_OUTPUT_DIR.rglob("splat.ply"), key=lambda path: path.stat().st_mtime)
    if not plys:
        raise RuntimeError(
            f"Training finished but no splat.ply was found under {TRAIN_OUTPUT_DIR}"
        )
    print(f"[pipeline] exporting from {plys[-1].parent}", flush=True)
    return image_count, registered, plys[-1], time.time() - timestamp_before


def _run_nerfstudio(video_path: Path | None) -> tuple[int, int, Path, float]:
    """COLMAP preprocessing + Splatfacto training + PLY export."""
    print("[pipeline] verifying runtime", flush=True)
    run_command(
        [
            "python3",
            "-c",
            (
                "import torch, gsplat; "
                "print('PyTorch:', torch.__version__); "
                "print('Torch CUDA:', torch.version.cuda); "
                "print('CUDA available:', torch.cuda.is_available()); "
                "print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); "
                "print('gsplat:', getattr(gsplat, '__version__', 'unknown'))"
            ),
        ]
    )

    if SOURCE_TYPE == "video":
        _, frames = _video_frame_plan(video_path, VIDEO_FPS)
        print(
            f"[pipeline] extracting about {frames} video frames ({VIDEO_FPS:g} fps) and "
            "processing them with COLMAP",
            flush=True,
        )
        process_args = [
            "ns-process-data", "video", "--data", str(video_path),
            "--output-dir", str(PROCESSED_DIR),
            "--num-frames-target", str(frames),
            "--num-downscales", str(NUM_DOWNSCALES),
        ]
    else:
        print(
            f"[pipeline] processing images with COLMAP "
            f"({MATCHING_METHOD} matching)",
            flush=True,
        )
        process_args = [
            "ns-process-data", "images", "--data", str(INPUT_DIR),
            "--output-dir", str(PROCESSED_DIR),
            "--num-downscales", str(NUM_DOWNSCALES),
            "--matching-method", MATCHING_METHOD,
        ]
    run_command(process_args, cwd=JOB_ROOT)
    image_count = sum(
        1 for path in (PROCESSED_DIR / "images").iterdir()
        if path.is_file() and path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
    ) if (PROCESSED_DIR / "images").is_dir() else 0
    if image_count < 3:
        raise RuntimeError(
            f"Preprocessing produced only {image_count} images; "
            "at least three overlapping views are required."
        )
    print(f"[pipeline] {image_count} images prepared for training", flush=True)

    transforms = PROCESSED_DIR / "transforms.json"
    if not transforms.exists():
        raise RuntimeError(
            f"COLMAP preprocessing completed without {transforms}"
        )

    # processed/images contains every input photo whether or not COLMAP found a
    # pose for it. Only registered images appear in transforms.json, so this is
    # the real measure of whether the reconstruction worked.
    registered = len(json.loads(transforms.read_text(encoding="utf-8"))["frames"])
    print(
        f"[pipeline] COLMAP registered {registered} of {image_count} images",
        flush=True,
    )
    if registered < MIN_REGISTERED_FRACTION * image_count:
        raise RuntimeError(
            f"COLMAP only registered {registered} of {image_count} images; "
            "camera poses are unreliable, aborting before training."
        )

    print(
        f"[pipeline] starting Splatfacto for {MAX_STEPS} iterations",
        flush=True,
    )

    timestamp_before = time.time()

    run_command(
        [
            "ns-train",
            "splatfacto",
            "--data",
            str(PROCESSED_DIR),
            "--output-dir",
            str(TRAIN_OUTPUT_DIR),
            "--max-num-iterations",
            str(MAX_STEPS),
            "--pipeline.model.use-scale-regularization",
            "True",
            "--viewer.quit-on-train-completion",
            "True",
            *(["--viewer.websocket-port", str(PREVIEW_PORT)] if PREVIEW else []),
        ],
        cwd=JOB_ROOT,
    )

    configs = sorted(
        TRAIN_OUTPUT_DIR.rglob("config.yml"),
        key=lambda path: path.stat().st_mtime,
    )

    if not configs:
        raise RuntimeError(
            f"Training finished but no config.yml was found under {TRAIN_OUTPUT_DIR}"
        )

    config = configs[-1]
    print(f"[pipeline] exporting from {config}", flush=True)

    run_command(
        [
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(config),
            "--output-dir",
            str(EXPORT_DIR),
        ],
        cwd=JOB_ROOT,
    )

    ply_files = sorted(
        EXPORT_DIR.rglob("*.ply"),
        key=lambda path: path.stat().st_mtime,
    )

    if not ply_files:
        raise RuntimeError(
            f"Export finished but no PLY was found under {EXPORT_DIR}"
        )
    return image_count, registered, ply_files[-1], time.time() - timestamp_before


# ---------------------------------------------------------------------------
# Google Drive input
# ---------------------------------------------------------------------------

DRIVE_MANIFEST = JOB_ROOT / "drive_manifest.json"
DRIVE_WORKERS = 8
DRIVE_ATTEMPTS = 5
DRIVE_RANGE_BYTES = 64 * 1024 ** 2
_DRIVE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_VIDEO_NAME = re.compile(r"^source_video(?:_([1-9][0-9]?))?\.[A-Za-z0-9]+$")


class _DriveFatal(RuntimeError):
    """An error that retrying cannot fix."""


_DRIVE_ABORT = threading.Event()


def _drive_url(api_base: str, item: dict) -> str:
    return (
        f"{api_base}/files/{urllib.parse.quote(item['id'], safe='')}"
        "?alt=media&supportsAllDrives=true"
    )


def _drive_http_failure(exc: urllib.error.HTTPError, name: str) -> str | None:
    """A fatal message for HTTP errors that retrying cannot fix, else None."""
    body = exc.read(400).decode("utf-8", "replace")
    if exc.code == 401:
        return ("Google Drive rejected the access token (it may have expired). "
                "Start the job again.")
    if exc.code == 404:
        return f"{name}: not found, or not shared with the service account."
    retryable_403 = exc.code == 403 and re.search(
        r"rateLimitExceeded|userRateLimitExceeded|backendError", body
    )
    if exc.code == 403 and not retryable_403:
        return f"{name}: access denied by Google Drive ({body[:160]!r})."
    if exc.code not in (408, 416, 429, 500, 502, 503, 504) and not retryable_403:
        return f"{name}: Google Drive returned HTTP {exc.code}."
    return None


def _drive_fetch_one(api_base: str, token: str, item: dict, progress) -> None:
    url = _drive_url(api_base, item)
    target = INPUT_DIR / item["name"]
    part = target.with_name(target.name + ".part")
    last = "unknown error"
    for attempt in range(1, DRIVE_ATTEMPTS + 1):
        received = 0
        if _DRIVE_ABORT.is_set():
            raise _DriveFatal("cancelled after another file failed")
        try:
            request = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "gaussian-studio-worker",
            })
            digest = hashlib.md5()
            with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as out:
                while True:
                    if _DRIVE_ABORT.is_set():
                        raise _DriveFatal("cancelled after another file failed")
                    chunk = response.read(1024 ** 2)
                    if not chunk:
                        break
                    out.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    progress(len(chunk))
            if received != item["size"]:
                raise OSError(f"got {received} bytes, expected {item['size']}")
            if item.get("md5") and digest.hexdigest() != item["md5"]:
                raise OSError("checksum mismatch")
            part.replace(target)
            return
        except urllib.error.HTTPError as exc:
            progress(-received)
            last = f"HTTP {exc.code}"
            fatal = _drive_http_failure(exc, item["name"])
            if fatal:
                raise _DriveFatal(fatal)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            progress(-received)
            last = str(exc)
        part.unlink(missing_ok=True)
        if attempt < DRIVE_ATTEMPTS:
            time.sleep(min(30, 2 ** attempt) + random.random())
    raise _DriveFatal(f"{item['name']}: failed after {DRIVE_ATTEMPTS} attempts ({last}).")


def _drive_fetch_range(
    api_base: str, token: str, item: dict, part: Path, first: int, last_byte: int, progress,
) -> None:
    """Download bytes first..last_byte of one Drive file into the same offsets of `part`."""
    url = _drive_url(api_base, item)
    length = last_byte - first + 1
    last = "unknown error"
    for attempt in range(1, DRIVE_ATTEMPTS + 1):
        received = 0
        if _DRIVE_ABORT.is_set():
            raise _DriveFatal("cancelled after another part failed")
        try:
            request = urllib.request.Request(url, headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "gaussian-studio-worker",
                "Range": f"bytes={first}-{last_byte}",
            })
            with urllib.request.urlopen(request, timeout=60) as response, part.open("r+b") as out:
                if response.status != 206:
                    raise _DriveFatal("Google Drive ignored the byte range request.")
                out.seek(first)
                while True:
                    if _DRIVE_ABORT.is_set():
                        raise _DriveFatal("cancelled after another part failed")
                    chunk = response.read(min(1024 ** 2, length - received))
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    progress(len(chunk))
                    if received >= length:
                        break
            if received != length:
                raise OSError(f"got {received} bytes of part at {first}, expected {length}")
            return
        except urllib.error.HTTPError as exc:
            progress(-received)
            last = f"HTTP {exc.code}"
            fatal = _drive_http_failure(exc, item["name"])
            if fatal:
                raise _DriveFatal(fatal)
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            progress(-received)
            last = str(exc)
        if attempt < DRIVE_ATTEMPTS:
            time.sleep(min(30, 2 ** attempt) + random.random())
    raise _DriveFatal(
        f"{item['name']}: part at byte {first} failed after {DRIVE_ATTEMPTS} attempts ({last})."
    )


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_index(name: str) -> int | None:
    """0 for source_video.mp4, n for source_video_n.mp4, None for anything else."""
    match = _VIDEO_NAME.match(name)
    if not match or Path(name).suffix.lower() not in VIDEO_EXTENSIONS:
        return None
    return int(match.group(1) or 0)


def _input_videos() -> list[Path]:
    """Uploaded videos in the order they were picked."""
    found = [(index, path) for path in INPUT_DIR.iterdir()
             if path.is_file() and (index := _video_index(path.name)) is not None]
    return [path for _, path in sorted(found)]


def _validate_drive_manifest(kind: str, files: list) -> None:
    for item in files:
        name = str(item.get("name", ""))
        suffix = Path(name).suffix.lower()
        is_video = _video_index(name) is not None
        allowed = (is_video if kind == "video" else suffix in IMAGE_EXTENSIONS if kind == "photos"
                   else is_video or suffix in IMAGE_EXTENSIONS)
        if (not _DRIVE_ID.match(str(item.get("id", ""))) or name != Path(name).name
                or name.startswith(".") or not allowed
                or not isinstance(item.get("size"), int) or item["size"] < 0):
            raise RuntimeError(f"The Drive manifest contains an invalid entry: {item!r}")
    if kind == "video" and not files:
        raise RuntimeError("A video job needs at least one video; the manifest lists none.")
    if kind == "video" and len(files) > 1 and any(
            _video_index(str(item["name"])) == 0 for item in files):
        raise RuntimeError("Several videos must be named source_video_<n>.")
    if len({str(item["name"]) for item in files}) != len(files):
        raise RuntimeError("The Drive manifest lists the same file name twice.")
    if kind == "photos" and len(files) < 3:
        raise RuntimeError(f"Only {len(files)} photos were listed; at least three are required.")
    if kind == "mixed":
        videos = [item for item in files if _video_index(str(item["name"])) is not None]
        if not videos or len(videos) == len(files):
            raise RuntimeError("A photos + videos job needs at least one of each in the manifest.")
        if any(_video_index(str(item["name"])) == 0 for item in videos):
            raise RuntimeError("Videos with photos must be named source_video_<n>.")


def _fetched_noun(kind: str, count: int) -> str:
    if kind == "photos":
        return f"all {count} photos were"
    if kind == "mixed":
        return f"all {count} photos and videos were"
    return "the video was" if count == 1 else f"all {count} videos were"


def fetch_drive_inputs() -> None:
    """Download the photos (or the one video) listed in the manifest the app uploaded."""
    if not DRIVE_MANIFEST.is_file():
        raise RuntimeError(f"Missing {DRIVE_MANIFEST}; the app did not send a Drive manifest.")
    try:
        manifest = json.loads(DRIVE_MANIFEST.read_text(encoding="utf-8"))
        api_base = str(manifest["api_base"]).rstrip("/")
        token = str(manifest["token"])
        files = list(manifest["files"])
        kind = str(manifest.get("kind", "photos"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"The Drive manifest is unreadable: {exc}") from exc
    finally:
        # The token is in memory now; do not leave it on disk.
        try:
            DRIVE_MANIFEST.write_bytes(b"")
            DRIVE_MANIFEST.unlink()
        except OSError:
            pass
    if not (api_base.startswith("https://") or api_base.startswith("http://127.0.0.1")):
        raise RuntimeError("Refusing to send the Drive token to a non-HTTPS address.")
    if kind not in {"photos", "video", "mixed"}:
        raise RuntimeError(f"Unknown Drive manifest kind: {kind!r}")
    _validate_drive_manifest(kind, files)
    noun = "photos" if kind == "photos" else "video" if len(files) == 1 else "videos"
    if kind == "mixed":
        videos = sum(_video_index(str(item["name"])) is not None for item in files)
        noun = (f"files ({len(files) - videos} photos and {videos} "
                f"video{'s' if videos != 1 else ''})")

    total = sum(item["size"] for item in files)
    free = shutil.disk_usage(JOB_ROOT if JOB_ROOT.exists() else JOB_ROOT.parent).free
    if free < total * 1.05 + 1024 ** 3:
        raise RuntimeError(
            f"Not enough disk space for {total / 1024**3:.2f} GiB of {noun} "
            f"({free / 1024**3:.1f} GiB free)."
        )
    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)
    INPUT_DIR.mkdir(parents=True)

    _DRIVE_ABORT.clear()
    lock = threading.Lock()
    received = 0

    def progress(delta: int) -> None:
        nonlocal received
        with lock:
            received += delta

    # One big file over one connection is slow; split each video into byte
    # ranges fetched in parallel, like the photos are.
    ranged = [item for item in files
              if _video_index(str(item["name"])) is not None
              and item["size"] > 2 * DRIVE_RANGE_BYTES]
    tasks = []
    for item in files:
        if item not in ranged:
            tasks.append((_drive_fetch_one, (api_base, token, item, progress)))
            continue
        part = INPUT_DIR / (item["name"] + ".part")
        with part.open("wb") as handle:
            handle.truncate(item["size"])
        tasks += [
            (_drive_fetch_range, (api_base, token, item, part, first,
                                  min(first + DRIVE_RANGE_BYTES, item["size"]) - 1, progress))
            for first in range(0, item["size"], DRIVE_RANGE_BYTES)
        ]
    unit = "parts" if ranged else {"video": "videos", "mixed": "files"}.get(kind, "photos")
    print(
        f"[drive] fetching {len(files)} {noun} ({total / 1024**2:.1f} MiB) "
        f"from Google Drive with {DRIVE_WORKERS} parallel connections",
        flush=True,
    )

    started = last_report = time.monotonic()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DRIVE_WORKERS) as pool:
        pending = {pool.submit(function, *args) for function, args in tasks}
        try:
            while pending:
                finished, pending = concurrent.futures.wait(
                    pending, timeout=5, return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in finished:
                    future.result()
                    done += 1
                now = time.monotonic()
                if now - last_report >= 5 or not pending:
                    with lock:
                        mib = received / 1024 ** 2
                    rate = mib / max(now - started, 0.001)
                    print(
                        f"[drive] fetched {done}/{len(tasks)} {unit} "
                        f"({mib:.0f} / {total / 1024**2:.0f} MiB, {rate:.1f} MiB/s)",
                        flush=True,
                    )
                    last_report = now
        except BaseException:
            _DRIVE_ABORT.set()
            for future in pending:
                future.cancel()
            raise
    for item in ranged:
        part = INPUT_DIR / (item["name"] + ".part")
        if part.stat().st_size != item["size"]:
            raise RuntimeError("Drive fetch verification failed: the video has the wrong size.")
        if item.get("md5"):
            print("[drive] verifying the video checksum", flush=True)
            if _md5_file(part) != item["md5"]:
                raise RuntimeError(
                    "Drive fetch verification failed: the video checksum does not match. "
                    "Start the job again."
                )
        part.replace(INPUT_DIR / item["name"])
    on_disk = [p for p in INPUT_DIR.iterdir() if p.is_file() and not p.name.endswith(".part")]
    if len(on_disk) != len(files) or sum(p.stat().st_size for p in on_disk) != total:
        raise RuntimeError(
            f"Drive fetch verification failed: expected {len(files)} files / {total} bytes, "
            f"found {len(on_disk)} files / {sum(p.stat().st_size for p in on_disk)} bytes."
        )
    print(
        f"[drive] {_fetched_noun(kind, len(files))} "
        f"fetched and verified in {time.monotonic() - started:.0f}s",
        flush=True,
    )


def run_pipeline() -> None:
    if ENGINE not in {"nerfstudio", "spirula"}:
        raise RuntimeError(f"Unknown processing engine: {ENGINE}")
    if SPIRULA_FLOATERS not in {"off", "mild", "strong"}:
        raise RuntimeError(f"Unknown floater suppression level: {SPIRULA_FLOATERS}")
    if SPIRULA_DISTRACTION not in {"off", "mild", "strong"}:
        raise RuntimeError(f"Unknown distraction robustness level: {SPIRULA_DISTRACTION}")
    if SPIRULA_GUIDANCE not in {"off", "surfaces", "depth"}:
        raise RuntimeError(f"Unknown geometry guidance level: {SPIRULA_GUIDANCE}")
    if not 0 <= SPIRULA_DEPTH_WEIGHT <= 1:
        raise RuntimeError(f"Depth guidance strength must be 0 to 1, got {SPIRULA_DEPTH_WEIGHT}")
    if SPIRULA_QUALITY not in SPIRULA_PRESETS:
        raise RuntimeError(f"Unknown quality level: {SPIRULA_QUALITY}")
    if SPIRULA_PRIMITIVE not in {"3dgs", "3dgut"}:
        raise RuntimeError(f"Unknown splat type: {SPIRULA_PRIMITIVE}")
    if SPIRULA_OBJECT and ENGINE != "spirula":
        raise RuntimeError("Object scans are only supported with Spirula Studio.")
    if SPIRULA_MATCHER not in SFM_MATCHERS:
        raise RuntimeError(f"Unknown feature matching: {SPIRULA_MATCHER}")
    if SPIRULA_TURNTABLE and not SPIRULA_OBJECT:
        raise RuntimeError("A turntable scan is an object scan; turn on the object scan too.")
    if FISHEYE and ENGINE != "spirula":
        raise RuntimeError("Fisheye photos are only supported with Spirula Studio.")
    if FISHEYE and not (1 <= FISHEYE_FOCAL_MM <= 50 and 5 <= SENSOR_WIDTH_MM <= 60):
        raise RuntimeError("The fisheye focal length or sensor width is out of range.")
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Missing input directory: {INPUT_DIR}")

    videos: list[Path] = []
    if SOURCE_TYPE == "mixed":
        if ENGINE != "spirula":
            raise RuntimeError("Photos and videos in one job are only supported with Spirula Studio.")
        if not _input_photo_count():
            raise RuntimeError("A photos + videos job found no photos in the input directory.")
    if SOURCE_TYPE in {"video", "mixed"}:
        videos = _input_videos()
        if not videos:
            raise RuntimeError("No uploaded video was found in the input directory.")
        if len(videos) > 1 and ENGINE != "spirula":
            raise RuntimeError("Several videos in one job are only supported with Spirula Studio.")
        if not 0.2 <= VIDEO_FPS <= 30:
            raise RuntimeError(f"Frames per second must be 0.2 to 30, got {VIDEO_FPS:g}.")
        # Nerfstudio images ship ffmpeg; Spirula's runtime installs it itself.
        if ENGINE == "nerfstudio" and not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise RuntimeError("The pod image needs ffmpeg and ffprobe for video processing.")
    elif SOURCE_TYPE == "images":
        image_count = sum(
            1 for path in INPUT_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower()
            in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
        )
        if image_count < 3:
            raise RuntimeError(
                f"Only {image_count} images found; COLMAP needs several overlapping views."
            )
    else:
        raise RuntimeError(f"Unknown input type: {SOURCE_TYPE}")

    for directory in (
        PROCESSED_DIR,
        TRAIN_OUTPUT_DIR,
        EXPORT_DIR,
    ):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[pipeline] engine: {ENGINE}", flush=True)
    if ENGINE == "spirula":
        image_count, registered, source_ply, training_seconds = _run_spirula(videos)
    else:
        image_count, registered, source_ply, training_seconds = _run_nerfstudio(
            videos[0] if videos else None)

    destination = OUTPUT_DIR / "gaussian_splat.ply"
    shutil.copy2(source_ply, destination)

    summary = {
        "pipeline": "spirula-studio" if ENGINE == "spirula" else "nerfstudio-splatfacto",
        "engine": ENGINE,
        "source_type": SOURCE_TYPE,
        "video_count": len(videos) if SOURCE_TYPE in {"video", "mixed"} else None,
        "image_count": image_count,
        "registered_images": registered,
        "matching_method": MATCHING_METHOD if SOURCE_TYPE == "images" and ENGINE == "nerfstudio" else None,
        "video_fps": VIDEO_FPS if SOURCE_TYPE in {"video", "mixed"} else None,
        "max_iterations": SPIRULA_RESULT.get("steps") if ENGINE == "spirula" else MAX_STEPS,
        "spirula_quality": SPIRULA_QUALITY if ENGINE == "spirula" else None,
        "spirula_vignetting_corrected": SPIRULA_VIGNETTING if ENGINE == "spirula" else None,
        "num_downscales": NUM_DOWNSCALES if ENGINE == "nerfstudio" else None,
        "spirula_version": SPIRULA_VERSION if ENGINE == "spirula" else None,
        "spirula_floater_suppression": SPIRULA_FLOATERS if ENGINE == "spirula" else None,
        "spirula_distraction_robustness": SPIRULA_DISTRACTION if ENGINE == "spirula" else None,
        "spirula_object_scan": SPIRULA_OBJECT if ENGINE == "spirula" else None,
        "spirula_turntable": SPIRULA_TURNTABLE if ENGINE == "spirula" else None,
        "spirula_main_scene": SPIRULA_RESULT.get("main_scene") if ENGINE == "spirula" else None,
        "spirula_feature_matching": SPIRULA_MATCHER if ENGINE == "spirula" else None,
        "spirula_object_masks": SPIRULA_RESULT.get("object_masks") if ENGINE == "spirula" else None,
        "spirula_guidance_requested": SPIRULA_GUIDANCE if ENGINE == "spirula" else None,
        "spirula_guidance_used": SPIRULA_RESULT.get("guidance_used") if ENGINE == "spirula" else None,
        "spirula_primitive": SPIRULA_PRIMITIVE if ENGINE == "spirula" else None,
        "fisheye": (
            {"focal_mm": FISHEYE_FOCAL_MM, "sensor_width_mm": SENSOR_WIDTH_MM,
             "focal_px": SPIRULA_RESULT.get("fisheye_focal_px")} if FISHEYE else None
        ),
        "spirula_depth_weight": (
            SPIRULA_DEPTH_WEIGHT
            if ENGINE == "spirula" and SPIRULA_RESULT.get("guidance_used") == "depth" else None
        ),
        "training_seconds": round(training_seconds, 3),
        "ply": str(destination),
    }

    (OUTPUT_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[pipeline] PLY ready: {destination}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["fetch-drive"]:
        try:
            fetch_drive_inputs()
        except (RuntimeError, _DriveFatal) as exc:
            print(f"[drive] error: {exc}", flush=True)
            sys.exit(1)
    else:
        sys.exit("usage: pipeline.py fetch-drive")