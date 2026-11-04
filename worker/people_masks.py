"""People masks for Spirula training: <dataset>/masks/<image name>.png, white = keep.

A person detector (Mask R-CNN, torchvision, COCO weights) runs on every photo the
camera solve placed. A detection counts as a person only if part of it reaches below
eye level: painted figures on a ceiling mural or a high poster are found too, and the
solved camera angles let them be told apart. Kept people are grown by about 2% of the
image width (hair, loose clothing, motion blur) and written as black on white.

Spirula leaves masked pixels out of every training comparison (colour, depth and
surface directions), so passers-by can't become floaters or ghosts.

Runs in the pod's PyTorch environment (torch_env.py), not the worker's:
    python people_masks.py <dataset> [--threshold 0.5] [--check-sheet out.jpg]
Prints [masks] progress lines and one "[stats] masks {json}" line at the end, and
saves the same stats as <dataset>/people_masks.json.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# COLMAP camera model ids: (name, parameter count).
MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12), 7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4), 9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
FISHEYE_MODELS = {"OPENCV_FISHEYE", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE"}
PERSON = 1                   # COCO label id
MIN_FEET_DEG = -5.0          # a person reaches at least this far below the horizon
GROW_SHARE = 0.02            # mask growth, as a share of the image's long side
WORK_LONG_SIDE = 1500        # photos are read at least this big, shrunk by 2 or 4 when larger
SHEET_TILES = 12


# ---- COLMAP model ------------------------------------------------------------------

def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            name, params = MODELS.get(model_id, ("UNKNOWN", 0))
            if not params:
                raise SystemExit(f"unsupported COLMAP camera model id {model_id}")
            values = struct.unpack(f"<{params}d", handle.read(8 * params))
            cameras[camera_id] = {"model": name, "width": width, "height": height,
                                  "params": values}
    return cameras


def read_images(path: Path) -> list[dict]:
    """Placed photos: name (relative to images/), camera id and world-to-camera rotation."""
    images = []
    with path.open("rb") as handle:
        (count,) = struct.unpack("<Q", handle.read(8))
        for _ in range(count):
            _, qw, qx, qy, qz, _, _, _, camera_id = struct.unpack("<i7di", handle.read(64))
            name = b""
            while (char := handle.read(1)) not in (b"\0", b""):
                name += char
            (points,) = struct.unpack("<Q", handle.read(8))
            handle.seek(points * 24, 1)
            rotation = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            images.append({"name": name.decode("utf-8"), "camera_id": camera_id,
                           "rotation": rotation})
    return sorted(images, key=lambda image: image["name"])


def largest_model(dataset: Path) -> Path:
    best, best_count = None, -1
    for model in sorted((dataset / "sparse").iterdir()):
        binary = model / "images.bin"
        if binary.is_file():
            with binary.open("rb") as handle:
                (count,) = struct.unpack("<Q", handle.read(8))
            if count > best_count:
                best, best_count = model, count
    if best is None:
        raise SystemExit(f"no camera model under {dataset / 'sparse'}")
    return best


def world_up(images: list[dict]) -> np.ndarray:
    """Spirula writes its models with +Z up (levelled on the ground, or on the cameras).
    If the cameras say otherwise (a model from elsewhere), trust their average up."""
    cameras_up = -np.mean([image["rotation"][1] for image in images], axis=0)
    cameras_up /= max(np.linalg.norm(cameras_up), 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    return z if cameras_up @ z > math.cos(math.radians(45)) else cameras_up


def rays(camera: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Camera-frame unit rays for pixel coordinates in the camera's own resolution.

    Lens distortion is left out (fisheyes are taken as equidistant): the eye-level test
    only needs a ray's angle to within a few degrees."""
    p, model = camera["params"], camera["model"]
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV",
                 "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    mx, my = (u - cx) / fx, (v - cy) / fy
    if model in FISHEYE_MODELS:
        theta = np.hypot(mx, my)
        scale = np.where(theta > 1e-12, np.sin(theta) / np.maximum(theta, 1e-12), 1.0)
        out = np.stack([mx * scale, my * scale, np.cos(theta)], axis=-1)
    else:
        out = np.stack([mx, my, np.ones_like(mx)], axis=-1)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def reaches_below_eye_level(mask: np.ndarray, camera: dict, rotation: np.ndarray,
                            up: np.ndarray) -> bool:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return False
    pick = np.linspace(0, len(ys) - 1, min(len(ys), 400)).astype(int)
    sx, sy = camera["width"] / mask.shape[1], camera["height"] / mask.shape[0]
    local = rays(camera, (xs[pick] + 0.5) * sx, (ys[pick] + 0.5) * sy)
    world = local @ rotation           # camera-to-world is the transpose of the rotation
    elevation = np.degrees(np.arcsin(np.clip(world @ up, -1, 1)))
    return float(elevation.min()) < MIN_FEET_DEG


# ---- photos ------------------------------------------------------------------------

def read_photo(path: Path) -> np.ndarray | None:
    """RGB, decoded at a reduced size when the photo is much larger than needed."""
    header = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
    if header is None:
        return None
    long_side = max(header.shape[:2]) * 8
    flag = cv2.IMREAD_COLOR
    for factor, reduced in ((4, cv2.IMREAD_REDUCED_COLOR_4), (2, cv2.IMREAD_REDUCED_COLOR_2)):
        if long_side / factor >= WORK_LONG_SIDE:
            flag = reduced
            break
    image = cv2.imread(str(path), flag)
    return None if image is None else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def check_sheet(rows: list[tuple[str, np.ndarray, np.ndarray, float]], target: Path) -> None:
    """The photos with the most masked area, people tinted red, three to a row."""
    tiles = []
    for name, photo, people, share in rows:
        tile = cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)
        tile[people] = (tile[people] * 0.35 + np.array([40, 40, 220]) * 0.65).astype(np.uint8)
        width = 480
        tile = cv2.resize(tile, (width, round(tile.shape[0] * width / tile.shape[1])),
                          interpolation=cv2.INTER_AREA)
        label = f"{name}  {share * 100:.1f}%"
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        tiles.append(tile)
    height = max(tile.shape[0] for tile in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(target), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ---- run ---------------------------------------------------------------------------

def run(dataset: Path, threshold: float, sheet: Path | None) -> dict:
    import torch
    import torchvision

    started = time.time()
    model_dir = largest_model(dataset)
    cameras = read_cameras(model_dir / "cameras.bin")
    images = read_images(model_dir / "images.bin")
    up = world_up(images)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=weights)
    model = model.to(device).eval()
    print(f"[masks] Mask R-CNN on {device}; {len(images)} placed photos", flush=True)
    masks_dir = dataset / "masks"
    written = with_people = dropped = 0
    shares: list[float] = []
    best: list[tuple[float, str, Path, Path]] = []
    for index, image in enumerate(images, start=1):
        source = dataset / "images" / image["name"]
        photo = read_photo(source)
        if photo is None:
            print(f"[masks] could not read {image['name']}; left unmasked", flush=True)
            continue
        tensor = torch.from_numpy(photo).permute(2, 0, 1).float().div(255).to(device)
        with torch.inference_mode():
            found = model([tensor])[0]
        keep = (found["labels"] == PERSON) & (found["scores"] >= threshold)
        people = np.zeros(photo.shape[:2], np.uint8)
        camera = cameras[image["camera_id"]]
        for mask in (found["masks"][keep, 0] > 0.5).cpu().numpy():
            if reaches_below_eye_level(mask, camera, image["rotation"], up):
                people |= mask.astype(np.uint8)
            else:
                dropped += 1
        if people.any():
            size = int(round(max(people.shape) * GROW_SHARE)) | 1
            people = cv2.dilate(people, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
            with_people += 1
        share = float(people.mean())
        shares.append(share)
        target = masks_dir / Path(image["name"]).with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), np.where(people > 0, 0, 255).astype(np.uint8))
        written += 1
        if share > 0:
            best.append((share, image["name"], source, target))
        if index % 50 == 0 or index == len(images):
            print(f"[masks] {index} / {len(images)} photos, {time.time() - started:.0f}s",
                  flush=True)
    if sheet is not None and best:
        best.sort(reverse=True)
        # The fullest photos, and some lighter ones spread through the rest.
        picks = best[:SHEET_TILES // 2] + best[SHEET_TILES // 2:][::max(1, len(best) // SHEET_TILES)]
        rows = []
        for share, name, source, target in sorted(picks[:SHEET_TILES], key=lambda pick: pick[1]):
            photo = read_photo(source)
            mask = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if photo is not None and mask is not None:
                rows.append((name, photo, mask < 128, share))
        if rows:
            check_sheet(rows, sheet)
    stats = {
        "detector": "maskrcnn_resnet50_fpn_v2", "photos": written, "with_people": with_people,
        "masked_share": round(float(np.mean(shares)), 4) if shares else 0.0,
        "dropped_high": dropped, "seconds": round(time.time() - started),
    }
    print(f"[stats] masks {json.dumps(stats)}", flush=True)
    (dataset / "people_masks.json").write_text(json.dumps(stats), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="detector confidence a person needs")
    parser.add_argument("--check-sheet", type=Path, help="write a JPEG of masked photos here")
    args = parser.parse_args()
    run(args.dataset, args.threshold, args.check_sheet)


if __name__ == "__main__":
    sys.exit(main())
