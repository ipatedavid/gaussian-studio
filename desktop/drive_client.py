"""Read-only Google Drive access through a service account.

The service-account key file never leaves this computer. The pod receives only
a one-hour access token (read-only scope) plus a list of the files to fetch.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import requests

from storage import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, probe_output

DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_API_BASE = "https://www.googleapis.com/drive/v3"
_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class DriveError(RuntimeError):
    pass


@dataclass(frozen=True)
class DriveFolder:
    id: str
    name: str
    is_shared_drive: bool = False


@dataclass(frozen=True)
class DriveFile:
    id: str
    name: str      # name as stored in Drive
    dest: str      # collision-free, filesystem-safe name used on the pod
    size: int
    md5: str
    # Videos only, from Drive's videoMediaMetadata (0 when Drive has not
    # processed the video yet). Drive never reports the frame rate; fps is
    # filled in by probe_video, which reads the file's own header.
    width: int = 0
    height: int = 0
    duration_ms: int = 0
    fps: float = 0.0


@dataclass(frozen=True)
class DriveListing:
    folder_id: str
    folder_name: str
    files: tuple[DriveFile, ...]          # photos
    skipped: int = 0   # files that are neither photos nor videos
    videos: tuple[DriveFile, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)


def _check_id(value: str) -> str:
    if not _ID.match(value or ""):
        raise DriveError(f"Invalid Google Drive ID: {value!r}")
    return value


def _safe_name(name: str) -> str:
    cleaned = "".join("_" if ch in '/\\\0' or ord(ch) < 32 else ch for ch in name).strip()
    cleaned = cleaned.lstrip(".") or "photo"
    return cleaned[:180]


def _unique_names(raw: list[tuple[str, str]]) -> dict[str, str]:
    """Map file id -> destination name, de-duplicating case-insensitively."""
    used: set[str] = set()
    result: dict[str, str] = {}
    for file_id, name in raw:
        base = _safe_name(name)
        stem, dot, ext = base.rpartition(".")
        if not dot:
            stem, ext = base, ""
        candidate, counter = base, 2
        while candidate.lower() in used:
            candidate = f"{stem}_{counter}" + (f".{ext}" if ext else "")
            counter += 1
        used.add(candidate.lower())
        result[file_id] = candidate
    return result


class DriveClient:
    def __init__(self, credentials_file: Path, api_base: str | None = None):
        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise DriveError(
                "The google-auth package is missing. Install it with: pip install google-auth"
            ) from exc
        credentials_file = Path(credentials_file)
        if not credentials_file.is_file():
            raise DriveError(f"Service account file not found: {credentials_file}")
        try:
            self._credentials = service_account.Credentials.from_service_account_file(
                str(credentials_file), scopes=[DRIVE_SCOPE],
            )
        except (ValueError, OSError, KeyError) as exc:
            raise DriveError(
                f"{credentials_file.name} is not a valid service account key file: {exc}"
            ) from exc
        self.email: str = self._credentials.service_account_email
        self.api_base = (api_base or os.getenv("GOOGLE_DRIVE_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ---- auth ---------------------------------------------------------
    def _refresh(self) -> None:
        from google.auth.exceptions import GoogleAuthError
        from google.auth.transport.requests import Request
        try:
            self._credentials.refresh(Request(session=self._session))
        except GoogleAuthError as exc:
            raise DriveError(f"Google rejected the service account credentials: {exc}") from exc
        except requests.RequestException as exc:
            raise DriveError(f"Could not reach Google to sign in: {exc}") from exc

    def _token(self) -> str:
        with self._lock:
            if not self._credentials.valid:
                self._refresh()
            return self._credentials.token

    def token_for_pod(self) -> tuple[str, str]:
        """A freshly minted token (about an hour of life) and its expiry in ISO form."""
        with self._lock:
            self._refresh()
            expiry = self._credentials.expiry
            return self._credentials.token, (expiry.isoformat() + "Z" if expiry else "")

    # ---- API ----------------------------------------------------------
    def _get(self, path: str, params: dict) -> dict:
        url = f"{self.api_base}/{path.lstrip('/')}"
        last = ""
        for attempt in range(1, 6):
            try:
                response = self._session.get(
                    url, params=params, timeout=30,
                    headers={"Authorization": f"Bearer {self._token()}"},
                )
            except requests.RequestException as exc:
                last = str(exc)
                time.sleep(min(8, 2 ** attempt * 0.5))
                continue
            if response.status_code == 401 and attempt < 5:
                with self._lock:
                    self._refresh()
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                last = f"HTTP {response.status_code}"
                time.sleep(min(8, 2 ** attempt * 0.5))
                continue
            if response.ok:
                return response.json()
            raise DriveError(self._explain(response))
        raise DriveError(f"Google Drive did not respond after several attempts ({last}).")

    def _explain(self, response: requests.Response) -> str:
        try:
            error = response.json().get("error", {})
            message = error.get("message") or response.text[:200]
            reason = (error.get("errors") or [{}])[0].get("reason", "")
        except ValueError:
            message, reason = response.text[:200], ""
        if response.status_code == 404:
            return (
                "Google Drive could not find that folder. Share it with "
                f"{self.email} (Viewer access is enough)."
            )
        if response.status_code == 403 and reason in {"accessNotConfigured", "serviceDisabled"}:
            return (
                "The Google Drive API is not enabled for this service account's project. "
                "Enable 'Google Drive API' in the Google Cloud console. "
                f"({message})"
            )
        return f"Google Drive error {response.status_code}: {message}"

    def _list(self, params: dict, key: str = "files") -> list[dict]:
        items: list[dict] = []
        page_token = None
        while True:
            query = dict(params)
            if page_token:
                query["pageToken"] = page_token
            data = self._get("files" if key == "files" else key, query)
            items.extend(data.get(key, []))
            page_token = data.get("nextPageToken")
            if not page_token:
                return items

    _ALL = {"supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}

    def list_folders(self, parent_id: str | None) -> list[DriveFolder]:
        """Folders inside parent_id, or (for None) everything shared with the account."""
        if parent_id is None:
            shared = self._list({
                **self._ALL, "pageSize": 1000, "orderBy": "name",
                "q": f"mimeType='{FOLDER_MIME}' and sharedWithMe=true and trashed=false",
                "fields": "nextPageToken, files(id,name)",
            })
            drives = self._list(
                {"pageSize": 100, "fields": "nextPageToken, drives(id,name)"}, key="drives",
            )
            seen = {d["id"] for d in drives}
            return (
                [DriveFolder(d["id"], d["name"], True) for d in drives]
                + [DriveFolder(f["id"], f["name"]) for f in shared if f["id"] not in seen]
            )
        _check_id(parent_id)
        files = self._list({
            **self._ALL, "corpora": "allDrives", "pageSize": 1000, "orderBy": "name",
            "q": f"'{parent_id}' in parents and mimeType='{FOLDER_MIME}' and trashed=false",
            "fields": "nextPageToken, files(id,name)",
        })
        return [DriveFolder(f["id"], f["name"]) for f in files]

    def folder_name(self, folder_id: str) -> str:
        _check_id(folder_id)
        info = self._get(f"files/{folder_id}", {"supportsAllDrives": "true", "fields": "id,name,mimeType"})
        if info.get("mimeType") != FOLDER_MIME:
            raise DriveError("That Google Drive item is not a folder.")
        return info["name"]

    def list_images(self, folder_id: str, folder_name: str = "") -> DriveListing:
        """Photos and videos directly inside the folder (subfolders are not searched)."""
        _check_id(folder_id)
        if not folder_name:
            folder_name = self.folder_name(folder_id)
        entries = self._list({
            **self._ALL, "corpora": "allDrives", "pageSize": 1000, "orderBy": "name",
            "q": f"'{folder_id}' in parents and trashed=false and mimeType!='{FOLDER_MIME}'",
            "fields": (
                "nextPageToken, files(id,name,size,md5Checksum,mimeType,"
                "videoMediaMetadata(width,height,durationMillis))"
            ),
        })
        if not entries:
            # Drive answers "nothing" (not 404) for a folder this account cannot
            # see, so confirm access to give the right message.
            self.folder_name(folder_id)
        photos, videos, skipped = [], [], 0
        for entry in entries:
            suffix = Path(entry.get("name", "")).suffix.lower()
            if not str(entry.get("size", "")).isdigit():
                skipped += 1
            elif suffix in IMAGE_EXTENSIONS:
                photos.append(entry)
            elif suffix in VIDEO_EXTENSIONS:
                videos.append(entry)
            else:
                skipped += 1
        names = _unique_names([(p["id"], p["name"]) for p in photos])
        files = tuple(
            DriveFile(p["id"], p["name"], names[p["id"]], int(p["size"]), p.get("md5Checksum", ""))
            for p in photos
        )
        return DriveListing(
            folder_id, folder_name, files, skipped,
            tuple(self._video_file(entry) for entry in videos),
        )

    @staticmethod
    def _video_file(entry: dict) -> DriveFile:
        meta = entry.get("videoMediaMetadata") or {}

        def number(key: str) -> int:
            value = str(meta.get(key, "")).strip()
            return int(value) if value.isdigit() else 0

        return DriveFile(
            entry["id"], entry["name"],
            "source_video" + Path(entry["name"]).suffix.lower(),
            int(entry["size"]), entry.get("md5Checksum", ""),
            number("width"), number("height"), number("durationMillis"),
        )

    def probe_video(self, video: DriveFile, timeout: float = 45) -> DriveFile:
        """Exact length, size and frame rate, read by ffprobe straight from Drive.

        ffprobe fetches only the byte ranges holding the video header (a few
        hundred kB at most), not the video. Returns the file unchanged if that
        fails; Drive's own metadata (if any) is still used then.
        """
        url = f"{self.api_base}/files/{_check_id(video.id)}?alt=media&supportsAllDrives=true"
        try:
            result = subprocess.run(
                [
                    # Length and rate come from the header; no need to sample frames.
                    "ffprobe", "-v", "error", "-probesize", "32768",
                    "-headers", f"Authorization: Bearer {self._token()}\r\n",
                    "-select_streams", "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration",
                    "-of", "json", url,
                ],
                capture_output=True, text=True, timeout=timeout,
            )
            info = probe_output(result.stdout) if result.returncode == 0 else None
        except (OSError, ValueError, subprocess.TimeoutExpired, DriveError):
            info = None
        if info is None:
            return video
        return replace(
            video, width=info.width, height=info.height,
            duration_ms=round(info.duration_seconds * 1000), fps=info.fps,
        )

    def build_manifest(self, listing: DriveListing) -> dict:
        """Everything the pod needs to fetch the photos itself."""
        return self._manifest("photos", listing.folder_name, listing.files)

    def build_video_manifest(self, folder_name: str, video) -> dict:
        """Everything the pod needs to fetch the videos: one is saved as
        source_video.<ext>, several as source_video_1.<ext>, source_video_2.<ext>..."""
        videos = tuple(video) if isinstance(video, (list, tuple)) else (video,)
        if len(videos) > 1:
            videos = tuple(
                replace(item, dest=f"source_video_{index}{Path(item.name).suffix.lower()}")
                for index, item in enumerate(videos, start=1)
            )
        return self._manifest("video", folder_name, videos)

    def build_mixed_manifest(self, folder_name: str, photos, videos) -> dict:
        """Photos plus videos from one folder; videos become source_video_<n>.<ext>."""
        videos = tuple(
            replace(item, dest=f"source_video_{index}{Path(item.name).suffix.lower()}")
            for index, item in enumerate(videos, start=1)
        )
        return self._manifest("mixed", folder_name, (*photos, *videos))

    def _manifest(self, kind: str, folder_name: str, files) -> dict:
        token, expires_at = self.token_for_pod()
        return {
            "kind": kind,
            "api_base": self.api_base,
            "token": token,
            "expires_at": expires_at,
            "folder_name": folder_name,
            "files": [
                {"id": f.id, "name": f.dest, "size": f.size, "md5": f.md5}
                for f in files
            ],
        }
