"""Add Gaussian Studio to the app menu and the desktop (no admin rights needed).

Run once:  python3 install_launcher.py
Remove:    python3 install_launcher.py --remove

Linux: writes ~/.local/share/applications/gaussian-studio.desktop and a copy on
the desktop. macOS: builds ~/Applications/Gaussian Studio.app and puts an alias
to it on the desktop. Windows: adds a Gaussian Studio shortcut to the Start menu
and the desktop. All start the app with this Python and this folder.
"""
from __future__ import annotations

import os
import plistlib
import shlex
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ICON = APP_DIR / "assets" / "gaussian-studio.png"
DESKTOP_ID = "gaussian-studio"
ENTRY = Path.home() / ".local" / "share" / "applications" / f"{DESKTOP_ID}.desktop"
MAC_APP = Path.home() / "Applications" / "Gaussian Studio.app"
MAC = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"
WINDOWS_ICON = APP_DIR / "assets" / "gaussian-studio.ico"
WINDOWS_LINK = "Gaussian Studio.lnk"
# Writes a .lnk through the Windows shell; the values come in environment variables
# so no path ever needs quoting.
SHORTCUT_SCRIPT = (
    "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:GS_LINK); "
    "$s.TargetPath = $env:GS_TARGET; $s.Arguments = $env:GS_ARGS; "
    "$s.WorkingDirectory = $env:GS_DIR; $s.IconLocation = $env:GS_ICON; "
    "$s.Description = 'Turn photos and videos into Gaussian splats on RunPod GPUs'; $s.Save()"
)


def desktop_dir() -> Path:
    """The user's desktop folder, which some Linux languages name differently."""
    try:
        found = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True,
                               check=False).stdout.strip()
    except OSError:
        found = ""
    return Path(found) if found and Path(found).is_dir() else Path.home() / "Desktop"


def draw_icon(path: Path, size: int = 256) -> None:
    """Soft overlapping blobs on a rounded tile: a few Gaussian splats."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtGui, QtWidgets

    import ui_theme as theme

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    image = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    tile = QtCore.QRectF(8, 8, size - 16, size - 16)
    painter.setPen(QtCore.Qt.NoPen)
    painter.setBrush(QtGui.QColor(theme.SURFACE_CONTAINER_HIGH))
    painter.drawRoundedRect(tile, 56, 56)
    blobs = (  # centre x, centre y, radius, colour
        (0.40, 0.44, 0.30, theme.PRIMARY),
        (0.62, 0.58, 0.26, "#E7B8F0"),
        (0.44, 0.66, 0.20, theme.SUCCESS),
    )
    painter.setCompositionMode(QtGui.QPainter.CompositionMode_Screen)
    for cx, cy, radius, colour in blobs:
        centre = QtCore.QPointF(size * cx, size * cy)
        gradient = QtGui.QRadialGradient(centre, size * radius)
        inner = QtGui.QColor(colour)
        outer = QtGui.QColor(colour)
        inner.setAlpha(235)
        outer.setAlpha(0)
        gradient.setColorAt(0.0, inner)
        gradient.setColorAt(0.55, QtGui.QColor(inner.red(), inner.green(), inner.blue(), 140))
        gradient.setColorAt(1.0, outer)
        painter.setBrush(gradient)
        painter.drawEllipse(centre, size * radius, size * radius)
    painter.end()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    del app


def desktop_quote(value: str) -> str:
    """Quote one Exec argument per the Desktop Entry spec."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$")
    return f'"{escaped}"'


def quiet_run(command: list[str]) -> None:
    """A helper some desktops have and others don't."""
    try:
        subprocess.run(command, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def install() -> None:
    sys.path.insert(0, str(APP_DIR))
    draw_icon(ICON)
    if WINDOWS:
        install_windows()
    elif MAC:
        install_mac()
    else:
        install_linux()


def install_linux() -> None:
    ENTRY.parent.mkdir(parents=True, exist_ok=True)
    ENTRY.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=Gaussian Studio\n"
        "Comment=Turn photos and videos into Gaussian splats on RunPod GPUs\n"
        f"Exec={desktop_quote(sys.executable)} {desktop_quote(str(APP_DIR / 'main.py'))}\n"
        f"Path={APP_DIR}\n"
        f"Icon={ICON}\n"
        "Terminal=false\n"
        "Categories=Graphics;3DGraphics;\n"
        # Lets the dock match the running window to this launcher.
        f"StartupWMClass={DESKTOP_ID}\n"
        "StartupNotify=true\n",
        encoding="utf-8",
    )
    ENTRY.chmod(0o755)
    quiet_run(["update-desktop-database", str(ENTRY.parent)])
    print(f"Installed {ENTRY}")
    shortcut = desktop_dir() / ENTRY.name
    if shortcut.parent.is_dir():
        shutil.copy2(ENTRY, shortcut)
        # GNOME only starts desktop launchers it has been told to trust.
        quiet_run(["gio", "set", str(shortcut), "metadata::trusted", "true"])
        print(f"Added {shortcut}")
    print("Open the app grid, search 'Gaussian Studio', then right-click it and pin it to the dock.")


def install_mac() -> None:
    contents = MAC_APP / "Contents"
    shutil.rmtree(MAC_APP, ignore_errors=True)
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()
    launcher = contents / "MacOS" / "gaussian-studio"
    launcher.write_text(
        "#!/bin/sh\n"
        f"cd {shlex.quote(str(APP_DIR))} || exit 1\n"
        f"exec {shlex.quote(sys.executable)} main.py\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    has_icon = make_icns(ICON, contents / "Resources" / "gaussian-studio.icns")
    info = {
        "CFBundleName": "Gaussian Studio",
        "CFBundleDisplayName": "Gaussian Studio",
        "CFBundleIdentifier": "local.gaussian-studio",
        "CFBundleExecutable": launcher.name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": "1.0",
        "NSHighResolutionCapable": True,
    }
    if has_icon:
        info["CFBundleIconFile"] = "gaussian-studio"
    with open(contents / "Info.plist", "wb") as handle:
        plistlib.dump(info, handle)
    print(f"Installed {MAC_APP}")
    alias = desktop_dir() / "Gaussian Studio"
    if alias.parent.is_dir():
        if alias.is_symlink():
            alias.unlink()
        if not alias.exists():
            alias.symlink_to(MAC_APP)
            print(f"Added {alias}")
    print("Open it from the desktop or Launchpad. To keep it in the Dock, right-click its "
          "Dock icon while it runs → Options → Keep in Dock.")


def windows_folders() -> list[Path]:
    """The Start menu's Programs folder and the desktop (which OneDrive may have moved)."""
    import ctypes

    folders = []
    for csidl in (0x02, 0x10):  # CSIDL_PROGRAMS, CSIDL_DESKTOPDIRECTORY
        buffer = ctypes.create_unicode_buffer(260)
        if ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer) == 0:
            folders.append(Path(buffer.value))
    return folders


def install_windows() -> None:
    from PySide6 import QtGui

    has_icon = QtGui.QImage(str(ICON)).save(str(WINDOWS_ICON))
    # pythonw starts the app without a console window; -X utf8 reads and writes
    # files as UTF-8, as on Linux and macOS.
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    for folder in windows_folders():
        link = folder / WINDOWS_LINK
        env = {**os.environ, "GS_LINK": str(link),
               "GS_TARGET": str(pythonw if pythonw.is_file() else sys.executable),
               "GS_ARGS": f'-X utf8 "{APP_DIR / "main.py"}"', "GS_DIR": str(APP_DIR),
               "GS_ICON": str(WINDOWS_ICON) if has_icon else ""}
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", SHORTCUT_SCRIPT],
            env=env, capture_output=True, text=True, check=False,
        )
        if result.returncode:
            print(f"Could not add {link}: {result.stderr.strip()}")
        else:
            print(f"Added {link}")
    print("Open it from the Start menu or the desktop. To pin it, right-click it > Pin to taskbar.")


def make_icns(png: Path, target: Path) -> bool:
    """Convert the icon with macOS's own sips and iconutil."""
    try:
        return _make_icns(png, target)
    except OSError:  # either tool missing; the app keeps the default icon
        return False


def _make_icns(png: Path, target: Path) -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256):
            for scale in (1, 2):
                pixels = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                result = subprocess.run(
                    ["sips", "-z", str(pixels), str(pixels), str(png), "--out", str(iconset / name)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                )
                if result.returncode:
                    return False
        result = subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(target)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return result.returncode == 0


def remove() -> None:
    if WINDOWS:
        for folder in windows_folders():
            (folder / WINDOWS_LINK).unlink(missing_ok=True)
            print(f"Removed {folder / WINDOWS_LINK}")
        return
    if MAC:
        shutil.rmtree(MAC_APP, ignore_errors=True)
        alias = desktop_dir() / "Gaussian Studio"
        if alias.is_symlink():
            alias.unlink()
        print(f"Removed {MAC_APP}")
        return
    ENTRY.unlink(missing_ok=True)
    (desktop_dir() / ENTRY.name).unlink(missing_ok=True)
    print(f"Removed {ENTRY}")


if __name__ == "__main__":
    remove() if "--remove" in sys.argv[1:] else install()
