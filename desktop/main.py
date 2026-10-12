from __future__ import annotations

import faulthandler
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from app import MainWindow

APP_DIR = Path(__file__).resolve().parent


def main() -> None:
    # Started from the dock there is no terminal: keep errors and crash reports
    # in a file instead of losing them.
    if not sys.stderr or not sys.stderr.isatty():
        log_dir = APP_DIR / "logs"
        log_dir.mkdir(exist_ok=True)
        sys.stderr = sys.stdout = open(log_dir / "app.log", "a", buffering=1, encoding="utf-8")
    faulthandler.enable(sys.stderr)
    # Required before the QApplication exists for the embedded training preview.
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    # The name and desktop file id let the dock match this window to its launcher
    # (WM_CLASS on X11, app id on Wayland); see install_launcher.py.
    app.setApplicationName("gaussian-studio")
    app.setDesktopFileName("gaussian-studio")
    icon = APP_DIR / "assets" / "gaussian-studio.png"
    if icon.is_file():
        app.setWindowIcon(QtGui.QIcon(str(icon)))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
