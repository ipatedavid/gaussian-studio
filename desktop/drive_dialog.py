"""Dialog for browsing the Google Drive folders a service account can see.

The user picks a folder; the New job page then lists its photos and videos
(directly inside it) with checkboxes.
"""
from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from drive_client import DriveClient, DriveListing
from ui_widgets import FramelessDialog

MIB = 1024 ** 2


class DriveFolderDialog(FramelessDialog):
    # request id, folders, listing of photos in the current folder, error text
    loaded = QtCore.Signal(int, object, object, str)

    def __init__(self, client: DriveClient, parent=None, start_id: str = "", start_name: str = ""):
        super().__init__(parent, "Choose a Google Drive folder")
        self.client = client
        self.selected: DriveListing | None = None
        self._path: list[tuple[str, str]] = [(start_id, start_name)] if start_id else []
        self._listing: DriveListing | None = None
        self._request = 0
        self.resize(600, 560)
        root = self.body

        info = QtWidgets.QLabel(
            "Showing folders shared with this service account. To make a folder appear, "
            "share it (Viewer access is enough) with:"
        )
        info.setWordWrap(True)
        info.setObjectName("helper")
        root.addWidget(info)
        email = QtWidgets.QLineEdit(client.email)
        email.setReadOnly(True)
        root.addWidget(email)

        nav = QtWidgets.QHBoxLayout()
        self.up_button = QtWidgets.QPushButton("↑ Up")
        self.up_button.clicked.connect(self._go_up)
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._navigate)
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setObjectName("muted")
        nav.addWidget(self.up_button)
        nav.addWidget(self.refresh_button)
        nav.addWidget(self.path_label, 1)
        root.addLayout(nav)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("gpuList")
        self.list.itemDoubleClicked.connect(self._open_item)
        self.list.itemActivated.connect(self._open_item)
        root.addWidget(self.list, 1)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("notice")
        root.addWidget(self.status)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.setObjectName("textButton")
        cancel.clicked.connect(self.reject)
        self.choose_button = QtWidgets.QPushButton("Use this folder")
        self.choose_button.setObjectName("primaryButton")
        self.choose_button.setEnabled(False)
        self.choose_button.clicked.connect(self._accept)
        buttons.addWidget(cancel)
        buttons.addWidget(self.choose_button)
        root.addLayout(buttons)

        self.loaded.connect(self._on_loaded)
        self._navigate()

    # ---- navigation ---------------------------------------------------
    def _current(self) -> tuple[str, str] | None:
        return self._path[-1] if self._path else None

    def _open_item(self, item: QtWidgets.QListWidgetItem | None):
        # A double-click also fires itemActivated; by then the first handler may
        # have opened the folder and cleared the list, leaving no item.
        if item is None:
            return
        target = item.data(QtCore.Qt.UserRole)
        if target:
            self._path.append((target.id, target.name))
            self._navigate()

    def _go_up(self):
        if self._path:
            self._path.pop()
            self._navigate()

    def _navigate(self, *_):
        self._request += 1
        request = self._request
        current = self._current()
        self._listing = None
        self.list.clear()
        self.choose_button.setEnabled(False)
        self.up_button.setEnabled(bool(self._path))
        self.path_label.setText(
            "Shared with you" if current is None else "Shared with you / " + " / ".join(n for _, n in self._path)
        )
        self.status.setText("Loading...")
        client = self.client

        def work():
            try:
                folders = client.list_folders(current[0] if current else None)
                listing = client.list_images(current[0], current[1]) if current else None
                payload = (folders, listing, "")
            except Exception as exc:  # shown to the user, never fatal
                payload = (None, None, str(exc))
            try:
                self.loaded.emit(request, *payload)
            except RuntimeError:
                pass  # dialog was closed while loading

        threading.Thread(target=work, daemon=True).start()

    def _on_loaded(self, request: int, folders, listing, error: str):
        if request != self._request:
            return
        if error:
            self.status.setText(error)
            return
        for folder in folders:
            label = f"🗂  {folder.name}  (shared drive)" if folder.is_shared_drive else f"📁  {folder.name}"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, folder)
            self.list.addItem(item)
        if listing is None:
            self.status.setText(
                f"{len(folders)} shared folder(s). Double-click a folder to open it."
                if folders else
                "Nothing is shared with this service account yet. Share a folder with the "
                "address above, then press Refresh."
            )
            return
        self._listing = listing
        photos, videos = len(listing.files), len(listing.videos)
        extra = f" ({listing.skipped} other files ignored)" if listing.skipped else ""
        parts = [f"{photos} photo{'s' if photos != 1 else ''}"] if photos else []
        if videos:
            parts.append(f"{videos} video{'s' if videos != 1 else ''}")
        if parts:
            size = listing.total_bytes + sum(video.size for video in listing.videos)
            self.status.setText(
                f"This folder has {' and '.join(parts)} · {size / MIB:,.1f} MiB{extra}. "
                "Subfolders are not included; you can untick items after choosing."
            )
            self.choose_button.setEnabled(True)
        else:
            self.status.setText(
                f"No photos or videos directly in this folder{extra}. "
                + ("Open a subfolder, or pick another folder." if folders else "Pick another folder.")
            )

    def _accept(self):
        if self._listing is not None and (self._listing.files or self._listing.videos):
            self.selected = self._listing
            self.accept()
