"""Dialog for picking a capture folder on Google Drive.

It opens on the folders inside the shared gaussian-splat folder; one click on a
folder picks it. The New job page then lists its photos and videos with checkboxes.
"""
from __future__ import annotations

import threading

from PySide6 import QtCore, QtWidgets

from drive_client import DriveClient, DriveFolder, DriveListing
from ui_widgets import FramelessDialog

MIB = 1024 ** 2

# The shared folder that holds one folder per capture (tools/drive_upload.py uploads there).
HOME_FOLDER = "gaussian-splat"


def home_folder(shared: list[DriveFolder]) -> DriveFolder | None:
    """The capture folder among those shared with the account: the one named
    gaussian-splat, or the only one shared."""
    for folder in shared:
        if folder.name.strip().lower().replace(" ", "-") in (HOME_FOLDER, HOME_FOLDER + "s"):
            return folder
    return shared[0] if len(shared) == 1 else None


class DriveFolderDialog(FramelessDialog):
    # request id, folders, path to show them under, error text
    loaded = QtCore.Signal(int, object, object, str)
    # request id, listing of the picked folder, error text
    picked = QtCore.Signal(int, object, str)

    def __init__(self, client: DriveClient, parent=None, start_id: str = "", start_name: str = ""):
        super().__init__(parent, "Choose a capture folder")
        self.client = client
        self.selected: DriveListing | None = None
        self._current_id = start_id
        # None until the first load finds the home folder; [] is the top level.
        self._path: list[tuple[str, str]] | None = None
        self._request = 0
        self.resize(520, 520)
        root = self.body

        # Only shown when nothing is shared yet: how to make folders appear.
        self.share_info = QtWidgets.QWidget()
        share_layout = QtWidgets.QVBoxLayout(self.share_info)
        share_layout.setContentsMargins(0, 0, 0, 0)
        info = QtWidgets.QLabel(
            "Share your capture folder with this service account (Viewer access is enough):"
        )
        info.setWordWrap(True)
        info.setObjectName("helper")
        share_layout.addWidget(info)
        email = QtWidgets.QLineEdit(client.email)
        email.setReadOnly(True)
        share_layout.addWidget(email)
        self.share_info.hide()
        root.addWidget(self.share_info)

        nav = QtWidgets.QHBoxLayout()
        self.up_button = QtWidgets.QPushButton("↑ Up")
        self.up_button.clicked.connect(self._go_up)
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setObjectName("muted")
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.refresh_button.clicked.connect(self._navigate)
        nav.addWidget(self.up_button)
        nav.addWidget(self.path_label, 1)
        nav.addWidget(self.refresh_button)
        root.addLayout(nav)

        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("gpuList")
        # One click picks a folder; Enter does too.
        self.list.itemClicked.connect(self._pick)
        self.list.itemActivated.connect(self._pick)
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
        buttons.addWidget(cancel)
        root.addLayout(buttons)

        self.loaded.connect(self._on_loaded)
        self.picked.connect(self._on_picked)
        self._navigate()

    # ---- folder list ----------------------------------------------------
    def _go_up(self):
        if self._path:
            self._path.pop()
            self._navigate()

    def _navigate(self, *_):
        self._request += 1
        request = self._request
        path = None if self._path is None else list(self._path)
        self.list.clear()
        self.list.setEnabled(True)
        self.up_button.setEnabled(False)
        self.status.setText("Loading...")
        client = self.client

        def work():
            try:
                folders_path = path
                if folders_path is None:
                    shared = client.list_folders(None)
                    home = home_folder(shared)
                    if home is None:
                        self._emit(self.loaded, request, shared, [], "")
                        return
                    folders_path = [(home.id, home.name)]
                folders = client.list_folders(folders_path[-1][0] if folders_path else None)
                self._emit(self.loaded, request, folders, folders_path, "")
            except Exception as exc:  # shown to the user, never fatal
                self._emit(self.loaded, request, None, path, str(exc))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _emit(signal, *args):
        try:
            signal.emit(*args)
        except RuntimeError:
            pass  # dialog was closed while loading

    def _on_loaded(self, request: int, folders, path, error: str):
        if request != self._request:
            return
        if error:
            self.status.setText(error)
            return
        self._path = path
        self.up_button.setEnabled(bool(path))
        self.path_label.setText(" / ".join(n for _, n in path) if path else "Shared with you")
        self.share_info.setVisible(not path and not folders)
        for folder in folders:
            label = f"🗂  {folder.name}  (shared drive)" if folder.is_shared_drive else f"📁  {folder.name}"
            if folder.id == self._current_id:
                label += "   · current"
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, folder)
            self.list.addItem(item)
            if folder.id == self._current_id:
                self.list.setCurrentItem(item)
        if folders:
            self.status.setText("Click a folder to use it.")
        elif path:
            self.status.setText("This folder has no folders inside it.")
        else:
            self.status.setText(
                "Nothing is shared with this service account yet. Share a folder with the "
                "address above, then press Refresh."
            )

    # ---- picking ----------------------------------------------------------
    def _pick(self, item: QtWidgets.QListWidgetItem | None):
        # Enter after a click fires itemActivated as well; ignore it while loading.
        if item is None or not self.list.isEnabled():
            return
        folder: DriveFolder = item.data(QtCore.Qt.UserRole)
        self._request += 1
        request = self._request
        self.list.setEnabled(False)
        self.status.setText(f"Loading {folder.name}...")
        client = self.client

        def work():
            try:
                self._emit(self.picked, request, client.list_images(folder.id, folder.name), "")
            except Exception as exc:  # shown to the user, never fatal
                self._emit(self.picked, request, None, str(exc))

        threading.Thread(target=work, daemon=True).start()

    def _on_picked(self, request: int, listing: DriveListing | None, error: str):
        if request != self._request:
            return
        self.list.setEnabled(True)
        if error:
            self.status.setText(error)
            return
        if listing.files or listing.videos:
            self.selected = listing
            self.accept()
            return
        extra = f" ({listing.skipped} other files ignored)" if listing.skipped else ""
        self.status.setText(f"No photos or videos in {listing.folder_name}{extra}. Pick another folder.")
