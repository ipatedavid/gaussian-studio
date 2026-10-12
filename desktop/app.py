from __future__ import annotations

import html
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import shiboken6
from PySide6 import QtCore, QtGui, QtWidgets

from config import (
    DEFAULT_SPIRULA_IMAGE, ENGINE_LABELS, GUIDANCE_LABELS, MAX_STEP_SCALE, QUALITY_LEVELS,
    PHOTOS_ITEM, STEP_REFERENCE_IMAGES, AppConfig, DriveKeyError, install_drive_key,
    quality_steps, service_account_email, write_env_value,
)
from drive_client import DriveClient, DriveError, DriveFile, DriveListing
from gpu_speed import SpeedModel, duration_text, runs_from_history
from drive_dialog import DriveFolderDialog
from job_controller import JobController
from job_manager import JobManager
from storage import (
    GIB, MAX_VIDEO_FPS, MIB, MIN_VIDEO_FPS, InputPlan, VideoInfo, folder_photos, list_videos,
    plan_drive_input, plan_drive_mixed, plan_drive_video, plan_input, plan_mixed, probe_video,
)
import ui_theme as theme
from ui_theme import APP_STYLE
from ui_widgets import (
    DockButton, EdgeResizer, ElidedLabel, FramelessDialog, ResponsiveRow, TitleBar, confirm,
    confirm_summary, disable_wheel_changes, icon, icon_button, inform,
)

try:  # the embedded browser is optional; without it the preview opens externally
    from PySide6.QtWebEngineCore import QWebEngineScript
    from PySide6.QtWebEngineWidgets import QWebEngineView
except ImportError:  # pragma: no cover - depends on the PySide6 install
    QWebEngineView = None
    QWebEngineScript = None

# Spirula's viewer has a Pause button that pauses training while the pod keeps
# billing; the app's own Stop button is the control for that.
HIDE_VIEWER_PAUSE = (
    "(function(){var s=document.createElement('style');"
    "s.textContent='#btn-pause{display:none!important}';"
    "(document.head||document.documentElement).appendChild(s);})();"
)


def label(text: str, role: str = "muted") -> QtWidgets.QLabel:
    widget = QtWidgets.QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(role not in {
        "brand", "pageTitle", "statusBadge", "cardTitle", "eyebrow", "statusDot", "jobTitle",
    })
    return widget


def card(title: str, description: str = "") -> tuple[QtWidgets.QFrame, QtWidgets.QVBoxLayout]:
    frame = QtWidgets.QFrame()
    frame.setObjectName("card")
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(20, 18, 20, 20)
    layout.setSpacing(13)
    layout.addWidget(label(title, "cardTitle"))
    if description:
        layout.addWidget(label(description, "helper"))
    return frame, layout


def tip_html(*paragraphs: str) -> str:
    """Tooltip text: rich text so Qt word-wraps it."""
    return "<qt>" + "<br><br>".join(html.escape(part) for part in paragraphs if part) + "</qt>"


def help_button(*paragraphs: str) -> QtWidgets.QToolButton:
    """A small '?' that explains something on hover (or click)."""
    button = QtWidgets.QToolButton()
    button.setObjectName("helpButton")
    button.setText("?")
    button.setCursor(QtCore.Qt.PointingHandCursor)
    button.setToolTip(tip_html(*paragraphs))
    button.clicked.connect(
        lambda: QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), button.toolTip(), button)
    )
    return button


def titled(title: str, *tip: str, role: str = "fieldLabel") -> QtWidgets.QWidget:
    """A label with a '?' next to it."""
    row = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(label(title, role))
    if tip:
        layout.addWidget(help_button(*tip))
    layout.addStretch(1)
    return row


def field(title: str, widget: QtWidgets.QWidget, *tip: str) -> QtWidgets.QWidget:
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(titled(title, *tip))
    layout.addWidget(widget)
    return container


def segment(text: str) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setText(text)
    button.setObjectName("segment")
    button.setCheckable(True)
    return button


FLOATER_HELP = {
    "off": "Off: trains exactly as before. Keep this for a baseline to compare against.",
    "mild": "Mild: removes most stray blobs and see-through patches. A safe first thing to try.",
    "strong": (
        "Strong: cleans more aggressively, but can also erase thin real things such as "
        "wires, railings, branches or hair. The best choice for shiny rooms."
    ),
}
REFLECTION_HELP = (
    "Shiny floors, metal and glass: a splat can only shift its colour a little with "
    "the viewing angle, so a sharp reflection can't be painted onto the surface. "
    "Training fakes it by building a mirrored copy of the room behind the surface, "
    "which looks right from where you filmed and breaks everywhere else: see-through "
    "floors, haze in front of metal doors. Mild or Strong makes every pixel commit to "
    "one surface and limits that colour shifting, so you get a solid floor with a "
    "softened reflection instead of a hole. Pair it with AI geometry guidance."
)
DISTRACTION_HELP = {
    "off": "Off: everything in the frames is treated as part of the scene.",
    "mild": "Mild: things that appear in only some frames count for less when placing splats.",
    "strong": (
        "Strong: ignores them more firmly. Can also lose real detail that only a few "
        "frames saw clearly."
    ),
}
PRIMITIVE_HELP = {
    "3dgs": (
        "Standard (3DGS): the usual splat. Opens in every splat viewer and trains fastest. "
        "Handles fisheye photos too, slightly less exactly near the edges."
    ),
    "3dgut": (
        "Lens-accurate (3DGUT): models wide and fisheye lenses exactly. Tests on real "
        "fisheye footage show the clearest gains in rooms and other compact indoor spaces, "
        "and it gives cleaner geometry. Trains somewhat slower. Saves the same PLY format "
        "as Standard, so SuperSplat and other viewers open it."
    ),
}
GUIDANCE_HELP = {
    "off": "Off: no hints and nothing extra runs on the pod.",
    "surfaces": (
        "Surfaces (recommended): uses only the surface-direction hints, with the same "
        "settings Spirula's own presets use for clean geometry. The AI step adds a few "
        "minutes on the pod."
    ),
    "depth": (
        "Surfaces + depth (experimental): also uses the AI's distance guesses. They are "
        "made one frame at a time and don't always agree with each other, so they can "
        "soften fine detail. Their strength is under Advanced options."
    ),
}


def segment_row(options: tuple[tuple[str, str], ...], parent) -> tuple[QtWidgets.QHBoxLayout, dict]:
    row = QtWidgets.QHBoxLayout()
    row.setSpacing(8)
    group = QtWidgets.QButtonGroup(parent)
    group.setExclusive(True)
    buttons = {}
    for key, text in options:
        button = segment(text)
        group.addButton(button)
        row.addWidget(button)
        buttons[key] = button
    row.addStretch(1)
    return row, buttons


def checked_key(buttons: dict, default: str) -> str:
    return next((key for key, button in buttons.items() if button.isChecked()), default)


def repolish(widget: QtWidgets.QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s" if minutes else f"{secs}s"


def size_text(size: int) -> str:
    return f"{size / GIB:.2f} GiB" if size >= GIB else f"{size / MIB:,.0f} MiB"


def video_row(name: str, size: int, seconds: float, width: int, height: int) -> str:
    """One video's line in the capture checklist."""
    details = [name]
    if seconds > 0:
        whole = round(seconds)
        details.append(f"{whole // 60}:{whole % 60:02d}")
    if width and height:
        details.append(f"{width}×{height}")
    details.append(size_text(size))
    return "  ·  ".join(details)


def job_elapsed(job: JobController) -> float:
    if job.started_at is None:
        return 0
    return (job.finished_at or time.time()) - job.started_at


class SettingsDialog(FramelessDialog):
    """Settings grouped into sections, with a section list on the left."""

    SECTIONS = ("Connection", "Pods & cost", "Engines", "Preview")

    def __init__(self, config: AppConfig, parent: QtWidgets.QWidget):
        super().__init__(parent, "Settings")
        self.resize(760, 520)
        self.body.addWidget(label("Changes apply to new jobs.", "helper"))
        split = QtWidgets.QHBoxLayout()
        split.setSpacing(20)
        self.nav = QtWidgets.QListWidget()
        self.nav.setObjectName("navList")
        self.nav.setFixedWidth(170)
        self.nav.addItems(self.SECTIONS)
        split.addWidget(self.nav)
        self.pages = QtWidgets.QStackedWidget()
        split.addWidget(self.pages, 1)
        self.body.addLayout(split, 1)

        # Connection
        page, layout = self._page()
        self.runpod_key = QtWidgets.QLineEdit(config.runpod_api_key)
        self.runpod_key.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(field(
            "RunPod API key", self.runpod_key,
            "Create one at runpod.io → Settings → API Keys. Saving stores it in "
            "desktop/.env on this computer, readable only by your user.",
        ))
        self.ssh_key = QtWidgets.QLineEdit(config.ssh_private_key)
        ssh_row = QtWidgets.QWidget()
        ssh_layout = QtWidgets.QHBoxLayout(ssh_row)
        ssh_layout.setContentsMargins(0, 0, 0, 0)
        ssh_layout.setSpacing(8)
        ssh_layout.addWidget(self.ssh_key, 1)
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse_ssh)
        ssh_layout.addWidget(browse)
        layout.addWidget(field(
            "SSH private key", ssh_row, "The matching .pub file must sit next to it.",
        ))
        current = config.drive_credentials_path()
        self.drive_key = QtWidgets.QLineEdit(str(current) if current else "")
        self.drive_key.setReadOnly(True)
        self.drive_key.setPlaceholderText("Not set up (optional)")
        drive_row = QtWidgets.QWidget()
        drive_layout = QtWidgets.QHBoxLayout(drive_row)
        drive_layout.setContentsMargins(0, 0, 0, 0)
        drive_layout.setSpacing(8)
        drive_layout.addWidget(self.drive_key, 1)
        drive_browse = QtWidgets.QPushButton("Choose file")
        drive_browse.clicked.connect(self._browse_drive_key)
        drive_layout.addWidget(drive_browse)
        layout.addWidget(field(
            "Google Drive key", drive_row,
            "The service-account .json file that lets the app read photos and videos from "
            "Drive folders shared with it. Saving copies it to desktop/service-account.json.",
        ))
        self.drive_email = label("", "helper")
        self.drive_email.setWordWrap(True)
        self.drive_email.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self.drive_email)
        self._show_drive_email()
        layout.addStretch(1)

        # Pods & cost
        page, layout = self._page()
        self.cloud_type = QtWidgets.QComboBox()
        for text, value in (
            ("Secure Cloud only (recommended)", "SECURE"),
            ("Any cloud (cheapest)", "ALL"),
            ("Community Cloud only", "COMMUNITY"),
        ):
            self.cloud_type.addItem(text, value)
        self.cloud_type.setCurrentIndex(max(0, self.cloud_type.findData(config.cloud_type)))
        layout.addWidget(field(
            "Cloud type", self.cloud_type,
            "The app needs direct SSH, which RunPod only offers on pods with a public IP. "
            "Secure Cloud always has one; some Community hosts don't.",
        ))
        self.max_pods = QtWidgets.QSpinBox()
        self.max_pods.setRange(1, 20)
        self.max_pods.setValue(max(1, int(config.max_concurrent_pods)))
        self.max_pods.setSuffix(" pods")
        self.max_pods.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        layout.addWidget(field(
            "Maximum pods at once", self.max_pods,
            "New jobs are refused while this many pods are in use, counting untracked "
            "gaussian-* pods in your RunPod account.",
        ))
        layout.addStretch(1)

        # Engines
        page, layout = self._page()
        self.spirula_image = QtWidgets.QLineEdit(config.spirula_image)
        self.spirula_image.setPlaceholderText(DEFAULT_SPIRULA_IMAGE)
        layout.addWidget(field(
            "Spirula Studio image", self.spirula_image,
            "Spirula needs Ubuntu 24.04 or newer. Its program is downloaded onto the pod "
            "when a job starts.",
        ))
        self.image = QtWidgets.QLineEdit(config.runpod_image)
        layout.addWidget(field(
            "Nerfstudio image", self.image, "Used only when a job's engine is set to Nerfstudio.",
        ))
        layout.addStretch(1)

        # Preview
        page, layout = self._page()
        self.live_preview = QtWidgets.QCheckBox("Show a live training preview")
        self.live_preview.setChecked(bool(config.live_preview))
        preview_row = QtWidgets.QHBoxLayout()
        preview_row.addWidget(self.live_preview)
        preview_row.addWidget(help_button(
            "Opens the engine's own training viewer in the job's Preview tab, through the "
            "job's SSH connection. Turn it off for slightly faster training."
        ))
        preview_row.addStretch(1)
        layout.addLayout(preview_row)
        layout.addStretch(1)

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.setObjectName("textButton")
        cancel.clicked.connect(self.reject)
        save = QtWidgets.QPushButton("Save")
        save.setObjectName("primaryButton")
        save.setDefault(True)
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        self.body.addLayout(buttons)

    def _page(self) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.pages.addWidget(page)
        return page, layout

    def _browse_ssh(self):
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose SSH private key", self.ssh_key.text()
        )
        if chosen:
            self.ssh_key.setText(chosen)

    def _browse_drive_key(self):
        chosen, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose Google service-account key", self.drive_key.text() or str(Path.home()),
            "JSON key (*.json)",
        )
        if not chosen:
            return
        try:
            service_account_email(Path(chosen))
        except DriveKeyError as exc:
            inform(self, "Not a Drive key", str(exc))
            return
        self.drive_key.setText(chosen)
        self._show_drive_email()

    def _show_drive_email(self):
        text = self.drive_key.text().strip()
        try:
            email = service_account_email(Path(text)) if text else ""
        except DriveKeyError:
            email = ""
        self.drive_email.setText(
            f"Share your Drive folders with {email} (Viewer is enough)." if email else ""
        )
        self.drive_email.setVisible(bool(email))


class UntrackedPodsDialog(FramelessDialog):
    """Pods named gaussian-* in RunPod that no running job owns."""

    def __init__(self, pods: list[dict], parent: QtWidgets.QWidget):
        super().__init__(parent, "Untracked pods")
        self.resize(780, 440)
        self.to_terminate: list[dict] = []
        self.to_reconnect: dict | None = None
        self._pods = pods
        self.body.addWidget(titled(
            "Pods this app created that no job is tracking. They may be billing.",
            "These pods' names start with gaussian-, but no running job owns them, for "
            "example after a crash, a failed download, or an uncertain creation.",
            "Reconnect and finish picks the job back up: it follows the training, downloads "
            "the result when it's ready, then terminates the pod. Terminate deletes the pod "
            "and everything on it.",
            role="helper",
        ))
        self.table = QtWidgets.QTableWidget(len(pods), 5)
        self.table.setHorizontalHeaderLabels(["Job", "Pod", "Status", "Rate", "Started"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for row, pod in enumerate(pods):
            started = (
                datetime.fromtimestamp(pod["created_at"]).strftime("%Y-%m-%d %H:%M")
                if pod.get("created_at") else "not in this app's records"
            )
            values = (
                pod.get("title") or pod["name"], pod["id"], pod["status"],
                f"${pod['cost']:.3f}/hr" if pod.get("cost") is not None else "unknown",
                started,
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self.table.selectAll()
        self.body.addWidget(self.table, 1)
        buttons = QtWidgets.QHBoxLayout()
        close = QtWidgets.QPushButton("Close")
        close.setObjectName("textButton")
        close.clicked.connect(self.reject)
        terminate = QtWidgets.QPushButton("Terminate selected")
        terminate.setObjectName("dangerButton")
        terminate.clicked.connect(self._terminate)
        self.reconnect_button = QtWidgets.QPushButton("Reconnect and finish")
        self.reconnect_button.setObjectName("primaryButton")
        self.reconnect_button.setToolTip("Select exactly one pod.")
        self.reconnect_button.clicked.connect(self._reconnect)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        buttons.addWidget(terminate)
        buttons.addStretch(1)
        buttons.addWidget(close)
        buttons.addWidget(self.reconnect_button)
        self._selection_changed()
        self.body.addLayout(buttons)

    def _selected_rows(self) -> list[int]:
        return sorted({index.row() for index in self.table.selectionModel().selectedRows()})

    def _selection_changed(self):
        self.reconnect_button.setEnabled(len(self._selected_rows()) == 1)

    def _reconnect(self):
        rows = self._selected_rows()
        if len(rows) != 1:
            return
        pod = self._pods[rows[0]]
        if confirm(
            self, "Reconnect to pod",
            f"Reconnect to pod {pod['id']} ({pod.get('title') or pod['name']})?\n\n"
            "The app waits for training to finish, downloads the result, then terminates "
            "the pod. If the pod's own job script has stopped, the app does the last step "
            "itself.",
            "Reconnect",
        ):
            self.to_reconnect = pod
            self.accept()

    def _terminate(self):
        rows = self._selected_rows()
        if not rows:
            return
        chosen = [self._pods[row] for row in rows]
        if confirm(
            self, "Terminate pods",
            "Terminate these pods now?\n\n"
            + "\n".join(f"{pod['id']}  ({pod.get('title') or pod['name']})" for pod in chosen)
            + "\n\nAnything on them that was not downloaded is lost.",
            "Terminate", danger=True,
        ):
            self.to_terminate = chosen
            self.accept()


def job_status(job: JobController, stage: str) -> tuple[str, str]:
    """(state, one-line description) shared by the job card and the dock icon."""
    if job.running:
        state = "stopping" if job.stop_in_progress else "running"
        text = f"{stage} · {format_duration(job_elapsed(job))}"
    elif job.active_pod_id:
        state = "kept"
        text = f"Pod kept, may be billing · {format_duration(job_elapsed(job))}"
    else:
        state = job.outcome or "failed"
        text = f"{state.capitalize()} · {format_duration(job_elapsed(job))}"
    if job.active_pod_id and job.cost_per_hour is not None:
        text += f" · ${job.cost_per_hour:.2f}/hr"
    return state, text


class JobCard(QtWidgets.QFrame):
    """One row of the sidebar job list."""

    def __init__(self, job: JobController, context: str):
        super().__init__()
        self.setObjectName("jobCard")
        self.job = job
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        top = QtWidgets.QHBoxLayout()
        top.setSpacing(8)
        self.dot = label("●", "statusDot")
        top.addWidget(self.dot)
        self.title = label(job.title, "jobTitle")
        self.title.setToolTip(job.title)
        top.addWidget(self.title, 1)
        layout.addLayout(top)
        self.context = label("", "cardMeta")
        self.context.setWordWrap(False)
        self.context.setToolTip(context)
        self.context.setText(
            self.context.fontMetrics().elidedText(context, QtCore.Qt.ElideRight, 260)
        )
        layout.addWidget(self.context)
        self.status = label("", "cardMeta")
        layout.addWidget(self.status)
        for widget in (self, self.dot, self.title, self.context, self.status):
            widget.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.refresh("Starting")

    def refresh(self, stage: str) -> None:
        job = self.job
        state, text = job_status(job, stage)
        title = self.title.fontMetrics().elidedText(job.title, QtCore.Qt.ElideMiddle, 210)
        if self.title.text() != title:
            self.title.setText(title)
        self.status.setText(text)
        if self.dot.property("state") != state:
            self.dot.setProperty("state", state)
            repolish(self.dot)


class JobView(QtWidgets.QWidget):
    """Detail page of one job: stage tracker, live log and its own controls."""

    STAGES = ("Start pod", "Upload", "Prepare", "Train", "Export", "Download")
    remove_requested = QtCore.Signal(object)

    def __init__(self, job: JobController, context: str, rate: str):
        super().__init__()
        self.job = job
        self._stage_index = -1
        self._error_shown = False
        # Scrolls instead of clipping when the window is shorter than the page.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("setupScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        content.setObjectName("appRoot")
        content.setMinimumHeight(480)
        scroll.setWidget(content)
        outer.addWidget(scroll)
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(24, 16, 24, 18)
        root.setSpacing(12)
        heading = QtWidgets.QVBoxLayout()
        heading.setSpacing(4)
        heading.addWidget(label(f"JOB {job.job_id}", "eyebrow"))
        heading.addWidget(label(job.title, "pageTitle"))
        heading.addWidget(label(context, "helper"))
        root.addLayout(heading)
        self.notice = label("", "notice")
        self.notice.hide()
        root.addWidget(self.notice)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        overview = QtWidgets.QFrame()
        overview.setObjectName("activityCard")
        overview_layout = QtWidgets.QVBoxLayout(overview)
        overview_layout.setContentsMargins(20, 17, 20, 18)
        overview_layout.setSpacing(12)
        # Side by side when there's room, one under the other at half-screen widths.
        stats = ResponsiveRow(640, spacing=4)
        self.gpu = label("GPU: " + (job.gpu_label or "waiting"), "helper")
        self.pod = label("Pod: not created", "helper")
        self.rate = label(f"Rate: {rate}", "helper")
        self.elapsed = label("Elapsed: 0s", "helper")
        stats.box.addWidget(self.gpu, 3)
        stats.box.addWidget(self.pod, 2)
        stats.box.addWidget(self.rate, 1)
        stats.box.addWidget(self.elapsed, 1)
        overview_layout.addWidget(stats)
        self.status = label("Starting job...", "cardTitle")
        overview_layout.addWidget(self.status)
        stage_row = QtWidgets.QHBoxLayout()
        stage_row.setSpacing(7)
        self.stage_labels = []
        for stage in self.STAGES:
            stage_label = label(stage, "stageBadge")
            stage_label.setAlignment(QtCore.Qt.AlignCenter)
            stage_label.setProperty("state", "pending")
            stage_row.addWidget(stage_label, 1)
            self.stage_labels.append(stage_label)
        overview_layout.addLayout(stage_row)
        splitter.addWidget(overview)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setObjectName("jobTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setDrawBase(False)

        preview_frame = QtWidgets.QFrame()
        preview_frame.setObjectName("activityCard")
        preview_layout = QtWidgets.QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(20, 16, 20, 18)
        preview_layout.setSpacing(10)
        preview_heading = QtWidgets.QHBoxLayout()
        preview_heading.addWidget(titled(
            "Live training preview",
            "Rendered on the pod's GPU and streamed through your SSH connection, so it can be "
            "slow on distant pods. Lower the resolution in the viewer's panel to speed it up. "
            "Nothing is rendered while this tab is hidden.",
            role="cardTitle",
        ))
        preview_heading.addStretch(1)
        self.reload_preview = QtWidgets.QPushButton("Reload")
        self.reload_preview.setObjectName("textButton")
        self.reload_preview.clicked.connect(self._reload_preview)
        preview_heading.addWidget(self.reload_preview)
        self.open_preview = QtWidgets.QPushButton("Open in browser")
        self.open_preview.setObjectName("textButton")
        self.open_preview.clicked.connect(self._open_preview_externally)
        preview_heading.addWidget(self.open_preview)
        preview_layout.addLayout(preview_heading)
        self.preview_stack = QtWidgets.QStackedWidget()
        self.preview_message = label("", "helper")
        self.preview_message.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_stack.addWidget(self.preview_message)
        self.preview_holder = QtWidgets.QWidget()
        holder_layout = QtWidgets.QVBoxLayout(self.preview_holder)
        holder_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_stack.addWidget(self.preview_holder)
        preview_layout.addWidget(self.preview_stack, 1)
        self.tabs.addTab(preview_frame, "Preview")
        self._web = None
        self._web_url: str | None = None
        self._syncing = False
        self._sync_pending = False
        self._preview_url: str | None = None
        self._auto_switched = False
        self._set_preview_message(
            "The live preview appears here once training starts. Until then, follow "
            "the Log tab." if job.config.live_preview else
            "Live preview is turned off for this job (Settings)."
        )

        log_frame = QtWidgets.QFrame()
        log_frame.setObjectName("activityCard")
        log_layout = QtWidgets.QVBoxLayout(log_frame)
        log_layout.setContentsMargins(20, 16, 20, 18)
        log_layout.setSpacing(10)
        log_heading = QtWidgets.QHBoxLayout()
        log_heading.addWidget(label("Live job log", "cardTitle"))
        log_heading.addStretch(1)
        open_log = QtWidgets.QPushButton("Open log file")
        open_log.setObjectName("textButton")
        open_log.setToolTip(str(job.log_path) if job.log_path else "No log file")
        open_log.setEnabled(job.log_path is not None)
        open_log.clicked.connect(self._open_log)
        log_heading.addWidget(open_log)
        log_layout.addLayout(log_heading)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setObjectName("jobLog")
        self.log.setReadOnly(True)
        # The complete log is also written to a file; keep the widget bounded.
        self.log.setMaximumBlockCount(20000)
        log_layout.addWidget(self.log, 1)
        self.tabs.addTab(log_frame, "Log")
        self.tabs.setCurrentIndex(1)
        self.tabs.currentChanged.connect(lambda _index: self._schedule_sync())
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([165, 420])
        root.addWidget(splitter, 1)

        actions = QtWidgets.QHBoxLayout()
        self.remove_button = QtWidgets.QPushButton("Remove from list")
        self.remove_button.setObjectName("textButton")
        self.remove_button.setToolTip(
            "Removes the job from the app and its history. Its results and log file stay on disk."
        )
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self.job))
        actions.addWidget(self.remove_button)
        actions.addStretch(1)
        self.open_results_button = QtWidgets.QPushButton("Open results")
        self.open_results_button.setObjectName("tonalButton")
        self.open_results_button.clicked.connect(self._open_results)
        actions.addWidget(self.open_results_button)
        self.stop_button = QtWidgets.QPushButton("Stop job")
        self.stop_button.setObjectName("dangerButton")
        self.stop_button.clicked.connect(self._stop)
        actions.addWidget(self.stop_button)
        root.addLayout(actions)

        job.log_line.connect(self._append_log)
        job.status_changed.connect(self._set_status)
        job.error.connect(self._error)
        job.active_pod_changed.connect(self._active_pod_changed)
        job.stop_started.connect(self.refresh)
        job.finished.connect(self._finished)
        job.preview_ready.connect(self._preview_ready)
        job.preview_closed.connect(self._preview_closed)
        if job.historic:
            self._show_history()
        self.refresh()

    HISTORY_STATUS = {
        "finished": "Finished",
        "stopped": "Stopped",
        "failed": "Failed",
        "interrupted": "Interrupted: the app closed while this job was running",
    }

    def _show_history(self):
        """A past job: its record, its log from disk, no live controls."""
        job = self.job
        self.pod.setText(f"Pod: {job.last_pod_id} (ended)" if job.last_pod_id else "Pod: none")
        self.status.setText(self.HISTORY_STATUS.get(job.outcome, job.status_text))
        self._set_preview_message("This job has ended, so there is no live preview.")
        if job.outcome == "interrupted":
            self.show_notice(
                "If this job's pod is still running, it's listed under untracked pods, where "
                "you can reconnect to it or terminate it."
            )
        lines = []
        if job.log_path and job.log_path.is_file():
            with job.log_path.open(encoding="utf-8", errors="replace") as handle:
                lines = handle.read().splitlines()[-20000:]
            self.log.setPlainText("\n".join(lines))
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())
        else:
            self.log.setPlainText("The log file for this job is no longer on disk.")
        for line in lines:
            # Lines are "[HH:MM:SS] message".
            self._track_stage(line.split("] ", 1)[-1])
        if job.outcome == "finished":
            self._render_stages(complete=True)

    # ---- live preview ------------------------------------------------
    PREVIEW_TAB = 0

    def _set_preview_message(self, text: str) -> None:
        self.preview_message.setText(text)
        self.preview_stack.setCurrentWidget(self.preview_message)
        self.reload_preview.setEnabled(False)
        self.open_preview.setEnabled(self._preview_url is not None)

    def _preview_ready(self, url: str) -> None:
        self._preview_url = url
        self.open_preview.setEnabled(True)
        if not self._auto_switched:
            self._auto_switched = True
            self.tabs.setCurrentIndex(self.PREVIEW_TAB)
        self._schedule_sync()

    def _preview_closed(self, reason: str) -> None:
        self._preview_url = None
        self._destroy_web()
        self._set_preview_message(reason)

    def _schedule_sync(self) -> None:
        # Never create or load the browser from inside a show/hide event: doing
        # so creates a native window, which sends this view another show event
        # before the first call has finished, and the recursion crashes the app.
        if not self._sync_pending:
            self._sync_pending = True
            QtCore.QTimer.singleShot(0, self._run_scheduled_sync)

    def _run_scheduled_sync(self) -> None:
        self._sync_pending = False
        self._sync_preview()

    def _sync_preview(self, *_):
        """Show the viewer only while it is visible, so the pod renders nothing otherwise."""
        if self._syncing:
            return
        self._syncing = True
        try:
            self._apply_preview_state()
        finally:
            self._syncing = False

    def _apply_preview_state(self) -> None:
        active = (
            self._preview_url is not None and self.isVisible()
            and self.tabs.currentIndex() == self.PREVIEW_TAB
        )
        if not active:
            if self._web is not None:
                self._destroy_web()
                if self._preview_url is not None:
                    self._set_preview_message("Preview paused while hidden.")
            return
        if QWebEngineView is None:
            self._set_preview_message(
                "This computer's Qt has no embedded browser. Use 'Open in browser'."
            )
            self.open_preview.setEnabled(True)
            return
        if self._web is None:
            self._web = QWebEngineView(self.preview_holder)
            script = QWebEngineScript()
            script.setName("hide-viewer-pause")
            script.setSourceCode(HIDE_VIEWER_PAUSE)
            script.setInjectionPoint(QWebEngineScript.DocumentReady)
            script.setWorldId(QWebEngineScript.MainWorld)
            script.setRunsOnSubFrames(False)
            self._web.page().scripts().insert(script)
            self.preview_holder.layout().addWidget(self._web)
            self._web_url = None
        if self._web_url != self._preview_url:
            # Remember what was requested: url() stays empty while a page loads.
            self._web_url = self._preview_url
            self._web.setUrl(QtCore.QUrl(self._preview_url))
        self.preview_stack.setCurrentWidget(self.preview_holder)
        self.reload_preview.setEnabled(True)

    def _destroy_web(self) -> None:
        # Deleted right away (not deleteLater): a web page still alive when Qt
        # shuts its browser engine down crashes the app on exit.
        if self._web is not None:
            web, self._web = self._web, None
            self._web_url = None
            web.stop()
            web.setParent(None)
            shiboken6.delete(web)

    def shutdown(self) -> None:
        self._preview_url = None
        self._destroy_web()

    def _reload_preview(self):
        if self._web is not None:
            self._web.reload()

    def _open_preview_externally(self):
        if self._preview_url:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._preview_url))

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_sync()

    def hideEvent(self, event):
        super().hideEvent(event)
        self._schedule_sync()

    @property
    def stage_name(self) -> str:
        if self._stage_index < 0:
            return "Starting"
        return self.STAGES[self._stage_index]

    def refresh(self, *_):
        job = self.job
        self.elapsed.setText(f"Elapsed: {format_duration(job_elapsed(job))}")
        if job.gpu_label:
            self.gpu.setText(f"GPU: {job.gpu_label}")
        if job.cost_per_hour is not None:
            self.rate.setText(f"Rate: ${job.cost_per_hour:.3f}/hr")
        can_stop = (job.running or job.active_pod_id is not None) and not job.stop_in_progress
        self.stop_button.setEnabled(can_stop)
        self.stop_button.setText(
            "Terminate pod" if job.active_pod_id and not job.running
            else "Stop job" if job.active_pod_id or not job.running
            else "Cancel job"
        )
        self.remove_button.setEnabled(not job.busy)
        self.open_results_button.setEnabled(
            job.results_dir is not None and job.results_dir.is_dir()
        )

    def _set_status(self, message: str):
        self.status.setText(message)
        lower = message.lower()
        if any(word in lower for word in ("creating", "startup", "pod ", "ssh")):
            self._advance_stage(0)
        elif "upload" in lower or "google drive" in lower:
            self._advance_stage(1)
        elif "processing" in lower:
            self._advance_stage(2)
        elif "download" in lower:
            self._advance_stage(5)
        self.refresh()

    def _advance_stage(self, index: int):
        if index > self._stage_index:
            self._stage_index = index
            self._render_stages()

    def _render_stages(self, complete: bool = False):
        for index, widget in enumerate(self.stage_labels):
            state = (
                "done" if complete or index < self._stage_index
                else "active" if index == self._stage_index
                else "pending"
            )
            widget.setProperty("state", state)
            repolish(widget)

    def _append_log(self, message: str):
        scrollbar = self.log.verticalScrollBar()
        follow = scrollbar.value() >= scrollbar.maximum() - 2
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {message}")
        if follow:
            scrollbar.setValue(scrollbar.maximum())
        self._track_stage(message)

    def _track_stage(self, message: str):
        lower = message.lower()
        if "uploading" in lower or "scp" in lower or "[drive]" in lower or "google drive" in lower:
            self._advance_stage(1)
        if "[pipeline] processing images" in lower or "[pipeline] extracting" in lower:
            self._advance_stage(2)
        if "[pipeline] starting splatfacto" in lower or "[pipeline] starting spirula" in lower:
            self._advance_stage(3)
        if "[pipeline] exporting" in lower:
            self._advance_stage(4)
        if (
            "downloading the result files" in lower
            or lower.startswith(("downloading gaussian_splat", "downloading pipeline_summary"))
        ):
            self._advance_stage(5)

    def show_notice(self, message: str, *, error: bool = False):
        self.notice.setText(message)
        self.notice.setProperty("state", "error" if error else "normal")
        repolish(self.notice)
        self.notice.setVisible(bool(message))

    def _error(self, message: str):
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] [ERROR] {message}")
        self.show_notice(message, error=True)

    def _active_pod_changed(self, pod_id):
        self.pod.setText(f"Pod: {pod_id}" if pod_id else "Pod: terminated")
        self.refresh()

    def _finished(self, success: bool):
        job = self.job
        if job.outcome == "stopped":
            self._append_log("Job stopped.")
        else:
            self._append_log("Pipeline completed." if success else "Pipeline failed.")
        if success:
            self._render_stages(complete=True)
            self.show_notice(f"Results saved to {job.results_dir}")
        self.refresh()

    def _stop(self):
        job = self.job
        if job.active_pod_id:
            question = (
                f"Terminate pod {job.active_pod_id} for '{job.title}' now?\n\n"
                "Processing stops. Results that have not been downloaded are lost."
            )
        else:
            question = (
                f"Cancel '{job.title}'?\n\nNo pod has been created yet; if a creation request "
                "is already in flight, its pod is terminated as soon as it appears."
            )
        if confirm(self, "Stop job", question, "Stop job", danger=True):
            self.stop_button.setEnabled(False)
            job.stop_pod()

    def _open_results(self):
        if self.job.results_dir and self.job.results_dir.is_dir():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.job.results_dir)))

    def _open_log(self):
        if self.job.log_path and self.job.log_path.is_file():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.job.log_path)))


class MainWindow(QtWidgets.QMainWindow):
    drive_listing_ready = QtCore.Signal(object, str)
    drive_videos_probed = QtCore.Signal(object)  # list of DriveFile with exact details

    # Below this window width the jobs panel collapses to the icon dock.
    DOCK_BREAKPOINT = 1180
    DOCK_EXPANDED_WIDTH = 288
    DOCK_COLLAPSED_WIDTH = 80

    def __init__(self):
        super().__init__()
        self.config = AppConfig()
        self.config.load_state()
        self.manager = JobManager(self.config)
        self._gpu_lookup_settings = None
        self._gpu_candidates: list[dict] = []
        self._current_plan: InputPlan | None = None
        self._shown_plan: InputPlan | None = None  # last capture that planned cleanly
        self._drive_listing: DriveListing | None = None
        # (path, size, mtime) -> VideoInfo, or None when ffprobe could not read it.
        self._local_video_infos: dict[tuple, VideoInfo | None] = {}
        self._probed_videos: dict[tuple, DriveFile] = {}  # (id, size, md5) -> probed copy
        self._views: dict[JobController, JobView] = {}
        self._cards: dict[JobController, tuple[QtWidgets.QListWidgetItem, JobCard]] = {}
        self._dock_buttons: dict[JobController, DockButton] = {}
        self._dock_expanded = True
        self._narrow: bool | None = None
        self.setWindowTitle("Gaussian Studio")
        # The app draws its own title bar; the window manager still moves,
        # resizes and snaps the window.
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.resize(1360, 840)
        # Small enough for a quarter-screen tile; pages scroll when shorter.
        self.setMinimumSize(600, 400)
        self.setStyleSheet(APP_STYLE)
        self._resizer = EdgeResizer(self)
        self._auto_gpu_timer = QtCore.QTimer(self)
        self._auto_gpu_timer.setSingleShot(True)
        self._auto_gpu_timer.setInterval(700)
        self._auto_gpu_timer.timeout.connect(self._auto_find_gpus)
        self._build_ui()
        disable_wheel_changes(self)
        self._load_settings()
        self._connect_signals()
        self._input_kind_changed()
        self._update_quality_summary()
        self._refresh_credentials_label()
        self._restore_drive_selection()
        self._ticker = QtCore.QTimer(self)
        self._ticker.setInterval(1000)
        self._ticker.timeout.connect(self._tick)
        self._ticker.start()
        self._tick()
        self._load_history()
        self.manager.start_monitoring()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.root_widget = QtWidgets.QWidget()
        self.root_widget.setObjectName("appRoot")
        self.root_widget.setProperty("framed", True)
        self.setCentralWidget(self.root_widget)
        root = QtWidgets.QVBoxLayout(self.root_widget)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)

        self.title_bar = TitleBar(self, "Gaussian Studio")
        mark = QtWidgets.QLabel("GS")
        mark.setObjectName("appMark")
        self.title_bar.leading.addWidget(mark)
        self.status = label("Ready", "statusBadge")
        self.status.setAlignment(QtCore.Qt.AlignCenter)
        self.title_bar.trailing.addWidget(self.status)
        self.credentials_label = QtWidgets.QPushButton("")
        self.credentials_label.setObjectName("credentialsChip")
        self.credentials_label.setCursor(QtCore.Qt.PointingHandCursor)
        self.credentials_label.clicked.connect(self._open_settings)
        self.title_bar.trailing.addWidget(self.credentials_label)
        self.settings_button = icon_button("settings", "Settings")
        self.settings_button.clicked.connect(self._open_settings)
        self.title_bar.trailing.addWidget(self.settings_button)
        self.title_bar.trailing.addSpacing(6)
        root.addWidget(self.title_bar)

        notice_holder = QtWidgets.QWidget()
        notice_layout = QtWidgets.QHBoxLayout(notice_holder)
        notice_layout.setContentsMargins(12, 0, 12, 6)
        self.global_notice = QtWidgets.QPushButton("")
        self.global_notice.setObjectName("globalNotice")
        self.global_notice.setToolTip("Click to dismiss")
        self.global_notice.clicked.connect(self.global_notice.hide)
        self.global_notice.hide()
        notice_layout.addWidget(self.global_notice)
        root.addWidget(notice_holder)

        body = QtWidgets.QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        self.pages = QtWidgets.QStackedWidget()
        self.setup_page = self._build_setup_page()
        self.pages.addWidget(self.setup_page)
        body.addWidget(self.pages, 1)
        root.addLayout(body, 1)

    def _build_sidebar(self) -> QtWidgets.QWidget:
        """The jobs panel: a full list when expanded, an icon dock when collapsed."""
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(self.DOCK_EXPANDED_WIDTH)
        outer = QtWidgets.QVBoxLayout(self.sidebar)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Expanded panel
        self.sidebar_full = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.sidebar_full)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(10)
        header = QtWidgets.QHBoxLayout()
        collapse = icon_button("dock", "Collapse to icons")
        collapse.clicked.connect(lambda: self._toggle_dock())
        header.addWidget(collapse)
        header.addWidget(label("Jobs", "sectionTitle"))
        header.addStretch(1)
        layout.addLayout(header)
        self.new_job_button = QtWidgets.QPushButton("  New job")
        self.new_job_button.setObjectName("fabButton")
        self.new_job_button.setIcon(icon("plus", theme.ON_PRIMARY_CONTAINER))
        self.new_job_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.new_job_button.clicked.connect(self._show_setup)
        layout.addWidget(self.new_job_button, alignment=QtCore.Qt.AlignLeft)
        self.job_list = QtWidgets.QListWidget()
        self.job_list.setObjectName("jobList")
        self.job_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.job_list.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.job_list.currentItemChanged.connect(self._job_selected)
        layout.addWidget(self.job_list, 1)
        self.empty_jobs = label(
            "No jobs yet. Jobs you start appear here; several can run at once.", "helper"
        )
        layout.addWidget(self.empty_jobs)
        self.untracked_button = QtWidgets.QPushButton("")
        self.untracked_button.setObjectName("warningBanner")
        self.untracked_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.untracked_button.clicked.connect(self._review_untracked)
        self.untracked_button.hide()
        layout.addWidget(self.untracked_button)
        self.totals = label("", "cardMeta")
        layout.addWidget(self.totals)
        outer.addWidget(self.sidebar_full)

        # Collapsed dock
        self.sidebar_rail = QtWidgets.QWidget()
        rail = QtWidgets.QVBoxLayout(self.sidebar_rail)
        rail.setContentsMargins(0, 10, 0, 12)
        rail.setSpacing(8)
        expand = icon_button("dock", "Show job details")
        expand.clicked.connect(lambda: self._toggle_dock())
        rail.addWidget(expand, alignment=QtCore.Qt.AlignHCenter)
        self.rail_new_job = QtWidgets.QPushButton("")
        self.rail_new_job.setObjectName("fabButton")
        self.rail_new_job.setIcon(icon("plus", theme.ON_PRIMARY_CONTAINER, 22))
        self.rail_new_job.setIconSize(QtCore.QSize(22, 22))
        self.rail_new_job.setFixedSize(52, 52)
        self.rail_new_job.setToolTip("New job")
        self.rail_new_job.setCursor(QtCore.Qt.PointingHandCursor)
        self.rail_new_job.clicked.connect(self._show_setup)
        rail.addWidget(self.rail_new_job, alignment=QtCore.Qt.AlignHCenter)
        rail.addSpacing(4)
        dock_scroll = QtWidgets.QScrollArea()
        dock_scroll.setWidgetResizable(True)
        dock_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        dock_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        dock_scroll.setStyleSheet("background: transparent;")
        dock_inner = QtWidgets.QWidget()
        self.dock_layout = QtWidgets.QVBoxLayout(dock_inner)
        self.dock_layout.setContentsMargins(0, 0, 0, 0)
        self.dock_layout.setSpacing(4)
        self.dock_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
        dock_scroll.setWidget(dock_inner)
        rail.addWidget(dock_scroll, 1)
        self.rail_warning = icon_button("warning", "", color=theme.WARNING, size=22)
        self.rail_warning.clicked.connect(self._review_untracked)
        self.rail_warning.hide()
        rail.addWidget(self.rail_warning, alignment=QtCore.Qt.AlignHCenter)
        self.rail_totals = label("", "cardMeta")
        self.rail_totals.setAlignment(QtCore.Qt.AlignCenter)
        self.rail_totals.setWordWrap(True)
        rail.addWidget(self.rail_totals)
        self.sidebar_rail.hide()
        outer.addWidget(self.sidebar_rail)
        return self.sidebar

    def _build_setup_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll = QtWidgets.QScrollArea()
        scroll.setObjectName("setupScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        content = QtWidgets.QWidget()
        content.setObjectName("appRoot")
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(24, 12, 24, 16)
        content_layout.setSpacing(14)
        content_layout.addWidget(label("New job", "pageTitle"))
        self.setup_notice = label("", "notice")
        self.setup_notice.hide()
        content_layout.addWidget(self.setup_notice)

        # Two columns on a wide window, one column at half-screen widths.
        columns = ResponsiveRow(900, spacing=16)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(16)

        source_card, source_layout = card("Capture")
        source_modes = QtWidgets.QHBoxLayout()
        source_modes.setSpacing(8)
        self.local_mode = segment("Local")
        self.drive_mode = segment("Google Drive")
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        for button in (self.local_mode, self.drive_mode):
            self.mode_group.addButton(button)
            source_modes.addWidget(button)
        source_modes.addStretch(1)
        source_layout.addLayout(source_modes)

        self.source_pages = QtWidgets.QStackedWidget()
        local_page = QtWidgets.QWidget()
        local_layout = QtWidgets.QHBoxLayout(local_page)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(8)
        self.input_folder = QtWidgets.QLineEdit()
        self.input_folder.setPlaceholderText("Folder with the photos and/or videos")
        self.input_folder.setToolTip(
            "Photos and videos directly in this folder are listed below; untick what to "
            "leave out. Photos and videos together become one splat."
        )
        local_layout.addWidget(self.input_folder, 1)
        local_browse = QtWidgets.QPushButton("Choose folder")
        local_browse.clicked.connect(self._browse_local)
        local_layout.addWidget(local_browse)
        self.source_pages.addWidget(local_page)

        drive_page = QtWidgets.QWidget()
        drive_layout = QtWidgets.QVBoxLayout(drive_page)
        drive_layout.setContentsMargins(0, 0, 0, 0)
        drive_layout.setSpacing(9)
        self.drive_selection = QtWidgets.QLineEdit()
        self.drive_selection.setReadOnly(True)
        self.drive_selection.setPlaceholderText("No Google Drive folder selected")
        drive_row = QtWidgets.QHBoxLayout()
        drive_row.setSpacing(8)
        drive_row.addWidget(self.drive_selection, 1)
        self.drive_browse = QtWidgets.QPushButton("Choose folder")
        self.drive_browse.clicked.connect(self._browse_drive)
        drive_row.addWidget(self.drive_browse)
        drive_layout.addLayout(drive_row)
        self.results_folder = QtWidgets.QLineEdit()
        self.results_folder.setPlaceholderText(str(self.config.results_root()))
        results_widget = QtWidgets.QWidget()
        results_row = QtWidgets.QHBoxLayout(results_widget)
        results_row.setContentsMargins(0, 0, 0, 0)
        results_row.setSpacing(8)
        results_row.addWidget(self.results_folder, 1)
        results_browse = QtWidgets.QPushButton("Browse")
        results_browse.clicked.connect(self._browse_results)
        results_row.addWidget(results_browse)
        drive_layout.addWidget(field(
            "Save results to", results_widget,
            "Results are saved on this computer. The pod downloads your files straight from "
            "Google; nothing is uploaded from here.",
        ))
        self.source_pages.addWidget(drive_page)
        source_layout.addWidget(self.source_pages)

        # What's in the chosen folder: one row for all its photos, one per video.
        self.capture_items = QtWidgets.QListWidget()
        self.capture_items.setObjectName("captureItems")
        self.capture_items.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.capture_items.setFocusPolicy(QtCore.Qt.NoFocus)
        self.capture_items.setToolTip("Untick anything to leave it out of this job.")
        self.capture_items.hide()
        source_layout.addWidget(self.capture_items)

        self.frames_box = QtWidgets.QWidget()
        frames_layout = QtWidgets.QVBoxLayout(self.frames_box)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(8)
        frame_row = QtWidgets.QWidget()
        frame_layout = QtWidgets.QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)
        self.video_fps = QtWidgets.QDoubleSpinBox()
        self.video_fps.setRange(MIN_VIDEO_FPS, MAX_VIDEO_FPS)
        self.video_fps.setDecimals(1)
        self.video_fps.setSingleStep(0.5)
        self.video_fps.setSuffix(" fps")
        self.video_fps.setToolTip("Frames taken per second of video, across the whole video.")
        frame_layout.addWidget(self.video_fps)
        self.frame_fps_label = QtWidgets.QLabel("")
        self.frame_fps_label.setToolTip("Frames the pod will take from all the videos together.")
        frame_layout.addWidget(self.frame_fps_label)
        frame_layout.addStretch(1)
        frames_layout.addWidget(field(
            "Frames per second", frame_row,
            "How many frames to take from each second of video, from start to end. "
            "3 suits a slow walk; use 2 for a small room, more if you moved quickly. "
            "More frames means longer camera solving on the pod.",
        ))
        self.frames_notice = label(
            "More frames than training can use well. Consider a lower fps.", "notice")
        self.frames_notice.setWordWrap(True)
        self.frames_notice.hide()
        frames_layout.addWidget(self.frames_notice)
        source_layout.addWidget(self.frames_box)

        self.fisheye_box = QtWidgets.QWidget()
        fisheye_layout = QtWidgets.QVBoxLayout(self.fisheye_box)
        fisheye_layout.setContentsMargins(0, 0, 0, 0)
        fisheye_layout.setSpacing(8)
        self.fisheye = QtWidgets.QCheckBox("Shot with a fisheye lens")
        fisheye_row = QtWidgets.QHBoxLayout()
        fisheye_row.addWidget(self.fisheye)
        fisheye_row.addWidget(help_button(
            "Manual lenses don't record their focal length in the photos, so enter it here "
            "with the sensor width. The photos are used as they are, never stretched flat.",
            "Spirula only. Fujifilm APS-C is 23.5 mm wide, Canon APS-C 22.3 mm.",
        ))
        fisheye_row.addStretch(1)
        fisheye_layout.addLayout(fisheye_row)
        lens_row = QtWidgets.QWidget()
        lens_layout = QtWidgets.QHBoxLayout(lens_row)
        lens_layout.setContentsMargins(0, 0, 0, 0)
        lens_layout.setSpacing(8)
        self.fisheye_focal = QtWidgets.QDoubleSpinBox()
        self.fisheye_focal.setRange(1.0, 50.0)
        self.fisheye_focal.setDecimals(1)
        self.fisheye_focal.setSuffix(" mm lens")
        self.fisheye_focal.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sensor_width = QtWidgets.QDoubleSpinBox()
        self.sensor_width.setRange(5.0, 60.0)
        self.sensor_width.setDecimals(1)
        self.sensor_width.setSuffix(" mm sensor width")
        self.sensor_width.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.sensor_width.setToolTip(
            "Fujifilm APS-C (X-T20 and most X bodies): 23.5 mm. Canon APS-C (M50): 22.3 mm."
        )
        lens_layout.addWidget(self.fisheye_focal)
        lens_layout.addWidget(self.sensor_width)
        lens_layout.addStretch(1)
        fisheye_layout.addWidget(lens_row)
        # Shown only when the choice conflicts with the engine.
        self.fisheye_help = label("", "notice")
        self.fisheye_help.hide()
        fisheye_layout.addWidget(self.fisheye_help)
        source_layout.addWidget(self.fisheye_box)
        self.source_preview = label("Choose an input to see its size and storage estimate.", "helper")
        source_layout.addWidget(self.source_preview)
        self.results_hint = label("", "helper")  # shown as the preview's tooltip
        left.addWidget(source_card)

        quality_card, quality_layout = card("Training")
        self.engine_hint = label("", "helper")
        quality_layout.addWidget(self.engine_hint)

        # Spirula options (the default engine).
        self.spirula_box = QtWidgets.QWidget()
        spirula_layout = QtWidgets.QVBoxLayout(self.spirula_box)
        spirula_layout.setContentsMargins(0, 0, 0, 0)
        spirula_layout.setSpacing(8)
        spirula_layout.addWidget(titled(
            "Quality",
            "How much detail the splat can hold. Higher levels allow more splats and train "
            "longer; Spirula stops adding splats at the level's budget, so more steps alone "
            "can't add detail.",
            "Draft: up to 200,000 splats, 15,000 steps. Quick tests.",
            "Standard: up to 1 million splats, 30,000 steps. Fine on most phones.",
            "High: up to 3 million splats, 50,000 steps. Portfolio work; publish with "
            "streamed detail so phones stay smooth.",
            "Ultra: up to 10 million splats, 80,000 steps. Hero pieces only: needs a GPU "
            "with about 48 GB, trains several times longer and makes large files.",
            "These steps suit up to 300 photos or frames. Each step learns from one image, so "
            "bigger captures train longer to see every image about as often: 450 frames on "
            "Standard trains 45,000 steps. The most is 2.5x the level's steps. Draft is never "
            "raised, so it stays a quick check; use a low fps with it.",
        ))
        quality_row, self.quality_buttons = segment_row(
            tuple((key, label) for key, (label, _steps, _splats) in QUALITY_LEVELS.items()), self,
        )
        spirula_layout.addLayout(quality_row)
        self.quality_notice = label("", "notice")
        self.quality_notice.hide()
        spirula_layout.addWidget(self.quality_notice)
        spirula_layout.addSpacing(4)
        spirula_layout.addWidget(titled("Splat type", *PRIMITIVE_HELP.values()))
        primitive_row, self.primitive_buttons = segment_row(
            (("3dgs", "Standard (3DGS)"), ("3dgut", "Lens-accurate (3DGUT)")), self,
        )
        spirula_layout.addLayout(primitive_row)
        vignetting_row = QtWidgets.QHBoxLayout()
        self.vignetting = QtWidgets.QCheckBox("Correct lens darkening at the edges")
        vignetting_row.addWidget(self.vignetting)
        vignetting_row.addWidget(help_button(
            "Many lenses darken the corners, strongly so for wide or fisheye lenses at wide "
            "apertures. Then a wall looks darker in one photo's corner than in the next "
            "photo's centre, and training can turn that into blotches or floaters.",
            "On: Spirula's camera model (NVIDIA's PPISP) also learns the lens's corner "
            "darkening and keeps it out of the splat. Turned on automatically with the "
            "fisheye option. Spirula leaves it off by default, so compare a run with and "
            "without.",
        ))
        vignetting_row.addStretch(1)
        spirula_layout.addLayout(vignetting_row)
        scene_row = QtWidgets.QHBoxLayout()
        self.main_scene = QtWidgets.QCheckBox("Keep to the main scene")
        scene_row.addWidget(self.main_scene)
        scene_row.addWidget(help_button(
            "Training normally tries to explain everything in every photo, so sky and distant "
            "buildings become big blurry splats far away. They also show behind and through "
            "the main scene, so deleting them later leaves holes.",
            "On: after the camera positions are solved, the pod works out the main scene (a "
            "cylinder around what you circled, or a box around the bulk of the scene) and "
            "trains only there. Far background is left out from the start: a compact splat "
            "with a clean edge, shown over your background colour or a skybox. Off keeps the "
            "surroundings.",
        ))
        scene_row.addStretch(1)
        spirula_layout.addLayout(scene_row)
        object_row = QtWidgets.QHBoxLayout()
        self.object_scan = QtWidgets.QCheckBox("Object scan: cut away the background")
        object_row.addWidget(self.object_scan)
        object_row.addWidget(help_button(
            "For one thing on a table, like a pastry, a shoe or a sculpture, photographed "
            "in rings all the way round. An AI model (BiRefNet) finds the object in every "
            "photo and masks out the table and the room, and Spirula trains with its object "
            "preset. No cleanup afterwards, and every splat goes into the object.",
            "The camera solver still uses the whole photos, so keep a patterned surface "
            "under the object: it is what places the photos. Costs about 2–3 extra minutes "
            "on the pod (a 424 MB model download, then masking each photo). If masking "
            "fails, the job trains normally and says so in the log.",
        ))
        object_row.addStretch(1)
        spirula_layout.addLayout(object_row)
        turntable_row = QtWidgets.QHBoxLayout()
        turntable_row.setContentsMargins(24, 0, 0, 0)
        self.turntable = QtWidgets.QCheckBox("Turntable: the object was turned, not the camera")
        turntable_row.addWidget(self.turntable)
        turntable_row.addWidget(help_button(
            "Tick this when the object was rotated (on a turntable, or by turning the plate or "
            "paper under it) while the camera stayed put. The background is then masked out "
            "before the camera positions are worked out, so the still background can't "
            "confuse them. If masking fails, the job stops before training.",
            "Keep the light soft and even: when the object turns and the lights don't, its "
            "shadows and highlights move, which can make shiny parts soft or blotchy. Photos "
            "from two or three camera heights give a full object; one height leaves the top "
            "or underside weak.",
        ))
        turntable_row.addStretch(1)
        spirula_layout.addLayout(turntable_row)
        spirula_layout.addSpacing(4)
        spirula_layout.addWidget(titled(
            "Floater suppression",
            "Floaters are blurry blobs hanging in mid-air, usually in spots the camera only "
            "saw from a few angles.",
            *FLOATER_HELP.values(),
            REFLECTION_HELP,
        ))
        floater_row, self.floater_buttons = segment_row(
            (("off", "Off"), ("mild", "Mild"), ("strong", "Strong")), self,
        )
        spirula_layout.addLayout(floater_row)
        spirula_layout.addSpacing(4)
        spirula_layout.addWidget(titled(
            "Ignore moving things",
            "People walking past, cars, or your own reflection sliding across a metal door "
            "or window. They are somewhere different in every frame, so training smears "
            "them into ghosts and haze.",
            *DISTRACTION_HELP.values(),
        ))
        distraction_row, self.distraction_buttons = segment_row(
            (("off", "Off"), ("mild", "Mild"), ("strong", "Strong")), self,
        )
        spirula_layout.addLayout(distraction_row)
        spirula_layout.addSpacing(4)
        spirula_layout.addWidget(titled(
            "AI geometry guidance",
            "Before training, an AI model looks at each frame and guesses which way every "
            "surface faces. Training uses those guesses as hints, so walls, floors and tables "
            "come out flat and solid instead of lumpy or hollow.",
            *GUIDANCE_HELP.values(),
        ))
        guidance_row, self.guidance_buttons = segment_row(
            (("off", "Off"), ("surfaces", "Surfaces"), ("depth", "Surfaces + depth")), self,
        )
        spirula_layout.addLayout(guidance_row)
        spirula_layout.addSpacing(4)
        spirula_layout.addWidget(titled(
            "Feature matching",
            "How the camera solver recognises the same spot in different frames. If a job stops "
            "because only part of the frames joined up (identical rods, tiles, windows or "
            "shiny surfaces make look-alike spots), pick a stronger option and run it again.",
            "Standard (SIFT): the classic method. Fast, nothing to download. Fine for most "
            "rooms, objects and outdoor scenes.",
            "Learned (ALIKED + LightGlue): AI features and matching, widely used since 2023. "
            "Much better at repetitive or shiny subjects. Camera solving takes roughly 10–15 "
            "minutes longer.",
            "LoMa: the newest matcher (2026), built for the hardest pairs. Its features take in "
            "a wider view around each spot, which helps tell repeated shapes apart. Downloads "
            "about 1.4 GB on each pod and solves slowest. Promising, but not yet widely "
            "compared.",
        ))
        matcher_row, self.matcher_buttons = segment_row(
            (("sift", "Standard"), ("aliked", "Learned"), ("loma", "LoMa")), self,
        )
        spirula_layout.addLayout(matcher_row)
        quality_layout.addWidget(self.spirula_box)

        self.quality_summary = label("", "helper")  # the summary bar shows the same
        self.advanced_button = QtWidgets.QToolButton()
        self.advanced_button.setObjectName("expander")
        self.advanced_button.setText("Advanced")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setIcon(icon("chevron_right", theme.PRIMARY, 18))
        self.advanced_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.advanced_button.toggled.connect(self._toggle_advanced)
        quality_layout.addWidget(self.advanced_button, alignment=QtCore.Qt.AlignLeft)
        self.advanced_widget = QtWidgets.QWidget()
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_widget)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(8)
        advanced_layout.addWidget(titled(
            "Engine",
            "Spirula Studio finds the cameras and trains on the pod with its own tools. "
            "Nerfstudio uses COLMAP and Splatfacto, which is slower and doesn't have "
            "Spirula's options.",
        ))
        engine_row, self.engine_buttons = segment_row(
            (("spirula", "Spirula Studio (recommended)"), ("nerfstudio", "Nerfstudio")), self,
        )
        advanced_layout.addLayout(engine_row)
        advanced_form = QtWidgets.QFormLayout()
        advanced_form.setContentsMargins(0, 6, 0, 0)
        advanced_form.setSpacing(10)
        self.steps = QtWidgets.QSpinBox()
        self.steps.setRange(10, 1_000_000)
        self.steps.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.downscales = QtWidgets.QSpinBox()
        self.downscales.setRange(0, 8)
        self.downscales.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.depth_weight = QtWidgets.QDoubleSpinBox()
        self.depth_weight.setRange(0.0, 1.0)
        self.depth_weight.setDecimals(2)
        self.depth_weight.setSingleStep(0.01)
        self.depth_weight.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.depth_weight.setToolTip(
            "How strongly the AI's distance guesses steer training (Spirula, 'Surfaces + "
            "depth' only). 0.05 is a gentle starting point; higher pulls the geometry harder "
            "toward the guesses, mistakes included."
        )
        self.steps_override = QtWidgets.QCheckBox("Set by hand")
        self.steps_override.setToolTip(
            "Off: Spirula trains for its quality level's steps, raised for captures of more "
            "than 300 images (up to 2.5x) so every image is seen about as often. "
            "Nerfstudio always uses this number."
        )
        steps_row = QtWidgets.QWidget()
        steps_layout = QtWidgets.QHBoxLayout(steps_row)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(10)
        steps_layout.addWidget(self.steps, 1)
        steps_layout.addWidget(self.steps_override)
        advanced_form.addRow("Training steps", steps_row)
        advanced_form.addRow("Image downscales", self.downscales)
        advanced_form.addRow("Depth guidance strength", self.depth_weight)
        self.downscales.setToolTip("Used by Nerfstudio preprocessing only.")
        advanced_layout.addLayout(advanced_form)
        self.advanced_widget.hide()
        quality_layout.addWidget(self.advanced_widget)
        left.addWidget(quality_card)
        left.addStretch(1)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        columns.box.addWidget(left_widget, 5)

        # GPU: picked automatically, changeable.
        gpu_card, gpu_layout = card("GPU")
        self.gpu_choice = label("Choose a capture first.", "gpuChoice")
        self.gpu_choice.setWordWrap(True)
        gpu_layout.addWidget(self.gpu_choice)
        self.gpu_hint = label(
            "The cheapest available GPU within your limits is picked for you.", "helper"
        )
        gpu_layout.addWidget(self.gpu_hint)
        gpu_buttons = QtWidgets.QHBoxLayout()
        self.gpu_change_button = QtWidgets.QPushButton("Change")
        self.gpu_change_button.setObjectName("tonalButton")
        self.gpu_change_button.setCheckable(True)
        self.gpu_change_button.toggled.connect(self._toggle_gpu_details)
        gpu_buttons.addWidget(self.gpu_change_button)
        self.find_gpus_button = QtWidgets.QPushButton("Refresh")
        self.find_gpus_button.setObjectName("textButton")
        self.find_gpus_button.setIcon(icon("refresh", theme.PRIMARY, 18))
        self.find_gpus_button.clicked.connect(lambda: self._find_gpus())
        gpu_buttons.addWidget(self.find_gpus_button)
        gpu_buttons.addStretch(1)
        gpu_layout.addLayout(gpu_buttons)
        self.gpu_details = QtWidgets.QWidget()
        details = QtWidgets.QVBoxLayout(self.gpu_details)
        details.setContentsMargins(0, 4, 0, 0)
        details.setSpacing(10)
        filters = QtWidgets.QHBoxLayout()
        filters.setSpacing(12)
        self.price = QtWidgets.QDoubleSpinBox()
        self.price.setRange(0.01, 100)
        self.price.setDecimals(3)
        self.price.setPrefix("$")
        self.price.setSuffix(" / hr")
        self.price.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.vram = QtWidgets.QSpinBox()
        self.vram.setRange(1, 512)
        self.vram.setSuffix(" GB")
        self.vram.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        filters.addWidget(field("Maximum price", self.price), 1)
        filters.addWidget(field("Minimum VRAM", self.vram), 1)
        details.addLayout(filters)
        self.gpu_list = QtWidgets.QListWidget()
        self.gpu_list.setObjectName("gpuList")
        self.gpu_list.setMinimumHeight(220)
        self.gpu_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.gpu_list.currentItemChanged.connect(self._update_run_enabled)
        details.addWidget(self.gpu_list, 1)
        self.gpu_details.hide()
        gpu_layout.addWidget(self.gpu_details, 1)
        gpu_layout.addStretch(0)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(gpu_card)
        right.addStretch(1)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        columns.box.addWidget(right_widget, 4)
        content_layout.addWidget(columns)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        page_layout.addWidget(scroll, 1)

        summary = QtWidgets.QFrame()
        summary.setObjectName("summaryBar")
        summary_layout = QtWidgets.QHBoxLayout(summary)
        summary_layout.setContentsMargins(22, 12, 22, 12)
        summary_layout.setSpacing(16)
        summary_copy = QtWidgets.QVBoxLayout()
        summary_copy.setSpacing(2)
        self.summary_label = ElidedLabel("Choose an input and GPU to start.", "helper")
        summary_copy.addWidget(self.summary_label)
        self.slots_label = ElidedLabel("", "cardMeta")
        summary_copy.addWidget(self.slots_label)
        summary_layout.addLayout(summary_copy, 1)
        self.run_button = QtWidgets.QPushButton("Start")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumWidth(140)
        self.run_button.setEnabled(False)
        self.run_button.clicked.connect(self._run)
        summary_layout.addWidget(self.run_button)
        page_layout.addWidget(summary)
        return page

    # ------------------------------------------------------------------
    # Window chrome and the jobs dock
    # ------------------------------------------------------------------

    def changeEvent(self, event):
        if event.type() == QtCore.QEvent.WindowStateChange:
            maximized = self.isMaximized() or self.isFullScreen()
            self.title_bar.sync_state()
            self.root_widget.setProperty("framed", not maximized)
            self.root_widget.layout().setContentsMargins(*(0, 0, 0, 0) if maximized else (1, 1, 1, 1))
            repolish(self.root_widget)
        super().changeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        narrow = event.size().width() < self.DOCK_BREAKPOINT
        if narrow != self._narrow:
            # Crossing the breakpoint resets to the automatic choice; a click on
            # the dock button overrides it until the next crossing.
            self._narrow = narrow
            self._set_dock_expanded(False if narrow else bool(self.config.dock_expanded),
                                    animate=self.isVisible())

    def _toggle_dock(self):
        expanded = not self._dock_expanded
        if not self._narrow:
            self.config.dock_expanded = expanded
        self._set_dock_expanded(expanded, animate=True)

    def _set_dock_expanded(self, expanded: bool, *, animate: bool) -> None:
        self._dock_expanded = expanded
        target = self.DOCK_EXPANDED_WIDTH if expanded else self.DOCK_COLLAPSED_WIDTH
        self.sidebar_full.setVisible(expanded)
        self.sidebar_rail.setVisible(not expanded)
        if not animate or self.sidebar.width() == target:
            self.sidebar.setFixedWidth(target)
            return
        animation = QtCore.QVariantAnimation(self)
        animation.setStartValue(self.sidebar.width())
        animation.setEndValue(target)
        animation.setDuration(160)
        animation.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        animation.valueChanged.connect(lambda value: self.sidebar.setFixedWidth(int(value)))
        animation.start(QtCore.QAbstractAnimation.DeleteWhenStopped)

    def _toggle_gpu_details(self, shown: bool):
        self.gpu_details.setVisible(shown)
        self.gpu_change_button.setText("Done" if shown else "Change")

    def _connect_signals(self):
        self.manager.status_changed.connect(self._set_status)
        self.manager.error.connect(self._manager_error)
        self.manager.notice.connect(self._show_global)
        self.manager.gpu_candidates_changed.connect(self._show_gpus)
        self.manager.gpu_lookup_finished.connect(self._gpu_lookup_finished)
        self.manager.job_added.connect(self._job_added)
        self.manager.untracked_pods_changed.connect(self._untracked_changed)
        self.manager.untracked_check_failed.connect(
            lambda message: self.totals.setToolTip(message)
        )
        for button in (*self.engine_buttons.values(), *self.floater_buttons.values(),
                       *self.distraction_buttons.values(), *self.matcher_buttons.values(),
                       *self.guidance_buttons.values(), *self.primitive_buttons.values(),
                       *self.quality_buttons.values()):
            button.clicked.connect(self._engine_changed)
        self.fisheye.toggled.connect(self._fisheye_changed)
        self.steps_override.toggled.connect(self._update_quality_summary)
        self.vignetting.toggled.connect(self._update_quality_summary)
        self.object_scan.toggled.connect(self._update_quality_summary)
        self.main_scene.toggled.connect(self._update_quality_summary)
        self.object_scan.toggled.connect(self.turntable.setEnabled)
        self.turntable.toggled.connect(self._update_quality_summary)
        # Fisheye lenses vignette strongly: switching fisheye on turns this on too.
        self.fisheye.clicked.connect(lambda on: on and self.vignetting.setChecked(True))
        self.fisheye_focal.valueChanged.connect(self._update_quality_summary)
        self.sensor_width.valueChanged.connect(self._update_quality_summary)
        self.depth_weight.valueChanged.connect(self._update_quality_summary)
        for button in (self.local_mode, self.drive_mode):
            button.clicked.connect(self._input_kind_changed)
        self.drive_listing_ready.connect(self._drive_listing_ready)
        self.drive_videos_probed.connect(self._drive_videos_probed)
        self.input_folder.editingFinished.connect(self._local_folder_changed)
        self.capture_items.itemChanged.connect(self._capture_ticks_changed)
        self.video_fps.valueChanged.connect(self._video_frames_changed)
        self.price.valueChanged.connect(self._invalidate_gpus)
        self.vram.valueChanged.connect(self._invalidate_gpus)
        self.vram.valueChanged.connect(self._update_quality_summary)
        self.downscales.valueChanged.connect(self._source_settings_changed)
        self.steps.valueChanged.connect(self._update_quality_summary)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self):
        self.input_folder.setText(self.config.input_folder)
        self.video_fps.setValue(self.config.video_fps)
        self.price.setValue(self.config.max_gpu_price)
        self.vram.setValue(self.config.min_gpu_vram_gb)
        self.steps.setValue(self.config.pipeline_max_steps)
        self.downscales.setValue(self.config.pipeline_num_downscales)
        self.drive_mode.setChecked(self.config.input_kind == "drive")
        self.local_mode.setChecked(not self.drive_mode.isChecked())
        self.results_folder.setText(self.config.results_folder)
        self._update_drive_selection_text()
        self.engine_buttons[self.config.engine].setChecked(True)
        self.floater_buttons[self.config.spirula_floaters].setChecked(True)
        self.distraction_buttons[self.config.spirula_distraction].setChecked(True)
        self.guidance_buttons[self.config.spirula_guidance].setChecked(True)
        self.matcher_buttons[self.config.spirula_matcher].setChecked(True)
        self.depth_weight.setValue(self.config.spirula_depth_weight)
        self.primitive_buttons[self.config.spirula_primitive].setChecked(True)
        self.quality_buttons[self.config.spirula_quality].setChecked(True)
        self.vignetting.setChecked(bool(self.config.spirula_vignetting))
        self.object_scan.setChecked(bool(self.config.spirula_object))
        self.main_scene.setChecked(bool(self.config.spirula_main_scene))
        self.turntable.setChecked(bool(self.config.spirula_turntable))
        self.turntable.setEnabled(self.object_scan.isChecked())
        self.steps_override.setChecked(bool(self.config.steps_override))
        self.fisheye.setChecked(bool(self.config.fisheye_lens))
        self.fisheye_focal.setValue(self.config.fisheye_focal_mm)
        self.sensor_width.setValue(self.config.sensor_width_mm)
        self._fisheye_changed()

    def _fisheye_changed(self, *_):
        on = self.fisheye.isChecked()
        self.fisheye_focal.setEnabled(on)
        self.sensor_width.setEnabled(on)
        self._update_quality_summary()

    def _quality(self) -> str:
        return checked_key(self.quality_buttons, "medium")

    def _effective_steps(self, plan: InputPlan | None = None) -> int:
        """Steps the job will train for (the pod confirms them from the real image count)."""
        if self._engine() == "spirula" and not self.steps_override.isChecked():
            plan = plan or self._shown_plan
            return quality_steps(self._quality(), plan.image_count if plan else None)
        return self.steps.value()

    def _steps_text(self, plan: InputPlan | None = None) -> str:
        """'45,000 steps (raised for 450 frames)', or just the steps."""
        steps = self._effective_steps(plan)
        plan = plan or self._shown_plan
        base = QUALITY_LEVELS[self._quality()][1]
        if self._engine() != "spirula" or self.steps_override.isChecked() or steps == base:
            return f"{steps:,} steps"
        noun = ("frames" if plan and plan.is_video else "photos and frames" if plan and plan.is_mixed
                else "photos")
        return f"{steps:,} steps (raised for {plan.image_count:,} {noun})"

    def _primitive(self) -> str:
        return checked_key(self.primitive_buttons, "3dgs")

    def _floaters(self) -> str:
        return checked_key(self.floater_buttons, "off")

    def _guidance(self) -> str:
        return checked_key(self.guidance_buttons, "off")

    def _matcher(self) -> str:
        return checked_key(self.matcher_buttons, "sift")

    def _distraction(self) -> str:
        return checked_key(self.distraction_buttons, "off")

    def _spirula_options_text(self) -> str:
        """Short description of the Spirula cleanup options, for summaries."""
        text = (
            f"{QUALITY_LEVELS[self._quality()][0]} · {self._primitive()} · "
            f"floaters: {self._floaters()} · guidance: {GUIDANCE_LABELS[self._guidance()]}"
        )
        if self.fisheye.isChecked():
            text = f"fisheye {self.fisheye_focal.value():g} mm · " + text
        if self._guidance() == "depth":
            text += f" ({self.depth_weight.value():g})"
        if self._distraction() != "off":
            text += f" · ignore moving: {self._distraction()}"
        if self.vignetting.isChecked():
            text += " · vignetting corrected"
        if self.main_scene.isChecked():
            text += " · main scene only"
        if self._matcher() != "sift":
            text += f" · matching: {self._matcher()}"
        if self.object_scan.isChecked():
            text = ("turntable object scan · " if self.turntable.isChecked() else "object scan · ") + text
        return text

    def _engine(self) -> str:
        return "spirula" if self.engine_buttons["spirula"].isChecked() else "nerfstudio"

    def _engine_changed(self, *_):
        self._update_quality_summary()

    def _save_settings(self):
        self.config.engine = self._engine()
        self.config.spirula_floaters = self._floaters()
        self.config.spirula_distraction = self._distraction()
        self.config.spirula_guidance = self._guidance()
        self.config.spirula_depth_weight = self.depth_weight.value()
        self.config.spirula_primitive = self._primitive()
        self.config.spirula_quality = self._quality()
        self.config.spirula_vignetting = self.vignetting.isChecked()
        self.config.spirula_object = self.object_scan.isChecked()
        self.config.spirula_main_scene = self.main_scene.isChecked()
        self.config.spirula_matcher = self._matcher()
        self.config.spirula_turntable = self.turntable.isChecked()
        self.config.steps_override = self.steps_override.isChecked()
        self.config.fisheye_lens = self.fisheye.isChecked()
        self.config.fisheye_focal_mm = self.fisheye_focal.value()
        self.config.sensor_width_mm = self.sensor_width.value()
        # input_folder and the ticks are saved as they change (_local_folder_changed).
        self.config.input_kind = self._source_kind()
        self.config.results_folder = self.results_folder.text().strip()
        self.config.video_fps = self.video_fps.value()
        self.config.max_gpu_price = self.price.value()
        self.config.min_gpu_vram_gb = self.vram.value()
        self.config.pipeline_max_steps = self.steps.value()
        self.config.pipeline_num_downscales = self.downscales.value()
        self.config.save_state()

    def _open_settings(self):
        dialog = SettingsDialog(self.config, self)
        disable_wheel_changes(dialog)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        api_key = dialog.runpod_key.text().strip()
        drive_key = dialog.drive_key.text().strip()
        current_drive = self.config.drive_credentials_path()
        try:
            if api_key != self.config.runpod_api_key:
                write_env_value("RUNPOD_API_KEY", api_key)
            if drive_key and Path(drive_key) != current_drive:
                self.config.google_service_account_file = str(install_drive_key(Path(drive_key)))
        except (OSError, DriveKeyError) as exc:
            self._error(f"Could not save your keys: {exc}")
        self.config.runpod_api_key = api_key
        self.config.ssh_private_key = dialog.ssh_key.text().strip()
        self.config.runpod_image = dialog.image.text().strip()
        self.config.cloud_type = dialog.cloud_type.currentData()
        self.config.max_concurrent_pods = dialog.max_pods.value()
        self.config.live_preview = dialog.live_preview.isChecked()
        self.config.spirula_image = (
            dialog.spirula_image.text().strip() or DEFAULT_SPIRULA_IMAGE
        )
        self._save_settings()
        self._invalidate_gpus()
        self._refresh_credentials_label()
        self.manager.check_untracked_pods()
        self._show_setup()
        self._show_notice("Settings saved. Find GPUs again to use the new connection.")

    def _refresh_credentials_label(self):
        private_key = Path(self.config.ssh_private_key).expanduser()
        public_key = Path(str(private_key) + ".pub")
        configured = (
            bool(self.config.runpod_api_key.strip())
            and private_key.is_file()
            and public_key.is_file()
        )
        self.credentials_label.setText("Set up credentials")
        self.credentials_label.setToolTip("Add your RunPod API key and SSH key in Settings.")
        self.credentials_label.setProperty("state", "missing")
        repolish(self.credentials_label)
        self.credentials_label.setVisible(not configured)

    # ------------------------------------------------------------------
    # Input selection
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Capture: a local or Drive folder, and which of its items to use
    # ------------------------------------------------------------------

    def _browse_local(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose the folder with your photos and/or videos", self.input_folder.text()
        )
        if chosen:
            self.input_folder.setText(chosen)
            self._local_folder_changed()

    def _local_folder_changed(self):
        folder = self.input_folder.text().strip()
        if folder != self.config.input_folder:
            self.config.input_folder = folder
            self.config.local_unticked = []
        self._refresh_capture_items()
        self._invalidate_gpus()
        self._update_source_preview()

    def _local_contents(self) -> tuple[Path | None, list[Path], list[Path]]:
        """The chosen local folder, and the photos and videos directly in it."""
        text = self.input_folder.text().strip()
        if not text:
            return None, [], []
        folder = Path(text).expanduser()
        try:
            return folder, folder_photos(folder), list_videos(folder)
        except (OSError, ValueError):
            return folder, [], []

    def _local_video_info(self, video: Path) -> VideoInfo | None:
        """Length and size of a local video for its checklist row (cached; None if unreadable)."""
        try:
            stat = video.stat()
        except OSError:
            return None
        key = (str(video), stat.st_size, stat.st_mtime_ns)
        if key not in self._local_video_infos:
            try:
                self._local_video_infos[key] = probe_video(video)
            except ValueError:
                self._local_video_infos[key] = None
        return self._local_video_infos[key]

    def _unticked(self) -> list:
        return self.config.drive_unticked if self.drive_mode.isChecked() else self.config.local_unticked

    def _ticked(self, key: str) -> bool:
        return key not in self._unticked()

    def _refresh_capture_items(self):
        """List the chosen folder's photos (one row) and videos, ticked unless left out."""
        rows: list[tuple[str, str]] = []
        if self.drive_mode.isChecked():
            listing = self._drive_listing
            if listing is not None:
                if listing.files:
                    count = len(listing.files)
                    rows.append((PHOTOS_ITEM, f"{count} photo{'s' if count != 1 else ''}  ·  "
                                              f"{size_text(listing.total_bytes)}"))
                for video in listing.videos:
                    video = self._probed_videos.get(self._video_key(video), video)
                    rows.append((video.id, video_row(
                        video.name, video.size, video.duration_ms / 1000, video.width, video.height,
                    )))
        else:
            _folder, photos, videos = self._local_contents()
            if photos:
                size = sum(photo.stat().st_size for photo in photos)
                rows.append((PHOTOS_ITEM, f"{len(photos)} photo{'s' if len(photos) != 1 else ''}"
                                          f"  ·  {size_text(size)}"))
            for video in videos:
                info = self._local_video_info(video)
                rows.append((video.name, video_row(
                    video.name, video.stat().st_size,
                    info.duration_seconds if info else 0, info.width if info else 0,
                    info.height if info else 0,
                )))
        unticked = set(self._unticked())
        self.capture_items.blockSignals(True)
        self.capture_items.clear()
        for key, text in rows:
            item = QtWidgets.QListWidgetItem(text)
            item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Unchecked if key in unticked else QtCore.Qt.Checked)
            item.setData(QtCore.Qt.UserRole, key)
            self.capture_items.addItem(item)
        self.capture_items.blockSignals(False)
        self.capture_items.setVisible(bool(rows))
        row_height = self.capture_items.sizeHintForRow(0) if rows else 0
        self.capture_items.setFixedHeight(min(len(rows), 6) * row_height + 8)

    def _capture_ticks_changed(self, *_):
        unticked = [
            item.data(QtCore.Qt.UserRole)
            for item in (self.capture_items.item(row) for row in range(self.capture_items.count()))
            if item.checkState() != QtCore.Qt.Checked
        ]
        if self.drive_mode.isChecked():
            self.config.drive_unticked = unticked
        else:
            self.config.local_unticked = unticked
        self._invalidate_gpus()
        self._update_source_preview()

    def _results_root(self) -> Path:
        text = self.results_folder.text().strip()
        return Path(text).expanduser() if text else self.config.results_root()

    def _browse_results(self):
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose where results are saved", str(self._results_root()),
        )
        if chosen:
            self.results_folder.setText(chosen)
            self._update_source_preview()

    def _drive_client(self) -> DriveClient | None:
        path = self.config.drive_credentials_path()
        if path is None:
            self._error(
                "Google Drive is not set up. Open Settings → Connection and choose your "
                "service-account key under Google Drive key."
            )
            return None
        try:
            return DriveClient(path)
        except DriveError as exc:
            self._error(str(exc))
            return None

    def _browse_drive(self):
        client = self._drive_client()
        if client is None:
            return
        dialog = DriveFolderDialog(
            client, self, self.config.drive_folder_id, self.config.drive_folder_name,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted or dialog.selected is None:
            return
        self._set_drive_listing(dialog.selected)

    def _set_drive_listing(self, listing: DriveListing):
        if listing.folder_id != self.config.drive_folder_id:
            self.config.drive_unticked = []
        self._drive_listing = listing
        self.config.drive_folder_id = listing.folder_id
        self.config.drive_folder_name = listing.folder_name
        # Drive's own metadata has no frame rate; read it from each video's header.
        self._probe_drive_videos(
            [v for v in listing.videos if self._video_key(v) not in self._probed_videos]
        )
        self._update_drive_selection_text()
        self._refresh_capture_items()
        self._invalidate_gpus()
        self._update_source_preview()

    def _drive_ticked_videos(self) -> list[DriveFile]:
        listing = self._drive_listing
        if listing is None:
            return []
        return [self._probed_videos.get(self._video_key(v), v)
                for v in listing.videos if self._ticked(v.id)]

    @staticmethod
    def _video_key(video: DriveFile) -> tuple:
        return (video.id, video.size, video.md5)

    def _probe_drive_videos(self, videos: list[DriveFile]):
        """Read each video's exact length and frame rate from its header, in the background."""
        path = self.config.drive_credentials_path()
        if not videos or path is None:
            return

        def work():
            try:
                client = DriveClient(path)
                probed = [client.probe_video(video) for video in videos]
            except Exception:  # Drive's own metadata still works without this
                return
            try:
                self.drive_videos_probed.emit(probed)
            except RuntimeError:
                pass  # window closed

        threading.Thread(target=work, daemon=True).start()

    def _drive_videos_probed(self, probed: list):
        for video in probed:
            if video.fps > 0:
                self._probed_videos[self._video_key(video)] = video
        listing = self._drive_listing
        if listing is not None and any(self._video_key(v) in self._probed_videos
                                       for v in listing.videos):
            self._refresh_capture_items()
            self._invalidate_gpus()
            self._update_source_preview()

    def _update_drive_selection_text(self):
        self.drive_selection.setText(self.config.drive_folder_name)

    def _restore_drive_selection(self):
        """Re-read the saved Drive folder in the background so the plan is ready."""
        if not self.drive_mode.isChecked() or not self.config.drive_folder_id:
            return
        path = self.config.drive_credentials_path()
        if path is None:
            return
        folder_id, folder_name = self.config.drive_folder_id, self.config.drive_folder_name

        def work():
            try:
                listing = DriveClient(path).list_images(folder_id, folder_name)
                result = (listing, "")
            except Exception as exc:
                result = (None, str(exc))
            try:
                self.drive_listing_ready.emit(*result)
            except RuntimeError:
                pass

        threading.Thread(target=work, daemon=True).start()

    def _drive_listing_ready(self, listing, error: str):
        if listing is None:
            if error:
                self.source_preview.setText(f"Could not read the saved Drive folder: {error}")
            return
        self._set_drive_listing(listing)

    def _source_kind(self) -> str:
        """Where the capture comes from: "local" or "drive"."""
        return "drive" if self.drive_mode.isChecked() else "local"

    def _video_fps(self) -> float:
        """Frames per second to take from the videos."""
        return self.video_fps.value()

    @staticmethod
    def _nothing_ticked(has_items: bool) -> str:
        return ("Tick the photos or at least one video." if has_items
                else "No photos or videos directly in this folder.")

    def _source_plan(self) -> InputPlan:
        """Plan the ticked items: photos, videos, or both."""
        fps, downscales, minimum = self._video_fps(), self.downscales.value(), self.config.volume_gb
        if self.drive_mode.isChecked():
            listing = self._drive_listing
            if listing is None:
                raise ValueError(
                    "Loading the saved Google Drive folder..." if self.config.drive_folder_id
                    else "Choose a Google Drive folder."
                )
            photos = listing.files if self._ticked(PHOTOS_ITEM) else ()
            videos = self._drive_ticked_videos()
            root = self._results_root()
            if photos and videos:
                return plan_drive_mixed(listing.folder_id, listing.folder_name, photos,
                                        tuple(videos), root, minimum, downscales, fps)
            if videos:
                return plan_drive_video(listing.folder_id, listing.folder_name, videos, root,
                                        minimum, downscales, fps)
            if photos:
                return plan_drive_input(listing.folder_id, listing.folder_name, photos, root,
                                        minimum)
            raise ValueError(self._nothing_ticked(bool(listing.files or listing.videos)))
        folder, photos, videos = self._local_contents()
        if folder is None:
            raise ValueError("Choose a folder with photos and/or videos.")
        if not folder.is_dir():
            raise ValueError("Select an existing folder.")
        ticked_photos = photos if self._ticked(PHOTOS_ITEM) else []
        ticked_videos = [video for video in videos if self._ticked(video.name)]
        if ticked_photos and ticked_videos:
            return plan_mixed(folder, tuple(ticked_photos), tuple(ticked_videos), minimum,
                              downscales, fps)
        if ticked_videos:
            return plan_input(ticked_videos[0], "video", minimum, downscales, fps,
                              tuple(ticked_videos[1:]))
        if ticked_photos:
            return plan_input(folder, "images", minimum, downscales, fps)
        raise ValueError(self._nothing_ticked(bool(photos or videos)))

    def _gpu_signature(self, plan: InputPlan):
        common = (
            self.config.runpod_api_key, self.price.value(), self.vram.value(),
            plan.kind, plan.volume_gb, plan.frame_target, self.downscales.value(),
        )
        if plan.is_drive:
            return common + (
                plan.drive_folder_id,
                tuple((f.id, f.size, f.md5) for f in (*plan.drive_files, *plan.drive_videos)),
            )
        videos = plan.videos or ((plan.source,) if plan.kind == "video" else ())
        files = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns)
            for path in (*plan.images, *videos)
        )
        return common + (str(plan.source), files)

    def _input_kind_changed(self, *_):
        drive = self.drive_mode.isChecked()
        self.source_pages.setCurrentIndex(1 if drive else 0)
        self.source_pages.setFixedHeight(self.source_pages.currentWidget().sizeHint().height())
        self.source_preview.setToolTip(
            "Results go in your results folder / Drive folder name / run time." if drive else
            "Results go inside the folder, in results / run time (beside the video, in "
            "results / video name / run time, for videos alone)."
        )
        self._refresh_capture_items()
        self._invalidate_gpus()
        self._update_source_preview()

    def _source_settings_changed(self, *_):
        self._invalidate_gpus()
        self._update_quality_summary()
        self._update_source_preview()

    def _video_frames_changed(self, *_):
        self._invalidate_gpus()
        if self._shown_plan is not None and self._shown_plan.has_videos:
            self._update_source_preview()

    def _toggle_advanced(self, expanded: bool):
        self.advanced_widget.setVisible(expanded)
        self.advanced_button.setIcon(
            icon("chevron_down" if expanded else "chevron_right", theme.PRIMARY, 18)
        )

    def _update_quality_summary(self, *_):
        engine = self._engine()
        self.downscales.setEnabled(engine == "nerfstudio")
        self.spirula_box.setVisible(engine == "spirula")
        self.steps.setEnabled(engine != "spirula" or self.steps_override.isChecked())
        if engine == "spirula" and not self.steps_override.isChecked():
            self.steps.blockSignals(True)
            self.steps.setValue(self._effective_steps())
            self.steps.blockSignals(False)
        ultra_short = self._quality() == "ultra" and self.vram.value() < 40
        self.quality_notice.setText(
            "Ultra needs a GPU with about 48 GB. Raise Minimum VRAM under GPU > Change."
        )
        self.quality_notice.setVisible(engine == "spirula" and ultra_short)
        conflict = self.fisheye.isChecked() and engine != "spirula"
        self.fisheye_help.setText(
            "Fisheye photos need Spirula Studio. Switch the engine under Training > Advanced."
        )
        self.fisheye_help.setVisible(conflict)
        self.depth_weight.setEnabled(engine == "spirula" and self._guidance() == "depth")
        # Only worth a line when it isn't the default.
        self.engine_hint.setVisible(engine != "spirula")
        if engine == "spirula":
            self.quality_summary.setText(
                f"{self._steps_text()} · {self._spirula_options_text()}"
            )
        else:
            self.engine_hint.setText("Using Nerfstudio. Switch back under Advanced.")
            self.quality_summary.setText(
                f"{self.steps.value():,} training steps · "
                f"{'Original images only' if self.downscales.value() == 0 else 'Downscaled copies available'}"
            )
        self._refresh_gpu_estimates()
        self._update_summary()

    def _update_source_preview(self):
        try:
            plan = self._source_plan()
        except (OSError, ValueError) as exc:
            self.source_preview.setText(str(exc))
            self.frame_fps_label.setText("")
            self._shown_plan = None
            self.frames_notice.hide()
            self.frames_box.hide()
            return
        self._shown_plan = plan
        self.frames_box.setVisible(plan.has_videos)
        self.frames_notice.setVisible(
            plan.has_videos and self._engine() == "spirula"
            and (plan.image_count or 0) > STEP_REFERENCE_IMAGES * MAX_STEP_SCALE
        )
        self._update_quality_summary()
        facts = [f"{plan.photo_count} photos"] if plan.has_photos else []
        if plan.has_videos:
            info = plan.video_info
            known = info.duration_seconds > 0
            self.frame_fps_label.setText(f"≈ {plan.frame_target:,} frames" if known else "")
            noun = "video" if plan.video_count == 1 else f"{plan.video_count} videos"
            facts.append(f"{info.duration_seconds:.0f}s of {noun}" if known
                         else f"{noun}, length not reported by Drive yet")
            if info.width and info.height:
                facts.append(f"{info.width}×{info.height}")
            facts.append(f"{info.fps:.1f} fps source" if info.fps > 0 else "frame rate read on the pod")
            facts.append(f"about {plan.frame_target:,} frames" if known
                         else f"{plan.extract_fps:g} frames per second")
        else:
            self.frame_fps_label.setText("")
        facts.append(size_text(plan.input_bytes))
        facts.append(f"{plan.volume_gb} GiB workspace")
        self.source_preview.setText(" · ".join(facts))

    # ------------------------------------------------------------------
    # GPU selection and starting jobs
    # ------------------------------------------------------------------

    def _placeholder_item(self, text: str) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem(text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
        item.setForeground(QtGui.QColor(theme.OUTLINE))
        item.setSizeHint(QtCore.QSize(0, 64))
        return item

    def _invalidate_gpus(self, *_):
        """The input or limits changed: forget the GPU list and look again shortly."""
        self._gpu_lookup_settings = None
        self._gpu_candidates = []
        self._current_plan = None
        self.gpu_list.clear()
        self.gpu_list.addItem(self._placeholder_item("Looking for GPUs..."))
        self.run_button.setEnabled(False)
        self._update_summary()
        self._auto_gpu_timer.start()

    def _auto_find_gpus(self):
        if not self.config.has_runpod():
            self.gpu_choice.setText("Add your RunPod API key in Settings to find GPUs.")
            return
        try:
            plan = self._source_plan()
            signature = self._gpu_signature(plan)
        except (OSError, ValueError):
            self.gpu_choice.setText("Choose a capture first.")
            self.gpu_hint.setText(
                "The cheapest available GPU within your limits is picked for you."
            )
            return
        if signature != self._gpu_lookup_settings:
            self._find_gpus(auto=True)

    def _selected_gpu(self):
        item = self.gpu_list.currentItem()
        return item.data(QtCore.Qt.UserRole) if item else None

    def _source_description(self, plan: InputPlan) -> str:
        parts = [f"{plan.photo_count} photos"] if plan.has_photos else []
        if plan.has_videos:
            noun = "video" if plan.video_count == 1 else f"{plan.video_count} videos"
            parts.append(f"{plan.extract_fps:g} fps from {noun} (≈ {plan.frame_target:,} frames)")
        return " + ".join(parts) + (" from Drive" if plan.is_drive else "")

    def _update_summary(self):
        self.slots_label.setText(self.manager.slots_text())
        if self._current_plan is None:
            self.summary_label.setText("Choose a capture to start.")
            return
        plan = self._current_plan
        item = self.gpu_list.currentItem()
        gpu = item.data(QtCore.Qt.UserRole + 2) if item and self._selected_gpu() else "no GPU yet"
        options = f"  ·  {self._spirula_options_text()}" if self._engine() == "spirula" else ""
        self.summary_label.setText(
            f"{ENGINE_LABELS[self._engine()]}{options}  ·  {self._source_description(plan)}  ·  "
            f"{self._steps_text(plan)}  ·  {plan.volume_gb} GiB  ·  {gpu}"
        )

    def _update_run_enabled(self, *_):
        selected = self._selected_gpu()
        self.run_button.setEnabled(selected is not None and self._current_plan is not None)
        item = self.gpu_list.currentItem()
        if selected is not None and item is not None:
            self.gpu_choice.setText(item.data(QtCore.Qt.UserRole + 1))
        self._update_summary()

    def _find_gpus(self, auto: bool = False):
        self._save_settings()
        try:
            plan = self._source_plan()
            signature = self._gpu_signature(plan)
        except (OSError, ValueError) as exc:
            if not auto:
                self._error(str(exc))
            return
        self._auto_gpu_timer.stop()
        self._gpu_lookup_settings = signature
        self._current_plan = plan
        self.gpu_list.clear()
        self.gpu_list.addItem(self._placeholder_item("Checking RunPod capacity..."))
        self.run_button.setEnabled(False)
        self.find_gpus_button.setEnabled(False)
        self.gpu_choice.setText("Finding a GPU...")
        self.gpu_hint.setText("Checking current RunPod capacity...")
        if not auto:
            self._show_notice("")
        self._set_status("Checking available GPUs...")
        self._update_summary()
        self.manager.refresh_gpus(plan.volume_gb)

    def _refresh_gpu_estimates(self):
        """Quality or engine changed: redo the time and cost lines, keeping the choice."""
        if self._gpu_candidates and self._gpu_lookup_settings is not None:
            self._show_gpus(self._gpu_candidates, keep_selection=True)

    def _show_gpus(self, candidates, keep_selection: bool = False):
        if self._gpu_lookup_settings is None:
            return
        try:
            plan = self._source_plan()
            if self._gpu_lookup_settings != self._gpu_signature(plan):
                return
        except (OSError, ValueError):
            return
        previous = self._selected_gpu() if keep_selection else None
        self._gpu_candidates = candidates
        self.gpu_list.clear()
        model = self._speed_model()
        spirula = self._engine() == "spirula"
        splats = QUALITY_LEVELS[self._quality()][2]
        steps = self._effective_steps(plan)
        kind = "video" if plan.is_video else "photos"
        options = []
        for gpu in candidates:
            price = gpu["_hourly_price"]
            speed = model.speed(gpu["id"])
            estimate = model.estimate(gpu["id"], steps, splats, kind) if spirula else None
            details = f"${price:.3f}/hr"
            outlook = "Speed unknown"
            if speed is not None:
                outlook = f"{speed.factor:.2f}× a 3090 ({'your jobs' if speed.measured else 'specs'})"
            if estimate is not None:
                cost = price * estimate.total_s / 3600
                outlook += f"  ·  ≈ {duration_text(estimate.total_s)}  ·  ≈ ${cost:.2f} per job"
                details = f"${price:.3f}/hr  ·  ≈ {duration_text(estimate.total_s)}  ·  ≈ ${cost:.2f} per job"
                rank = (0, cost)
            elif speed is not None:
                rank = (1, price / speed.factor)
            else:
                rank = (2, price)
            text = (
                f"{gpu['_display_name']}\n"
                f"{gpu['memoryInGb']} GB VRAM  ·  {gpu['_cloud_type']}  ·  "
                f"${price:.3f}/hr  ·  {gpu['_stock_status']}\n{outlook}"
            )
            item = QtWidgets.QListWidgetItem(text)
            item.setSizeHint(QtCore.QSize(0, 84))
            item.setToolTip(self._gpu_speed_tooltip(model, gpu["id"], estimate, kind, splats))
            selection = (gpu["id"], gpu["_cloud_type"]) if gpu["_available"] else None
            item.setData(QtCore.Qt.UserRole, selection)
            item.setData(QtCore.Qt.UserRole + 1, f"{gpu['_display_name']}  ·  {details}")
            item.setData(QtCore.Qt.UserRole + 2, f"{gpu['_display_name']} at ${price:.3f}/hr")
            if not gpu["_available"]:
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEnabled)
                item.setForeground(QtGui.QColor(theme.OUTLINE))
            else:
                options.append((rank, estimate.total_s if estimate else math.inf, item))
            self.gpu_list.addItem(item)
        if not candidates:
            self.gpu_list.addItem(self._placeholder_item("No GPUs match these limits right now."))
        if options:
            # Picked for you: the lowest cost for this job; within 5% of it, the
            # quickest. "Change" shows the full list.
            best = min(rank for rank, _, _ in options)
            close = [o for o in options if o[0][0] == best[0] and o[0][1] <= best[1] * 1.05]
            pick = min(close, key=lambda o: o[1])[2]
            kept = [o[2] for o in options if o[2].data(QtCore.Qt.UserRole) == previous]
            self.gpu_list.setCurrentItem(kept[0] if kept else pick)
            basis = "Lowest estimated cost for this job" if best[0] == 0 else "Cheapest"
            available = len(options)
            if kept and kept[0] is not pick:
                self.gpu_hint.setText(
                    f"Your pick. {basis}: {pick.data(QtCore.Qt.UserRole + 1)}."
                )
            else:
                self.gpu_hint.setText(
                    f"{basis} of {available} available option{'s' if available != 1 else ''} "
                    f"within ${self.price.value():.2f}/hr and {self.vram.value()} GB VRAM."
                )
        else:
            self.gpu_list.setCurrentRow(-1)
            self.gpu_choice.setText("No GPU available within your limits.")
            self.gpu_hint.setText(
                "Raise the price limit or lower the VRAM under Change, then Refresh."
            )
        self._update_run_enabled()

    def _speed_model(self) -> SpeedModel:
        """GPU speeds and job estimates from the finished jobs in history."""
        try:
            records = self.manager.history.recent(500)
        except Exception:
            records = []
        return SpeedModel(runs_from_history(records))

    @staticmethod
    def _gpu_speed_tooltip(model: SpeedModel, gpu_id: str, estimate, kind: str,
                           splats: int) -> str:
        speed = model.speed(gpu_id)
        if speed is None:
            return "No speed figures for this GPU yet."
        lines = [
            f"Speed: {speed.factor:.2f}× an RTX 3090, "
            + ("measured on your own jobs." if speed.measured
               else "estimated from its compute and memory bandwidth.")
        ]
        runs = model.runs_on(gpu_id)
        if runs:
            lines.append("Your finished jobs on it:")
            lines += [
                f"  {run.splats / 1e6:g}M splats, {run.kind}"
                f"{' ' + run.resolution if run.resolution else ''}: {run.ms_per_step:.0f} ms per step"
                for run in runs[-5:]
            ]
        if estimate is not None:
            noun = "photo" if kind == "photos" else "video"
            basis = (f"your past {noun} jobs with {splats / 1e6:g}M splats" if estimate.from_history
                     else "typical figures (no past jobs like this one yet)")
            lines.append(
                f"Estimate: training ≈ {duration_text(estimate.train_s)}, plus "
                f"≈ {duration_text(estimate.total_s - estimate.train_s)} for pod start, "
                f"download and camera solving. Based on {basis}."
            )
        return "\n".join(lines)

    def _gpu_lookup_finished(self):
        self.find_gpus_button.setEnabled(True)
        if self._selected_gpu() is None and self.gpu_choice.text() == "Finding a GPU...":
            self.gpu_choice.setText("Couldn't check RunPod. Try Refresh.")

    def _run(self):
        self._save_settings()
        gpu_selection = self._selected_gpu()
        if not gpu_selection:
            self._error("No GPU is selected yet. Wait for the search, or use Change.")
            return
        try:
            plan = self._source_plan()
            if self._gpu_lookup_settings != self._gpu_signature(plan):
                self._invalidate_gpus()
                raise ValueError("The capture changed, so the GPU search is running again.")
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        if self.fisheye.isChecked() and self._engine() != "spirula":
            self._error("Fisheye photos need Spirula Studio. Switch the engine, or untick the fisheye option.")
            return
        if self.object_scan.isChecked() and self._engine() != "spirula":
            self._error("Object scans need Spirula Studio. Switch the engine, or untick object scan.")
            return
        if plan.is_mixed and self._engine() != "spirula":
            self._error("Photos and videos in one job need Spirula Studio. Switch the engine, "
                        "or untick the photos or the videos.")
            return
        if plan.video_count > 1 and self._engine() != "spirula":
            self._error("Several videos in one job need Spirula Studio. Switch the engine, or tick one video.")
            return
        parts = [f"{plan.photo_count} photos"] if plan.has_photos else []
        if plan.has_videos:
            parts.append(plan.display_name if plan.is_video else
                         f"{plan.video_count} video{'s' if plan.video_count != 1 else ''}")
        where = f" from Drive folder “{plan.drive_folder_name}”" if plan.is_drive else ""
        headline = f"{' + '.join(parts)}{where}  ·  {size_text(plan.input_bytes)}"
        selected_item = self.gpu_list.currentItem()
        gpu_text = selected_item.text().splitlines()[0]
        rate = selected_item.text().splitlines()[1].split("  ·  ")[2]
        engine = self._engine()
        rows = []
        if plan.has_videos:
            known = plan.video_info.duration_seconds > 0
            rows.append(("Frames", f"{plan.extract_fps:g} per second of video"
                         + (f"  ·  about {plan.frame_target:,}" if known else "")))
        if engine == "spirula":
            splat = {"3dgs": "Standard (3DGS)", "3dgut": "Lens-accurate (3DGUT)"}[self._primitive()]
            rows.append(("Engine", f"Spirula Studio  ·  {splat}"))
            label_, _steps, splats = QUALITY_LEVELS[self._quality()]
            rows.append(("Quality", f"{label_}  ·  up to {splats:,} splats  ·  "
                                    f"{self._steps_text(plan)}"))
            rows.append(("Cleanup", f"floaters {self._floaters()}  ·  guidance "
                                    f"{GUIDANCE_LABELS[self._guidance()]}"
                                    + (f"  ·  ignore moving {self._distraction()}"
                                       if self._distraction() != "off" else "")
                                    + ("  ·  vignetting corrected"
                                       if self.vignetting.isChecked() else "")))
        else:
            rows.append(("Engine", "Nerfstudio"))
        if self.fisheye.isChecked():
            rows.append(("Lens", f"fisheye, {self.fisheye_focal.value():g} mm"))
        if engine == "spirula" and self._matcher() != "sift":
            rows.append(("Matching", {"aliked": "Learned (ALIKED + LightGlue)  ·  about 10–15 min more solving",
                                      "loma": "LoMa  ·  1.4 GB download, slowest solving"}[self._matcher()]))
        if self.object_scan.isChecked() and engine == "spirula":
            rows.append(("Object", "background cut away by AI masks"
                         + ("  ·  turntable: masked before solving the cameras"
                            if self.turntable.isChecked() else "")
                         + "  ·  about 2–3 min extra"))
        rows.append(("GPU", f"{gpu_text}  ·  {rate}"))
        rows.append(("Storage", f"{plan.volume_gb} GiB workspace"))
        if not confirm_summary(
            self, "Start this job?", headline, rows,
            f"{self.manager.slots_text()} now. RunPod may bill storage separately.",
            "Create pod and start",
        ):
            return
        context = f"{ENGINE_LABELS[engine]}  ·  {self._source_description(plan)}"
        if plan.is_video:
            # The title is the folder; the files say which capture this is.
            context = f"{plan.display_name}  ·  {context}"
        if engine == "spirula":
            context += f"  ·  {self._spirula_options_text()}"
        try:
            self.manager.start_job(
                title=plan.job_title,
                context=context,
                rate=rate,
                gpu_label=gpu_text,
                run_args=(plan.source, gpu_selection),
                run_kwargs={
                    "source_type": plan.kind,
                    "video_fps": plan.extract_fps,
                    "engine": engine,
                    "drive_folder": (
                        (plan.drive_folder_id, plan.drive_folder_name) if plan.is_drive else None
                    ),
                    "results_root": self._results_root() if plan.is_drive else None,
                    "drive_video_id": ",".join(
                        f.id for f in (plan.drive_files if plan.kind == "drive_video"
                                       else plan.drive_videos)),
                    # A mixed job's source is the folder, so every video is extra.
                    **({"more_videos": plan.videos} if plan.kind == "mixed"
                       else {"more_videos": plan.videos[1:]} if len(plan.videos) > 1 else {}),
                },
            )
        except ValueError as exc:
            self._error(str(exc))
            return
        self._set_status("Ready")

    # ------------------------------------------------------------------
    # Job list
    # ------------------------------------------------------------------

    def _job_added(self, job: JobController):
        # A reconnected job replaces its own entry from history.
        for old in [other for other in self._views if other.historic and other.job_id == job.job_id]:
            self._remove_job_widgets(old)
        self._add_job_widgets(job, select=True, top=True)

    def _add_job_widgets(self, job: JobController, *, select: bool, top: bool):
        context = job.context or ENGINE_LABELS.get(job.engine, "")
        view = JobView(job, context, job.rate_text or "pending")
        disable_wheel_changes(view)
        view.remove_requested.connect(self._remove_job)
        self._views[job] = view
        self.pages.addWidget(view)
        card_widget = JobCard(job, context)
        item = QtWidgets.QListWidgetItem()
        item.setSizeHint(QtCore.QSize(0, 86))
        item.setData(QtCore.Qt.UserRole, job)
        self.job_list.insertItem(0 if top else self.job_list.count(), item)
        self.job_list.setItemWidget(item, card_widget)
        self._cards[job] = (item, card_widget)
        dock_button = DockButton(job.title)
        dock_button.clicked.connect(lambda _checked=False, j=job: self._select_job(j))
        self.dock_layout.insertWidget(0 if top else self.dock_layout.count(), dock_button,
                                      alignment=QtCore.Qt.AlignHCenter)
        self._dock_buttons[job] = dock_button
        job.finished.connect(lambda _success: self._tick())
        job.active_pod_changed.connect(lambda _pod: self._tick())
        if select:
            self.job_list.setCurrentItem(item)
        self._tick()

    def _load_history(self):
        for job in self.manager.load_history():
            self._add_job_widgets(job, select=False, top=False)

    def _select_job(self, job: JobController):
        item = self._cards.get(job, (None, None))[0]
        if item is not None:
            self.job_list.setCurrentItem(item)

    def _sync_dock_selection(self, selected: JobController | None):
        for job, button in self._dock_buttons.items():
            button.setChecked(job is selected)

    def _job_selected(self, current, _previous=None):
        job = current.data(QtCore.Qt.UserRole) if current else None
        self._sync_dock_selection(job)
        view = self._views.get(job)
        if view is not None:
            self.pages.setCurrentWidget(view)
            self.new_job_button.setEnabled(True)
            self._set_status("Ready")

    def _show_setup(self):
        self.job_list.clearSelection()
        self.job_list.setCurrentRow(-1)
        self._sync_dock_selection(None)
        self.pages.setCurrentWidget(self.setup_page)
        self._refresh_credentials_label()
        self._update_summary()

    def _remove_job(self, job: JobController):
        if not self.manager.remove_job(job):
            return
        self._remove_job_widgets(job)
        if self.job_list.count():
            self.job_list.setCurrentRow(0)
        else:
            self._show_setup()
        self._tick()

    def _remove_job_widgets(self, job: JobController):
        item, _card = self._cards.pop(job)
        self.job_list.takeItem(self.job_list.row(item))
        dock_button = self._dock_buttons.pop(job)
        dock_button.setParent(None)
        dock_button.deleteLater()
        view = self._views.pop(job)
        view.shutdown()
        self.pages.removeWidget(view)
        view.deleteLater()

    def _tick(self):
        running = 0
        for job, (_item, card_widget) in self._cards.items():
            view = self._views[job]
            card_widget.refresh(view.stage_name)
            state, text = job_status(job, view.stage_name)
            self._dock_buttons[job].set_state(state, f"{job.title}\n{text}")
            view.refresh()
            running += job.running
        pods = sum(1 for job in self._cards if job.active_pod_id) + len(self.manager.untracked)
        self.empty_jobs.setVisible(not self._cards)
        cost = self.manager.hourly_cost()
        self.totals.setText(
            f"{running} running · {pods} pod{'s' if pods != 1 else ''} · ${cost:.2f}/hr"
        )
        self.rail_totals.setText(f"{pods} pod{'s' if pods != 1 else ''}\n${cost:.2f}/h")
        self.rail_totals.setToolTip(self.totals.text())
        self.slots_label.setText(self.manager.slots_text())

    def _untracked_changed(self, pods):
        self.totals.setToolTip("")
        self.rail_warning.setVisible(bool(pods))
        if pods:
            self.rail_warning.setToolTip(
                f"{len(pods)} untracked pod{'s' if len(pods) != 1 else ''}: click to review"
            )
        if not pods:
            self.untracked_button.hide()
        else:
            cost = sum(pod.get("cost") or 0 for pod in pods)
            self.untracked_button.setText(
                f"⚠  {len(pods)} untracked pod{'s' if len(pods) != 1 else ''} · ${cost:.2f}/hr  ›"
            )
            self.untracked_button.setToolTip("Pods this app created that no job is tracking. Click to review.")
            self.untracked_button.show()
        self._tick()

    def _review_untracked(self):
        pods = list(self.manager.untracked)
        if not pods:
            self.untracked_button.hide()
            return
        dialog = UntrackedPodsDialog(pods, self)
        if dialog.exec() == QtWidgets.QDialog.Accepted and dialog.to_reconnect is not None:
            self.manager.reattach(dialog.to_reconnect)
            return
        if dialog.result() == QtWidgets.QDialog.Accepted:
            for pod in dialog.to_terminate:
                self.manager.terminate_untracked(pod["id"], pod["name"])
            self._show_global(
                f"Terminating {len(dialog.to_terminate)} untracked pod(s)...", error=False,
            )

    # ------------------------------------------------------------------
    # Notices
    # ------------------------------------------------------------------

    def _set_status(self, message: str):
        lower = message.lower()
        badge_text = (
            message if message in {"Ready", "Error"}
            else "Checking GPUs" if "gpu" in lower
            else message[:28]
        )
        self.status.setText(badge_text)
        self.status.setProperty("state", "error" if message == "Error" else "normal")
        repolish(self.status)

    def _show_notice(self, message: str, *, error: bool = False):
        self.setup_notice.setText(message)
        self.setup_notice.setProperty("state", "error" if error else "normal")
        repolish(self.setup_notice)
        self.setup_notice.setVisible(bool(message))

    def _show_global(self, message: str, error: bool = False):
        self.global_notice.setText(message + "   ✕")
        self.global_notice.setProperty("state", "error" if error else "normal")
        repolish(self.global_notice)
        self.global_notice.setVisible(bool(message))

    def _error(self, message: str):
        self._set_status("Error")
        self._show_notice(message, error=True)
        if self.pages.currentWidget() is not self.setup_page:
            self._show_global(message, error=True)
        if message.startswith((
            "No capacity for the selected GPU", "Selected GPU no longer has matching"
        )):
            self._invalidate_gpus()

    def _manager_error(self, message: str):
        if message.startswith("GPU lookup failed"):
            self._error(message)
        else:
            self._show_global(message, error=True)

    def closeEvent(self, event):
        busy = self.manager.busy_jobs()
        # Jobs running on their pod carry on without the app, which reconnects
        # when it opens again. Jobs still setting up or shutting a pod down don't.
        blocking = [job for job in busy if not job.detachable]
        if blocking:
            lines = "\n".join(
                f"•  {job.title}: "
                + (f"pod {job.active_pod_id} kept after a failed download" if not job.running
                   else job.status_text)
                for job in blocking
            )
            inform(
                self, "Jobs are setting up or shutting down a pod",
                "These jobs are creating, uploading to or terminating a pod:\n\n"
                f"{lines}\n\nWait until they are processing on the pod (or finished), or stop "
                "them first. Closing now could leave a pod running with nothing to do.",
            )
            event.ignore()
            return
        if busy:
            names = "\n".join(f"•  {job.title}" for job in busy)
            if not confirm(
                self, "Leave jobs running?",
                f"These jobs keep running on their pods, and keep costing money:\n\n{names}\n\n"
                "Open the app again to follow them and download the results.",
                "Close and leave running",
            ):
                event.ignore()
                return
            for job in busy:
                job.detach()
        for view in self._views.values():
            view.shutdown()
        self._save_settings()
        event.accept()
