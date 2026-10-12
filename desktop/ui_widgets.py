"""Window chrome and small reusable widgets for the Material 3 look.

The main window and dialogs are frameless: they draw their own title bar and
hand moving/resizing back to the window manager (startSystemMove/Resize), so
snapping to half the screen and keyboard tiling keep working.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtSvg, QtWidgets

import ui_theme as theme

# ---------------------------------------------------------------------------
# Icons: 24x24 stroke drawings, coloured when rendered.
# ---------------------------------------------------------------------------

_ICONS = {
    "minimize": '<path d="M6 12h12"/>',
    "maximize": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "restore": '<rect x="5" y="9" width="10" height="10" rx="2"/><path d="M9 9V7a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2h-2"/>',
    "close": '<path d="M6 6l12 12M18 6L6 18"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "settings": (
        '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/>'
        '<circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>'
    ),
    "dock": '<rect x="3.5" y="4.5" width="17" height="15" rx="2.5"/><path d="M9.5 4.5v15"/>',
    "chevron_left": '<path d="M14.5 6.5L9 12l5.5 5.5"/>',
    "chevron_right": '<path d="M9.5 6.5L15 12l-5.5 5.5"/>',
    "chevron_down": '<path d="M6.5 9.5L12 15l5.5-5.5"/>',
    "warning": '<path d="M12 4L2.8 19.5h18.4z"/><path d="M12 10v4.5M12 17.2v.3"/>',
    "refresh": '<path d="M19 12a7 7 0 1 1-2.05-4.95"/><path d="M19 4.5v4h-4"/>',
    "external": '<path d="M14 5h5v5M19 5l-8 8"/><path d="M17 14v4a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 5 18V8.5A1.5 1.5 0 0 1 6.5 7H10"/>',
    "folder": '<path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4l2 2h8a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19H5a1.5 1.5 0 0 1-1.5-1.5z"/>',
    "stop": '<rect x="7" y="7" width="10" height="10" rx="1.5"/>',
}
_icon_cache: dict[tuple[str, str, int], QtGui.QIcon] = {}


class _NoWheelChanges(QtCore.QObject):
    """Scrolling over a field, dropdown, slider or tab bar must not change it.

    The wheel event is refused, so Qt hands it on to the parent: the page
    around the field scrolls instead.
    """

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel:
            event.ignore()
            return True
        return False


_WHEEL_TARGETS = (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QTabBar,
                  QtWidgets.QSlider, QtWidgets.QDial)


def disable_wheel_changes(root: QtWidgets.QWidget) -> None:
    """Guard the fields, dropdowns, sliders and tab bars inside root (and root itself).

    Installed on each of those widgets, never on the application: an
    application-wide Python event filter sees the embedded browser's internal
    objects too, and showing the training preview then crashes the app.
    """
    guard = root.findChild(_NoWheelChanges, "noWheelChanges", QtCore.Qt.FindDirectChildrenOnly)
    if guard is None:
        guard = _NoWheelChanges(root)
        guard.setObjectName("noWheelChanges")
    widgets = [root] if isinstance(root, _WHEEL_TARGETS) else []
    for kind in _WHEEL_TARGETS:
        widgets += root.findChildren(kind)
    for widget in widgets:
        widget.installEventFilter(guard)  # installing twice keeps a single entry


def icon(name: str, color: str = theme.ON_SURFACE_VARIANT, size: int = 20) -> QtGui.QIcon:
    key = (name, color, size)
    if key in _icon_cache:
        return _icon_cache[key]
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{_ICONS[name]}</svg>"
    )
    renderer = QtSvg.QSvgRenderer(QtCore.QByteArray(svg.encode()))
    result = QtGui.QIcon()
    for scale in (1, 2):
        pixmap = QtGui.QPixmap(size * scale, size * scale)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        result.addPixmap(pixmap)
    _icon_cache[key] = result
    return result


def icon_button(name: str, tooltip: str, *, object_name: str = "iconButton",
                color: str = theme.ON_SURFACE_VARIANT, size: int = 20) -> QtWidgets.QToolButton:
    button = QtWidgets.QToolButton()
    button.setObjectName(object_name)
    button.setIcon(icon(name, color, size))
    button.setIconSize(QtCore.QSize(size, size))
    button.setToolTip(tooltip)
    button.setCursor(QtCore.Qt.PointingHandCursor)
    button.setAutoRaise(True)
    return button


# ---------------------------------------------------------------------------
# Title bar and frameless resizing
# ---------------------------------------------------------------------------

class TitleBar(QtWidgets.QFrame):
    """App-coloured title bar: drag to move, double-click to maximise."""

    def __init__(self, window: QtWidgets.QWidget, title: str, *, closable_only: bool = False):
        super().__init__()
        self.setObjectName("titleBar")
        self._window = window
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 6, 6)
        layout.setSpacing(8)
        self.leading = QtWidgets.QHBoxLayout()
        self.leading.setSpacing(10)
        layout.addLayout(self.leading)
        self.title = QtWidgets.QLabel(title)
        self.title.setObjectName("windowTitle")
        layout.addWidget(self.title)
        layout.addStretch(1)
        self.trailing = QtWidgets.QHBoxLayout()
        self.trailing.setSpacing(6)
        layout.addLayout(self.trailing)
        self.max_button = None
        if not closable_only:
            minimize = icon_button("minimize", "Minimise", object_name="windowButton", size=18)
            minimize.clicked.connect(window.showMinimized)
            layout.addWidget(minimize)
            self.max_button = icon_button("maximize", "Maximise", object_name="windowButton", size=18)
            self.max_button.clicked.connect(self.toggle_maximized)
            layout.addWidget(self.max_button)
        close = icon_button("close", "Close", object_name="closeButton", size=18)
        close.clicked.connect(window.close)
        layout.addWidget(close)

    def toggle_maximized(self):
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def sync_state(self):
        if self.max_button is not None:
            maximized = self._window.isMaximized()
            self.max_button.setIcon(icon("restore" if maximized else "maximize", size=18))
            self.max_button.setToolTip("Restore" if maximized else "Maximise")

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.max_button is not None:
            self.toggle_maximized()
            event.accept()


class EdgeResizer(QtCore.QObject):
    """Lets a frameless window be resized from its edges and corners."""

    MARGIN = 6

    def __init__(self, window: QtWidgets.QWidget):
        super().__init__(window)
        self._window = window
        self._cursor_set = False
        # Watch only this window's native handle, which receives every mouse
        # event for the window before its widgets do. An application-wide filter
        # also sees the embedded browser's internal objects and crashes Qt.
        window.winId()
        window.windowHandle().installEventFilter(self)

    def _edges(self, pos: QtCore.QPoint) -> QtCore.Qt.Edges:
        """Which window edges `pos` (window coordinates) is close to."""
        if self._window.isMaximized() or self._window.isFullScreen():
            return QtCore.Qt.Edges()
        rect = self._window.rect()
        if not rect.adjusted(-1, -1, 1, 1).contains(pos):
            return QtCore.Qt.Edges()
        edges = QtCore.Qt.Edges()
        m = self.MARGIN
        if pos.x() <= m:
            edges |= QtCore.Qt.LeftEdge
        elif pos.x() >= rect.width() - m:
            edges |= QtCore.Qt.RightEdge
        if pos.y() <= m:
            edges |= QtCore.Qt.TopEdge
        elif pos.y() >= rect.height() - m:
            edges |= QtCore.Qt.BottomEdge
        return edges

    @staticmethod
    def _cursor_for(edges) -> QtCore.Qt.CursorShape:
        left, right = bool(edges & QtCore.Qt.LeftEdge), bool(edges & QtCore.Qt.RightEdge)
        top, bottom = bool(edges & QtCore.Qt.TopEdge), bool(edges & QtCore.Qt.BottomEdge)
        if (left and top) or (right and bottom):
            return QtCore.Qt.SizeFDiagCursor
        if (right and top) or (left and bottom):
            return QtCore.Qt.SizeBDiagCursor
        if left or right:
            return QtCore.Qt.SizeHorCursor
        return QtCore.Qt.SizeVerCursor

    def eventFilter(self, obj, event):
        kind = event.type()
        if kind not in (QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonPress):
            return False
        edges = self._edges(event.position().toPoint())
        if kind == QtCore.QEvent.MouseButtonPress:
            if edges and event.button() == QtCore.Qt.LeftButton:
                self._clear_cursor()
                return bool(obj.startSystemResize(edges))
            return False
        if edges:
            shape = self._cursor_for(edges)
            if not self._cursor_set:
                QtWidgets.QApplication.setOverrideCursor(shape)
                self._cursor_set = True
            else:
                QtWidgets.QApplication.changeOverrideCursor(shape)
        else:
            self._clear_cursor()
        return False

    def _clear_cursor(self):
        if self._cursor_set:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._cursor_set = False


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------

class FramelessDialog(QtWidgets.QDialog):
    """A Material dialog surface with its own title row. Add content to `body`."""

    def __init__(self, parent: QtWidgets.QWidget | None, title: str):
        super().__init__(parent)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowTitle(title)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        self.surface = QtWidgets.QFrame()
        self.surface.setObjectName("dialogSurface")
        shadow = QtWidgets.QGraphicsDropShadowEffect(self.surface)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 6)
        shadow.setColor(QtGui.QColor(0, 0, 0, 150))
        self.surface.setGraphicsEffect(shadow)
        outer.addWidget(self.surface)
        surface_layout = QtWidgets.QVBoxLayout(self.surface)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        surface_layout.setSpacing(0)
        self.title_bar = TitleBar(self, title, closable_only=True)
        self.title_bar.setObjectName("dialogTitleBar")
        surface_layout.addWidget(self.title_bar)
        content = QtWidgets.QWidget()
        self.body = QtWidgets.QVBoxLayout(content)
        self.body.setContentsMargins(24, 4, 24, 22)
        self.body.setSpacing(14)
        surface_layout.addWidget(content, 1)


def _message_dialog(parent, title: str, text: str, buttons: list[tuple[str, str, int]]) -> int:
    dialog = FramelessDialog(parent, title)
    dialog.setMinimumWidth(440)
    message = QtWidgets.QLabel(text)
    message.setObjectName("dialogText")
    message.setWordWrap(True)
    message.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    dialog.body.addWidget(message)
    row = QtWidgets.QHBoxLayout()
    row.addStretch(1)
    result = {"code": 0}
    for text_, role, code in buttons:
        button = QtWidgets.QPushButton(text_)
        button.setObjectName(role)
        button.clicked.connect(lambda _=False, c=code: (result.update(code=c), dialog.accept()))
        row.addWidget(button)
        if role in {"primaryButton", "dangerButton"}:
            button.setDefault(True)
    dialog.body.addLayout(row)
    dialog.exec()
    return result["code"]


def confirm(parent, title: str, text: str, confirm_text: str = "Continue", *,
            danger: bool = False) -> bool:
    """Ask a yes/no question; True when the user confirms."""
    return _message_dialog(parent, title, text, [
        ("Cancel", "textButton", 0),
        (confirm_text, "dangerButton" if danger else "primaryButton", 1),
    ]) == 1


def confirm_summary(parent, title: str, headline: str, rows: list[tuple[str, str]],
                    note: str, confirm_text: str) -> bool:
    """A confirmation laid out as a headline, aligned label/value rows and fine print."""
    dialog = FramelessDialog(parent, title)
    dialog.setMinimumWidth(460)
    lead = QtWidgets.QLabel(headline)
    lead.setObjectName("sectionTitle")
    lead.setWordWrap(True)
    dialog.body.addWidget(lead)
    panel = QtWidgets.QFrame()
    panel.setObjectName("subtlePanel")
    grid = QtWidgets.QGridLayout(panel)
    grid.setContentsMargins(16, 12, 16, 12)
    grid.setHorizontalSpacing(18)
    grid.setVerticalSpacing(8)
    for row, (name, value) in enumerate(rows):
        key = QtWidgets.QLabel(name)
        key.setObjectName("fieldLabel")
        text = QtWidgets.QLabel(value)
        text.setWordWrap(True)
        grid.addWidget(key, row, 0, QtCore.Qt.AlignTop)
        grid.addWidget(text, row, 1)
    grid.setColumnStretch(1, 1)
    dialog.body.addWidget(panel)
    if note:
        fine = QtWidgets.QLabel(note)
        fine.setObjectName("cardMeta")
        fine.setWordWrap(True)
        dialog.body.addWidget(fine)
    buttons = QtWidgets.QHBoxLayout()
    buttons.addStretch(1)
    cancel = QtWidgets.QPushButton("Cancel")
    cancel.setObjectName("textButton")
    cancel.clicked.connect(dialog.reject)
    go = QtWidgets.QPushButton(confirm_text)
    go.setObjectName("primaryButton")
    go.setDefault(True)
    go.clicked.connect(dialog.accept)
    buttons.addWidget(cancel)
    buttons.addWidget(go)
    dialog.body.addLayout(buttons)
    return dialog.exec() == QtWidgets.QDialog.Accepted


def inform(parent, title: str, text: str) -> None:
    _message_dialog(parent, title, text, [("OK", "primaryButton", 1)])


# ---------------------------------------------------------------------------
# Dock items and responsive layout
# ---------------------------------------------------------------------------

STATE_COLORS = {
    # (container, on-container) pairs from the theme
    "running": (theme.PRIMARY_CONTAINER, theme.ON_PRIMARY_CONTAINER),
    "stopping": (theme.WARNING_CONTAINER, theme.ON_WARNING_CONTAINER),
    "kept": (theme.WARNING_CONTAINER, theme.ON_WARNING_CONTAINER),
    "finished": (theme.SUCCESS_CONTAINER, theme.ON_SUCCESS_CONTAINER),
    "failed": (theme.ERROR_CONTAINER, theme.ON_ERROR_CONTAINER),
    "stopped": (theme.SURFACE_CONTAINER_HIGHEST, theme.ON_SURFACE_VARIANT),
    "interrupted": (theme.WARNING_CONTAINER, theme.ON_WARNING_CONTAINER),
}
STATE_DOTS = {
    "running": theme.PRIMARY, "stopping": theme.WARNING, "kept": theme.WARNING,
    "finished": theme.SUCCESS, "failed": theme.ERROR, "stopped": theme.OUTLINE,
    "interrupted": theme.WARNING,
}


def initials(title: str) -> str:
    stem = title.rsplit(".", 1)[0] if "." in title[-6:] else title
    words = [w for w in "".join(ch if ch.isalnum() else " " for ch in stem).split() if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


class DockButton(QtWidgets.QAbstractButton):
    """A job shown like an app in a dock: initials, a status colour and a dot."""

    SIZE = 56

    def __init__(self, title: str):
        super().__init__()
        self.setCheckable(True)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.text_ = initials(title)
        self.state = "running"
        self.progress: float | None = None

    def set_state(self, state: str, tooltip: str) -> None:
        if state != self.state:
            self.state = state
            self.update()
        self.setToolTip(tooltip)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QtCore.QRectF(self.rect())
        if self.isChecked() or self.underMouse():
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(
                theme.SECONDARY_CONTAINER if self.isChecked() else theme.HOVER_OVERLAY_SOLID))
            painter.drawRoundedRect(rect.adjusted(2, 2, -2, -2), 16, 16)
        bg, fg = STATE_COLORS.get(self.state, STATE_COLORS["stopped"])
        circle = rect.adjusted(10, 10, -10, -10)
        painter.setBrush(QtGui.QColor(bg))
        painter.drawEllipse(circle)
        font = QtGui.QFont(theme.FONT_FAMILY.split(",")[0].strip('"'))
        font.setPixelSize(13)
        font.setWeight(QtGui.QFont.DemiBold)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(fg))
        painter.drawText(circle, QtCore.Qt.AlignCenter, self.text_)
        dot = QtCore.QRectF(circle.right() - 9, circle.bottom() - 9, 11, 11)
        painter.setPen(QtGui.QPen(QtGui.QColor(theme.SURFACE_CONTAINER_LOW), 2.5))
        painter.setBrush(QtGui.QColor(STATE_DOTS.get(self.state, theme.OUTLINE)))
        painter.drawEllipse(dot)

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


class ElidedLabel(QtWidgets.QLabel):
    """One line of text that shortens with '…' instead of widening its container."""

    def __init__(self, text: str = "", role: str = "helper"):
        super().__init__()
        self.setObjectName(role)
        self._full = ""
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt naming
        self._full = text
        self.setToolTip(text)
        self._elide()

    def text(self) -> str:  # noqa: D401 - Qt naming
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        shown = self.fontMetrics().elidedText(self._full, QtCore.Qt.ElideRight, max(0, self.width()))
        super().setText(shown)


class ResponsiveRow(QtWidgets.QWidget):
    """Side-by-side columns that stack into one column when narrower than `breakpoint`."""

    def __init__(self, breakpoint: int, spacing: int = 16):
        super().__init__()
        self.breakpoint = breakpoint
        self.box = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.box.setContentsMargins(0, 0, 0, 0)
        self.box.setSpacing(spacing)

    def resizeEvent(self, event):
        direction = (QtWidgets.QBoxLayout.TopToBottom if event.size().width() < self.breakpoint
                     else QtWidgets.QBoxLayout.LeftToRight)
        if self.box.direction() != direction:
            self.box.setDirection(direction)
        super().resizeEvent(event)

