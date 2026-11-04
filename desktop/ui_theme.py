"""Material 3 dark theme (blue seed) for the desktop application.

Colours are Material 3 roles. Components follow M3 shapes: filled/tonal/
outlined/text buttons, filter chips for choices, filled text fields, tonal
surface containers instead of borders.
"""
from pathlib import Path

# the combo box arrow: a stylesheet can only draw it from a file
_ASSETS = Path(__file__).resolve().parent / "assets"
_CHEVRON_DOWN = (_ASSETS / "chevron-down.svg").as_posix()
_CHECK = (_ASSETS / "check.svg").as_posix()
FONT_DIR = _ASSETS / "fonts"

# ---- colour roles --------------------------------------------------------
# Neutral greys (no blue tint) with one soft blue accent; hairline borders
# separate surfaces instead of heavy fills.
PRIMARY = "#9DB8FF"
ON_PRIMARY = "#0B1530"
PRIMARY_CONTAINER = "#24345A"
ON_PRIMARY_CONTAINER = "#D8E2FF"
SECONDARY_CONTAINER = "#2A2C33"
ON_SECONDARY_CONTAINER = "#EDEDF0"
TERTIARY_CONTAINER = "#573E5C"
ERROR = "#FFB4AB"
ERROR_CONTAINER = "#7A1A1A"
ON_ERROR_CONTAINER = "#FFDAD6"
SUCCESS = "#8DD5A6"
SUCCESS_CONTAINER = "#1B3F2B"
ON_SUCCESS_CONTAINER = "#A9F2C1"
WARNING = "#F2C26B"
WARNING_CONTAINER = "#4A3A10"
ON_WARNING_CONTAINER = "#FFDFA0"
SURFACE = "#0E0F11"
SURFACE_CONTAINER_LOWEST = "#0A0B0C"
SURFACE_CONTAINER_LOW = "#151619"
SURFACE_CONTAINER = "#1A1B1F"
SURFACE_CONTAINER_HIGH = "#202126"
SURFACE_CONTAINER_HIGHEST = "#1E1F24"
ON_SURFACE = "#EDEDF0"
ON_SURFACE_VARIANT = "#A3A3AD"
OUTLINE = "#74747E"
OUTLINE_VARIANT = "#2B2C31"
BORDER = "rgba(255, 255, 255, 0.07)"
BORDER_STRONG = "rgba(255, 255, 255, 0.14)"
HOVER_OVERLAY = "rgba(255, 255, 255, 0.06)"
HOVER_OVERLAY_SOLID = "#212227"
PRIMARY_HOVER = "rgba(157, 184, 255, 0.10)"
DISABLED_BG = "rgba(255, 255, 255, 0.08)"
DISABLED_FG = "rgba(237, 237, 240, 0.35)"

FONT_FAMILY = '"Inter", "Noto Sans", "Roboto", "DejaVu Sans", sans-serif'
MONO_FAMILY = '"Noto Sans Mono", "DejaVu Sans Mono", monospace'

APP_STYLE = f"""
QWidget {{
    color: {ON_SURFACE};
    font-family: {FONT_FAMILY};
    font-size: 9.5pt;
}}
QMainWindow, QWidget#appRoot, QScrollArea#setupScroll, QScrollArea#setupScroll > QWidget > QWidget {{
    background: {SURFACE};
}}
QWidget#appRoot[framed="true"] {{
    border: 1px solid {OUTLINE_VARIANT};
}}

/* ---- window chrome ---- */
QFrame#titleBar {{
    background: {SURFACE};
}}
QFrame#dialogTitleBar {{
    background: transparent;
}}
QLabel#windowTitle {{
    font-size: 10.5pt;
    font-weight: 600;
    color: {ON_SURFACE};
}}
QLabel#appMark {{
    background: {PRIMARY_CONTAINER};
    color: {ON_PRIMARY_CONTAINER};
    border-radius: 8px;
    font-weight: 700;
    font-size: 9pt;
    min-width: 26px; max-width: 26px; min-height: 26px; max-height: 26px;
    qproperty-alignment: AlignCenter;
}}
QToolButton#windowButton, QToolButton#closeButton, QToolButton#iconButton {{
    background: transparent;
    border: none;
    border-radius: 16px;
    padding: 6px;
}}
QToolButton#windowButton:hover, QToolButton#iconButton:hover {{
    background: {HOVER_OVERLAY};
}}
QToolButton#closeButton:hover {{
    background: {ERROR_CONTAINER};
}}
QFrame#dialogSurface {{
    background: {SURFACE_CONTAINER_HIGH};
    border: 1px solid {BORDER_STRONG};
    border-radius: 16px;
}}
QLabel#dialogText {{
    color: {ON_SURFACE_VARIANT};
    font-size: 10.5pt;
}}

/* ---- surfaces ---- */
QFrame#card, QFrame#activityCard {{
    background: {SURFACE_CONTAINER_LOW};
    border: 1px solid {BORDER};
    border-radius: 14px;
}}
QFrame#summaryBar {{
    background: {SURFACE_CONTAINER};
    border: none;
    border-top: 1px solid {OUTLINE_VARIANT};
}}
QFrame#sidebar {{
    background: {SURFACE_CONTAINER_LOW};
    border: none;
    border-right: 1px solid {BORDER};
}}
QFrame#subtlePanel {{
    background: {SURFACE_CONTAINER};
    border-radius: 12px;
}}

/* ---- type ---- */
QLabel#brand, QLabel#pageTitle {{
    color: {ON_SURFACE};
    font-size: 16pt;
    font-weight: 600;
}}
QLabel#cardTitle {{
    color: {ON_SURFACE};
    font-size: 10.5pt;
    font-weight: 600;
}}
QLabel#sectionTitle {{
    color: {ON_SURFACE};
    font-size: 10.5pt;
    font-weight: 600;
}}
QLabel#muted, QLabel#helper {{
    color: {ON_SURFACE_VARIANT};
}}
QLabel#fieldLabel {{
    color: {ON_SURFACE_VARIANT};
    font-size: 9.5pt;
    font-weight: 500;
}}
QLabel#eyebrow {{
    color: {PRIMARY};
    font-size: 9pt;
    font-weight: 600;
    letter-spacing: 0.4px;
}}
QLabel#gpuChoice {{
    color: {ON_SURFACE};
    font-size: 11pt;
    font-weight: 600;
}}
QLabel#statusBadge {{
    background: {SURFACE_CONTAINER_HIGH};
    color: {ON_SURFACE_VARIANT};
    border: 1px solid {OUTLINE_VARIANT};
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 9pt;
    font-weight: 600;
}}
QLabel#statusBadge[state="success"] {{
    background: {SUCCESS_CONTAINER}; color: {ON_SUCCESS_CONTAINER}; border-color: {SUCCESS_CONTAINER};
}}
QLabel#statusBadge[state="error"] {{
    background: {ERROR_CONTAINER}; color: {ON_ERROR_CONTAINER}; border-color: {ERROR_CONTAINER};
}}
QPushButton#credentialsChip {{
    background: transparent;
    color: {ON_SURFACE_VARIANT};
    border: 1px solid {OUTLINE_VARIANT};
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 9pt;
    font-weight: 600;
    min-height: 0;
}}
QPushButton#credentialsChip[state="missing"] {{
    background: {WARNING_CONTAINER}; color: {ON_WARNING_CONTAINER}; border-color: {WARNING_CONTAINER};
}}
QLabel#notice {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
    border-radius: 12px;
    padding: 10px 14px;
}}
QLabel#notice[state="error"] {{
    background: {ERROR_CONTAINER};
    color: {ON_ERROR_CONTAINER};
}}
QLabel#stageBadge {{
    background: {SURFACE_CONTAINER_HIGH};
    color: {ON_SURFACE_VARIANT};
    border-radius: 8px;
    padding: 8px 4px;
    font-size: 9pt;
    font-weight: 600;
}}
QLabel#stageBadge[state="active"] {{
    background: {PRIMARY_CONTAINER};
    color: {ON_PRIMARY_CONTAINER};
}}
QLabel#stageBadge[state="done"] {{
    background: {SUCCESS_CONTAINER};
    color: {ON_SUCCESS_CONTAINER};
}}

/* ---- text fields: rounded, hairline border ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE_CONTAINER_HIGHEST};
    color: {ON_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    min-height: 20px;
    padding: 6px 11px;
    selection-background-color: {PRIMARY_CONTAINER};
}}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {BORDER_STRONG};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QComboBox:on {{
    border-color: {PRIMARY};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {DISABLED_FG};
    border-color: {BORDER};
}}
QLineEdit[readOnly="true"] {{ color: {ON_SURFACE_VARIANT}; }}
QComboBox {{ combobox-popup: 0; padding-right: 30px; }}
QComboBox::drop-down {{ border: none; width: 28px; subcontrol-position: center right; }}
QComboBox::down-arrow {{ image: url("{_CHEVRON_DOWN}"); width: 18px; height: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_CONTAINER_HIGH};
    color: {ON_SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 10px;
    padding: 4px;
    outline: none;
}}
QComboBox QAbstractItemView::item {{
    min-height: 22px;
    padding: 5px 10px;
    border-radius: 6px;
}}
QComboBox QAbstractItemView::item:hover {{ background: {HOVER_OVERLAY}; }}
QComboBox QAbstractItemView::item:selected {{ background: {SECONDARY_CONTAINER}; color: {ON_SURFACE}; }}
QCheckBox {{ spacing: 9px; color: {ON_SURFACE}; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    background: {SURFACE_CONTAINER_HIGHEST};
}}
QCheckBox::indicator:hover {{ border-color: {OUTLINE}; }}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    image: url("{_CHECK}");
}}
QCheckBox:disabled {{ color: {OUTLINE}; }}
QCheckBox::indicator:disabled {{ border-color: {BORDER}; background: transparent; }}
QCheckBox::indicator:checked:disabled {{ background: {OUTLINE_VARIANT}; border-color: {OUTLINE_VARIANT}; }}

/* ---- buttons ---- */
QPushButton {{
    background: {HOVER_OVERLAY};
    color: {ON_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 14px;
    min-height: 20px;
    font-weight: 500;
}}
QPushButton:hover {{ background: rgba(255, 255, 255, 0.10); border-color: {BORDER_STRONG}; }}
QPushButton:disabled {{ color: {DISABLED_FG}; border-color: {BORDER}; background: transparent; }}
QPushButton#primaryButton {{
    background: {PRIMARY};
    color: {ON_PRIMARY};
    border: none;
    padding: 8px 22px;
    font-weight: 600;
}}
QPushButton#primaryButton:hover {{ background: #B6CAFF; }}
QPushButton#primaryButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}
QPushButton#tonalButton {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
    border: 1px solid {BORDER};
}}
QPushButton#tonalButton:hover {{ background: #33363E; }}
QPushButton#tonalButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}
QPushButton#textButton {{
    background: transparent;
    color: {PRIMARY};
    border: none;
    padding: 8px 12px;
}}
QPushButton#textButton:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton#textButton:disabled {{ color: {DISABLED_FG}; }}
QPushButton#starButton {{
    background: transparent;
    color: {WARNING};
    border: none;
    padding: 2px 3px;
    font-size: 20px;
}}
QPushButton#starButton:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton#dangerButton {{
    background: {ERROR_CONTAINER};
    color: {ON_ERROR_CONTAINER};
    border: none;
}}
QPushButton#dangerButton:hover {{ background: #A8121B; }}
QPushButton#dangerButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}
QPushButton#fabButton {{
    background: {PRIMARY_CONTAINER};
    color: {ON_PRIMARY_CONTAINER};
    border: 1px solid rgba(157, 184, 255, 0.18);
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: 600;
}}
QPushButton#fabButton:hover {{ background: #355489; }}
QPushButton#warningBanner {{
    background: {WARNING_CONTAINER};
    color: {ON_WARNING_CONTAINER};
    border: none;
    border-radius: 12px;
    text-align: left;
    padding: 10px 14px;
}}
QPushButton#warningBanner:hover {{ background: #6B5100; }}
QPushButton#globalNotice {{
    background: {ON_SURFACE};
    color: {SURFACE};
    border: none;
    border-radius: 8px;
    text-align: left;
    padding: 10px 16px;
    font-weight: 500;
}}
QPushButton#globalNotice[state="error"] {{
    background: {ERROR_CONTAINER};
    color: {ON_ERROR_CONTAINER};
}}

/* filter chips (single choice) */
QToolButton#segment {{
    background: transparent;
    color: {ON_SURFACE_VARIANT};
    border: 1px solid {OUTLINE_VARIANT};
    border-radius: 8px;
    padding: 6px 14px;
    font-weight: 600;
}}
QToolButton#segment:hover {{ background: {HOVER_OVERLAY}; }}
QToolButton#segment:checked {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
    border-color: {SECONDARY_CONTAINER};
}}
QToolButton#helpButton {{
    background: {HOVER_OVERLAY};
    color: {OUTLINE};
    border: none;
    border-radius: 8px;
    min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px;
    padding: 0px;
    font-size: 7.5pt;
    font-weight: 600;
}}
QToolButton#helpButton:hover {{
    color: {PRIMARY};
    border-color: {PRIMARY};
}}
QToolButton#expander {{
    background: transparent;
    color: {PRIMARY};
    border: none;
    border-radius: 16px;
    padding: 6px 10px;
    font-weight: 600;
}}
QToolButton#expander:hover {{ background: {PRIMARY_HOVER}; }}

/* ---- lists ---- */
QListWidget#gpuList, QListWidget#jobList, QListWidget#navList {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#gpuList::item {{
    background: {SURFACE_CONTAINER};
    border: 1px solid {BORDER};
    border-radius: 10px;
    margin: 3px 0px;
    padding: 10px 12px;
    color: {ON_SURFACE};
}}
QListWidget#gpuList::item:hover {{ background: {HOVER_OVERLAY_SOLID}; }}
QListWidget#gpuList::item:selected {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
}}
QListWidget#gpuList::item:disabled {{ color: {OUTLINE}; background: {SURFACE_CONTAINER_LOW}; }}
QListWidget#jobList::item {{
    background: transparent;
    border-radius: 16px;
    margin: 0px 0px 4px 0px;
}}
QListWidget#jobList::item:hover {{ background: {HOVER_OVERLAY_SOLID}; }}
QListWidget#jobList::item:selected {{ background: {SECONDARY_CONTAINER}; }}
QListWidget#navList::item {{
    border-radius: 18px;
    padding: 9px 14px;
    margin: 2px 0px;
    color: {ON_SURFACE_VARIANT};
    font-weight: 600;
}}
QListWidget#navList::item:hover {{ background: {HOVER_OVERLAY}; }}
QListWidget#navList::item:selected {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
}}
QListWidget#captureItems {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#captureItems::item {{
    background: {SURFACE_CONTAINER_HIGHEST};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin: 2px 0px;
    padding: 7px 10px;
    color: {ON_SURFACE};
}}
QListWidget#captureItems::item:hover {{ background: {HOVER_OVERLAY_SOLID}; }}
QListWidget#captureItems::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    background: {SURFACE_CONTAINER_HIGHEST};
}}
QListWidget#captureItems::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
    image: url("{_CHECK}");
}}
QFrame#jobCard {{ background: transparent; border: none; }}
QLabel#jobTitle {{ color: {ON_SURFACE}; font-weight: 600; }}
QLabel#cardMeta {{ color: {ON_SURFACE_VARIANT}; font-size: 9pt; }}
QLabel#statusDot {{ color: {PRIMARY}; font-size: 11pt; }}
QLabel#statusDot[state="finished"] {{ color: {SUCCESS}; }}
QLabel#statusDot[state="failed"] {{ color: {ERROR}; }}
QLabel#statusDot[state="stopped"] {{ color: {OUTLINE}; }}
QLabel#statusDot[state="kept"], QLabel#statusDot[state="stopping"],
QLabel#statusDot[state="interrupted"] {{ color: {WARNING}; }}

/* ---- job view ---- */
QTabWidget#jobTabs::pane {{ border: none; }}
QTabWidget#jobTabs QTabBar::tab {{
    background: transparent;
    color: {ON_SURFACE_VARIANT};
    border: none;
    border-bottom: 3px solid transparent;
    padding: 9px 18px;
    font-weight: 600;
}}
QTabWidget#jobTabs QTabBar::tab:selected {{
    color: {PRIMARY};
    border-bottom-color: {PRIMARY};
}}
QTabWidget#jobTabs QTabBar::tab:hover {{ color: {ON_SURFACE}; }}
QPlainTextEdit#jobLog {{
    background: {SURFACE_CONTAINER_LOWEST};
    color: {ON_SURFACE_VARIANT};
    border: none;
    border-radius: 12px;
    padding: 12px;
    font-family: {MONO_FAMILY};
    font-size: 9pt;
    selection-background-color: {PRIMARY_CONTAINER};
}}
QSplitter::handle {{ background: transparent; height: 8px; }}

/* ---- tables (untracked pods) ---- */
QTableWidget {{
    background: {SURFACE_CONTAINER};
    color: {ON_SURFACE};
    border: none;
    border-radius: 12px;
    gridline-color: {OUTLINE_VARIANT};
    selection-background-color: {SECONDARY_CONTAINER};
    selection-color: {ON_SECONDARY_CONTAINER};
}}
QHeaderView::section {{
    background: {SURFACE_CONTAINER};
    color: {ON_SURFACE_VARIANT};
    border: none;
    border-bottom: 1px solid {OUTLINE_VARIANT};
    padding: 8px;
    font-weight: 600;
}}

/* ---- misc ---- */
QToolTip {{
    background: {SURFACE_CONTAINER_HIGH};
    color: {ON_SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 7px 10px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {OUTLINE_VARIANT}; border-radius: 3px; min-height: 28px; margin: 0 2px; }}
QScrollBar::handle:vertical:hover {{ background: {OUTLINE}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {OUTLINE_VARIANT}; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""


def load_fonts() -> None:
    """Inter (SIL Open Font License, assets/fonts) for the whole app; the stylesheet falls back
    to Noto Sans when the files are missing. Needs the QApplication to exist."""
    from PySide6 import QtGui
    for path in sorted(FONT_DIR.glob("*.ttf")):
        QtGui.QFontDatabase.addApplicationFont(str(path))
