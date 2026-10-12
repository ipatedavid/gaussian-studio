"""Material 3 dark theme (blue seed) for the desktop application.

Colours are Material 3 roles. Components follow M3 shapes: filled/tonal/
outlined/text buttons, filter chips for choices, filled text fields, tonal
surface containers instead of borders.
"""
from pathlib import Path

# the combo box arrow: a stylesheet can only draw it from a file
_CHEVRON_DOWN = (Path(__file__).resolve().parent / "assets" / "chevron-down.svg").as_posix()

# ---- colour roles --------------------------------------------------------
PRIMARY = "#ADC6FF"
ON_PRIMARY = "#102F60"
PRIMARY_CONTAINER = "#2B4678"
ON_PRIMARY_CONTAINER = "#D8E2FF"
SECONDARY_CONTAINER = "#3E4759"
ON_SECONDARY_CONTAINER = "#DAE2F9"
TERTIARY_CONTAINER = "#573E5C"
ERROR = "#FFB4AB"
ERROR_CONTAINER = "#93000A"
ON_ERROR_CONTAINER = "#FFDAD6"
SUCCESS = "#8DD5A6"
SUCCESS_CONTAINER = "#1D4D33"
ON_SUCCESS_CONTAINER = "#A9F2C1"
WARNING = "#F2C26B"
WARNING_CONTAINER = "#5A4300"
ON_WARNING_CONTAINER = "#FFDFA0"
SURFACE = "#111318"
SURFACE_CONTAINER_LOWEST = "#0C0E13"
SURFACE_CONTAINER_LOW = "#191C20"
SURFACE_CONTAINER = "#1D2024"
SURFACE_CONTAINER_HIGH = "#282A2F"
SURFACE_CONTAINER_HIGHEST = "#33353A"
ON_SURFACE = "#E2E2E9"
ON_SURFACE_VARIANT = "#C4C6D0"
OUTLINE = "#8E9099"
OUTLINE_VARIANT = "#44474F"
HOVER_OVERLAY = "rgba(226, 226, 233, 0.08)"
HOVER_OVERLAY_SOLID = "#24272C"
PRIMARY_HOVER = "rgba(173, 198, 255, 0.10)"
DISABLED_BG = "rgba(226, 226, 233, 0.12)"
DISABLED_FG = "rgba(226, 226, 233, 0.38)"

FONT_FAMILY = '"Noto Sans", "Roboto", "DejaVu Sans", sans-serif'
MONO_FAMILY = '"Noto Sans Mono", "DejaVu Sans Mono", monospace'

APP_STYLE = f"""
QWidget {{
    color: {ON_SURFACE};
    font-family: {FONT_FAMILY};
    font-size: 10pt;
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
    border: 1px solid {OUTLINE_VARIANT};
    border-radius: 24px;
}}
QLabel#dialogText {{
    color: {ON_SURFACE_VARIANT};
    font-size: 10.5pt;
}}

/* ---- surfaces ---- */
QFrame#card, QFrame#activityCard {{
    background: {SURFACE_CONTAINER_LOW};
    border: none;
    border-radius: 16px;
}}
QFrame#summaryBar {{
    background: {SURFACE_CONTAINER};
    border: none;
    border-top: 1px solid {OUTLINE_VARIANT};
}}
QFrame#sidebar {{
    background: {SURFACE_CONTAINER_LOW};
    border: none;
}}
QFrame#subtlePanel {{
    background: {SURFACE_CONTAINER};
    border-radius: 12px;
}}

/* ---- type ---- */
QLabel#brand, QLabel#pageTitle {{
    color: {ON_SURFACE};
    font-size: 17pt;
    font-weight: 500;
}}
QLabel#cardTitle {{
    color: {ON_SURFACE};
    font-size: 12pt;
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
    font-size: 9pt;
    font-weight: 600;
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

/* ---- text fields (M3 filled) ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {SURFACE_CONTAINER_HIGHEST};
    color: {ON_SURFACE};
    border: none;
    border-bottom: 1px solid {ON_SURFACE_VARIANT};
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    min-height: 22px;
    padding: 8px 12px;
    selection-background-color: {PRIMARY_CONTAINER};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-bottom: 2px solid {PRIMARY};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {DISABLED_FG};
    border-bottom-color: {OUTLINE_VARIANT};
}}
QComboBox::drop-down {{ border: none; width: 30px; }}
QComboBox::down-arrow {{ image: url("{_CHEVRON_DOWN}"); width: 18px; height: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_CONTAINER};
    color: {ON_SURFACE};
    border: 1px solid {OUTLINE_VARIANT};
    selection-background-color: {SECONDARY_CONTAINER};
    outline: none;
}}
QCheckBox {{ spacing: 10px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {ON_SURFACE_VARIANT};
    border-radius: 3px;
    background: transparent;
}}
QCheckBox::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
}}

/* ---- buttons ---- */
QPushButton {{
    background: transparent;
    color: {PRIMARY};
    border: 1px solid {OUTLINE};
    border-radius: 20px;
    padding: 8px 20px;
    min-height: 22px;
    font-weight: 600;
}}
QPushButton:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton:disabled {{ color: {DISABLED_FG}; border-color: {DISABLED_BG}; }}
QPushButton#primaryButton {{
    background: {PRIMARY};
    color: {ON_PRIMARY};
    border: none;
    padding: 9px 24px;
}}
QPushButton#primaryButton:hover {{ background: #BFD2FF; }}
QPushButton#primaryButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}
QPushButton#tonalButton {{
    background: {SECONDARY_CONTAINER};
    color: {ON_SECONDARY_CONTAINER};
    border: none;
}}
QPushButton#tonalButton:hover {{ background: #4A5468; }}
QPushButton#tonalButton:disabled {{ background: {DISABLED_BG}; color: {DISABLED_FG}; }}
QPushButton#textButton {{
    background: transparent;
    color: {PRIMARY};
    border: none;
    padding: 8px 12px;
}}
QPushButton#textButton:hover {{ background: {PRIMARY_HOVER}; }}
QPushButton#textButton:disabled {{ color: {DISABLED_FG}; }}
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
    border: none;
    border-radius: 16px;
    padding: 12px 18px;
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
    background: transparent;
    color: {ON_SURFACE_VARIANT};
    border: 1px solid {OUTLINE_VARIANT};
    border-radius: 8px;
    min-width: 14px; max-width: 14px; min-height: 14px; max-height: 14px;
    padding: 0px;
    font-size: 7.5pt;
    font-weight: 700;
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
    border-radius: 12px;
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
    background: {SURFACE_CONTAINER};
    border-radius: 10px;
    margin: 2px 0px;
    padding: 7px 10px;
    color: {ON_SURFACE};
}}
QListWidget#captureItems::item:hover {{ background: {HOVER_OVERLAY_SOLID}; }}
QListWidget#captureItems::indicator {{
    width: 16px; height: 16px;
    border: 2px solid {ON_SURFACE_VARIANT};
    border-radius: 3px;
    background: transparent;
}}
QListWidget#captureItems::indicator:checked {{
    background: {PRIMARY};
    border-color: {PRIMARY};
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
    background: {ON_SURFACE};
    color: {SURFACE};
    border: none;
    border-radius: 4px;
    padding: 5px 8px;
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {OUTLINE_VARIANT}; border-radius: 4px; min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {OUTLINE}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {OUTLINE_VARIANT}; border-radius: 4px; min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
"""
