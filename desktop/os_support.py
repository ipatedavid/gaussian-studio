"""What differs when the app runs on Windows."""
from __future__ import annotations

import os
import subprocess

WINDOWS = os.name == "nt"

# A windowed app on Windows opens a console window for every console program it
# starts (ssh, ffprobe); pass these to subprocess to keep them hidden.
HIDDEN: dict = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}
