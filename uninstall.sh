#!/usr/bin/env bash
# Removes Gaussian Studio from Linux or macOS, asking before touching your results or
# your SSH key. For scripts: ./uninstall.sh --yes [--delete-results] [--delete-key] [--ignore-pods]
# This only finds Python; desktop/installer.py does the rest, on every system.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="$(command -v python3 || true)"
[[ -n "$PYTHON" ]] || PYTHON=desktop/.venv/bin/python
exec "$PYTHON" desktop/installer.py uninstall "$@"
