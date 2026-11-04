#!/usr/bin/env bash
# Sets up Gaussian Studio on Linux or macOS and adds its desktop shortcut.
# Run it again after updating (git pull); your keys and settings are kept.
# This only finds Python; desktop/installer.py does the rest, on every system.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    echo "Python 3.10+ is missing. Install Homebrew (https://brew.sh), then run: brew install python@3.12" >&2
  else
    echo "Python 3.10+ is missing. On Ubuntu/Debian run: sudo apt install python3 python3-venv" >&2
  fi
  exit 1
fi
echo "Using $PYTHON ($("$PYTHON" --version))"
exec "$PYTHON" desktop/installer.py install
