#!/usr/bin/env bash
# Sets up Gaussian Studio on Linux or macOS and adds its desktop shortcut.
# Run it again after updating (git pull); your keys and settings are kept.
set -euo pipefail

cd "$(dirname "$0")/desktop"
OS="$(uname -s)"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
fail() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

say "Looking for Python 3.10 or newer…"
PYTHON=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  if [[ "$OS" == Darwin ]]; then
    fail "Python 3.10+ is missing. Install Homebrew (https://brew.sh), then run: brew install python@3.12"
  fi
  fail "Python 3.10+ is missing. On Ubuntu/Debian run: sudo apt install python3 python3-venv"
fi
echo "Using $PYTHON ($("$PYTHON" --version))"

say "Installing the app's Python packages (first run takes a few minutes)…"
if [[ ! -x .venv/bin/python ]]; then
  rm -rf .venv
  "$PYTHON" -m venv .venv ||
    fail "Could not create the virtual environment. On Ubuntu/Debian run: sudo apt install python3-venv"
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r requirements.txt

if [[ "$OS" == Linux ]] && command -v ldconfig >/dev/null && ! ldconfig -p | grep libxcb-cursor >/dev/null; then
  echo "Note: Qt needs libxcb-cursor0 to open windows. If the app doesn't start, run:"
  echo "      sudo apt install libxcb-cursor0"
fi
if ! command -v ffprobe >/dev/null 2>&1; then
  if [[ "$OS" == Darwin ]]; then hint="brew install ffmpeg"; else hint="sudo apt install ffmpeg"; fi
  echo "Note: ffprobe is missing. Photos work without it; to use videos, run: $hint"
fi

say "Checking for an SSH key…"
KEY="$HOME/.ssh/id_ed25519"
if [[ -f "$KEY" && -f "$KEY.pub" ]]; then
  echo "Using your existing key $KEY"
  NEW_KEY=0
else
  [[ -e "$KEY" ]] && fail "$KEY exists without $KEY.pub. Recreate it with: ssh-keygen -y -f $KEY > $KEY.pub"
  mkdir -p "$HOME/.ssh" && chmod 700 "$HOME/.ssh"
  ssh-keygen -q -t ed25519 -N "" -C "gaussian-studio" -f "$KEY"
  echo "Created $KEY"
  NEW_KEY=1
fi

say "Adding the Gaussian Studio shortcut…"
.venv/bin/python install_launcher.py

if [[ "$NEW_KEY" == 1 ]]; then
  copied=""
  # wl-copy and xclip stay running to serve the clipboard; detach them from this terminal.
  if command -v pbcopy >/dev/null; then pbcopy < "$KEY.pub" && copied=1
  elif command -v wl-copy >/dev/null; then wl-copy < "$KEY.pub" >/dev/null 2>&1 && copied=1
  elif command -v xclip >/dev/null; then xclip -selection clipboard < "$KEY.pub" >/dev/null 2>&1 && copied=1
  fi
  say "One step on RunPod: add your new public key"
  echo "Open https://console.runpod.io/user/settings → SSH Public Keys and paste this line:"
  echo
  cat "$KEY.pub"
  [[ -n "$copied" ]] && echo && echo "(It's already on your clipboard.)"
fi

say "Done."
echo "Open Gaussian Studio from your desktop, then go to Settings → Connection and paste"
echo "your RunPod API key. The setup guide in this folder (SETUP.md) has the details."
