"""Installs and uninstalls Gaussian Studio on Linux, macOS and Windows.

The scripts in the app's folder only find Python and run this:
    install.sh / install.cmd      ->  python desktop/installer.py install
    uninstall.sh / uninstall.cmd  ->  python desktop/installer.py uninstall [--yes ...]

It uses only the standard library, so it runs before the app's packages are installed
and after they're gone. Messages are plain ASCII, which every console can print.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
WINDOWS = os.name == "nt"
MAC = sys.platform == "darwin"
VENV = APP_DIR / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
KEY_LABEL = "gaussian-studio"  # the comment on SSH keys the installer makes
CLOSED_STATES = {"terminated", "not_created", "gone"}  # as in pod_ledger.py
COLOUR = sys.stdout.isatty() and not WINDOWS
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if WINDOWS else {}


def say(text: str) -> None:
    print(f"\n\033[1m{text}\033[0m" if COLOUR else f"\n{text}", flush=True)


def fail(text: str) -> None:
    raise SystemExit(f"\n{text}")


def run(command: list, failure: str, **kwargs) -> None:
    if subprocess.run([str(part) for part in command], **kwargs).returncode:
        fail(failure)


def ask(question: str, default: bool, assume: bool) -> bool:
    """Yes or no; with --yes, the default answer."""
    if assume:
        return default
    reply = input(f"{question} {'[Y/n]' if default else '[y/N]'} ").strip()
    return reply.lower().startswith("y") if reply else default


def windows_folder(csidl: int) -> Path:
    import ctypes

    buffer = ctypes.create_unicode_buffer(260)
    ctypes.windll.shell32.SHGetFolderPathW(None, csidl, None, 0, buffer)
    return Path(buffer.value)


# ---- install --------------------------------------------------------------

def default_key() -> Path:
    return Path.home() / ".ssh" / "id_ed25519"


def install() -> None:
    say("Installing the app's Python packages (first run takes a few minutes)...")
    if not VENV_PYTHON.is_file():
        shutil.rmtree(VENV, ignore_errors=True)
        run([sys.executable, "-m", "venv", VENV],
            "Could not create the virtual environment."
            + ("" if WINDOWS or MAC else " On Ubuntu/Debian run: sudo apt install python3-venv"))
    run([VENV_PYTHON, "-m", "pip", "install", "--quiet", "--upgrade", "pip"],
        "Updating pip failed; see the messages above.")
    run([VENV_PYTHON, "-m", "pip", "install", "--quiet", "-r", APP_DIR / "requirements.txt"],
        "Installing the Python packages failed; see the messages above.")
    check_tools()

    say("Checking for an SSH key...")
    key, new_key = default_key(), False
    public = Path(f"{key}.pub")
    if key.is_file() and public.is_file():
        print(f"Using your existing key {key}")
    elif key.exists():
        fail(f'{key} exists without {public.name}. Recreate it with: ssh-keygen -y -f "{key}" > "{public}"')
    else:
        key.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", KEY_LABEL, "-f", key],
            "Could not create an SSH key.")
        print(f"Created {key}")
        new_key = True

    say("Adding the Gaussian Studio shortcut...")
    run([VENV_PYTHON, "-X", "utf8", "install_launcher.py"], "Adding the shortcut failed.",
        cwd=APP_DIR)

    if new_key:
        line = public.read_text(encoding="utf-8").strip()
        copied = copy_to_clipboard(line)
        say("One step on RunPod: add your new public key")
        print("Open https://console.runpod.io/user/settings > SSH Public Keys and paste this line:")
        print(f"\n{line}")
        if copied:
            print("\n(It's already on your clipboard.)")

    say("Done.")
    where = "the Start menu or your desktop" if WINDOWS else "your desktop"
    print(f"Open Gaussian Studio from {where}, then go to Settings > Connection and paste")
    print("your RunPod API key. The setup guide in this folder (SETUP.md) has the details.")


def check_tools() -> None:
    """Hints for system tools the app needs; only ssh is required."""
    if not (shutil.which("ssh") and shutil.which("ssh-keygen")):
        if WINDOWS:
            fail("Windows' OpenSSH Client is missing. Add it in Settings > System > Optional "
                 'features > View features > "OpenSSH Client", then run this again.')
        fail("ssh is missing. On Ubuntu/Debian run: sudo apt install openssh-client")
    if not WINDOWS and not MAC:
        try:
            libraries = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
        except OSError:
            libraries = "libxcb-cursor"
        if "libxcb-cursor" not in libraries:
            print("Note: Qt needs libxcb-cursor0 to open windows. If the app doesn't start, run:")
            print("      sudo apt install libxcb-cursor0")
    if not shutil.which("ffprobe"):
        hint = ("winget install -e --id Gyan.FFmpeg" if WINDOWS
                else "brew install ffmpeg" if MAC else "sudo apt install ffmpeg")
        print(f"Note: ffprobe is missing. Photos work without it; to use videos, run: {hint}")


def copy_to_clipboard(text: str) -> bool:
    if WINDOWS:
        commands = [["clip"]]
    elif MAC:
        commands = [["pbcopy"]]
    else:
        commands = [["wl-copy"], ["xclip", "-selection", "clipboard"]]
    for command in commands:
        if not shutil.which(command[0]):
            continue
        try:
            # wl-copy and xclip stay running to serve the clipboard; don't wait on their output.
            subprocess.run(command, input=text.encode(), stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10, check=True)
            return True
        except (OSError, subprocess.SubprocessError):
            continue
    return False


# ---- uninstall ------------------------------------------------------------

def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_env(path: Path) -> dict[str, str]:
    values = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        name, sep, value = line.strip().partition("=")
        if sep and not name.startswith("#"):
            values[name.strip()] = value.strip().strip("'\"")
    return values


def setting(state: dict, env: dict, field: str, name: str) -> str:
    """A saved setting wins over .env, as in config.py."""
    value = state.get(field) if field in state else env.get(name, "")
    return str(value or "").strip()


def where_things_are(app_dir: Path = APP_DIR) -> tuple[Path, Path, list[str]]:
    """The results folder, the SSH key, and the pods that may still be running."""
    state = read_json(app_dir / "app_state.json")
    env = read_env(app_dir / ".env")
    results = setting(state, env, "results_folder", "RESULTS_FOLDER")
    key = setting(state, env, "ssh_private_key", "SSH_PRIVATE_KEY")
    pods = [
        "  ".join(str(part) for part in (entry.get("pod_name") or entry.get("pod_id") or "?",
                                         entry.get("title") or "", entry.get("state") or "?") if part)
        for entry in read_json(app_dir / "pod_ledger.json").values()
        if isinstance(entry, dict) and entry.get("state") not in CLOSED_STATES
    ]
    return ((app_dir / Path(results).expanduser()).resolve() if results else app_dir / "results",
            Path(key).expanduser() if key else default_key(), pods)


def made_by_installer(key: Path) -> bool:
    try:
        return Path(f"{key}.pub").read_text(encoding="utf-8").split()[-1] == KEY_LABEL
    except (OSError, IndexError):
        return False


def app_running() -> bool:
    """Whether the app is open: a Python running main.py from this folder."""
    if WINDOWS:
        try:
            found = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
                 "ForEach-Object CommandLine"],
                capture_output=True, text=True, timeout=60, **HIDDEN).stdout
        except (OSError, subprocess.SubprocessError):
            return False
        return str(APP_DIR / "main.py").lower() in found.lower()
    try:
        pids = subprocess.run(["pgrep", "-f", r"main\.py"], capture_output=True, text=True).stdout.split()
    except OSError:
        return False
    for pid in pids:
        try:  # the shortcuts start the app inside its folder
            cwd = os.readlink(f"/proc/{pid}/cwd") if Path("/proc").is_dir() else subprocess.run(
                ["lsof", "-a", "-p", pid, "-d", "cwd", "-Fn"], capture_output=True, text=True,
            ).stdout.partition("\nn")[2].strip()
        except OSError:
            continue
        if cwd and Path(cwd) == APP_DIR:
            return True
    return False


def results_home() -> Path:
    base = windows_folder(0x05) if WINDOWS else Path.home()  # CSIDL_PERSONAL: Documents
    target = base / "Gaussian Studio results"
    return target.with_name(f"{target.name} {time.strftime('%Y-%m-%d-%H%M%S')}") if target.exists() else target


def _force_remove(function, path, _error) -> None:
    """Windows won't delete read-only files (git's objects); clear the flag and retry."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def delete_folder(folder: Path) -> bool:
    if folder.exists():
        shutil.rmtree(folder, **{"onexc" if sys.version_info >= (3, 12) else "onerror": _force_remove})
    return not folder.exists()


def uninstall(args: argparse.Namespace) -> None:
    assume = args.yes
    if not (APP_DIR / "main.py").is_file():
        fail(f"This doesn't look like the Gaussian Studio folder ({ROOT}); nothing was removed.")
    if app_running():
        fail("Gaussian Studio is open. Close it first, then run this again.")
    results, key, pods = where_things_are()

    if pods:
        say("These GPUs may still be running, and RunPod bills them until they're stopped:")
        for pod in pods:
            print(f"  {pod}")
        print("Open the app and stop them, or terminate them at https://console.runpod.io/pods.")
        if assume:
            if not args.ignore_pods:
                fail("Nothing was removed. Add --ignore-pods to remove the app anyway.")
        elif not ask("Remove the app anyway?", False, False):
            fail("Nothing was removed.")

    # Results inside the app folder would go with it: move them out first, unless told otherwise.
    results_plan, move_to = "keep", None
    if results.is_relative_to(ROOT) and results.is_dir() and any(results.iterdir()):
        move_to = results_home()
        say(f"Your finished splats are inside the app folder: {results}")
        if assume:
            results_plan = "delete" if args.delete_results else "move"
        elif ask(f'Move them to "{move_to}" so they\'re kept?', True, False):
            results_plan = "move"
        elif ask("Delete them with the app?", False, False):
            results_plan = "delete"
        else:
            fail("Nothing was removed. Copy the results somewhere safe, then run this again.")

    # Only offer to delete an SSH key the installer made.
    delete_key = False
    if made_by_installer(key):
        say(f"The installer made the SSH key {key} for this app.")
        delete_key = args.delete_key if assume else ask(
            "Delete it too? (Keep it if you use it for anything else.)", False, False)

    say("This removes:")
    print("  - the Gaussian Studio shortcuts")
    print(f"  - {ROOT} (the app, its Python packages, your saved RunPod key, settings and logs)")
    if results_plan == "move":
        print(f"  - after moving your results to {move_to}")
    elif results_plan == "delete":
        print(f"  - including your results in {results}")
    if delete_key:
        print(f"  - the SSH key {key} and {key}.pub")
    if not ask("Go ahead?", True, assume):
        fail("Nothing was removed.")

    if results_plan == "move":
        shutil.move(str(results), str(move_to))
        print(f"Moved your results to {move_to}")
    sys.path.insert(0, str(APP_DIR))
    import install_launcher  # standard library only, like this file

    try:
        install_launcher.remove()
    except OSError as exc:
        print(f"Could not remove the shortcuts ({exc}); delete them by hand.")
    if delete_key:
        for path in (key, Path(f"{key}.pub")):
            path.unlink(missing_ok=True)
        print(f"Deleted {key}")
    os.chdir(Path.home())
    try:
        removed = delete_folder(ROOT)
    except OSError:
        removed = False
    if removed:
        print(f"Deleted {ROOT}")
    elif WINDOWS:
        # The Python running this may be the app's own, which Windows can't delete while it
        # runs: finish once it has exited.
        subprocess.Popen(f'cmd /c ping -n 4 127.0.0.1 >nul & rmdir /s /q "{ROOT}"',
                         cwd=str(Path.home()), **HIDDEN)
        print(f"Deleting {ROOT} once this window closes")
    else:
        print(f"Some files could not be removed. Delete the folder {ROOT} yourself.")

    say("Gaussian Studio is removed.")
    print("Left for you to remove on the websites, if you like:")
    print("  - your RunPod API key: https://console.runpod.io/user/settings > API Keys")
    print("  - the SSH public key you added there: same page > SSH Public Keys")
    print("  - if you set up Google Drive: the service account key in Google Cloud Console")
    print("Python, ffmpeg and git stay installed, since other apps may use them.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("install", help="set up the app and its shortcuts; run again after updating")
    remove = commands.add_parser("uninstall", help="remove the app, asking about results and the SSH key")
    remove.add_argument("--yes", action="store_true",
                        help="no questions: results move out of the app folder, the SSH key stays")
    remove.add_argument("--delete-results", action="store_true", help="with --yes: delete the results")
    remove.add_argument("--delete-key", action="store_true",
                        help="with --yes: delete the SSH key, if the installer made it")
    remove.add_argument("--ignore-pods", action="store_true",
                        help="with --yes: remove the app even if a GPU may still be running")
    args = parser.parse_args()
    try:
        install() if args.command == "install" else uninstall(args)
    except KeyboardInterrupt:
        fail("Stopped.")


if __name__ == "__main__":
    main()
