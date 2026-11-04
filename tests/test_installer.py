"""desktop/installer.py: what the uninstaller finds, and a whole uninstall in a sandbox home."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "desktop"))

import installer  # noqa: E402


class WhereThingsAreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = Path(self.tmp.name).resolve()

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_without_settings(self):
        results, key, pods = installer.where_things_are(self.app)
        self.assertEqual(results, self.app / "results")
        self.assertEqual(key, Path.home() / ".ssh" / "id_ed25519")
        self.assertEqual(pods, [])

    def test_saved_settings_win_over_env_and_open_pods_are_listed(self):
        (self.app / ".env").write_text('RESULTS_FOLDER="/from/env"\nSSH_PRIVATE_KEY=/keys/env\n')
        (self.app / "app_state.json").write_text(json.dumps({"results_folder": "splats"}))
        (self.app / "pod_ledger.json").write_text(json.dumps({
            "a": {"pod_name": "gaussian-a", "title": "Museum", "state": "running"},
            "b": {"pod_name": "gaussian-b", "state": "terminated"},
        }))
        results, key, pods = installer.where_things_are(self.app)
        self.assertEqual(results, self.app / "splats")  # relative to the app folder
        self.assertEqual(key, Path("/keys/env"))
        self.assertEqual(pods, ["gaussian-a  Museum  running"])

    def test_only_keys_the_installer_labelled_count_as_its_own(self):
        key = self.app / "id_ed25519"
        Path(f"{key}.pub").write_text("ssh-ed25519 AAAA gaussian-studio\n")
        self.assertTrue(installer.made_by_installer(key))
        Path(f"{key}.pub").write_text("ssh-ed25519 AAAA me@laptop\n")
        self.assertFalse(installer.made_by_installer(key))
        self.assertFalse(installer.made_by_installer(self.app / "missing"))


@unittest.skipIf(os.name == "nt", "HOME doesn't move Windows' own folders; the Windows CI job runs it for real")
class UninstallTests(unittest.TestCase):
    """The real uninstaller, on a copy of the app in a sandbox home folder."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name).resolve()
        self.root = self.home / "gaussian studio"
        (self.root / "desktop" / "results" / "job 1").mkdir(parents=True)
        for name in ("installer.py", "install_launcher.py", "main.py"):
            shutil.copy(REPO / "desktop" / name, self.root / "desktop" / name)
        shutil.copy(REPO / "uninstall.sh", self.root / "uninstall.sh")
        (self.root / "desktop" / "results" / "job 1" / "gaussian_splat.ply").write_text("ply")
        (self.home / ".ssh").mkdir()
        (self.home / ".ssh" / "id_ed25519").write_text("private")
        (self.home / ".ssh" / "id_ed25519.pub").write_text("ssh-ed25519 AAAA gaussian-studio\n")
        self.entry = self.home / ".local" / "share" / "applications" / "gaussian-studio.desktop"
        self.entry.parent.mkdir(parents=True)
        self.entry.write_text("[Desktop Entry]\n")

    def tearDown(self):
        self.tmp.cleanup()

    def uninstall(self, *args, answers=""):
        env = {**os.environ, "HOME": str(self.home), "XDG_CONFIG_HOME": str(self.home / ".config")}
        return subprocess.run(["bash", str(self.root / "uninstall.sh"), *args], input=answers,
                              capture_output=True, text=True, env=env, cwd=self.root, timeout=60)

    def test_answers_keep_results_and_delete_the_key(self):
        # Move the results out (Enter), delete the key (y), go ahead (Enter).
        result = self.uninstall(answers="\ny\n\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.exists())
        self.assertFalse(self.entry.exists())
        self.assertFalse((self.home / ".ssh" / "id_ed25519").exists())
        kept = self.home / "Gaussian Studio results" / "job 1" / "gaussian_splat.ply"
        self.assertEqual(kept.read_text(), "ply")

    def test_an_open_pod_stops_it_unless_ignored(self):
        (self.root / "desktop" / "pod_ledger.json").write_text(
            json.dumps({"a": {"pod_name": "gaussian-a", "state": "running"}}))
        result = self.uninstall("--yes")
        self.assertEqual(result.returncode, 1)
        self.assertIn("--ignore-pods", result.stderr)
        self.assertTrue(self.root.exists())
        result = self.uninstall("--yes", "--ignore-pods", "--delete-results")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.root.exists())
        self.assertFalse((self.home / "Gaussian Studio results").exists())
        self.assertTrue((self.home / ".ssh" / "id_ed25519").exists())  # kept by default

    def test_saying_no_removes_nothing(self):
        result = self.uninstall(answers="\n\nn\n")
        self.assertEqual(result.returncode, 1)
        self.assertTrue((self.root / "desktop" / "results" / "job 1").is_dir())
        self.assertTrue(self.entry.exists())


if __name__ == "__main__":
    unittest.main()
