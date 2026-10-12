"""Settings saves the RunPod key to .env and installs a Drive service-account key."""
from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "desktop"))

import config
from dotenv import dotenv_values


class SettingsKeyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".env"
        self.key = self.tmp / "service-account.json"
        patcher = patch.multiple(config, ENV_FILE=self.env, DRIVE_KEY_FILE=self.key)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _download(self, name="downloaded.json", **fields):
        data = {"type": "service_account", "client_email": "reader@proj.iam.gserviceaccount.com"}
        data.update(fields)
        path = self.tmp / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_api_key_is_written_private_and_keeps_other_values(self):
        self.env.write_text("MAX_GPU_PRICE_PER_HOUR=0.55\n", encoding="utf-8")
        config.write_env_value("RUNPOD_API_KEY", "rpa_test")
        config.write_env_value("RUNPOD_API_KEY", "rpa_new")
        values = dotenv_values(self.env)
        self.assertEqual(values["RUNPOD_API_KEY"], "rpa_new")
        self.assertEqual(values["MAX_GPU_PRICE_PER_HOUR"], "0.55")
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)

    def test_creates_env_when_missing(self):
        config.write_env_value("RUNPOD_API_KEY", "rpa_test")
        self.assertEqual(dotenv_values(self.env)["RUNPOD_API_KEY"], "rpa_test")
        self.assertEqual(stat.S_IMODE(self.env.stat().st_mode), 0o600)

    def test_drive_key_is_copied_and_env_points_at_it(self):
        installed = config.install_drive_key(self._download())
        self.assertEqual(installed, self.key)
        self.assertEqual(json.loads(self.key.read_text())["type"], "service_account")
        self.assertEqual(stat.S_IMODE(self.key.stat().st_mode), 0o600)
        self.assertEqual(dotenv_values(self.env)["GOOGLE_SERVICE_ACCOUNT_FILE"], "service-account.json")

    def test_choosing_the_installed_key_again_keeps_it(self):
        config.install_drive_key(self._download())
        config.install_drive_key(self.key)
        self.assertEqual(config.service_account_email(self.key), "reader@proj.iam.gserviceaccount.com")

    def test_rejects_files_that_are_not_service_account_keys(self):
        oauth = self._download("client.json", type="authorized_user")
        broken = self.tmp / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        for path in (oauth, broken, self.tmp / "missing.json"):
            with self.assertRaises(config.DriveKeyError):
                config.install_drive_key(path)
        self.assertFalse(self.key.exists())
        self.assertFalse(self.env.exists())


if __name__ == "__main__":
    unittest.main()
