import unittest
from pathlib import Path

from scripts.cortex_optimizer_worker import (
    RemoteTarget,
    UnsafeRemoteRoot,
    admin_commands,
    remote_install_command,
    remote_optimizer_command,
    require_safe_remote_root,
    rsync_pull_command,
    rsync_push_command,
)


class CortexOptimizerWorkerTests(unittest.TestCase):
    def test_safe_remote_root_accepts_imaopt_child(self):
        root = require_safe_remote_root(Path("/home/imaopt/iMa"))
        self.assertEqual(Path("/home/imaopt/iMa"), root)

    def test_safe_remote_root_rejects_protected_paths(self):
        for path in ["/", "/home", "/home/imaopt", "/srv/apps/cortex", "/tmp", "/home/cortex/iMa"]:
            with self.subTest(path=path):
                with self.assertRaises(UnsafeRemoteRoot):
                    require_safe_remote_root(Path(path))

    def test_push_excludes_stateful_and_secret_paths(self):
        command = rsync_push_command(RemoteTarget())
        rendered = " ".join(command)
        self.assertIn("--delete", command)
        self.assertIn("--exclude .venv/", rendered)
        self.assertIn("--exclude artifacts/", rendered)
        self.assertIn("--exclude .env.local", rendered)
        self.assertTrue(command[-1].endswith("imaopt@100.95.24.121:/home/imaopt/iMa/"))

    def test_install_uses_supported_python_and_remote_root(self):
        command = remote_install_command(RemoteTarget(), python_bin="python3.12")
        rendered = " ".join(command)
        self.assertIn("python3.12 -m venv .venv", rendered)
        self.assertIn("cd /home/imaopt/iMa", rendered)
        self.assertIn("pip install -e .", rendered)

    def test_optimizer_command_defaults_to_sequential_local_policy(self):
        command = remote_optimizer_command(RemoteTarget(), "artifacts/agentic-learning/cortex-smoke", dry_run=True)
        rendered = " ".join(command)
        self.assertIn("--max-concurrent-trials 1", rendered)
        self.assertIn("--policy local", rendered)
        self.assertIn("--dry-run", rendered)

    def test_pull_rejects_nested_campaign_names(self):
        with self.assertRaises(ValueError):
            rsync_pull_command(RemoteTarget(), "../bad")
        with self.assertRaises(ValueError):
            rsync_pull_command(RemoteTarget(), "nested/bad")

    def test_admin_commands_are_scoped_to_imaopt(self):
        rendered = admin_commands()
        self.assertIn("useradd --create-home --shell /bin/bash imaopt", rendered)
        self.assertIn("/home/imaopt/.ssh/authorized_keys", rendered)
        self.assertIn("/home/imaopt/iMa", rendered)
        self.assertNotIn("/srv/apps/cortex", rendered)


if __name__ == "__main__":
    unittest.main()
