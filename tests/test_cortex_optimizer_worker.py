import unittest
from pathlib import Path

from scripts.cortex_optimizer_worker import (
    RemoteTarget,
    UnsafeRemoteRoot,
    admin_commands,
    remote_install_command,
    remote_install_command_with_options,
    remote_optimizer_command,
    remote_chown_command,
    require_safe_remote_root,
    rsync_pull_command,
    rsync_push_command,
    worker_shell_command,
)


class CortexOptimizerWorkerTests(unittest.TestCase):
    def test_safe_remote_root_accepts_imaopt_child(self):
        root = require_safe_remote_root(Path("/home/imaopt/research-v2"))
        self.assertEqual(Path("/home/imaopt/research-v2"), root)

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
        self.assertIn("--exclude .venv*/", rendered)
        self.assertIn("--exclude artifacts/", rendered)
        self.assertIn("--exclude .env.local", rendered)
        self.assertTrue(command[-1].endswith("root@100.95.24.121:/home/imaopt/research-v2/"))

    def test_sync_chowns_remote_root_for_worker(self):
        command = remote_chown_command(RemoteTarget())
        rendered = " ".join(command)
        self.assertIn("root@100.95.24.121", rendered)
        self.assertIn("chown -R imaopt:imaopt /home/imaopt/research-v2", rendered)

    def test_worker_shell_wraps_when_ssh_user_differs(self):
        command = worker_shell_command(RemoteTarget(), "set -eu; cd /home/imaopt/iMa; whoami")
        self.assertIn("sudo -u imaopt -H env HOME=/home/imaopt bash -lc", command)
        self.assertIn("whoami", command)

    def test_worker_shell_direct_when_ssh_user_matches(self):
        command = worker_shell_command(
            RemoteTarget(ssh_user="imaopt"),
            "set -eu; cd /home/imaopt/iMa; whoami",
        )
        self.assertEqual("set -eu; cd /home/imaopt/iMa; whoami", command)

    def test_install_uses_supported_python_and_remote_root(self):
        command = remote_install_command(RemoteTarget(), python_bin="python3.12")
        rendered = " ".join(command)
        self.assertIn("python3.12 -m venv .venv", rendered)
        self.assertIn("cd /home/imaopt/research-v2", rendered)
        self.assertIn("pip install -e .", rendered)
        self.assertIn("sudo -u imaopt -H env HOME=/home/imaopt bash -lc", rendered)

    def test_install_can_use_mirror_python_override_and_no_deps(self):
        command = remote_install_command_with_options(
            RemoteTarget(),
            python_bin="python3",
            venv_dir=".venv314",
            pip_index_url="https://pypi.tuna.tsinghua.edu.cn/simple",
            pip_trusted_host="pypi.tuna.tsinghua.edu.cn",
            upgrade_pip=False,
            ignore_requires_python=True,
            no_deps=True,
        )
        rendered = " ".join(command)
        self.assertIn("python3 -m venv .venv314", rendered)
        self.assertIn("PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple", rendered)
        self.assertIn("PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn", rendered)
        self.assertIn("pip install --ignore-requires-python --no-deps -e .", rendered)
        self.assertNotIn("pip install --upgrade pip", rendered)

    def test_optimizer_command_defaults_to_sequential_local_policy(self):
        command = remote_optimizer_command(RemoteTarget(), "artifacts/agentic-learning/cortex-smoke", dry_run=True)
        rendered = " ".join(command)
        self.assertIn("--max-concurrent-trials 1", rendered)
        self.assertIn("--policy local", rendered)
        self.assertIn("--dry-run", rendered)

    def test_optimizer_command_can_use_agentic_policy(self):
        command = remote_optimizer_command(
            RemoteTarget(),
            "artifacts/agentic-learning/cortex-agentic-smoke",
            policy="agentic",
            dry_run=True,
        )
        rendered = " ".join(command)
        self.assertIn("--policy agentic", rendered)

    def test_optimizer_command_forwards_relative_config(self):
        command = remote_optimizer_command(
            RemoteTarget(),
            "artifacts/agentic-learning/cortex-agentic-smoke",
            policy="agentic",
            config_path="config/cortex-agentic.json",
            dry_run=True,
        )
        self.assertIn("--config config/cortex-agentic.json", " ".join(command))
        with self.assertRaisesRegex(ValueError, "relative"):
            remote_optimizer_command(
                RemoteTarget(),
                "artifacts/agentic-learning/cortex-agentic-smoke",
                config_path="../secret.json",
            )

    def test_optimizer_command_can_use_alternate_venv(self):
        command = remote_optimizer_command(
            RemoteTarget(),
            "artifacts/agentic-learning/cortex-smoke",
            venv_dir=".venv314",
            dry_run=True,
        )
        self.assertIn(".venv314/bin/ima-optimize", " ".join(command))

    def test_pull_rejects_nested_campaign_names(self):
        with self.assertRaises(ValueError):
            rsync_pull_command(RemoteTarget(), "../bad")
        with self.assertRaises(ValueError):
            rsync_pull_command(RemoteTarget(), "nested/bad")

    def test_admin_commands_are_scoped_to_imaopt(self):
        rendered = admin_commands()
        self.assertIn("useradd --create-home --shell /bin/bash imaopt", rendered)
        self.assertIn("/home/imaopt/.ssh/authorized_keys", rendered)
        self.assertIn("/home/imaopt/research-v2", rendered)
        self.assertNotIn("/srv/apps/cortex", rendered)


if __name__ == "__main__":
    unittest.main()
