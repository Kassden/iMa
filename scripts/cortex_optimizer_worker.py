from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_HOST = "100.95.24.121"
DEFAULT_ADMIN_USER = "cortex"
DEFAULT_SSH_USER = "root"
DEFAULT_WORKER_USER = "imaopt"
DEFAULT_REMOTE_ROOT = Path("/home/imaopt/iMa")
DEFAULT_LOCAL_PULL_ROOT = Path("artifacts/remote-cortex")
DEFAULT_VENV_DIR = ".venv"


class UnsafeRemoteRoot(ValueError):
    pass


@dataclass(frozen=True)
class RemoteTarget:
    host: str = DEFAULT_HOST
    user: str = DEFAULT_WORKER_USER
    remote_root: Path = DEFAULT_REMOTE_ROOT
    ssh_user: str = DEFAULT_SSH_USER

    @property
    def login(self) -> str:
        return f"{self.ssh_user}@{self.host}"


def require_safe_remote_root(remote_root: Path, worker_user: str = DEFAULT_WORKER_USER) -> Path:
    path = Path(str(remote_root))
    raw = path.as_posix()
    allowed_prefix = f"/home/{worker_user}/"
    forbidden = {
        "/",
        "/home",
        f"/home/{worker_user}",
        "/srv",
        "/srv/apps",
        "/srv/apps/cortex",
        "/tmp",
    }
    if not raw.startswith(allowed_prefix) or raw in forbidden or ".." in path.parts:
        raise UnsafeRemoteRoot(
            f"remote root must be a child of {allowed_prefix} and not a protected path: {raw}"
        )
    return path


def ssh_command(login: str, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        login,
        remote_command,
    ]


def worker_shell_command(target: RemoteTarget, inner_command: str) -> str:
    worker = shlex.quote(target.user)
    if target.ssh_user == target.user:
        return inner_command
    home = shlex.quote(f"/home/{target.user}")
    return f"sudo -u {worker} -H env HOME={home} bash -lc {shlex.quote(inner_command)}"


def check_command(host: str = DEFAULT_HOST, admin_user: str = DEFAULT_ADMIN_USER) -> list[str]:
    remote = (
        "set -eu; "
        "echo HOST=$(hostname); "
        "echo USER=$(whoami); "
        "id; "
        "echo ---users; "
        "getent passwd | awk -F: '$3>=1000 && $3<60000 {print $1\":\"$3\":\"$6\":\"$7}'; "
        "echo ---services; "
        "systemctl list-units --type=service --state=running --no-pager --plain "
        "| awk '{print $1}' | grep -E 'cortex|nginx|postgres|tailscale|ssh' || true; "
        "echo ---resources; free -h; nproc; df -h / /home /srv 2>/dev/null || df -h /; "
        "echo ---tools; "
        "command -v python3 || true; python3 --version || true; "
        "command -v git || true; command -v rsync || true; command -v tmux || true; "
        "echo ---sudo; sudo -n true && echo sudo_nopasswd=yes || echo sudo_nopasswd=no"
    )
    return ssh_command(f"{admin_user}@{host}", remote)


def admin_commands(worker_user: str = DEFAULT_WORKER_USER, source_user: str = DEFAULT_ADMIN_USER) -> str:
    quoted_worker = shlex.quote(worker_user)
    quoted_source = shlex.quote(source_user)
    home = f"/home/{worker_user}"
    return "\n".join([
        f"sudo useradd --create-home --shell /bin/bash {quoted_worker}",
        f"sudo install -d -m 700 -o {quoted_worker} -g {quoted_worker} {home}/.ssh",
        f"sudo install -m 600 -o {quoted_worker} -g {quoted_worker} /home/{quoted_source}/.ssh/authorized_keys {home}/.ssh/authorized_keys",
        f"sudo install -d -m 755 -o {quoted_worker} -g {quoted_worker} {home}/iMa",
        f"getent passwd {quoted_worker}",
        f"sudo -u {quoted_worker} -H bash -lc 'whoami; pwd; ls -ld {home} {home}/iMa'",
    ])


def rsync_push_command(target: RemoteTarget, local_root: Path = Path(".")) -> list[str]:
    remote_root = require_safe_remote_root(target.remote_root, target.user)
    return [
        "rsync",
        "-az",
        "--delete",
        "--exclude",
        ".git/",
        "--exclude",
        ".venv/",
        "--exclude",
        ".venv*/",
        "--exclude",
        ".mega/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        "*.pyc",
        "--exclude",
        "artifacts/",
        "--exclude",
        ".env",
        "--exclude",
        ".env.local",
        f"{local_root.as_posix().rstrip('/')}/",
        f"{target.login}:{remote_root.as_posix()}/",
    ]


def remote_chown_command(target: RemoteTarget) -> list[str]:
    remote_root = require_safe_remote_root(target.remote_root, target.user)
    root = shlex.quote(remote_root.as_posix())
    worker = shlex.quote(target.user)
    remote = f"set -eu; chown -R {worker}:{worker} {root}"
    return ssh_command(target.login, remote)


def remote_install_command(target: RemoteTarget, python_bin: str = "python3.12") -> list[str]:
    return remote_install_command_with_options(target, python_bin=python_bin)


def remote_install_command_with_options(
    target: RemoteTarget,
    python_bin: str = "python3.12",
    venv_dir: str = DEFAULT_VENV_DIR,
    pip_index_url: str | None = None,
    pip_trusted_host: str | None = None,
    upgrade_pip: bool = True,
    ignore_requires_python: bool = False,
    no_deps: bool = False,
) -> list[str]:
    remote_root = require_safe_remote_root(target.remote_root, target.user)
    root = shlex.quote(remote_root.as_posix())
    py = shlex.quote(python_bin)
    venv = shlex.quote(venv_dir)
    env_parts = []
    if pip_index_url:
        env_parts.append(f"PIP_INDEX_URL={shlex.quote(pip_index_url)}")
    if pip_trusted_host:
        env_parts.append(f"PIP_TRUSTED_HOST={shlex.quote(pip_trusted_host)}")
    pip_env = (" ".join(env_parts) + " ") if env_parts else ""
    install_flags = []
    if ignore_requires_python:
        install_flags.append("--ignore-requires-python")
    if no_deps:
        install_flags.append("--no-deps")
    rendered_install_flags = " ".join(install_flags)
    if rendered_install_flags:
        rendered_install_flags = f"{rendered_install_flags} "
    upgrade = f"{pip_env}{venv}/bin/python -m pip install --upgrade pip; " if upgrade_pip else ""
    inner = (
        f"set -eu; cd {root}; "
        f"{py} -m venv {venv}; "
        f"{upgrade}"
        f"{pip_env}{venv}/bin/python -m pip install {rendered_install_flags}-e .; "
        f"{venv}/bin/python --version; "
        f"{venv}/bin/ima-optimize --help >/dev/null; "
        "echo install_ok"
    )
    remote = worker_shell_command(target, inner)
    return ssh_command(target.login, remote)


def remote_optimizer_command(
    target: RemoteTarget,
    campaign: str,
    max_trials: str = "1",
    proposal_batch_size: int = 1,
    max_concurrent_trials: str = "1",
    timeout_minutes: int | None = None,
    spec_profile: str = "default",
    venv_dir: str = DEFAULT_VENV_DIR,
    dry_run: bool = False,
) -> list[str]:
    remote_root = require_safe_remote_root(target.remote_root, target.user)
    root = shlex.quote(remote_root.as_posix())
    campaign_arg = shlex.quote(campaign)
    venv = shlex.quote(venv_dir)
    args = [
        f"{venv}/bin/ima-optimize",
        "run",
        "--campaign",
        campaign_arg,
        "--max-trials",
        shlex.quote(str(max_trials)),
        "--proposal-batch-size",
        str(proposal_batch_size),
        "--max-concurrent-trials",
        shlex.quote(str(max_concurrent_trials)),
        "--spec-profile",
        shlex.quote(spec_profile),
        "--policy",
        "local",
    ]
    if timeout_minutes is not None:
        args.extend(["--timeout-minutes", str(timeout_minutes)])
    if dry_run:
        args.append("--dry-run")
    inner = f"set -eu; cd {root}; {' '.join(args)}"
    remote = worker_shell_command(target, inner)
    return ssh_command(target.login, remote)


def rsync_pull_command(
    target: RemoteTarget,
    campaign_name: str,
    local_pull_root: Path = DEFAULT_LOCAL_PULL_ROOT,
) -> list[str]:
    remote_root = require_safe_remote_root(target.remote_root, target.user)
    safe_campaign = Path(campaign_name)
    if safe_campaign.is_absolute() or ".." in safe_campaign.parts or len(safe_campaign.parts) != 1:
        raise ValueError(f"campaign name must be one path segment: {campaign_name}")
    remote_path = remote_root / "artifacts" / "agentic-learning" / campaign_name
    local_path = local_pull_root / campaign_name
    return [
        "rsync",
        "-az",
        "--delete",
        f"{target.login}:{remote_path.as_posix()}/",
        f"{local_path.as_posix().rstrip('/')}/",
    ]


def run(command: list[str], dry_run: bool = False) -> int:
    print(shlex.join(command))
    if dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Operate the iMa optimizer worker on cortex-server")
    root.add_argument("--host", default=DEFAULT_HOST)
    sub = root.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Read-only shared-host inventory")
    check.add_argument("--admin-user", default=DEFAULT_ADMIN_USER)

    admin = sub.add_parser("print-admin-commands", help="Print scoped sudo commands for imaopt setup")
    admin.add_argument("--worker-user", default=DEFAULT_WORKER_USER)
    admin.add_argument("--source-user", default=DEFAULT_ADMIN_USER)

    for name in ("sync", "install", "remote-dry-run", "remote-run", "pull"):
        item = sub.add_parser(name)
        item.add_argument("--worker-user", default=DEFAULT_WORKER_USER)
        item.add_argument("--ssh-user", default=DEFAULT_SSH_USER)
        item.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
        item.add_argument("--dry-run-command", action="store_true")
        if name in {"remote-dry-run", "remote-run"}:
            item.add_argument("--campaign", default="artifacts/agentic-learning/cortex-smoke")
            item.add_argument("--max-trials", default="1")
            item.add_argument("--proposal-batch-size", type=int, default=1)
            item.add_argument("--max-concurrent-trials", default="1")
            item.add_argument("--timeout-minutes", type=int)
            item.add_argument("--spec-profile", choices=("default", "long", "adaptive"), default="default")
        if name == "pull":
            item.add_argument("--campaign-name", default="cortex-smoke")
            item.add_argument("--local-pull-root", type=Path, default=DEFAULT_LOCAL_PULL_ROOT)
        if name == "install":
            item.add_argument("--python-bin", default="python3.12")
            item.add_argument("--venv-dir", default=DEFAULT_VENV_DIR)
            item.add_argument("--pip-index-url")
            item.add_argument("--pip-trusted-host")
            item.add_argument("--no-pip-upgrade", action="store_true")
            item.add_argument("--ignore-requires-python", action="store_true")
            item.add_argument("--no-deps", action="store_true")
        if name in {"remote-dry-run", "remote-run"}:
            item.add_argument("--venv-dir", default=DEFAULT_VENV_DIR)

    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "check":
        return run(check_command(args.host, args.admin_user))
    if args.command == "print-admin-commands":
        print(admin_commands(args.worker_user, args.source_user))
        return 0

    target = RemoteTarget(args.host, args.worker_user, args.remote_root, args.ssh_user)
    if args.command == "sync":
        code = run(rsync_push_command(target), args.dry_run_command)
        if code != 0:
            return code
        return run(remote_chown_command(target), args.dry_run_command)
    if args.command == "install":
        return run(
            remote_install_command_with_options(
                target,
                python_bin=args.python_bin,
                venv_dir=args.venv_dir,
                pip_index_url=args.pip_index_url,
                pip_trusted_host=args.pip_trusted_host,
                upgrade_pip=not args.no_pip_upgrade,
                ignore_requires_python=args.ignore_requires_python,
                no_deps=args.no_deps,
            ),
            args.dry_run_command,
        )
    if args.command == "remote-dry-run":
        return run(
            remote_optimizer_command(
                target,
                args.campaign,
                args.max_trials,
                args.proposal_batch_size,
                args.max_concurrent_trials,
                args.timeout_minutes,
                args.spec_profile,
                args.venv_dir,
                dry_run=True,
            ),
            args.dry_run_command,
        )
    if args.command == "remote-run":
        return run(
            remote_optimizer_command(
                target,
                args.campaign,
                args.max_trials,
                args.proposal_batch_size,
                args.max_concurrent_trials,
                args.timeout_minutes,
                args.spec_profile,
                args.venv_dir,
            ),
            args.dry_run_command,
        )
    if args.command == "pull":
        return run(rsync_pull_command(target, args.campaign_name, args.local_pull_root), args.dry_run_command)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
