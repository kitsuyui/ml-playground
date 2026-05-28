import os
import subprocess
from pathlib import Path


def test_run_each_passes_subcommand_as_literal_argument(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workspace = tmp_path / "workspace"
    package_dir = workspace / "packages" / "sample"
    dev_shared_dir = workspace / "packages" / "dev-shared"
    fake_bin_dir = tmp_path / "bin"
    fake_uv = fake_bin_dir / "uv"
    argv_log = tmp_path / "uv-argv.log"

    package_dir.mkdir(parents=True)
    dev_shared_dir.mkdir(parents=True)
    fake_bin_dir.mkdir()
    fake_uv.write_text(
        "#!/usr/bin/env sh\n"
        "set -eu\n"
        "printf '%s\\n' \"$@\" >> \"$UV_ARGV_LOG\"\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin_dir}{os.pathsep}{env['PATH']}"
    env["UV_ARGV_LOG"] = str(argv_log)

    subprocess.run(  # noqa: S603
        [str(repo_root / "bin" / "run-each"), "safe; touch pwned"],
        cwd=workspace,
        env=env,
        check=True,
    )

    assert not (workspace / "pwned").exists()
    assert argv_log.read_text(encoding="utf-8").splitlines() == [
        "run",
        "poe",
        "safe; touch pwned",
    ]
