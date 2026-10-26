from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]
RUNNER = ROOT / "scripts/ascend/bench_sparse_kv_950.sh"


def _fake_python(tmp_path: Path, status: int) -> Path:
    script = tmp_path / f"python-{status}.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "${FAKE_ARGS_FILE}"\n'
        f"exit {status}\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _isolated_runner(tmp_path: Path, status: int) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "checkout"
    runner = repo / "scripts/ascend/bench_sparse_kv_950.sh"
    runner.parent.mkdir(parents=True)
    shutil.copy2(RUNNER, runner)
    opp = (
        repo
        / "sgl-kernel-npu-sparse-kv/src/sgl_kernel_npu_sparse_kv/_cann_ops_custom/vendors/sgl_sparse_kv_transformer/bin/set_env.bash"
    )
    opp.parent.mkdir(parents=True)
    opp.write_text(
        'export ASCEND_CUSTOM_OPP_PATH="custom:${ASCEND_CUSTOM_OPP_PATH}"\nexport LD_LIBRARY_PATH="lib:${LD_LIBRARY_PATH}"\n'
    )
    env = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path, status)),
        "FAKE_ARGS_FILE": str(tmp_path / "args"),
        "SGL_SPARSE_KV_BENCH_DIR": str(tmp_path / "run"),
    }
    env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    env.pop("LD_LIBRARY_PATH", None)
    return runner, env


@pytest.mark.parametrize("status", [0, 42, 134])
@pytest.mark.parametrize("custom_output", [False, True])
def test_runner_preserves_process_exit_after_teardown(
    tmp_path: Path,
    status: int,
    custom_output: bool,
) -> None:
    runner, env = _isolated_runner(tmp_path, status)
    args = ["--output-dir", str(tmp_path / "custom results")] if custom_output else []
    result = subprocess.run(
        [str(runner), *args], env=env, text=True, capture_output=True
    )
    assert result.returncode == status, result.stderr
    assert (tmp_path / "run/process_exit_code").read_text().strip() == str(status)
    expected = tmp_path / ("custom results" if custom_output else "run/results")
    assert f"results={expected}" in result.stdout
    assert str(expected) in (tmp_path / "args").read_text().splitlines()


def test_dry_run_does_not_require_opp_or_create_artifacts(tmp_path: Path) -> None:
    args_file = tmp_path / "args"
    python = _fake_python(tmp_path, 0)
    env = {**os.environ, "PYTHON_BIN": str(python), "FAKE_ARGS_FILE": str(args_file)}
    result = subprocess.run(
        [str(RUNNER), "--dry-run", "--preset", "smoke"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines()[:4] == [
        "-m",
        "sparse_kv.perf",
        "--dry-run",
        "--preset",
    ]
    assert not list(tmp_path.glob("sglang-sparse-kv-bench-*"))


def test_help_is_available_without_npu_environment(tmp_path: Path) -> None:
    args_file = tmp_path / "args"
    python = _fake_python(tmp_path, 0)
    env = {**os.environ, "PYTHON_BIN": str(python), "FAKE_ARGS_FILE": str(args_file)}
    result = subprocess.run(
        [str(RUNNER), "--help"], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "sparse_kv.perf",
        "--help",
    ]


def test_bash_syntax_is_valid() -> None:
    result = subprocess.run(["bash", "-n", str(RUNNER)], cwd=ROOT, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_run_directory_is_not_modified(tmp_path: Path) -> None:
    runner, env = _isolated_runner(tmp_path, 0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    marker = run_dir / "marker"
    marker.write_text("keep", encoding="utf-8")
    result = subprocess.run([str(runner), "--preset", "smoke"], env=env, text=True)
    assert result.returncode != 0
    assert marker.read_text(encoding="utf-8") == "keep"


def test_runner_records_tee_failure_separately(tmp_path: Path) -> None:
    runner, env = _isolated_runner(tmp_path, 0)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tee = fake_bin / "tee"
    fake_tee.write_text(
        "#!/usr/bin/env bash\ncat >/dev/null\nexit 23\n", encoding="utf-8"
    )
    fake_tee.chmod(0o755)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run([str(runner)], env=env, text=True, capture_output=True)
    assert result.returncode == 23
    run_dir = tmp_path / "run"
    assert (run_dir / "process_exit_code").read_text().strip() == "0"
    assert (run_dir / "tee_exit_code").read_text().strip() == "23"
    assert (run_dir / "runner_exit_code").read_text().strip() == "23"
