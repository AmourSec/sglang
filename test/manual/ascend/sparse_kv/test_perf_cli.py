from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ASCEND_TESTS = Path(__file__).resolve().parents[1]


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONPATH": str(ASCEND_TESTS)}
    # No site-packages: --help and --dry-run must not import torch/NPU/MemFabric.
    return subprocess.run(
        [sys.executable, "-S", "-m", "sparse_kv.perf", *arguments],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_full_dry_run_has_expected_matrix_and_memory(tmp_path: Path) -> None:
    output = tmp_path / "no-output"
    result = run_cli("--dry-run", "--output-dir", str(output))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["workloads"] == 560
    assert summary["result_rows"] == 1680
    assert summary["mtps"] == [0, 1, 2, 3]
    assert summary["batch_sizes"][-1] == 32
    assert 140000 in summary["seq_lens"]
    assert summary["host_payload_bytes"] == 32 * 143488 * 1152
    assert summary["resident_payload_bytes_max"] == 128 * 4096 * 1152
    assert not output.exists()


def test_smoke_and_custom_matrix() -> None:
    smoke = run_cli("--preset", "smoke", "--dry-run")
    assert smoke.returncode == 0, smoke.stderr
    assert json.loads(smoke.stdout)["workloads"] == 16
    custom = run_cli(
        "--batch-sizes",
        "32",
        "--mtps",
        "1,2",
        "--seq-lens",
        "4096,140000",
        "--hit-ratios",
        "0.333",
        "--capacities",
        "2048,8192",
        "--ops",
        "transfer",
        "--dry-run",
    )
    assert custom.returncode == 0, custom.stderr
    assert json.loads(custom.stdout)["result_rows"] == 8


@pytest.mark.parametrize(
    "arguments",
    [
        ["--mtps", "4"],
        ["--batch-sizes", "33"],
        ["--hit-ratios", "nan"],
        ["--capacities", "1024"],
        ["--ops", "attention"],
        ["--iterations", "0"],
        ["--warmup", "-1"],
        ["--seq-lens", "1000"],
        ["--mtps", "0,0"],
    ],
)
def test_invalid_matrix_fails_before_device_import(arguments: list[str]) -> None:
    result = run_cli(*arguments, "--dry-run")
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_help_without_device_dependencies() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--hit-ratios" in result.stdout
