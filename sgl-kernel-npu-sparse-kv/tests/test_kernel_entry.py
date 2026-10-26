from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).parents[1]
CONFIG_HEADER = (
    PACKAGE_ROOT / "cann_ops/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_config.h"
)


@pytest.mark.parametrize(
    "tiling_key_macro",
    [None, "sparse_kv_plan=sparse_kv_plan_0_tilingkey"],
    ids=["normal", "cann_tiling_key"],
)
def test_plan_config_namespace_does_not_collide_with_tiling_key_macro(
    tiling_key_macro: str | None,
) -> None:
    """Compile the real Plan header beside the CANN-generated entry symbol."""
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required for the entry-name regression")

    source = """
#include "sparse_kv_plan_config.h"
extern "C" void sparse_kv_plan();
int main() { return static_cast<int>(sparse_kv_plan_config::PLAN_THREADS); }
"""
    with tempfile.TemporaryDirectory(prefix="sgl-sparse-kv-entry-") as directory:
        source_path = Path(directory) / "entry.cpp"
        source_path.write_text(source)
        command = [compiler, "-std=c++17", "-fsyntax-only", f"-I{CONFIG_HEADER.parent}"]
        if tiling_key_macro is not None:
            command.append(f"-D{tiling_key_macro}")
        command.append(str(source_path))
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_plan_kernel_keeps_exported_entry_name_and_uses_private_config_namespace() -> (
    None
):
    kernel = (
        PACKAGE_ROOT
        / "cann_ops/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_apt.cpp"
    ).read_text()
    tiling = (
        PACKAGE_ROOT
        / "cann_ops/attention/sparse_kv_plan/op_host/sparse_kv_plan_tiling.cpp"
    ).read_text()
    assert "void sparse_kv_plan(" in kernel
    assert "namespace sparse_kv_plan_config" in CONFIG_HEADER.read_text()
    assert "using sparse_kv_plan::" not in kernel
    assert "sparse_kv_plan::" not in tiling
