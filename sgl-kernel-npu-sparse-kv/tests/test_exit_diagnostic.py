from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path


def _run_diagnostic(
    tmp_path: Path, *selectors: str, **settings: str
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    repo = Path(__file__).parents[2]
    script = repo / "scripts/ascend/diagnose_sparse_kv_950.sh"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        'arguments="$*"\n'
        'if [[ "$arguments" == *"npu_device_name"* && "${FAKE_BASELINE_FAIL:-0}" == 1 ]]; then exit 7; fi\n'
        'if [[ "$arguments" == *"import IPython"* && "${FAKE_RUNTIME_FAIL:-0}" == 1 ]]; then exit 1; fi\n'
        'if [[ "$arguments" == *"test_npu_plan.py::first[param]"* ]]; then exit 134; fi\n'
        "exit 0\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    opp_environment = tmp_path / "set_env.bash"
    opp_environment.write_text("export ASCEND_CUSTOM_OPP_PATH=/custom\n")
    evidence = tmp_path / "evidence"
    environment = os.environ | {
        "PYTHON_BIN": str(fake_python),
        "SGL_SPARSE_KV_OPP_ENV": str(opp_environment),
        "SGL_SPARSE_KV_DIAG_DIR": str(evidence),
        **settings,
    }
    result = subprocess.run(
        ["bash", str(script), *selectors],
        capture_output=True,
        text=True,
        env=environment,
    )
    return result, (evidence / "status.tsv").read_text().splitlines(), evidence


def test_exit_diagnostic_continues_after_same_module_selector_failure(
    tmp_path: Path,
) -> None:
    result, rows, evidence = _run_diagnostic(
        tmp_path,
        "test/manual/ascend/sparse_kv/test_npu_plan.py::first[param]",
        "test/manual/ascend/sparse_kv/test_npu_plan.py::later[param]",
    )

    assert result.returncode == 134
    assert rows[0] == "stage\texit_code\tlog"
    assert rows[1].startswith("baseline_950\t0\t")
    assert rows[-2].startswith("pytest_01_test_npu_plan\t134\t")
    assert rows[-1].startswith("pytest_02_test_npu_plan\t0\t")
    assert (evidence / "stages/pytest_01_test_npu_plan.log").is_file()
    environment_log = (evidence / "environment.log").read_text()
    assert "python_realpath=" in environment_log


def test_exit_diagnostic_preserves_dependency_failure_after_later_success(
    tmp_path: Path,
) -> None:
    result, rows, _ = _run_diagnostic(
        tmp_path,
        "test/manual/ascend/sparse_kv/test_npu_plan.py::later[param]",
        FAKE_RUNTIME_FAIL="1",
    )

    assert result.returncode == 1
    assert any(row.startswith("runtime_dependencies\t1\t") for row in rows)
    assert rows[-1].startswith("pytest_01_test_npu_plan\t0\t")


def test_exit_diagnostic_stops_after_baseline_failure(tmp_path: Path) -> None:
    result, rows, _ = _run_diagnostic(
        tmp_path,
        "test/manual/ascend/sparse_kv/test_npu_plan.py::later[param]",
        FAKE_BASELINE_FAIL="1",
    )

    assert result.returncode == 7
    assert len(rows) == 2
    assert rows[-1].startswith("baseline_950\t7\t")


def test_exit_diagnostic_returns_zero_when_all_stages_pass(tmp_path: Path) -> None:
    result, rows, _ = _run_diagnostic(
        tmp_path, "test/manual/ascend/sparse_kv/test_npu_plan.py::later[param]"
    )

    assert result.returncode == 0
    assert all("\t0\t" in row for row in rows[1:])


def test_exit_diagnostic_loads_host_planes_without_test_package_import(
    tmp_path: Path,
) -> None:
    stub_path = tmp_path / "stubs"
    shadow_test = tmp_path / "shadow" / "test"
    stub_path.mkdir()
    shadow_test.mkdir(parents=True)
    (shadow_test / "__init__.py").write_text("SHADOWED = True\n")
    (stub_path / "torch.py").write_text(
        "class _DType:\n"
        "    itemsize = 2\n"
        "\n"
        "bfloat16 = _DType()\n"
        "\n"
        "class _Npu:\n"
        "    @staticmethod\n"
        "    def current_device():\n"
        "        return 0\n"
        "\n"
        "npu = _Npu()\n"
    )
    (stub_path / "torch_npu.py").write_text("")
    (stub_path / "memfabric_hybrid.py").write_text(
        "class _Tensor:\n"
        "    device = type('Device', (), {'type': 'cpu'})()\n"
        "    def __getitem__(self, _):\n"
        "        return self\n"
        "    def data_ptr(self):\n"
        "        return 1\n"
        "\n"
        "class _Config:\n"
        "    pass\n"
        "\n"
        "class _Offload:\n"
        "    OffloadConfig = _Config\n"
        "    Scene = type('Scene', (), {'LOCAL': 1})\n"
        "    OFFLOAD_FLAG_GIANT_PAGE = 1\n"
        "    @staticmethod\n"
        "    def initialize(_):\n"
        "        return 0\n"
        "    @staticmethod\n"
        "    def empty(*_, **__):\n"
        "        return _Tensor()\n"
        "    @staticmethod\n"
        "    def get_dva(_):\n"
        "        return 1\n"
        "\n"
        "offload = _Offload()\n"
    )
    fake_python = tmp_path / "probe_python"
    fake_python.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "import types\n"
        "from pathlib import Path\n"
        "code = sys.argv[sys.argv.index('-c') + 1] if '-c' in sys.argv else ''\n"
        "if 'HostPlanes(128)' in code:\n"
        "    import test\n"
        "    assert Path(test.__file__).parent == Path(os.environ['SHADOW_TEST_PACKAGE'])\n"
        "    if 'ensure_loaded' in code:\n"
        "        extension = types.ModuleType('sgl_kernel_npu_sparse_kv')\n"
        "        extension.ensure_loaded = lambda: print('extension_stub=loaded')\n"
        "        sys.modules['sgl_kernel_npu_sparse_kv'] = extension\n"
        "    exec(code, {'__name__': '__main__'})\n"
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
    opp_environment = tmp_path / "set_env.bash"
    opp_environment.write_text("export ASCEND_CUSTOM_OPP_PATH=/custom\n")
    evidence = tmp_path / "evidence"
    environment = os.environ | {
        "PYTHON_BIN": str(fake_python),
        "PYTHONPATH": f"{stub_path}:{shadow_test.parent}",
        "SGL_SPARSE_KV_OPP_ENV": str(opp_environment),
        "SGL_SPARSE_KV_DIAG_DIR": str(evidence),
        "SHADOW_TEST_PACKAGE": str(shadow_test),
    }

    legacy = subprocess.run(
        [
            sys.executable,
            "-c",
            "from test.manual.ascend.sparse_kv.host_memory import HostPlanes; HostPlanes(128)",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert legacy.returncode != 0
    assert "test.manual" in legacy.stderr

    result = subprocess.run(
        [
            "bash",
            str(Path(__file__).parents[2] / "scripts/ascend/diagnose_sparse_kv_950.sh"),
            "test/manual/ascend/sparse_kv/test_npu_plan.py::later[param]",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    rows = (evidence / "status.tsv").read_text().splitlines()
    assert any(row.startswith("memfabric_host_planes\t0\t") for row in rows)
    for stage in (
        "memfabric_host_planes",
        "extension_and_memfabric",
        "memfabric_then_extension",
    ):
        log = (evidence / "stages" / f"{stage}.log").read_text()
        assert "host_memory_probe=loaded" in log
    for stage in ("extension_and_memfabric", "memfabric_then_extension"):
        log = (evidence / "stages" / f"{stage}.log").read_text()
        assert "extension_stub=loaded" in log
