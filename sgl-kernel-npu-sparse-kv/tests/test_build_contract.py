from __future__ import annotations

import subprocess
from pathlib import Path


def test_build_selects_both_cmake_operator_names() -> None:
    build_script = Path(__file__).parents[1].joinpath("build.sh").read_text()

    assert '--ops="sparse_kv_plan;sparse_kv_transfer"' in build_script


def test_build_help_is_platform_independent() -> None:
    root = Path(__file__).parents[1]

    result = subprocess.run(
        ["bash", str(root / "build.sh"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "SOC_VERSION" in result.stdout
    assert "ASCEND_HOME_PATH" in result.stdout


def test_build_rejects_noncanonical_950_target() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        ["bash", str(root / "build.sh")],
        check=False,
        capture_output=True,
        env={"SOC_VERSION": "ascend950pr_9599"},
        text=True,
    )

    assert result.returncode == 2
    assert "SOC_VERSION must be ascend950" in result.stderr


def test_vendored_source_licenses_are_distributed() -> None:
    root = Path(__file__).parents[1]
    notice = root.joinpath("NOTICE").read_text()
    cann_license = root.joinpath(
        "THIRD_PARTY_LICENSES/CANN-Open-Software-License-Agreement-v2.0.txt"
    ).read_text()
    mulan_license = root.joinpath("THIRD_PARTY_LICENSES/MulanPSL-2.0.txt").read_text()

    assert "CANN Open Software License Agreement Version 2.0" in cann_license
    assert "木兰宽松许可证" in mulan_license
    assert (
        "cann_ops/attention/sparse_kv_plan/op_kernel/sparse_kv_plan_apt.cpp" in notice
    )
    assert (
        "cann_ops/attention/sparse_kv_transfer/op_kernel/sparse_kv_transfer_apt.cpp"
        in notice
    )
    assert (
        "THIRD_PARTY_LICENSES/CANN-Open-Software-License-Agreement-v2.0.txt" in notice
    )
    assert "THIRD_PARTY_LICENSES/Protobuf-BSD-3-Clause.txt" in notice
    assert root.joinpath("THIRD_PARTY_LICENSES/Protobuf-BSD-3-Clause.txt").is_file()


def test_downloaded_build_dependencies_are_hash_pinned() -> None:
    root = Path(__file__).parents[1]
    cmake_files = {
        root
        / "cann_ops/cmake/third_party/json.cmake": "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
        root
        / "cann_ops/cmake/third_party/ascend_protobuf.cmake": "9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42",
        root
        / "cann_ops/cmake/third_party/abseil-cpp.cmake": "987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed",
        root
        / "cann_ops/cmake/third_party/makeself-fetch.cmake": "bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a",
    }

    for cmake_file, expected_hash in cmake_files.items():
        content = cmake_file.read_text()
        assert "URL_HASH" in content
        assert expected_hash in content
        assert "TLS_VERIFY OFF" not in content
        assert "CMAKE_TLS_VERIFY=0" not in content


def test_binding_guards_device_and_int32_block_size() -> None:
    root = Path(__file__).parents[1]
    binding = root.joinpath("csrc/binding.cpp").read_text()
    plan_tiling = root.joinpath(
        "cann_ops/attention/sparse_kv_plan/op_host/sparse_kv_plan_tiling.cpp"
    ).read_text()
    transfer_tiling = root.joinpath(
        "cann_ops/attention/sparse_kv_transfer/op_host/sparse_kv_transfer_tiling.cpp"
    ).read_text()

    assert "c10::DeviceGuard" in binding
    assert "must be on the same NPU device" in binding
    assert "block_size exceeds the int32 kernel limit" in binding
    assert "host cache byte range overflows uint64" in binding
    assert "blockSize > std::numeric_limits<int32_t>::max()" in plan_tiling
    assert "blockSize > std::numeric_limits<int32_t>::max()" in transfer_tiling
    assert "host cache byte range overflows uint64" in transfer_tiling


def test_plan_transfer_validates_both_phases_before_launch() -> None:
    binding = Path(__file__).parents[1].joinpath("csrc/binding.cpp").read_text()
    start = binding.index("void PlanTransfer")
    end = binding.index("void MetaPlan", start)
    body = binding[start:end]

    assert body.index("ValidatePlan(") < body.index("ValidateTransfer(")
    assert body.index("ValidateTransfer(") < body.index("\n  Plan(")
