from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _generated_environment(tmp_path: Path, **environment: str) -> dict[str, str]:
    root = Path(__file__).parents[1]
    installer = root / "cann_ops/cmake/scripts/custom/install.sh"
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    output = bin_path / "set_env.bash"
    script = f"""
set -euo pipefail
_ASCEND_CUSTOM_OPP_PATH=/opt/sglang/vendors/custom
bin_path='{bin_path}'
set_env_variable=$(sed -n '/^[[:space:]]*set_env_variable=/p' '{installer}' | head -n 1)
eval "$set_env_variable"
emitter=$(sed -n '/^[[:space:]]*printf .*set_env_variable.*bin_path/p' '{installer}' | head -n 1)
eval "$emitter"
"""
    env = os.environ.copy()
    env.update(environment)
    subprocess.run(["bash", "-euo", "pipefail", "-c", script], check=True, env=env)
    source_env = env.copy()
    if not environment.get("ASCEND_CUSTOM_OPP_PATH"):
        source_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    if not environment.get("LD_LIBRARY_PATH"):
        source_env.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", f"source '{output}'; env"],
        check=True,
        capture_output=True,
        text=True,
        env=source_env,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def test_generated_environment_is_safe_with_unset_paths(tmp_path: Path) -> None:
    values = _generated_environment(
        tmp_path,
        ASCEND_CUSTOM_OPP_PATH="",
        LD_LIBRARY_PATH="",
    )

    assert values["ASCEND_CUSTOM_OPP_PATH"] == "/opt/sglang/vendors/custom"
    assert values["LD_LIBRARY_PATH"] == "/opt/sglang/vendors/custom/op_api/lib"


def test_generated_environment_preserves_existing_paths(tmp_path: Path) -> None:
    values = _generated_environment(
        tmp_path,
        ASCEND_CUSTOM_OPP_PATH="/opt/ascend/opp:/opt/ascend/extra path",
        LD_LIBRARY_PATH="/opt/ascend/lib:/opt/ascend/extra lib",
    )

    assert (
        values["ASCEND_CUSTOM_OPP_PATH"]
        == "/opt/sglang/vendors/custom:/opt/ascend/opp:/opt/ascend/extra path"
    )
    assert (
        values["LD_LIBRARY_PATH"]
        == "/opt/sglang/vendors/custom/op_api/lib:/opt/ascend/lib:/opt/ascend/extra lib"
    )
