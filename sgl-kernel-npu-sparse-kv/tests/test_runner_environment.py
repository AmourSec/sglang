from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("existing", [False, True])
def test_runner_can_source_a_legacy_opp_environment(
    tmp_path: Path, existing: bool
) -> None:
    runner = (
        Path(__file__).parents[2] / "scripts/ascend/run_sparse_kv_950.sh"
    ).read_text()
    branch = runner.split('if [[ -f "${OPP_ENV}" ]]; then\n', 1)[1].split("\nfi", 1)[0]
    function = ""
    if "source_opp_environment() {" in runner:
        function = (
            "source_opp_environment() {"
            + runner.split("source_opp_environment() {", 1)[1].split("\n}", 1)[0]
            + "\n}\n"
        )
    legacy = tmp_path / "set_env.bash"
    legacy.write_text(
        'export ASCEND_CUSTOM_OPP_PATH="/custom:${ASCEND_CUSTOM_OPP_PATH}"\n'
        'export LD_LIBRARY_PATH="/custom/lib:${LD_LIBRARY_PATH}"\n'
    )
    environment = os.environ.copy()
    for variable in ("ASCEND_CUSTOM_OPP_PATH", "LD_LIBRARY_PATH"):
        environment.pop(variable, None)
        if existing:
            environment[variable] = "/existing path:/second"
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            function
            + "OPP_ENV=$1\n"
            + branch
            + '\nprintf "%s\\n" "$ASCEND_CUSTOM_OPP_PATH" "$LD_LIBRARY_PATH"\n',
            "runner-env-test",
            str(legacy),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    suffix = "/existing path:/second" if existing else ""
    assert result.stdout.splitlines() == [f"/custom:{suffix}", f"/custom/lib:{suffix}"]
