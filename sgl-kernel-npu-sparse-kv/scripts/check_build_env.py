#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# ///
# ─── How to run ───
# python3 scripts/check_build_env.py

from __future__ import annotations

import os
import re
from importlib import import_module
from pathlib import Path

EXPECTED_VERSION = (2, 10)
_CANN_VERSION = re.compile(r"(?<![\d.])9\.1\.0(?![\d.])")
_MAX_VERSION_FILE_BYTES = 64 * 1024


def _major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        message = f"cannot parse package version: {version}"
        raise RuntimeError(message)
    return (int(match.group(1)), int(match.group(2)))


def _find_cann_version(ascend_home: Path) -> Path | None:
    preferred = (
        ascend_home / "version.info",
        ascend_home / "version.cfg",
        ascend_home / "install.info",
    )
    discovered = sorted(ascend_home.rglob("version.info"))
    candidates = dict.fromkeys((*preferred, *discovered))
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(errors="replace") as stream:
            version_text = stream.read(_MAX_VERSION_FILE_BYTES)
        if _CANN_VERSION.search(version_text) is not None:
            return path
    return None


def main() -> None:
    torch = import_module("torch")
    torch_npu = import_module("torch_npu")
    torch_version = str(torch.__version__)
    torch_npu_version = str(torch_npu.__version__)
    if _major_minor(torch_version) != EXPECTED_VERSION:
        message = f"torch 2.10.x is required, got {torch_version}"
        raise RuntimeError(message)
    if _major_minor(torch_npu_version) != EXPECTED_VERSION:
        message = f"torch_npu 2.10.x is required, got {torch_npu_version}"
        raise RuntimeError(message)

    ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
    cann_version_file = _find_cann_version(ascend_home)
    if cann_version_file is None:
        message = f"CANN 9.1.0 is required; no matching version.info found below {ascend_home}"
        raise RuntimeError(message)
    print(
        f"torch={torch_version} torch_npu={torch_npu_version} "
        f"CANN=9.1.0 ({cann_version_file})"
    )


if __name__ == "__main__":
    main()
