from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_checker():
    script = Path(__file__).parents[1] / "scripts" / "check_build_env.py"
    spec = importlib.util.spec_from_file_location("check_build_env", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cann_version_search_is_not_limited_to_first_four_files(tmp_path: Path) -> None:
    checker = _load_checker()
    for index in range(5):
        component = tmp_path / f"component-{index}"
        component.mkdir()
        component.joinpath("version.info").write_text("Version=8.5.0\n")
    expected = tmp_path / "toolkit" / "version.info"
    expected.parent.mkdir()
    expected.write_text("Version=9.1.0\n")

    assert checker._find_cann_version(tmp_path) == expected


def test_cann_version_requires_9_1_0(tmp_path: Path) -> None:
    tmp_path.joinpath("version.info").write_text("Version=9.1.1\n")

    assert _load_checker()._find_cann_version(tmp_path) is None
