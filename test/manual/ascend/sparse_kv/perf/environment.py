from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .results import JsonValue

REPO_ROOT = Path(__file__).resolve().parents[5]
SOURCE_SCOPE = (
    "test/manual/ascend/sparse_kv/perf",
    "test/manual/ascend/sparse_kv/npu_harness.py",
    "test/manual/ascend/sparse_kv/host_memory.py",
    "test/manual/ascend/sparse_kv/reference.py",
    "scripts/ascend/bench_sparse_kv_950.sh",
    "sgl-kernel-npu-sparse-kv",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(
    repo_root: Path = REPO_ROOT,
    package_dir: Path | None = None,
) -> dict[str, JsonValue]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout

    result: dict[str, JsonValue] = {
        "git_commit": None,
        "scoped_dirty": None,
        "source_scope": list(SOURCE_SCOPE),
    }
    try:
        result["git_commit"] = git("rev-parse", "HEAD").strip()
        status = git(
            "status", "--porcelain=v1", "--untracked-files=all", "--", *SOURCE_SCOPE
        )
        result["scoped_dirty"] = bool(status.strip())
        result["scoped_git_status"] = status
        paths = git(
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *SOURCE_SCOPE,
        )
        result["source_sha256"] = {
            name: _sha256(repo_root / name)
            for name in sorted(set(paths.split("\0")))
            if name and (repo_root / name).is_file()
        }
    except (OSError, subprocess.CalledProcessError) as error:
        result["git_error"] = str(error)
    package_dir = (
        package_dir
        or repo_root / "sgl-kernel-npu-sparse-kv/src/sgl_kernel_npu_sparse_kv"
    )
    result["bundled_binary_sha256"] = {
        str(path.relative_to(package_dir)): _sha256(path)
        for path in sorted(package_dir.rglob("*.so*"))
        if path.is_file()
    }
    return result


def environment_metadata() -> dict[str, JsonValue]:
    versions: dict[str, JsonValue] = {}
    for package in (
        "torch",
        "torch-npu",
        "memfabric-hybrid",
        "sgl-kernel-npu-sparse-kv",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed-as-distribution"
    toolkit = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    cann_versions = {
        str(path): path.read_text(errors="replace")
        for path in (toolkit / "version.cfg", toolkit / "version.info")
        if path.is_file()
    }
    meminfo = Path("/proc/meminfo")
    hugepages = (
        [line for line in meminfo.read_text().splitlines() if "Huge" in line]
        if meminfo.exists()
        else []
    )
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "provenance": source_provenance(),
        "versions": versions,
        "cann_version_files": cann_versions,
        "hugepages": hugepages,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
                "LD_LIBRARY_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "NPU_VISIBLE_DEVICES",
                "ACL_LAUNCH_BLOCKING",
                "TASK_QUEUE_ENABLE",
                "OMP_NUM_THREADS",
            )
        },
    }
