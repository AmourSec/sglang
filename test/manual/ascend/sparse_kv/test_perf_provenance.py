from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> None:
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Benchmark Test",
            "-c",
            "user.email=benchmark@example.invalid",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_provenance_distinguishes_head_worktree_and_binaries(tmp_path: Path) -> None:
    from .perf.environment import source_provenance

    repo = tmp_path / "repo"
    source = repo / "test/manual/ascend/sparse_kv/perf/cases.py"
    source.parent.mkdir(parents=True)
    source.write_text("original source\n")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    package = repo / "installed"
    package.mkdir()
    binary = package / "_C.so"
    binary.write_bytes(b"original binary")
    original = source_provenance(repo, package)
    assert original["scoped_dirty"] is False
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert original["source_sha256"][str(source.relative_to(repo))] == original_hash
    (repo / "unrelated.txt").write_text("unrelated change")
    assert source_provenance(repo, package)["scoped_dirty"] is False
    source.write_text("modified source\n")
    added = source.parent / "new.py"
    added.write_text("new benchmark helper\n")
    binary.write_bytes(b"rebuilt binary")
    changed = source_provenance(repo, package)
    assert changed["git_commit"] == original["git_commit"]
    assert changed["scoped_dirty"] is True
    assert changed["source_sha256"][str(source.relative_to(repo))] != original_hash
    assert str(added.relative_to(repo)) in changed["source_sha256"]
    assert changed["bundled_binary_sha256"] != original["bundled_binary_sha256"]
    assert "new.py" in changed["scoped_git_status"]


def test_missing_git_metadata_is_explicit(tmp_path: Path) -> None:
    from .perf.environment import source_provenance

    result = source_provenance(tmp_path)
    assert result["git_commit"] is None
    assert result["scoped_dirty"] is None
    assert result["git_error"]
