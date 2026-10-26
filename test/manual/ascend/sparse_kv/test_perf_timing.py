from __future__ import annotations

import csv
import io
import json
import sys
import warnings
from pathlib import Path

import pytest

from .perf.results import ResultWriter
from .perf.timing import Samples, measure, summarize


class FakeEvent:
    def __init__(self, clock: list[float], log: list[str]) -> None:
        self.clock = clock
        self.log = log
        self.value = 0.0

    def record(self) -> None:
        self.log.append("record")
        self.value = self.clock[0]

    def synchronize(self) -> None:
        self.log.append("event_sync")
        self.clock[0] += 1.0

    def elapsed_time(self, end: FakeEvent) -> float:
        return end.value - self.value


def test_measure_restores_before_warmup_and_each_sample() -> None:
    clock = [0.0]
    restores: list[int] = []
    calls: list[int] = []
    log: list[str] = []

    def tick() -> None:
        clock[0] += 2.5

    def event_factory() -> FakeEvent:
        return FakeEvent(clock, log)

    def restore() -> None:
        log.append("restore")
        restores.append(len(calls))
        clock[0] += 5.0

    def synchronize() -> None:
        log.append("sync")
        clock[0] += 3.0

    def call() -> None:
        log.append("call")
        calls.append(1)
        tick()

    samples = measure(
        call,
        restore,
        synchronize=synchronize,
        event_factory=event_factory,
        warmup=2,
        iterations=3,
        clock_ns=lambda: int(clock[0] * 1_000_000),
    )
    assert len(calls) == 5
    assert restores == [0, 1, 2, 3, 4]
    assert samples.device_us == [2500.0] * 3
    assert samples.wall_us == [3500.0] * 3
    assert log == (
        ["record", "record", "event_sync"]
        + ["restore", "sync", "call", "sync"] * 2
        + ["restore", "sync", "record", "call", "record", "event_sync"] * 3
    )


@pytest.mark.parametrize("warmup,iterations", [(-1, 1), (0, 0), (0, -1)])
def test_invalid_iteration_counts(warmup: int, iterations: int) -> None:
    with pytest.raises(ValueError):
        measure(
            lambda: None,
            lambda: None,
            synchronize=lambda: None,
            event_factory=lambda: FakeEvent([0.0], []),
            warmup=warmup,
            iterations=iterations,
        )


def test_operation_failure_is_not_reported_as_a_sample() -> None:
    def fail() -> None:
        raise RuntimeError("device failed")

    with pytest.raises(RuntimeError, match="device failed"):
        measure(
            fail,
            lambda: None,
            synchronize=lambda: None,
            event_factory=lambda: FakeEvent([0.0], []),
            warmup=0,
            iterations=1,
        )


def test_summarize_percentiles_and_rejects_bad_values() -> None:
    stats = summarize([1.0, 2.0, 4.0, 8.0])
    assert stats["mean"] == pytest.approx(3.75)
    assert stats["p50"] == pytest.approx(3.0)
    assert stats["p90"] == pytest.approx(6.8)
    assert stats["p99"] == pytest.approx(7.88)
    assert stats["stddev"] == pytest.approx(2.6809513237)
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([float("nan")])
    with pytest.raises(ValueError):
        summarize([-1.0])
    with pytest.raises(ValueError):
        summarize([float("inf")])


def test_result_writer_incremental_roundtrip(tmp_path: Path) -> None:
    destination = tmp_path / "run"
    with ResultWriter(destination, {"device": "Ascend950"}) as writer:
        writer.write(
            {"batch": 2, "payload_bytes": 1152},
            "transfer",
            Samples([10.0, 20.0], [12.0, 24.0]),
        )
        writer.write({"batch": 2, "payload_bytes": 0}, "plan", Samples([3.0], [4.0]))
        assert len((destination / "samples.jsonl").read_text().splitlines()) == 2
        assert len(list(csv.DictReader((destination / "summary.csv").open()))) == 2
        assert (
            json.loads((destination / "metadata.json").read_text())["status"]
            == "running"
        )
    rows = list(csv.DictReader((destination / "summary.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert float(rows[0]["logical_payload_gb_per_s"]) > 0
    assert rows[1]["logical_payload_gb_per_s"] == ""
    raw = [
        json.loads(line)
        for line in (destination / "samples.jsonl").read_text().splitlines()
    ]
    assert raw[0]["operation"] == "transfer"
    assert len(raw) == 2
    assert (
        json.loads((destination / "metadata.json").read_text())["status"]
        == "measurements_finished"
    )
    with pytest.raises(FileExistsError):
        ResultWriter(destination, {})


def test_result_writer_marks_failed_context(tmp_path: Path) -> None:
    destination = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        with ResultWriter(destination, {}) as writer:
            writer.write({"payload_bytes": 0}, "transfer", Samples([1.0], [1.0]))
            raise RuntimeError("boom")
    assert json.loads((destination / "metadata.json").read_text())["status"] == "failed"


def test_result_writer_rejects_mismatched_samples_and_zero_duration(
    tmp_path: Path,
) -> None:
    with ResultWriter(tmp_path / "mismatch", {}) as writer:
        with pytest.raises(ValueError):
            writer.write({"payload_bytes": 1}, "transfer", Samples([1.0], []))
    with ResultWriter(tmp_path / "zero", {}) as writer:
        writer.write({"payload_bytes": 1}, "transfer", Samples([0.0], [0.0]))
    row = next(
        csv.DictReader((tmp_path / "zero" / "summary.csv").open(encoding="utf-8"))
    )
    assert row["logical_payload_gb_per_s"] == ""


def test_metadata_replace_is_atomic_on_success_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "atomic"
    writer = ResultWriter(destination, {"run": "test"})

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        writer.finish()
    assert (
        json.loads((destination / "metadata.json").read_text())["status"] == "running"
    )


@pytest.mark.parametrize("warning_filter", ["default", "error"])
def test_failure_status_error_preserves_original_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    warning_filter: str,
) -> None:
    destination = tmp_path / "failed_atomic"
    writer = ResultWriter(destination, {})

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with warnings.catch_warnings():
        warnings.simplefilter(warning_filter, RuntimeWarning)
        with pytest.raises(RuntimeError, match="original"):
            with writer:
                raise RuntimeError("original")
    assert "failure status" in capfd.readouterr().err


def test_failure_diagnostic_bypasses_broken_python_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    class BrokenStderr(io.StringIO):
        def write(self, text: str) -> int:
            raise RuntimeError("broken Python stderr")

    writer = ResultWriter(tmp_path / "broken-stderr", {})

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_replace)
        patch.setattr(sys, "stderr", BrokenStderr())
        with pytest.raises(RuntimeError, match="original"):
            with writer:
                raise RuntimeError("original")
    assert "failure status" in capfd.readouterr().err
