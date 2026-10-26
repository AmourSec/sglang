"""Small, deterministic timing primitives for the sparse KV benchmarks."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Protocol


class Event(Protocol):
    def record(self) -> None: ...

    def synchronize(self) -> None: ...

    def elapsed_time(self, end: "Event") -> float: ...


@dataclass(frozen=True)
class Samples:
    device_us: list[float]
    wall_us: list[float]


def _check_value(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def measure(
    call: Callable[[], None],
    restore: Callable[[], None],
    *,
    synchronize: Callable[[], None],
    event_factory: Callable[[], Event],
    warmup: int,
    iterations: int,
    clock_ns: Callable[[], int] = perf_counter_ns,
) -> Samples:
    """Measure an already-prepared operation with NPU events and wall time.

    ``restore`` is deliberately called before every warmup and measured sample,
    so a stateful Plan operation cannot turn a controlled miss workload into a
    sequence of increasingly warm calls.
    """
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")
    start = event_factory()
    end = event_factory()
    # CANN may lazily allocate event resources on the first record. Prime that
    # cost outside the measured loop.
    start.record()
    end.record()
    end.synchronize()

    def run_once(measured: bool) -> tuple[float, float] | None:
        restore()
        synchronize()
        if measured:
            wall_start = clock_ns()
            start.record()
            call()
            end.record()
            end.synchronize()
            wall_stop = clock_ns()
            elapsed = _check_value(start.elapsed_time(end) * 1000.0, "device time")
            return elapsed, _check_value((wall_stop - wall_start) / 1000.0, "wall time")
        call()
        synchronize()
        return None

    for _ in range(warmup):
        run_once(False)
    device_us: list[float] = []
    wall_us: list[float] = []
    for _ in range(iterations):
        result = run_once(True)
        assert result is not None
        device, wall = result
        device_us.append(device)
        wall_us.append(wall)
    return Samples(device_us, wall_us)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample set")
    clean = [_check_value(float(value), "sample") for value in values]
    ordered = sorted(clean)

    def percentile(percent: float) -> float:
        index = (len(ordered) - 1) * percent
        lower = math.floor(index)
        upper = math.ceil(index)
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "mean": statistics.fmean(clean),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "min": ordered[0],
        "max": ordered[-1],
        "stddev": statistics.pstdev(clean),
    }
