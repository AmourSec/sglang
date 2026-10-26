from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

TOPK = 2048
BLOCK_SIZE = 128
TOKEN_BYTES = 1152
MAX_SEQ_LEN = 143360
T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkCase:
    batch_size: int
    mtp: int
    seq_len: int
    hit_ratio: float
    capacity: int = 4096
    seed: int = 2026

    def __post_init__(self) -> None:
        if not 1 <= self.batch_size <= 32:
            raise ValueError("batch_size must be in [1, 32]")
        if not 0 <= self.mtp <= 3:
            raise ValueError("mtp must be in [0, 3]")
        if not TOPK <= self.seq_len <= MAX_SEQ_LEN:
            raise ValueError(f"seq_len must be in [{TOPK}, {MAX_SEQ_LEN}]")
        if not math.isfinite(self.hit_ratio) or not 0 <= self.hit_ratio <= 1:
            raise ValueError("hit_ratio must be finite and in [0, 1]")
        if self.capacity not in (2048, 4096, 8192):
            raise ValueError("capacity must be one of 2048, 4096, or 8192")

    @property
    def width(self) -> int:
        return self.mtp + 1

    @property
    def rows(self) -> int:
        return self.batch_size * self.width

    @property
    def hits_per_row(self) -> int:
        return round(TOPK * self.hit_ratio)

    @property
    def misses_per_row(self) -> int:
        return TOPK - self.hits_per_row

    @property
    def actual_hit_ratio(self) -> float:
        return self.hits_per_row / TOPK

    @property
    def pages_per_request(self) -> int:
        return (self.seq_len + self.width + BLOCK_SIZE - 1) // BLOCK_SIZE

    @property
    def host_num_blocks(self) -> int:
        return self.batch_size * self.pages_per_request

    @property
    def host_tokens(self) -> int:
        return self.host_num_blocks * BLOCK_SIZE

    @property
    def payload_bytes(self) -> int:
        return self.rows * self.misses_per_row * TOKEN_BYTES

    @property
    def case_id(self) -> str:
        ratio = f"{self.hit_ratio:.17g}"
        return f"b{self.batch_size}_mtp{self.mtp}_s{self.seq_len}_h{ratio}_c{self.capacity}"


def _axis(values: list[T] | tuple[T, ...], name: str) -> tuple[T, ...]:
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be non-empty and contain no duplicates")
    return result


def make_cases(
    batches: list[int] | tuple[int, ...],
    mtps: list[int] | tuple[int, ...],
    seq_lens: list[int] | tuple[int, ...],
    hit_ratios: list[float] | tuple[float, ...],
    capacities: list[int] | tuple[int, ...] = (4096,),
    seed: int = 2026,
) -> list[BenchmarkCase]:
    axes = (
        _axis(batches, "batches"),
        _axis(mtps, "mtps"),
        _axis(seq_lens, "seq_lens"),
        _axis(hit_ratios, "hit_ratios"),
        _axis(capacities, "capacities"),
    )
    return [
        BenchmarkCase(batch, mtp, seq, ratio, capacity, seed)
        for batch in axes[0]
        for mtp in axes[1]
        for seq in axes[2]
        for ratio in axes[3]
        for capacity in axes[4]
    ]
