from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cases import TOKEN_BYTES, BenchmarkCase, make_cases
from .results import JsonValue

OPERATIONS = ("plan", "transfer", "plan_transfer")


@dataclass(frozen=True)
class RunConfig:
    cases: list[BenchmarkCase]
    operations: tuple[str, ...]
    warmup: int
    iterations: int
    device: int
    output_dir: Path
    dry_run: bool

    def summary(self) -> dict[str, JsonValue]:
        host_bytes = max(case.host_tokens for case in self.cases) * TOKEN_BYTES
        return {
            "workloads": len(self.cases),
            "result_rows": len(self.cases) * len(self.operations),
            "batch_sizes": sorted({case.batch_size for case in self.cases}),
            "mtps": sorted({case.mtp for case in self.cases}),
            "seq_lens": sorted({case.seq_len for case in self.cases}),
            "hit_ratios": sorted({case.hit_ratio for case in self.cases}),
            "capacities": sorted({case.capacity for case in self.cases}),
            "seed": self.cases[0].seed,
            "operations": list(self.operations),
            "warmup": self.warmup,
            "iterations": self.iterations,
            "device_index": self.device,
            "host_payload_bytes": host_bytes,
            "host_pool_reserve_bytes": max(8 * 1024**2, host_bytes + 16 * 1024**2),
            "resident_payload_bytes_max": max(
                case.rows * case.capacity * TOKEN_BYTES for case in self.cases
            ),
            "output_dir": str(self.output_dir),
        }


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="GLM-shaped Ascend 950 sparse KV eager microbenchmark"
    )
    parser.add_argument("--preset", choices=("full", "smoke"), default="full")
    for option, help_text in (
        ("batch-sizes", "comma-separated requests per batch, 1..32"),
        ("mtps", "comma-separated draft counts, 0..3; rows=batch*(mtp+1)"),
        ("seq-lens", "comma-separated stable prefix lengths, 2048..143360"),
        ("hit-ratios", "comma-separated logical cache hit ratios, 0..1"),
        ("capacities", "comma-separated resident capacities: 2048,4096,8192"),
    ):
        parser.add_argument(f"--{option}", help=help_text)
    parser.add_argument(
        "--ops",
        default=",".join(OPERATIONS),
        help="comma-separated plan,transfer,plan_transfer",
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print matrix/memory estimates without device imports or allocations",
    )
    args = parser.parse_args(argv)
    smoke = args.preset == "smoke"
    try:
        cases = make_cases(
            [
                int(value)
                for value in (
                    args.batch_sizes or ("1,32" if smoke else "1,2,4,8,16,31,32")
                ).split(",")
            ],
            [int(value) for value in (args.mtps or "0,1,2,3").split(",")],
            [
                int(value)
                for value in (
                    args.seq_lens or ("140000" if smoke else "8192,32768,140000,143360")
                ).split(",")
            ],
            [
                float(value)
                for value in (
                    args.hit_ratios or ("0,1" if smoke else "0,0.25,0.5,0.75,1")
                ).split(",")
            ],
            [int(value) for value in (args.capacities or "4096").split(",")],
            args.seed,
        )
        operations = tuple(part.strip() for part in args.ops.split(","))
        if len(set(operations)) != len(operations) or any(
            op not in OPERATIONS for op in operations
        ):
            raise ValueError(f"--ops must contain unique entries from {OPERATIONS}")
        warmup = args.warmup if args.warmup is not None else (3 if smoke else 10)
        iterations = (
            args.iterations if args.iterations is not None else (10 if smoke else 50)
        )
        if warmup < 0 or iterations <= 0 or args.device < 0:
            raise ValueError("warmup/device must be >=0 and iterations must be >0")
    except ValueError as error:
        parser.error(str(error))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        args.output_dir
        or Path(tempfile.gettempdir()) / f"sglang-sparse-kv-perf-{stamp}-{os.getpid()}"
    )
    return RunConfig(
        cases, operations, warmup, iterations, args.device, output, args.dry_run
    )
