# Sparse KV operator benchmark

Use Python 3.10+ (the target environment uses PyTorch/torch_npu 2.10 and CANN
9.1.0). Set `PYTHON_BIN` to the intended virtual environment's Python when needed.
Run from the SGLang repository root on an Ascend 950 host after building the
custom operators; the benchmark does not rebuild them:

```bash
bash scripts/ascend/bench_sparse_kv_950.sh --preset smoke
bash scripts/ascend/bench_sparse_kv_950.sh --preset full
```

The smoke preset covers batch 1 and 32, MTP 0 through 3, a 140,000-token
prefix, and 0% and 100% logical cache hit ratios. The full preset uses batch
1, 2, 4, 8, 16, 31, and 32; all four MTP widths; prefixes 8,192, 32,768,
140,000, and 143,360; and hit ratios 0, .25, .5, .75, and 1. Capacity is
4,096 by default. Full uses 10 warmups and 50 samples per operation, producing
560 workloads and 1,680 summary rows. Smoke uses 3 warmups and 10 samples,
producing 16 workloads and 48 rows. Axes can be narrowed for a focused run:

```bash
bash scripts/ascend/bench_sparse_kv_950.sh \
  --batch-sizes 1,8,16,32 --mtps 0,1,2,3 --seq-lens 140000 \
  --hit-ratios 0,.5,1 --warmup 10 --iterations 50
```

On a Mac or another CPU-only machine, validate case construction and output
without loading NPU libraries:

```bash
bash scripts/ascend/bench_sparse_kv_950.sh --dry-run --preset smoke
```

On macOS, select a Python 3.10+ environment through `PYTHON_BIN` if the system
`python3` is older. Dry-run prints the matrix and memory estimates; it creates
no result files and does not load torch, torch_npu or MemFabric.

Use `--capacities 2048,4096,8192` to sweep cache capacity, `--ops plan,transfer`
to select operations, and `--device 0` to select the logical device index.
`--help` lists all flags. For CPU regression tests, use the existing
`bash scripts/ascend/run_sparse_kv_950.sh cpu` entry point.

Each run creates a new directory (or uses a new `SGL_SPARSE_KV_BENCH_DIR`) containing
`results/summary.csv`, `results/samples.jsonl`, `results/metadata.json`, the
console `benchmark.log`, and `process_exit_code`. The CSV has one row for each
case and operation (`plan`, `transfer`, and `plan_transfer`) with device-event
microsecond statistics (mean, p50, p90, p99, min, max, standard deviation) and
synchronized wall time. Logical payload bandwidth is reported for Transfer
and the combined call as `logical_payload_gb_per_s`, in decimal GB/s; it is
empty for Plan or an unresolved zero-duration sample, and zero when there are
no misses. This counts useful bytes, not total physical memory traffic.
`--output-dir` can relocate the three result files; the wrapper prints their
actual path. Existing output directories are rejected. Partial records are
flushed after each operation. Success requires `process_exit_code` to be `0`,
including interpreter/library teardown; a 134 abort is never ignored.
The wrapper also records `tee_exit_code` and `runner_exit_code`; the latter
must be `0` for a complete successful run, including log capture. Metadata is
replaced atomically and includes scoped Git changes, individual source SHA-256
hashes, bundled operator `.so` hashes, versions and environment settings.
Dirty runs are permitted and explicitly labeled for comparison.

The workload uses GLM-5.2 C8 operator dimensions: Top-K 2,048, page size 128,
BF16 K 512 and RoPE 64 (1,152 bytes per miss). `seq_len` is the stable prefix;
query `j` within each request sees `seq_len + j + 1` tokens. Each row selects
2,048 unique existing prefix tokens (new/draft tokens are excluded), permitting
exact 100% logical hits. Queries are deterministic and reused across hit-ratio
and capacity cases. Ratios are rounded to an integer hit count per row, with
requested and actual ratios recorded. `occupancy_min/max` records available
resident entries when a short prefix cannot fill all cache slots. All seven
Plan state tensors are restored before every
warmup and sample, outside the timed event interval. Transfer uses the fixed
metadata produced by a validation Plan call. The host pool is allocated once
for the largest case through MemFabric LOCAL giant-page DVA and is not replaced
with HBM memory.

`plan_transfer` is the existing combined Python/operator API: it invokes Plan
and then Transfer sequentially. MTP 1 and 2 are benchmarked as raw query widths
2 and 3; this does not change production decode guards. Event timings are eager
NPU elapsed times, while wall timings include event recording, launch and
synchronization. NPU Event `elapsed_time` returns milliseconds, converted here
to microseconds ([Ascend implementation](https://github.com/Ascend/pytorch/blob/master/torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp)).
The eager event interval can include host dispatch gaps; it is not a hardware
profiler's kernel-only duration. Allocation, host payload preparation, oracle
checks and state restoration are excluded from both reported intervals. The
same addresses are intentionally reused between samples, so the requested
logical hit ratio is not a claim about hardware cache state.

The full matrix reserves about 5.31 GB (4.94 GiB) of host giant-page memory and roughly
576 MiB of resident NPU storage for batch 32, MTP 3, capacity 4,096. Ensure
huge pages and available process memory are configured before starting it.
Sample payloads and untouched-slot sentinels are checked outside the timed
interval (up to eight miss slots and eight untouched slots per row); all seven
Plan metadata tensors are compared with the scalar oracle. The existing
correctness suite covers complete payload matrices. Hit slots carry sentinels
because these benchmarks do not run attention; the C8 indexer is not timed.

This Mac-side checkout validates case construction, argument preparation and
the output format only. It does not provide hardware timings; collect those on
the Ascend 950 host and retain the generated `metadata.json` with the results.
