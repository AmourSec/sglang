from __future__ import annotations

import json
from functools import partial

from .cases import BLOCK_SIZE, TOPK
from .cli import RunConfig, parse_args
from .environment import REPO_ROOT, environment_metadata
from .results import ResultWriter, Scalar
from .timing import measure, summarize


def run(config: RunConfig) -> None:
    import sgl_kernel_npu_sparse_kv
    import torch
    import torch_npu

    from ..host_memory import HostPlanes
    from ..npu_harness import load_ops
    from .device import DeviceWorkload
    from .workload import build_workload

    if not torch.npu.is_available():
        raise RuntimeError("an Ascend 950 NPU is required")
    torch.npu.set_device(config.device)
    device_name = torch.npu.get_device_name(config.device)
    if "950" not in device_name:
        raise RuntimeError(f"Ascend 950 required, detected {device_name}")
    package_path = str(sgl_kernel_npu_sparse_kv.__file__)
    if not package_path.startswith(str(REPO_ROOT / "sgl-kernel-npu-sparse-kv")):
        raise RuntimeError(f"extension loaded from another checkout: {package_path}")
    ops = load_ops()
    metadata = environment_metadata()
    metadata.update(
        {
            "configuration": config.summary(),
            "device_name": device_name,
            "torch_npu_version": str(torch_npu.__version__),
            "extension_path": package_path,
            "timing": "eager NPU Event elapsed_time milliseconds converted to microseconds; wall includes event record, API dispatch and end synchronization",
            "cache_method": "restore seven metadata tensors before every warmup/sample; repeated addresses, no hardware cache flush; fixed Plan outputs for Transfer",
            "plan_transfer": "combined API, sequential Plan and Transfer kernels",
            "sequence_definition": "seq_len is stable prefix; query j sees seq_len+j+1; unique TopK from existing prefix only",
            "validation": "all Plan metadata versus scalar oracle; up to 8 miss payloads and 8 untouched slots per row outside timing",
        }
    )
    with ResultWriter(config.output_dir, metadata) as writer, torch.inference_mode():
        print(f"device={device_name}; allocating MemFabric LOCAL DDR pool", flush=True)
        host = HostPlanes(max(case.host_tokens for case in config.cases))
        for index, case in enumerate(config.cases, 1):
            print(f"[{index}/{len(config.cases)}] preparing {case.case_id}", flush=True)
            workload = build_workload(case)
            device = DeviceWorkload(workload, host, ops)
            fields: dict[str, Scalar] = {
                "case_id": case.case_id,
                "batch_size": case.batch_size,
                "mtp": case.mtp,
                "width": case.width,
                "rows": case.rows,
                "seq_len": case.seq_len,
                "visible_seq_len_max": case.seq_len + case.width,
                "topk": TOPK,
                "capacity": case.capacity,
                "block_size": BLOCK_SIZE,
                "k_dim": 512,
                "rope_dim": 64,
                "dtype": "bfloat16",
                "requested_hit_ratio": case.hit_ratio,
                "actual_hit_ratio": case.actual_hit_ratio,
                "hits_per_row": case.hits_per_row,
                "misses_per_row": case.misses_per_row,
                "total_misses": case.misses_per_row * case.rows,
                "payload_bytes": case.payload_bytes,
                "occupancy_min": workload.occupancy_min,
                "occupancy_max": workload.occupancy_max,
                "host_tokens": case.host_tokens,
                "seed": case.seed,
                "warmup": config.warmup,
                "iterations": config.iterations,
                "timing_mode": "eager",
            }
            for operation in config.operations:
                device.prepare(operation)
                device.calls[operation]()
                torch.npu.synchronize()
                device.check(operation)
                samples = measure(
                    device.calls[operation],
                    partial(device.restore, operation),
                    synchronize=torch.npu.synchronize,
                    event_factory=partial(torch.npu.Event, enable_timing=True),
                    warmup=config.warmup,
                    iterations=config.iterations,
                )
                device.check(operation)
                writer.write(fields, operation, samples)
                stats = summarize(samples.device_us)
                print(
                    f"  {operation}: mean={stats['mean']:.3f} us p50={stats['p50']:.3f} us p99={stats['p99']:.3f} us",
                    flush=True,
                )
            del device
        torch.npu.synchronize()
    print(
        f"measurements_finished={config.output_dir}; process exit must also succeed",
        flush=True,
    )


def main() -> None:
    config = parse_args()
    print(json.dumps(config.summary(), indent=2), flush=True)
    if not config.dry_run:
        run(config)


if __name__ == "__main__":
    main()
