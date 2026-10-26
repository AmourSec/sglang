# Ascend 950 Sparse KV operator tests

The CPU suite validates the independent scalar Plan oracle and paged Transfer oracle without an NPU:

```bash
python -m pytest -q test/manual/ascend/sparse_kv/test_reference.py
```

On an Ascend 950 host with CANN 9.1.0, torch/torch_npu 2.10.0, and MemFabric installed, use the runner from the repository root:

```bash
bash scripts/ascend/run_sparse_kv_950.sh cpu
bash scripts/ascend/run_sparse_kv_950.sh quick
bash scripts/ascend/run_sparse_kv_950.sh full
bash scripts/ascend/run_sparse_kv_950.sh host
bash scripts/ascend/run_sparse_kv_950.sh graph
```

`cpu` runs the independent oracle, production metadata/configuration and ModelSlim C8 descriptor tests, plus the extension's static build/package/schema contracts without importing torch_npu. `quick` adds representative Plan cases. `full` adds the BS1/2/4/8/16/31/32, MTP widths 1/4, sequence-boundary matrix, C2048/C4096/C8192 workspace cases, B32 TopK2048 real DDR Transfer, fused graph replay, selected-KV attention comparison against an FP32 oracle, and a weight-free real-manager lifecycle covering chunked prefill, decode, request reuse, MTP suffix replacement, layer offsets, target/draft isolation, and factory cache-hit sharing. `host` runs the large and focused real DDR DVA Transfer cases. `graph` runs fused Plan/Transfer capture and replay while changing metadata and host payload.

The runner imports `sgl_kernel_npu_sparse_kv` and builds `sgl-kernel-npu-sparse-kv` with `bash build.sh` if it is not loadable. Host modes require `memfabric_hybrid.offload` with LOCAL scene, `OFFLOAD_FLAG_GIANT_PAGE`, and nonzero `get_dva()` results. They fail when those capabilities are unavailable; device HBM is never substituted for host DDR.

The full B32/S143360 case reserves about 4.92 GiB of MemFabric host planes and the width-4 case allocates about 0.56 GiB of resident NPU storage. Exact-bit payload generation and comparison currently bring peak process host memory above 10 GiB, so run `full` and `host` with at least 16 GiB of available process RAM in addition to the MemFabric reservation.

Each invocation writes an environment report and pytest output under `${TMPDIR:-/tmp}`. Set `SGL_SPARSE_KV_EVIDENCE_DIR` to select a different artifact directory and `PYTHON_BIN` to select Python.

## Build and run after pulling onto the 950 host

Use the existing GLM-5.2 C8 Python environment. The repository's [Ascend installation guide](../../../../docs/docs/hardware-platforms/ascend-npus/getting-started/installation.mdx) currently specifies Python 3.12, CANN 9.1.0, torch 2.10.0, torch_npu 2.10.0.post6, and the **950 build** of sgl-kernel-npu 2026.9.0.post9. The runner records the installed versions; it does not upgrade them. MemFabric must provide the LOCAL giant-page/DVA API used by `clean_simt`.

From the SGLang repository root:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
(
  cd sgl-kernel-npu-sparse-kv
  SOC_VERSION=ascend950 bash build.sh
  python3 -m pip install --no-build-isolation -v .
)
bash scripts/ascend/run_sparse_kv_950.sh cpu
bash scripts/ascend/run_sparse_kv_950.sh quick
bash scripts/ascend/run_sparse_kv_950.sh full
```

The first build downloads hash-pinned dependencies. See the [extension build instructions](../../../../sgl-kernel-npu-sparse-kv/README.md) for offline preparation. The operator tests generate tensors and do not need model weights. `full` includes the host and graph cases; run those modes separately when diagnosing a failure. Preserve both `environment.log` and `pytest.log` from a failed run.

### Rebuild after the initial 950 device fixes

If the server still has the temporary `sparse_kv_plan_apt_entry` workaround, save
and set aside those local edits before pulling. The repository now renames only
the internal namespace, keeping the original `sparse_kv_plan` entry and default
`opInterface.value`. Do not reapply the temporary entry rename onto this fix.

CANN can retain generated `aic-ascend950-ops-info.ini` metadata across an
incremental build. From the SGLang repository root, remove the generated build
directory once and rebuild/reinstall after pulling:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
rm -rf -- sgl-kernel-npu-sparse-kv/cann_ops/build
(
  cd sgl-kernel-npu-sparse-kv
  SOC_VERSION=ascend950 bash build.sh
  python3 -m pip install --no-build-isolation -v .
)
bash scripts/ascend/run_sparse_kv_950.sh cpu
bash scripts/ascend/run_sparse_kv_950.sh quick
bash scripts/ascend/run_sparse_kv_950.sh full
```

The new installer generates environment exports that are safe under `set -u`,
preserve existing paths, and omit empty trailing path entries. The runner also
initializes the two optional path variables before sourcing older generated
scripts, so a stale script cannot fail preflight merely because a variable is
unset.

The batch-shrink regression keeps its four-row allocation and passes three-row
views for the smaller call, then restores four-row views. It checks the entire
state against the reference, including the inactive tail. The CPU suite also
checks storage identity, tail preservation, and restoration for both compact
and fixed-shape calls and both UB/GM workspace sizes. The NPU binding continues
to reject unsliced state with a mismatched row count.

### Runtime prerequisites and full-mode follow-up

`sgl_kernel_npu_sparse_kv` and `sgl_kernel_npu` are separate packages. The
real-manager test also needs a built, compatible `sgl_kernel_npu` containing
`sparsity_driven_kv_offload`. Building this repository's sparse-KV extension
does not build that dependency. Build/install the external checkout using its
own version's instructions, then check the **same Python environment** used by
the runner. A source directory without `lib/libsgl_kernel_npu.so`, or an older
installed package without this module, is insufficient. The following check
prints the resolved package path before importing the required capabilities:

```bash
python3 - <<'PY'
import importlib.util
import sys
import torch
import torch_npu
import IPython

print("python:", sys.executable)
spec = importlib.util.find_spec("sgl_kernel_npu")
print("sgl_kernel_npu:", None if spec is None else spec.origin, flush=True)
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor, slot_map_lookup, unidex_copy_inplace,
)
print("runtime kernel imports passed")
PY
```

`IPython` is already declared in SGLang's Python dependencies and is needed by
the attention/runtime import chain. Installing it into a different Python
environment does not make it available to the test interpreter.

Check free hugepages as well as ordinary RAM before the large host test:

```bash
grep -E '^(MemAvailable|HugePages_|Hugepagesize|Hugetlb)' /proc/meminfo
```

The reported 950 run required increasing its hugepage pool from 2 GiB to 7 GiB
for an approximately 5.3 GiB reservation. This is an observed configuration,
not a fixed requirement for every MemFabric version; alignment, free pages and
other processes also matter. The scripts do not modify the system hugepage pool.

The two guarded BF16 allocations now pass tuple dimensions to `torch.full`.
This is a Python test-only fix and does not require another CANN rebuild.
Run `host` to recheck the B32, S143360 cases at query widths 1 and 4, then `full`.

If pytest prints its results but exits with `corrupted size`, `double free` or
status 134, retain that failure. Passing assertions do not establish a clean
process exit, and the message alone does not identify the corrupting component.
Use the diagnostic runner to separate import, allocation and operator stages:

```bash
bash scripts/ascend/diagnose_sparse_kv_950.sh
# Or restrict the pytest stages while keeping the baseline probes:
bash scripts/ascend/diagnose_sparse_kv_950.sh \
  'test/manual/ascend/sparse_kv/test_npu_full_host.py::test_full_host_plan_transfer_exact_payload_and_guards[1]'
```

This requires the already built extension and the same CANN/Python environment
as the normal runner. It runs each stage in a separate process, enables Python
faulthandler, saves separate logs and exact exit codes in `status.tsv`, and
continues gathering evidence after failed stages. Its final status remains
nonzero if any stage failed. Preserve the whole evidence directory and the
original combined `full` log; isolated success does not replace the full run.

The reported minimal reproducer triggers exit 134 when `libcust_opapi.so` is
loaded with `RTLD_GLOBAL` before MemFabric initialization. The loader now uses
explicit `RTLD_LOCAL`; the C++ adapter already resolves ACLNN symbols through
library handles. The specific conflicting native symbol/destructor is not yet
identified, and the local-visibility fix still needs fresh-process verification
on 950. See the [dlopen visibility rules](https://man7.org/linux/man-pages/man3/dlopen.3.html).

This Python loader change does not require rebuilding the CANN operators. The
test runners import this checkout's source package; applications using an
installed wheel must update that Python package. Start a new Python process:
local loading cannot undo an earlier global load in the same process. An older
standalone reproducer that still hardcodes `RTLD_GLOBAL` continues to exercise
the original trigger; use `ensure_loaded()` to test the fixed loader.

The diagnostic probes now load this checkout's `host_memory.py` by absolute
file path, avoiding an installed or standard-library `test` package that can
shadow `test.manual`. They check both extension-before-MemFabric and the reverse
initialization order. These probes do not require the separate `sgl_kernel_npu`
package; a missing `runtime_dependencies` stage remains a distinct failure and
does not prevent collecting the coexistence probe results.

## Enable the SGLang runtime path

After installing the extension and this SGLang checkout, set:

```bash
export ASCEND_USE_FIA=1
export SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1
export SGLANG_NPU_SPARSE_KV_SIMT=1
export SGLANG_NPU_SPARSE_KV_CAPACITY=4096
export SGLANG_NPU_USE_MLAPO=0
```

Keep the model path, ModelSlim weight configuration, and TP/DP settings from the deployment command. Add these arguments for the local offload path:

```text
--attention-backend ascend
--kv-cache-dtype bfloat16
--disable-radix-cache
--max-running-requests 32
--context-length 143360
```

MTP=0 omits speculative arguments. MTP=3 adds:

```text
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-eagle-topk 1
--speculative-num-draft-tokens 4
```

Here `speculative-eagle-topk=1` selects a linear prediction chain. It is separate from GLM-5.2's attention `index_topk=2048`: target verification processes four query rows per request, each with 2048 selected indices. The model's single NextN layer is iterated for the three predictions.

Main MLA KV remains BF16 K512 + rope64. On 950 the C8 indexer uses FP8 E4M3FN payload and FP32 scales, following the source implementation. The runtime rejects P/D disaggregation, context parallelism, HiCache, HiSparse, prefix caching, and MLAPO combinations that have not been integrated into this path.

The standalone test memory figures above are for one layer. A full model allocation is much larger: at B32/S140000, 78 backbone layers need about 374.91 GiB of host KV per process before request padding and draft storage. With capacity 4096 and four query rows, backbone resident KV alone is about 43.88 GiB per process before padding, metadata, and the indexer. Actual allocation follows the configured context length and per-process request count. Start a model smoke test at lower concurrency if the deployment does not have this memory budget, then increase it to the required target.

After the single-operator suite passes, validate the deployed model in MTP=0 and MTP=3 with chunked prefill, consecutive decoding, and mixed-length requests, first in eager execution and then with ACL graph enabled. Full-model serving, CANN compilation, DVA behavior, and NPU numerical results cannot be validated on the development Mac; they remain device-side acceptance checks.
