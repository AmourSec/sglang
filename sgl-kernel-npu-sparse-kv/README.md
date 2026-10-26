# SGLang Ascend 950 sparse KV operators

This package builds the `SparseKvPlan` and `SparseKvTransfer` CANN custom
operators and registers these PyTorch entry points:

- `torch.ops.sgl_sparse_kv.plan`
- `torch.ops.sgl_sparse_kv.transfer`
- `torch.ops.sgl_sparse_kv.plan_transfer`

The package has no vLLM runtime dependency. It targets CANN 9.1.0,
torch 2.10.x, torch_npu 2.10.x, and Ascend 950 (`Ascend950PR_9599`).

## Build on Ascend 950

Load the CANN environment, activate the Python environment containing the
matching torch and torch_npu builds, then run:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd sglang/sgl-kernel-npu-sparse-kv
SOC_VERSION=ascend950 bash build.sh
python3 -m pip install --no-build-isolation -v .
```

Run `bash build.sh --help` to show the supported environment variables without
requiring Linux or a CANN installation. `SOC_VERSION` must be exactly
`ascend950`; `ASCEND_HOME_PATH` defaults to the latest system toolkit.

The first clean build downloads pinned json 3.11.3, Protobuf 25.1, Abseil
20230802.1, and makeself 2.5.0 archives over HTTPS. Every archive is checked
against a SHA-256 digest before use. For an offline build, prepopulate the
vendored CMake dependency locations under `cann_ops/third_party/` before
running `build.sh`.

`build.sh` uses the vendored CANN `opbuild` project to generate ACLNN APIs,
compile the dynamic AscendC/SIMT kernels, create the custom OPP installer,
install it under the Python package, and build the Torch binding in place.
It does not use `ascendc_library` as a substitute for ACLNN generation.

## Load

```python
import sgl_kernel_npu_sparse_kv

sgl_kernel_npu_sparse_kv.ensure_loaded()
```

For a source-tree import check before installation, use
`PYTHONPATH=src python3 -c 'import sgl_kernel_npu_sparse_kv'`.

`ensure_loaded()` prepends the packaged vendor OPP and loads its `libcust_opapi.so`
with explicit `RTLD_LOCAL` visibility before importing the C++ registration
module. The adapter resolves ACLNN entry points with `dlsym` on a library
handle, so it does not require process-global symbol export. The Python loader
retains the handle for the lifetime of the module. This avoids requesting the
global visibility that triggered exit-time corruption with MemFabric in the
reported 950 reproduction; device validation must include normal process exit.

`host_cache_bases` is a contiguous NPU int64 tensor with
shape `[2]`; its values are the host K-plane and rope/V-plane device virtual
addresses. CPU host virtual addresses are invalid.

## Positional ABI

`plan` takes 15 tensors in this order: `req_ids`, `topk_indices`,
`stable_prefix_lens`, `visible_seq_lens`, `token_to_req`, `block_table`,
`active_rows`, `last_req_ids`, `slot_to_token`, `lru_slots`, `current_slots`,
`miss_count`, `miss_tokens`, `miss_slots`, `compact_workspace`; followed by
`topk`, `capacity`, `max_token`, `block_size`, `host_num_blocks`.

`transfer` takes `miss_count`, `miss_tokens`, `miss_slots`, `token_to_req`,
`block_table`, `host_cache_bases`, `active_rows`, `resident_k`, `resident_v`;
followed by `topk`, `capacity`, `block_size`, `host_num_blocks`,
`token_size_bytes_k`, `token_size_bytes_v`.

`plan_transfer` concatenates the 15 plan tensors, `host_cache_bases`,
`resident_k`, `resident_v`, then the plan attributes and transfer byte sizes.
Plan state/workspace tensors and resident tensors are declared mutable in the
Torch schemas and have Meta implementations for graph/export shape analysis.

All tensor inputs must be contiguous NPU tensors with the declared dtype and
shape. The binding verifies the conditional GM workspace capacity and the exact
BF16 resident payload byte size without reading device values or synchronizing
the device.

For each Plan call, all per-row tensors have the same leading dimension as
`topk_indices`. Persistent allocations may be larger: pass contiguous prefix
views such as `last_req_ids[:rows]` and `slot_to_token[:rows]`, as the SGLang
runtime does. These views retain the backing storage, so rows outside the call
remain unchanged and can be reused in a later larger batch. With fixed tensor
shapes, `active_rows` selects the processed prefix; the kernel clamps it to the
call's row count. Inactive rows do not participate in Plan/LRU updates.

## Source provenance

The CANN project, host tiling, and AscendC kernels are copied from
`vllmascend_sep_fif` commit `cd84b83c566e62ac26934758cc563b152292f41e`.
Only the sparse KV Plan/Transfer operators and their CANN build dependency
closure are included. Original file-level copyright and license notices are
preserved. Package-authored and Apache-2.0 files are covered by `LICENSE`;
the two SIMT kernel files use Mulan PSL v2, and the vendored CANN build support
uses the CANN Open Software License Agreement Version 2.0. See `NOTICE` and
`THIRD_PARTY_LICENSES/` for the exact mapping and license texts.

The internal Plan constants namespace is named `sparse_kv_plan_config` to avoid
colliding with CANN's generated tiling-key entry macro. The exported kernel
entry remains `sparse_kv_plan`; no `opInterface.value` override is required.

macOS checks include CPU references and native-library loader visibility using
a small compiled probe. CANN compilation and NPU numeric/teardown tests require
a Linux Ascend 950 host with CANN 9.1.0.
