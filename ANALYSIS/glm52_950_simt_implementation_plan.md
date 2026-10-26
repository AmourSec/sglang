# GLM-5.2 950 SIMT KV offload implementation plan

> For agentic workers: use superpowers:subagent-driven-development. The initial implementation was delivered uncommitted; the user subsequently authorized committing and pushing to AmourSec/sglang on wzj_kv_offload. Work in the requested existing checkout and preserve other edits.

**Goal:** Migrate the proven 950 Plan/Transfer operators into SGLang local decode and MTP, with comprehensive standalone correctness tests.

**Architecture:** A standalone CANN/PyTorch extension stored in this repository exposes the source operator contract. SGLang owns split BF16 host/resident cache planes and persistent query-row LRU, shared by attention backends of the same KV pool. Indexer C8 remains separate.

**Tech stack:** CANN9.1.0 AscendC/SIMT, torch/torch_npu2.10.0, Python, pytest.

**Spec:** `ANALYSIS/glm52_950_simt_migration_design.md`.

## Global constraints

- Source logical baseline cd84b83c566e62ac26934758cc563b152292f41e; target initial HEAD 904691e0b5dc4516c5bc9771874d03dd0cadb470.
- Local offload only; no P/D integration.
- K512/rope64 BF16, indexer32x128, TopK2048; BS1..32, MTP widths1 and4, seq includes140000 and143360.
- Mac cannot run NPU compilation/tests. Write reference-driven tests first where possible; report static validation and device validation separately.
- Independent workers own disjoint files and must accommodate each other's changes. No refactor of unrelated code.

## Review focus

- Request-slot reuse and dormant MTP rows must never reuse stale KV.
- Target/draft pools must not alias; multistep draft backends must share their own store.
- BF16 main storage must not inherit FP8/int8 indexer dtype.
- Host HVA must never be passed as DVA; interleaved split views must never masquerade as contiguous planes.
- Graph replay must consume changed metadata, and full140k test must exercise actual host transfers.

## Tasks and ownership

- [x] **1. Standalone extension.** Own `sgl-kernel-npu-sparse-kv/`: kernel/tiling sources, minimal build support, validation, torch schemas/Meta, Python loader `sgl_kernel_npu_sparse_kv.ensure_loaded()`. Expose `torch.ops.sgl_sparse_kv.plan`, `transfer`, `plan_transfer` using source positional contract; plan omits host bases/resident tensors and transfer byte attrs. Check shell/Python syntax and dependency closure; document950 build command.
- [x] **2. Runtime integration.** Own `sparsity_driven_kv_offload/` except `config.py`, plus `attention/ascend_backend.py`: split host store, pool-owned manager, generation/frontier, query metadata, prefill writes, decode/verify/draft historical materialization and attention. Consume the extension API above. Supply meaningful CPU-checkable metadata tests where possible.
- [x] **3. C8 and memory configuration.** Own `memory_pool_npu.py`, `dsa_npu_indexer.py`, ModelSlim config, `sparsity_driven_kv_offload/config.py`, `environ.py`, and necessary pool configuration: separate index dtype/scales, faithful source C8 path, BF16 main cache, early resident memory reservation and incompatible-feature guards. Coordinate exact config/helper names with runtime owner.
- [x] **4. Correctness suite and runner.** Own `test/manual/ascend/sparse_kv/` and `scripts/ascend/`: independent scalar Plan oracle, transfer payload/guard checks, actual host DVA cases, multistep invalidation, MTP four-query causal checks, graph replay, BS32/140k. Add quick/full/host/graph selection and environment reports; no model weights required.
- [x] **5. Integration review and static checks.** Root reconciles APIs, checks full diff, invokes read-only review, runs available Python/shell/C++ static checks, validates test collection/CPU references where available, repairs actual findings.
- [x] **6. Delivery.** Update design/status and execution README; provide exact build and test commands, list checks run and NPU checks not run. The initial delivery left changes uncommitted; subsequent Git publication follows the user's explicit request.

## Decisions

- User clarification completes the earlier discovery questions and authorizes implementation; do not add another approval round.
- Preserve the requested checkout instead of creating a separate worktree; no history changes.
- NPU red/green and manual device QA are explicitly deferred by the user. Static success is never represented as device correctness.

## Completion evidence

Implementation and Mac-side verification completed on 2026-10-24. The combined CPU runner passed 51 tests. The manual directory passed 8 CPU reference tests and skipped its 8 NPU-only modules because torch_npu is unavailable on this Mac; these 8 passing tests are already included in the combined 51. Forty authored/modified Python files passed AST, Ruff, formatting, and import-order checks. Shell syntax, runner/build help, invalid runner argument handling, and authored-code diff checks passed. Staging the complete vendor sources additionally exposes inherited whitespace warnings in 46 CANN support/license files; their original bytes are preserved. Four independent review lanes found no remaining blocking issue.

The full runner now includes a separate-process, weight-free real-manager lifecycle test using MemFabric DVA. Real CANN compilation, NPU numerical checks, graph replay, and full-model serving remain the user-arranged 950 acceptance step. See `test/manual/ascend/sparse_kv/README.md` for build, tests, runtime switches, and memory requirements. No commit or push was made during implementation; the subsequent publication request targets AmourSec/sglang on branch wzj_kv_offload.
