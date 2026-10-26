#!/usr/bin/env bash
# Isolate native process-exit failures in the Ascend 950 sparse-KV test stack.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_DIR="$(dirname "$(command -v "${PYTHON_BIN}")")"
export PATH="${PYTHON_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/sgl-kernel-npu-sparse-kv/src${PYTHONPATH:+:${PYTHONPATH}}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${SGL_SPARSE_KV_DIAG_DIR:-${TMPDIR:-/tmp}/sglang-sparse-kv-diagnose-${STAMP}}"
OPP_ENV="${SGL_SPARSE_KV_OPP_ENV:-${REPO_ROOT}/sgl-kernel-npu-sparse-kv/src/sgl_kernel_npu_sparse_kv/_cann_ops_custom/vendors/sgl_sparse_kv_transformer/bin/set_env.bash}"
HOST_MEMORY_PATH="${REPO_ROOT}/test/manual/ascend/sparse_kv/host_memory.py"
export SGL_SPARSE_KV_HOST_MEMORY_PATH="${HOST_MEMORY_PATH}"
mkdir -p "${EVIDENCE_DIR}"
EVIDENCE_DIR="$(cd "${EVIDENCE_DIR}" && pwd)"
LOG_DIR="${EVIDENCE_DIR}/stages"
STATUS_FILE="${EVIDENCE_DIR}/status.tsv"
mkdir -p "${LOG_DIR}"
printf 'stage\texit_code\tlog\n' >"${STATUS_FILE}"
export PYTHONUNBUFFERED=1

source_opp_environment() {
  export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  source "$1"
}

run_stage() {
  local stage="$1"
  shift
  local log="${LOG_DIR}/${stage}.log"
  local status
  printf 'stage=%s start log=%s\n' "${stage}" "${log}"
  if "$@" >"${log}" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\t%s\t%s\n' "${stage}" "${status}" "${log}" >>"${STATUS_FILE}"
  if (( status != 0 && FIRST_FAILURE == 0 )); then
    FIRST_FAILURE=${status}
  fi
  printf 'stage=%s exit_code=%s log=%s\n' "${stage}" "${status}" "${log}"
  return 0
}

FIRST_FAILURE=0
printf 'python=%s\npython_realpath=%s\nopp_env=%s\n' "${PYTHON_BIN}" "$(command -v "${PYTHON_BIN}")" "${OPP_ENV}" >"${EVIDENCE_DIR}/environment.log"
if [[ -r /proc/meminfo ]]; then
  awk '/^(MemAvailable:|HugePages_|Hugepagesize:|Hugetlb:)/' /proc/meminfo >>"${EVIDENCE_DIR}/environment.log"
fi
if [[ ! -f "${OPP_ENV}" ]]; then
  printf 'missing built custom OPP environment: %s\n' "${OPP_ENV}" >&2
  exit 2
fi
source_opp_environment "${OPP_ENV}"
cd "${REPO_ROOT}"

run_stage baseline_950 "${PYTHON_BIN}" -X faulthandler -c '
import torch
import torch_npu
import sys
if not torch_npu.npu.is_available():
    raise SystemExit("Ascend NPU is unavailable")
name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"sys_executable={sys.executable}")
print(f"npu_device_name={name}")
if "950" not in name:
    raise SystemExit(f"Ascend 950 is required, detected {name}")
'
if (( FIRST_FAILURE != 0 )); then
  printf 'baseline failed; inspect %s\n' "${STATUS_FILE}" >&2
  exit "${FIRST_FAILURE}"
fi

run_stage sparse_extension "${PYTHON_BIN}" -X faulthandler -c '
import sgl_kernel_npu_sparse_kv
sgl_kernel_npu_sparse_kv.ensure_loaded()
'
run_stage runtime_dependencies "${PYTHON_BIN}" -X faulthandler -c '
import importlib.util
import IPython
import torch
import torch_npu

spec = importlib.util.find_spec("sgl_kernel_npu")
if spec is None:
    raise SystemExit("sgl_kernel_npu is not importable")
print(f"sgl_kernel_npu_origin={spec.origin}")
from sgl_kernel_npu.sparsity_driven_kv_offload import (
    create_shm_tensor,
    slot_map_lookup,
    unidex_copy_inplace,
)
print(f"IPython={IPython.__version__}")
print("sgl_kernel_npu sparse-KV symbols=available")
'
run_stage memfabric_host_planes "${PYTHON_BIN}" -X faulthandler -c '
import importlib.util
import os
import torch
import torch_npu

host_memory_path = os.environ["SGL_SPARSE_KV_HOST_MEMORY_PATH"]
spec = importlib.util.spec_from_file_location("sgl_sparse_kv_host_memory", host_memory_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load host-memory probe from {host_memory_path}")
host_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_memory)
host_memory.HostPlanes(128)
print(f"host_memory_probe=loaded path={host_memory_path}")
'
run_stage extension_and_memfabric "${PYTHON_BIN}" -X faulthandler -c '
import importlib.util
import os
import sgl_kernel_npu_sparse_kv
import torch
import torch_npu

host_memory_path = os.environ["SGL_SPARSE_KV_HOST_MEMORY_PATH"]
spec = importlib.util.spec_from_file_location("sgl_sparse_kv_host_memory", host_memory_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load host-memory probe from {host_memory_path}")
host_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_memory)
sgl_kernel_npu_sparse_kv.ensure_loaded()
host_memory.HostPlanes(128)
print(f"host_memory_probe=loaded path={host_memory_path}")
'
run_stage memfabric_then_extension "${PYTHON_BIN}" -X faulthandler -c '
import importlib.util
import os
import sgl_kernel_npu_sparse_kv
import torch
import torch_npu

host_memory_path = os.environ["SGL_SPARSE_KV_HOST_MEMORY_PATH"]
spec = importlib.util.spec_from_file_location("sgl_sparse_kv_host_memory", host_memory_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"cannot load host-memory probe from {host_memory_path}")
host_memory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host_memory)
host_memory.HostPlanes(128)
print(f"host_memory_probe=loaded path={host_memory_path}")
sgl_kernel_npu_sparse_kv.ensure_loaded()
'

DEFAULT_TESTS=(
  test/manual/ascend/sparse_kv/test_npu_binding.py
  test/manual/ascend/sparse_kv/test_npu_plan.py
  test/manual/ascend/sparse_kv/test_npu_full_host.py
  test/manual/ascend/sparse_kv/test_npu_host_matrix.py
  test/manual/ascend/sparse_kv/test_npu_transfer.py
  test/manual/ascend/sparse_kv/test_npu_graph.py
  test/manual/ascend/sparse_kv/test_npu_attention.py
  test/manual/ascend/sparse_kv/test_npu_runtime.py
)
if (( $# )); then
  TESTS=("$@")
else
  TESTS=("${DEFAULT_TESTS[@]}")
fi
test_index=0
for test_path in "${TESTS[@]}"; do
  ((test_index += 1))
  test_file="${test_path%%::*}"
  stage="$(printf 'pytest_%02d_%s' "${test_index}" "$(basename "${test_file}" .py)")"
  run_stage "${stage}" env SGL_SPARSE_KV_TEST_MODE=full "${PYTHON_BIN}" -X faulthandler -m pytest -vv -rs "${test_path}"
done

printf 'evidence=%s\n' "${EVIDENCE_DIR}"
exit "${FIRST_FAILURE}"
