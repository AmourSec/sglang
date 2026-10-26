#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {cpu|quick|full|host|graph}"
}

source_opp_environment() {
  export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  source "$1"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MODE="${1:-quick}"
case "${MODE}" in
  cpu|quick|full|host|graph) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_DIR="$(dirname "$(command -v "${PYTHON_BIN}")")"
export PATH="${PYTHON_DIR}:${PATH}"
export PYTHONPATH="${REPO_ROOT}/python:${REPO_ROOT}/sgl-kernel-npu-sparse-kv/src${PYTHONPATH:+:${PYTHONPATH}}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EVIDENCE_DIR="${SGL_SPARSE_KV_EVIDENCE_DIR:-${TMPDIR:-/tmp}/sglang-sparse-kv-${MODE}-${STAMP}}"
mkdir -p "${EVIDENCE_DIR}"
ENV_LOG="${EVIDENCE_DIR}/environment.log"
TEST_LOG="${EVIDENCE_DIR}/pytest.log"
: >"${TEST_LOG}"
finish() {
  status=$?
  if [[ ! -s "${TEST_LOG}" ]]; then
    echo "pytest_not_started=preflight_failed" >"${TEST_LOG}"
  fi
  if (( status != 0 )); then
    echo "sparse-kv runner failed with status ${status}; inspect ${ENV_LOG} and ${TEST_LOG}" >&2
  fi
  echo "evidence=${EVIDENCE_DIR}"
}
trap finish EXIT

{
  echo "mode=${MODE}"
  echo "utc=${STAMP}"
  echo "repo=$(basename "${REPO_ROOT}")"
  echo "python_command=${PYTHON_BIN##*/}"
  uname -srm
  "${PYTHON_BIN}" --version
  for variable in ASCEND_HOME_PATH ASCEND_OPP_PATH LD_LIBRARY_PATH NPU_VISIBLE_DEVICES SOC_VERSION; do
    value="${!variable:-unset}"
    echo "${variable}=${value}"
  done
  if command -v npu-smi >/dev/null 2>&1; then
    echo "npu-smi=present"
  else
    echo "npu-smi=missing"
  fi
} >"${ENV_LOG}" 2>&1

if [[ "${MODE}" == "cpu" ]]; then
  cd "${REPO_ROOT}"
  "${PYTHON_BIN}" -m pytest -vv -rs \
    test/manual/ascend/sparse_kv/test_reference.py \
    test/manual/ascend/sparse_kv/test_plan_call_views.py \
    test/registered/unit/npu/test_sparse_kv_simt_config.py \
    test/registered/unit/npu/test_sparse_kv_simt_metadata.py \
    test/registered/unit/npu/test_modelslim_indexer_cache.py \
    sgl-kernel-npu-sparse-kv/tests \
    2>&1 | tee "${TEST_LOG}"
  exit 0
fi

"${PYTHON_BIN}" - <<'PY' >>"${ENV_LOG}" 2>&1
import torch
import torch_npu

print(f"torch={torch.__version__}")
print(f"torch_npu={torch_npu.__version__}")
print(f"npu_available={torch_npu.npu.is_available()}")
if not torch_npu.npu.is_available():
    raise SystemExit("Ascend NPU is required by the 950 runner")
device_name = torch_npu.npu.get_device_name(torch_npu.npu.current_device())
print(f"npu_device_name={device_name}")
if "950" not in device_name:
    raise SystemExit(f"Ascend 950 is required, detected {device_name}")
PY

OPP_ENV="${REPO_ROOT}/sgl-kernel-npu-sparse-kv/src/sgl_kernel_npu_sparse_kv/_cann_ops_custom/vendors/sgl_sparse_kv_transformer/bin/set_env.bash"
if [[ -f "${OPP_ENV}" ]]; then
  source_opp_environment "${OPP_ENV}"
fi
if ! "${PYTHON_BIN}" - <<'PY' >>"${ENV_LOG}" 2>&1
from importlib import metadata
import os
from pathlib import Path

import sgl_kernel_npu_sparse_kv

sgl_kernel_npu_sparse_kv.ensure_loaded()
package = Path(sgl_kernel_npu_sparse_kv.__file__).resolve()
expected = Path(os.environ["PYTHONPATH"].split(":")[1]).resolve()
if expected not in package.parents:
    raise SystemExit(f"extension came from another checkout: {package}")
try:
    print(f"sgl_kernel_npu_sparse_kv={metadata.version('sgl-kernel-npu-sparse-kv')}")
except metadata.PackageNotFoundError:
    print("sgl_kernel_npu_sparse_kv=source-tree")
PY
then
  (cd "${REPO_ROOT}/sgl-kernel-npu-sparse-kv" && SOC_VERSION=ascend950 bash build.sh) >>"${ENV_LOG}" 2>&1
  if [[ ! -f "${OPP_ENV}" ]]; then
    echo "built custom OPP did not produce ${OPP_ENV}" >>"${ENV_LOG}"
    exit 1
  fi
  source_opp_environment "${OPP_ENV}"
  "${PYTHON_BIN}" - <<'PY' >>"${ENV_LOG}" 2>&1
import sgl_kernel_npu_sparse_kv

sgl_kernel_npu_sparse_kv.ensure_loaded()
PY
fi

if [[ "${MODE}" == "full" || "${MODE}" == "host" || "${MODE}" == "graph" ]]; then
  "${PYTHON_BIN}" - <<'PY' >>"${ENV_LOG}" 2>&1
from importlib import metadata

from memfabric_hybrid import offload

required = ("OffloadConfig", "OFFLOAD_FLAG_GIANT_PAGE", "get_dva", "empty")
missing = [name for name in required if not hasattr(offload, name)]
if missing:
    raise SystemExit(f"real DDR DVA testing unsupported: memfabric_hybrid.offload missing {missing}")
print("host_provider=memfabric_hybrid.offload LOCAL giant-page/get_dva")
try:
    print(f"memfabric_hybrid={metadata.version('memfabric-hybrid')}")
except metadata.PackageNotFoundError:
    print("memfabric_hybrid=unknown")
PY
fi

for version_file in \
  "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/version.cfg" \
  "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/version.info"; do
  if [[ -f "${version_file}" ]]; then
    echo "cann_version_file=${version_file}" >>"${ENV_LOG}"
    cat "${version_file}" >>"${ENV_LOG}"
  fi
done

cd "${REPO_ROOT}"
export SGL_SPARSE_KV_TEST_MODE="${MODE}"
case "${MODE}" in
  quick)
    TESTS=(
      test/manual/ascend/sparse_kv/test_reference.py
      test/manual/ascend/sparse_kv/test_npu_plan.py
    )
    ;;
  full)
    TESTS=(
      test/manual/ascend/sparse_kv/test_reference.py
      test/manual/ascend/sparse_kv/test_npu_binding.py
      test/manual/ascend/sparse_kv/test_npu_plan.py
      test/manual/ascend/sparse_kv/test_npu_full_host.py
      test/manual/ascend/sparse_kv/test_npu_host_matrix.py
      test/manual/ascend/sparse_kv/test_npu_transfer.py
      test/manual/ascend/sparse_kv/test_npu_graph.py
      test/manual/ascend/sparse_kv/test_npu_attention.py
    )
    ;;
  host)
    TESTS=(
      test/manual/ascend/sparse_kv/test_npu_full_host.py
      test/manual/ascend/sparse_kv/test_npu_host_matrix.py
      test/manual/ascend/sparse_kv/test_npu_transfer.py
    )
    ;;
  graph)
    TESTS=(test/manual/ascend/sparse_kv/test_npu_graph.py)
    ;;
esac

"${PYTHON_BIN}" -m pytest -vv -rs "${TESTS[@]}" 2>&1 | tee "${TEST_LOG}"

if [[ "${MODE}" == "full" ]]; then
  "${PYTHON_BIN}" -m pytest -vv -rs \
    test/manual/ascend/sparse_kv/test_npu_runtime.py \
    2>&1 | tee -a "${TEST_LOG}"
fi
