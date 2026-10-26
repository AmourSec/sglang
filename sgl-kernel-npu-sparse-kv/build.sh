#!/usr/bin/env bash
# Copyright (c) 2026 SGLang Team
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage: bash build.sh [--help]

Build the SparseKvPlan and SparseKvTransfer custom operators and Torch binding.

Environment:
  SOC_VERSION       Ascend 950 target passed to CANN opbuild (default: ascend950)
  ASCEND_HOME_PATH  CANN 9.1.0 toolkit root

The active python3 must provide torch 2.10.x and torch_npu 2.10.x.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi

SOC_VERSION=${SOC_VERSION:-ascend950}
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
export ASCEND_HOME_PATH

if [[ ${SOC_VERSION} != ascend950 ]]; then
    echo "ERROR: SOC_VERSION must be ascend950, got ${SOC_VERSION}" >&2
    exit 2
fi
if [[ $(uname -s) != Linux ]]; then
    echo "ERROR: Ascend custom operators must be built on Linux with CANN 9.1.0" >&2
    exit 2
fi
if [[ ! -d ${ASCEND_HOME_PATH} ]]; then
    echo "ERROR: CANN toolkit not found at ${ASCEND_HOME_PATH}" >&2
    exit 2
fi

python3 "${ROOT_DIR}/scripts/check_build_env.py"

cd "${ROOT_DIR}/cann_ops"
bash build.sh --pkg --ops="sparse_kv_plan;sparse_kv_transfer" --soc="${SOC_VERSION}" --vendor_name=sgl_sparse_kv

shopt -s nullglob
installers=("${ROOT_DIR}"/cann_ops/build/cann-ops-transformer*.run)
shopt -u nullglob
if (( ${#installers[@]} != 1 )); then
    echo "ERROR: expected one CANN installer, found ${#installers[@]}" >&2
    exit 1
fi

install_root="${ROOT_DIR}/src/sgl_kernel_npu_sparse_kv/_cann_ops_custom"
mkdir -p "${install_root}"
find "${install_root}" -mindepth 1 -maxdepth 1 -exec rm -rf -- {} +
"${installers[0]}" --install-path="${install_root}"

cd "${ROOT_DIR}"
python3 setup.py build_ext --inplace
