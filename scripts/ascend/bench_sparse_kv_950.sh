#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
export PYTHONPATH="${repo_root}/test/manual/ascend:${repo_root}/sgl-kernel-npu-sparse-kv/src${PYTHONPATH:+:${PYTHONPATH}}"

dry_run=0
show_help=0
result_dir_override=""
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  arg="${args[index]}"
  [[ "$arg" == "--dry-run" ]] && dry_run=1
  [[ "$arg" == "--help" || "$arg" == "-h" ]] && show_help=1
  if [[ "$arg" == "--output-dir" && $((index + 1)) -lt ${#args[@]} ]]; then
    result_dir_override="${args[index + 1]}"
  elif [[ "$arg" == --output-dir=* ]]; then
    result_dir_override="${arg#--output-dir=}"
  fi
done
if (( show_help )); then
  exec "${python_bin}" -m sparse_kv.perf --help
fi
if (( dry_run )); then
  exec "${python_bin}" -m sparse_kv.perf "$@"
fi

export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
opp_env="${repo_root}/sgl-kernel-npu-sparse-kv/src/sgl_kernel_npu_sparse_kv/_cann_ops_custom/vendors/sgl_sparse_kv_transformer/bin/set_env.bash"
if [[ ! -f "${opp_env}" ]]; then
  echo "missing custom OPP environment: ${opp_env}; build/install the extension first" >&2
  exit 1
fi
source "${opp_env}"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="${SGL_SPARSE_KV_BENCH_DIR:-${TMPDIR:-/tmp}/sglang-sparse-kv-bench-${stamp}-$$}"
mkdir -p "$(dirname "${run_root}")"
mkdir "${run_root}"
result_dir="${result_dir_override:-${run_root}/results}"
log_file="${run_root}/benchmark.log"
set +e
if [[ -n "${result_dir_override}" ]]; then
  "${python_bin}" -m sparse_kv.perf "$@" 2>&1 | tee "${log_file}"
else
  "${python_bin}" -m sparse_kv.perf --output-dir "${result_dir}" "$@" 2>&1 | tee "${log_file}"
fi
pipeline_status=("${PIPESTATUS[@]}")
set -e
python_status="${pipeline_status[0]}"
tee_status="${pipeline_status[1]}"
printf '%s\n' "${python_status}" > "${run_root}/process_exit_code"
printf '%s\n' "${tee_status}" > "${run_root}/tee_exit_code"
runner_status="${python_status}"
if (( runner_status == 0 && tee_status != 0 )); then
  runner_status="${tee_status}"
fi
printf '%s\n' "${runner_status}" > "${run_root}/runner_exit_code"
printf 'results=%s\nlog=%s\nprocess_exit_code=%s\ntee_exit_code=%s\nrunner_exit_code=%s\n' \
  "${result_dir}" "${log_file}" "${python_status}" "${tee_status}" "${runner_status}"
if (( python_status != 0 )); then
  exit "${python_status}"
fi
if (( tee_status != 0 )); then
  exit "${tee_status}"
fi
