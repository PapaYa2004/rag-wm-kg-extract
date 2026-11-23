# Sourced by every job script. Bash only.
set -euo pipefail

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SLURM_DIR}/config.env" ]]; then
  echo "Missing ${SLURM_DIR}/config.env (copy config.env.example and edit it)" >&2
  exit 2
fi
# shellcheck disable=SC1091
source "${SLURM_DIR}/config.env"

export HF_HOME NLTK_DATA DSPY_CACHEDIR APPTAINER_CACHEDIR KG_MODEL KG_RETRIEVAL_MODEL KG_EVAL_EMBED_MODEL
export PYTHONPATH="${KE_DIR}:${KGGEN_SRC}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0            # set iteration order -> reproducible SemHash / KMeans prompts
export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_DIR}" "${DSPY_CACHEDIR}"

kgx_offline() {                    # compute nodes: use only pre-downloaded models
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
}

kgx_client_env() {
  module load "${PY_MODULE}"
  # shellcheck disable=SC1091
  source "${VENV_CLIENT}/bin/activate"
}

kgx_gpu_info() {
  export JOB_T0="${JOB_T0:-$(date +%s.%N)}"
  export N_GPUS
  N_GPUS=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES:-0}")
  export GPU_NAME
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)
  echo "[job] host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-none} (${GPU_NAME})"
}

kgx_free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()'
}

# Start vLLM (OpenAI-compatible) on this node's GPU and wait until it serves the model.
kgx_start_vllm() {
  module load apptainer
  local port; port=$(kgx_free_port)
  local tag="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
  VLLM_LOG="${LOG_DIR}/vllm_${tag}.log"
  local scratch="${JOBFS:-/tmp/${USER}_${tag}}"
  mkdir -p "${scratch}/vllm_cache" "${scratch}/xdg"
  export KG_API_BASE="http://127.0.0.1:${port}/v1"

  local t0; t0=$(date +%s)
  # NB: Apptainer passes the host environment through, incl. HF_HOME / HF_HUB_OFFLINE.
  VLLM_CACHE_ROOT="${scratch}/vllm_cache" XDG_CACHE_HOME="${scratch}/xdg" \
  apptainer exec --nv -B /fred "${VLLM_SIF}" \
    vllm serve "${HF_MODEL}" \
      --host 127.0.0.1 --port "${port}" \
      --served-model-name "${HF_MODEL}" \
      --dtype bfloat16 --seed 0 \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
      > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  trap kgx_stop_vllm EXIT

  echo "[job] waiting for vLLM on port ${port} (log: ${VLLM_LOG})"
  for _ in $(seq 1 180); do
    if curl -sf "${KG_API_BASE}/models" > /dev/null; then
      export VLLM_READY_S=$(( $(date +%s) - t0 ))
      echo "[job] vLLM ready after ${VLLM_READY_S}s"
      return 0
    fi
    if ! kill -0 "${VLLM_PID}" 2> /dev/null; then
      echo "[job] vLLM exited during start-up; tail of log:" >&2
      tail -n 50 "${VLLM_LOG}" >&2
      return 1
    fi
    sleep 5
  done
  echo "[job] vLLM not ready after 15 min" >&2
  tail -n 50 "${VLLM_LOG}" >&2
  return 1
}

kgx_stop_vllm() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2> /dev/null; then
    kill "${VLLM_PID}" 2> /dev/null || true
    wait "${VLLM_PID}" 2> /dev/null || true
  fi
}

# Run a python command in the background so the USR1 signal Slurm sends before
# the time limit (--signal=B:USR1@600) can requeue the job; progress is
# checkpointed per doc, so the requeued job resumes where it stopped.
kgx_run_requeueable() {
  local requeue_id="${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
  requeue_id="${requeue_id:-${SLURM_JOB_ID:-}}"
  "$@" &
  local pid=$!
  trap 'echo "[job] time limit near: requeueing ${requeue_id}"; kill -TERM '"${pid}"' 2>/dev/null; scontrol requeue '"${requeue_id}"' || echo "[job] requeue refused; resubmit manually"' USR1
  local rc=0
  wait "${pid}" || rc=$?
  # wait returns >128 when interrupted by the trap; wait again for the real exit code
  if kill -0 "${pid}" 2> /dev/null; then wait "${pid}" || rc=$?; fi
  return "${rc}"
}
