#!/bin/bash
#SBATCH --job-name=kgx-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=20G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%A_%a.out
# One array task = one shard = one vLLM server on one A100.
# Required env (set by submit.sh): OUT_DIR, NUM_SHARDS. Optional: SAMPLE.

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_offline
kgx_gpu_info
kgx_client_env
kgx_start_vllm

kgx_run_requeueable python -m kg_extract.extract \
  --kb "${KB_JSONL}" --dataset "${DATASET}" --out-dir "${OUT_DIR}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "${NUM_SHARDS}" \
  ${SAMPLE:+--sample "${SAMPLE}"} \
  --model "${KG_MODEL}" --api-base "${KG_API_BASE}" \
  --temperature "${KG_TEMPERATURE}" --concurrency "${CONCURRENCY}"
