#!/bin/bash
#SBATCH --job-name=kgx-merge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=20G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%j.out
# Aggregate shards -> raw graph (saved first) -> KGGen clustering -> clustered graph + stats.
# Clustering LLM calls are cached in $DSPY_CACHEDIR, so a requeued merge replays
# finished calls from cache instead of recomputing them.
# Required env: OUT_DIR. Optional: MERGE_ARGS (e.g. "--no-cluster" or "--allow-missing").

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_offline
kgx_gpu_info
kgx_client_env
kgx_start_vllm

kgx_run_requeueable python -m kg_extract.merge \
  --out-dir "${OUT_DIR}" --kb "${KB_JSONL}" \
  --model "${KG_MODEL}" --api-base "${KG_API_BASE}" --temperature "${KG_TEMPERATURE}" \
  --retrieval-model "${KG_RETRIEVAL_MODEL}" ${MERGE_ARGS:-}
