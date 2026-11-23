#!/bin/bash
#SBATCH --job-name=kgx-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
# Ground-truth evaluation. Runs only after extraction + merge, in its own process.
# Required env: OUT_DIR. Optional: EVAL_ARGS (e.g. "--tau-e 0.8 --tau-r 0.6").

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_offline
kgx_client_env
# Ground truth must be in the same doc-id space as the extracted KB.
if [[ "${KB_SOURCE}" == "beir" ]]; then
  GT="${GT_BEIR}"
else
  GT="${RAGWM_OUTPUT}/wm_generate/${DATASET}/${MUTUAL_TIMES}/wmuint_inject.json"
fi
python -m kg_eval.evaluate --out-dir "${OUT_DIR}" --gt "${GT}" \
  --embed-model "${KG_EVAL_EMBED_MODEL}" ${EVAL_ARGS:-}
