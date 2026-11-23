#!/bin/bash
#SBATCH --job-name=kgx-datasets
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --tmp=20G
#SBATCH --output=%x_%j.out
# Build datasets/<ds>/<ds>_watermarked.jsonl (+ eval-only ground truth) from RAG-WM's
# injection log, then verify it passage by passage against the real Chroma store.
# This job reads ground truth on purpose: it is a data-preparation (defender) step, not extraction.

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_load_python
INJECT_JSON="${RAGWM_OUTPUT}/wm_generate/${DATASET}/${MUTUAL_TIMES}/wmuint_inject.json"
ORIG="${DATASETS_DIR}/${DATASET}/${DATASET}_original.jsonl"

VERIFY=()
if [[ -d "${CHROMA_DIR}" ]]; then
  source "${VENV_EXPORT}/bin/activate"
  COPY="${JOBFS:-/tmp/${USER}_${SLURM_JOB_ID}}/chroma_copy"
  mkdir -p "${COPY}" "$(dirname "${CHROMA_KB_JSONL}")"
  cp -a "${CHROMA_DIR}/." "${COPY}/"
  python -m kg_extract.export_kb --chroma-path "${COPY}" --collection "${COLLECTION}" --out "${CHROMA_KB_JSONL}"
  deactivate
  VERIFY=(--verify-kb "${CHROMA_KB_JSONL}")
else
  echo "[datasets] CHROMA_DIR not found: building without verification against the store"
fi

source "${VENV_CLIENT}/bin/activate"
python -m kg_eval.build_corpus_variants --dataset "${DATASET}" \
  --beir-corpus "${ORIG}" --inject-json "${INJECT_JSON}" \
  --datasets-dir "${DATASETS_DIR}" "${VERIFY[@]}"
