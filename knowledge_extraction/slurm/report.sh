#!/bin/bash
# Print 20 sample triplets, stats.json and the full-run estimate for a finished run.
# Light (reads small JSON files): fine on a login node.
#   bash slurm/report.sh                       # pilot dir
#   bash slurm/report.sh /fred/.../nfcorpus    # any out dir
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source slurm/config.env
module load "${PY_MODULE}"
# shellcheck disable=SC1091
source "${VENV_CLIENT}/bin/activate"
export PYTHONPATH="${KE_DIR}:${KGGEN_SRC}"
OUT="${1:-${OUT_ROOT}/${DATASET}_pilot}"
N_FULL=$(wc -l < "${KB_JSONL}")
python -m kg_extract.report samples --out-dir "${OUT}" --n 20
echo
python -m kg_extract.report estimate --out-dir "${OUT}" --full-docs "${N_FULL}"
