#!/bin/bash
#SBATCH --job-name=kgx-export
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --tmp=20G
#SBATCH --output=%x_%j.out
# Export the watermarked KB as black-box (doc_id, text) records into $KB_JSONL.
#   KB_SOURCE=beir   -> from datasets/<ds>/<ds>_watermarked.jsonl
#   KB_SOURCE=chroma -> from a COPY of the Chroma store in $JOBFS (original never opened/migrated)

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_load_python
source "${VENV_EXPORT}/bin/activate"
mkdir -p "$(dirname "${KB_JSONL}")"

if [[ "${KB_SOURCE}" == "beir" ]]; then
  [[ -s "${WM_CORPUS}" ]] || { echo "missing ${WM_CORPUS}: run 'slurm/submit.sh datasets' first" >&2; exit 2; }
  python -m kg_extract.export_kb --beir-corpus "${WM_CORPUS}" --out "${KB_JSONL}"
else
  COPY="${JOBFS:-/tmp/${USER}_${SLURM_JOB_ID}}/chroma_copy"
  mkdir -p "${COPY}"
  cp -a "${CHROMA_DIR}/." "${COPY}/"
  python -m kg_extract.export_kb --chroma-path "${COPY}" --collection "${COLLECTION}" --out "${KB_JSONL}"
fi
