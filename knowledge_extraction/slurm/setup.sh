#!/bin/bash
# One-time setup on an NT login node (tooarrana1/2). Light work only: creates
# venvs, pulls the vLLM container, downloads model weights (network I/O).
# No GPU / heavy compute here. Safe to re-run: finished steps are skipped.
#
#   export HF_TOKEN=hf_xxx        # needed once for the gated Llama-3.1 weights; not written to disk by this script
#   bash slurm/setup.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f slurm/config.env ]] || { echo "cp slurm/config.env.example slurm/config.env and edit it first" >&2; exit 2; }
# shellcheck disable=SC1091
source slurm/config.env
export HF_HOME NLTK_DATA APPTAINER_CACHEDIR
export APPTAINER_TMPDIR="${APPTAINER_CACHEDIR}/tmp"
mkdir -p "${WORK}/venvs" "${WORK}/containers" "${HF_HOME}" "${NLTK_DATA}" "${APPTAINER_TMPDIR}" \
         "${OUT_ROOT}/kb" "${LOG_DIR}" "${DSPY_CACHEDIR}"

module load "${PY_MODULE}"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "PY_MODULE must provide Python >= 3.10" >&2; exit 2; }

echo "== 1/5 export venv (chromadb==0.5.20) =="
if [[ ! -x "${VENV_EXPORT}/bin/python" ]]; then
  python3 -m venv "${VENV_EXPORT}"
fi
"${VENV_EXPORT}/bin/pip" install -q --upgrade pip
"${VENV_EXPORT}/bin/pip" install -q -r requirements-export.txt

echo "== 2/5 client venv (kg-gen deps, dspy, sentence-transformers) =="
if [[ ! -x "${VENV_CLIENT}/bin/python" ]]; then
  python3 -m venv "${VENV_CLIENT}"
fi
"${VENV_CLIENT}/bin/pip" install -q --upgrade pip
"${VENV_CLIENT}/bin/pip" install -q -r requirements-client.txt
"${VENV_CLIENT}/bin/pip" freeze > "${OUT_ROOT}/requirements-client.lock.txt"

echo "== 3/5 vLLM container =="
module load apptainer
if [[ ! -s "${VLLM_SIF}" ]]; then
  apptainer pull "${VLLM_SIF}" "${VLLM_IMAGE}"
fi
ls -lh "${VLLM_SIF}"

echo "== 4/5 model weights + NLTK data into ${HF_HOME} =="
"${VENV_CLIENT}/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN")
snapshot_download("${HF_MODEL}", token=tok,
                  allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"])
for m in {"${KG_RETRIEVAL_MODEL}", "${KG_EVAL_EMBED_MODEL}", "${SEMHASH_MODEL}"}:
    snapshot_download(m)
import nltk
for r in ("punkt", "punkt_tab"):
    nltk.download(r, download_dir="${NLTK_DATA}", quiet=True)
print("downloads OK")
PY

echo "== 5/5 kg-gen API check (vendored source at ${KGGEN_SRC}) =="
PYTHONPATH="${KE_DIR}:${KGGEN_SRC}" HF_HUB_OFFLINE=1 "${VENV_CLIENT}/bin/python" -m kg_extract.check_kggen \
  | tee "${OUT_ROOT}/kggen_check.txt"
echo "setup done"
