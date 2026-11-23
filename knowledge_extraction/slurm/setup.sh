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
export HF_HOME NLTK_DATA APPTAINER_CACHEDIR TIKTOKEN_CACHE_DIR
export APPTAINER_TMPDIR="${APPTAINER_CACHEDIR}/tmp"
mkdir -p "${WORK}/venvs" "${WORK}/containers" "${HF_HOME}" "${NLTK_DATA}" "${APPTAINER_TMPDIR}" \
         "${OUT_ROOT}/kb" "${LOG_DIR}" "${DSPY_CACHEDIR}"

module load ${PY_MODULE}   # may list several modules
unset PYTHONPATH            # toolchain numpy/scipy must not satisfy pip installs into the venvs
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "PY_MODULE must provide Python >= 3.10" >&2; exit 2; }

echo "== $(date +%H:%M:%S) 1/5 export venv (chromadb==0.5.20) =="
if [[ ! -x "${VENV_EXPORT}/bin/python" ]]; then
  python3 -m venv "${VENV_EXPORT}"
fi
"${VENV_EXPORT}/bin/pip" install --progress-bar off --upgrade pip
"${VENV_EXPORT}/bin/pip" install --progress-bar off -r requirements-export.txt

echo "== $(date +%H:%M:%S) 2/5 client venv (kg-gen deps, dspy, sentence-transformers) =="
if [[ ! -x "${VENV_CLIENT}/bin/python" ]]; then
  python3 -m venv "${VENV_CLIENT}"
fi
"${VENV_CLIENT}/bin/pip" install --progress-bar off --upgrade pip
"${VENV_CLIENT}/bin/pip" install --progress-bar off -r requirements-client.txt
"${VENV_CLIENT}/bin/pip" freeze > "${OUT_ROOT}/requirements-client.lock.txt"

echo "== $(date +%H:%M:%S) 2b/5 RAG-WM venv (to generate the watermarked KB with a local LLM) =="
if [[ ! -x "${VENV_RAGWM}/bin/python" ]]; then
  python3 -m venv "${VENV_RAGWM}"
fi
"${VENV_RAGWM}/bin/pip" install --progress-bar off --upgrade pip
"${VENV_RAGWM}/bin/pip" install --progress-bar off -r requirements-ragwm.txt
"${VENV_RAGWM}/bin/pip" freeze > "${OUT_ROOT}/requirements-ragwm.lock.txt"
mkdir -p "${TIKTOKEN_CACHE_DIR}"
TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR}" "${VENV_RAGWM}/bin/python" -c \
  "import tiktoken; [tiktoken.get_encoding(e) for e in ('cl100k_base', 'o200k_base')]; print('tiktoken OK')"

echo "== $(date +%H:%M:%S) 3/5 vLLM container =="
module load apptainer
if [[ ! -s "${VLLM_SIF}" ]]; then
  apptainer pull "${VLLM_SIF}" "${VLLM_IMAGE}"
fi
ls -lh "${VLLM_SIF}"

echo "== $(date +%H:%M:%S) 4/5 model weights + NLTK data into ${HF_HOME} =="
"${VENV_CLIENT}/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download
tok = os.environ.get("HF_TOKEN") or None
pat = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
snapshot_download("${RAGWM_HF_MODEL}", allow_patterns=pat)          # RAG-WM's LLM (not gated)
try:                                                                 # extractor LLM (gated)
    snapshot_download("${HF_MODEL}", token=tok, allow_patterns=pat)
    print("extractor model ${HF_MODEL}: OK")
except Exception as e:  # GatedRepoError / 401 / 403 while access is pending, or no token
    print(f"WARNING: could not download extractor model ${HF_MODEL} ({type(e).__name__}).")
    print("  RAG-WM (submit.sh ragwm) does not need it. Once your access is ACCEPTED on")
    print("  huggingface.co/settings/gated-repos, export HF_TOKEN and re-run: bash slurm/setup.sh")
for m in {"${KG_RETRIEVAL_MODEL}", "${KG_EVAL_EMBED_MODEL}", "${SEMHASH_MODEL}", "facebook/contriever"}:
    snapshot_download(m)
import nltk
for r in ("punkt", "punkt_tab"):
    nltk.download(r, download_dir="${NLTK_DATA}", quiet=True)
print("downloads OK")
PY

echo "== $(date +%H:%M:%S) 5/5 kg-gen API check (vendored source at ${KGGEN_SRC}) =="
PYTHONPATH="${KE_DIR}:${KGGEN_SRC}" HF_HUB_OFFLINE=1 "${VENV_CLIENT}/bin/python" -m kg_extract.check_kggen \
  | tee "${OUT_ROOT}/kggen_check.txt"
echo "setup done"
