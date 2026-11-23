#!/bin/bash
#SBATCH --job-name=ragwm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=20G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=%x_%j.out
# One RAG-WM step (README steps 3-6) with the unmodified RAG-WM code. Required env: STEP
#   vectorstore | entity | doc | inject | verify
# The LLM is a local vLLM server (Qwen2.5-7B-Instruct) answering to the name RAG-WM asks for.
# Every step resumes from RAG-WM's own checkpoints when re-run or requeued.

source "${SLURM_SUBMIT_DIR}/slurm/common.sh"
kgx_offline
kgx_gpu_info
kgx_load_python
# shellcheck disable=SC1091
source "${VENV_RAGWM}/bin/activate"
export RAGWM_DIR RAGWM_DATASETS_DIR
export RAGWM_CHROMA_PATH="${CHROMA_DIR}"
mkdir -p "${CHROMA_DIR}" "${RAGWM_OUTPUT}/wm_prepare"

# BEIR folder in the layout RAG-WM's loader expects (corpus.jsonl = the ORIGINAL corpus).
D="${RAGWM_DATASETS_DIR}/${DATASET}"
if [[ ! -s "${D}/corpus.jsonl" ]]; then
  mkdir -p "${D}"
  cp "${DATASETS_DIR}/${DATASET}/${DATASET}_original.jsonl" "${D}/corpus.jsonl"
  cp "${DATASETS_DIR}/${DATASET}/queries.jsonl" "${D}/"
  cp -r "${DATASETS_DIR}/${DATASET}/qrels" "${D}/"
fi

if [[ "${STEP}" != "vectorstore" ]]; then
  VLLM_MODEL="${RAGWM_HF_MODEL}" \
  VLLM_SERVED="${RAGWM_SERVED_NAME} ${RAGWM_HF_MODEL}" \
  VLLM_EXTRA="--enable-auto-tool-choice --tool-call-parser hermes" \
    kgx_start_vllm
  # RAG-WM reads model_configs/gpt3.5_config.json (git-ignored). Write it for the local server;
  # refuse to overwrite a config we did not write (it might hold a real API key).
  python - <<PY
import json, os
p = os.path.join("${RAGWM_DIR}", "model_configs", "gpt3.5_config.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
if os.path.exists(p) and "_written_by_kgx" not in open(p).read():
    raise SystemExit(f"{p} exists and was not written by this script; move it away first")
cfg = {"_written_by_kgx": "local vLLM for RAG-WM on OzSTAR",
       "model_info": {"provider": "gpt", "name": "${RAGWM_SERVED_NAME}"},
       "api_key_info": {"api_keys": ["EMPTY"], "api_base": ["${KG_API_BASE}"], "api_key_use": 0},
       "params": {"temperature": 0.01, "seed": 100, "gpus": [], "max_output_tokens": 400}}
json.dump(cfg, open(p, "w"), indent=2)
print("[ragwm] wrote", p)
PY
fi

L=(python "${KE_DIR}/ragwm_run/launch.py")
COMMON=(--eval_dataset "${DATASET}" --eval_model_code contriever --score_function cosine)
MAIN=(src/main.py "${COMMON[@]}" --mutual_times "${MUTUAL_TIMES}" --basepath "${RAGWM_OUTPUT}")
echo "[ragwm] step=${STEP}"

case "${STEP}" in
  vectorstore)
    "${L[@]}" rag/vectorstore.py "${COMMON[@]}" ;;
  entity)
    kgx_run_requeueable "${L[@]}" entity_generate/generate_entity_llm_check.py \
      --eval_dataset "${DATASET}" --dataset_prob 1 --basepath "${RAGWM_OUTPUT}/wm_prepare" \
      --model_config_path model_configs/gpt3.5_config.json ;;
  doc)
    # Watermark tuples: generated ONCE (fresh random secret key each time it runs).
    if [[ ! -s "${RAGWM_OUTPUT}/wm_prepare/${DATASET}/wmunit.json" ]]; then
      "${L[@]}" entity_generate/generate_hash_entity.py --eval_dataset "${DATASET}" -t scratch \
        --entity_num "${RAGWM_ENTITY_NUM}" --edge_prob "${RAGWM_EDGE_PROB}" \
        --basepath "${RAGWM_OUTPUT}/wm_prepare"
    fi
    kgx_run_requeueable "${L[@]}" "${MAIN[@]}" --doc 1 --inject 0 --verify 0 --stat 0 ;;
  inject)
    "${L[@]}" "${MAIN[@]}" --doc 0 --inject 1 --verify 0 --stat 0 ;;
  verify)
    kgx_run_requeueable "${L[@]}" "${MAIN[@]}" --doc 0 --inject 0 --verify 1 --stat 1 ;;
  *)
    echo "unknown STEP=${STEP}" >&2; exit 2 ;;
esac
echo "[ragwm] step ${STEP} done"
