#!/bin/bash
# Submit the Knowledge Extraction pipeline as a chain of Slurm jobs.
# Run from anywhere on an NT login node (tooarrana1/2):
#
#   slurm/submit.sh ragwm [--from STEP]                      # run RAG-WM (local LLM) then build the datasets
#   slurm/submit.sh datasets                                 # build datasets/<ds>/<ds>_watermarked.jsonl from RAG-WM outputs
#   slurm/submit.sh export                                   # (re)export the KB only
#   slurm/submit.sh pilot                                    # export (if needed) -> 1 shard on PILOT_SAMPLE docs -> merge
#   slurm/submit.sh full --shards N [--time HH:MM:SS] [--max-parallel M]
#   slurm/submit.sh merge <out_dir> [--time HH:MM:SS] [-- extra merge args]
#   slurm/submit.sh eval  <out_dir> [-- extra evaluate args]
#
# Re-running the same `pilot` / `full` command resumes: extraction skips doc_ids
# that already have a record in <out_dir>/shards/.
set -euo pipefail
KE_DIR_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${KE_DIR_HERE}"
# shellcheck disable=SC1091
source slurm/config.env
mkdir -p "${LOG_DIR}"

ACC=()
[[ -n "${SLURM_ACCOUNT:-}" ]] && ACC=(--account="${SLURM_ACCOUNT}")
sb() { sbatch --parsable "${ACC[@]}" "$@" | cut -d';' -f1; }

submit_export_if_needed() {
  if [[ -s "${KB_JSONL}" && "${1:-}" != "force" ]]; then
    echo ""   # no dependency
  else
    sb --output="${LOG_DIR}/%x_%j.out" slurm/export_kb.sbatch
  fi
}

chain() {  # $1 out_dir  $2 num_shards  $3 sample(or "")  $4 time  $5 max_parallel
  local out="$1" shards="$2" sample="$3" time="$4" maxpar="$5"
  local dep_export; dep_export=$(submit_export_if_needed)
  local dep=(); [[ -n "${dep_export}" ]] && dep=(--dependency=afterok:"${dep_export}")
  local arr="0-$((shards - 1))"; [[ -n "${maxpar}" ]] && arr="${arr}%${maxpar}"
  local ex
  ex=$(sb "${dep[@]}" --array="${arr}" --time="${time}" \
        --output="${LOG_DIR}/%x_%A_%a.out" \
        --export=ALL,OUT_DIR="${out}",NUM_SHARDS="${shards}",SAMPLE="${sample}" \
        slurm/extract.sbatch)
  local mg
  mg=$(sb --dependency=afterok:"${ex}" --kill-on-invalid-dep=yes \
        --output="${LOG_DIR}/%x_%j.out" --export=ALL,OUT_DIR="${out}" slurm/merge.sbatch)
  echo "export: ${dep_export:-skipped (KB exists)}  extract array: ${ex} [${arr}]  merge: ${mg}"
  echo "out_dir: ${out}"
  echo "watch:   squeue -u \$USER ;  tail -f ${LOG_DIR}/kgx-extract_${ex}_0.out"
}

cmd="${1:-}"; shift || true
case "${cmd}" in
  ragwm)
    # RAG-WM chain: vectorstore -> entity -> doc -> inject -> verify -> datasets
    # `submit.sh ragwm --from doc` restarts from a step (finished steps resume from checkpoints anyway).
    from="vectorstore"; [[ "${1:-}" == "--from" ]] && from="$2"
    steps=(vectorstore entity doc inject verify)
    declare -A tlim=([vectorstore]=01:00:00 [entity]=08:00:00 [doc]=12:00:00 [inject]=01:00:00 [verify]=02:00:00)
    started=0; dep=(); ids=""
    for s in "${steps[@]}"; do
      [[ "${s}" == "${from}" ]] && started=1
      [[ ${started} -eq 0 ]] && continue
      j=$(sb "${dep[@]}" --job-name="ragwm-${s}" --time="${tlim[$s]}" \
            --output="${LOG_DIR}/%x_%j.out" --export=ALL,STEP="${s}" slurm/ragwm.sbatch)
      dep=(--dependency=afterok:"${j}" --kill-on-invalid-dep=yes); ids="${ids} ${s}=${j}"
    done
    [[ ${started} -eq 0 ]] && { echo "unknown step ${from}" >&2; exit 2; }
    j=$(sb "${dep[@]}" --output="${LOG_DIR}/%x_%j.out" slurm/build_datasets.sbatch)
    echo "jobs:${ids} datasets=${j}"
    echo "logs: ${LOG_DIR}/ragwm-<step>_<jobid>.out (+ vllm_<jobid>.log)" ;;
  datasets)
    j=$(sb --output="${LOG_DIR}/%x_%j.out" slurm/build_datasets.sbatch); echo "datasets job: ${j}" ;;
  export)
    j=$(submit_export_if_needed force); echo "export job: ${j}" ;;
  pilot)
    chain "${OUT_ROOT}/${DATASET}_pilot" 1 "${PILOT_SAMPLE}" "01:00:00" "" ;;
  full)
    shards=""; time="02:00:00"; maxpar=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --shards) shards="$2"; shift 2 ;;
        --time) time="$2"; shift 2 ;;
        --max-parallel) maxpar="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
      esac
    done
    [[ -z "${shards}" ]] && { echo "--shards N is required (see the pilot estimate)" >&2; exit 2; }
    chain "${OUT_ROOT}/${DATASET}" "${shards}" "" "${time}" "${maxpar}" ;;
  merge)
    out="$1"; shift; time="04:00:00"; extra=""
    while [[ $# -gt 0 ]]; do
      case "$1" in --time) time="$2"; shift 2 ;; --) shift; extra="$*"; break ;; *) echo "unknown arg $1" >&2; exit 2 ;; esac
    done
    j=$(sb --time="${time}" --output="${LOG_DIR}/%x_%j.out" \
          --export=ALL,OUT_DIR="${out}",MERGE_ARGS="${extra}" slurm/merge.sbatch)
    echo "merge job: ${j}" ;;
  eval)
    out="$1"; shift; extra=""
    [[ "${1:-}" == "--" ]] && { shift; extra="$*"; }
    j=$(sb --output="${LOG_DIR}/%x_%j.out" --export=ALL,OUT_DIR="${out}",EVAL_ARGS="${extra}" slurm/evaluate.sbatch)
    echo "eval job: ${j}" ;;
  *)
    sed -n '2,15p' "$0"; exit 2 ;;
esac
