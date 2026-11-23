# Changelog

Semantic versioning; the version lives in `kg_extract/__init__.py` and is written
into every output record, run file, `stats.json` and evaluation report.

## 0.3.2 (2026-11-23)
- setup.sh: a gated/unavailable extractor model no longer aborts setup (RAG-WM's Qwen is
  downloaded first); re-run setup once access is granted.

## 0.3.1 (2026-11-23)
- setup.sh: timestamps per stage and visible pip progress (no more silent `-q`).

## 0.3.0 (2026-11-23)
- Generate the watermarked KB on OzSTAR: `slurm/submit.sh ragwm` runs the unmodified
  RAG-WM pipeline (vectorstore -> entity -> tuples + watermark texts -> inject -> verify)
  as a resumable Slurm chain, then builds `datasets/<ds>/<ds>_watermarked.jsonl`.
- LLM for RAG-WM: local Qwen2.5-7B-Instruct on vLLM, served as `gpt-3.5-turbo` (the name
  RAG-WM requests); config written per job to the git-ignored `rag-wm/model_configs/`.
- `ragwm_run/launch.py`: redirects RAG-WM's hard-coded Chroma path in memory and stubs
  unused providers (fastchat, google.generativeai); RAG-WM files are not modified.
- `kgx_start_vllm` accepts VLLM_MODEL / VLLM_SERVED / VLLM_EXTRA.
- CHROMA_DIR / RAGWM_OUTPUT now default to `$WORK/ragwm/...` (no manual edit).

## 0.2.0 (2026-11-23)
- Dataset layout: `datasets/<ds>/<ds>_original.jsonl` + `<ds>_watermarked.jsonl`
  (built by `kg_eval.build_corpus_variants`, optionally verified against the Chroma
  store); eval-only ground truth in `datasets/_ground_truth/<ds>/`.
- KB source switch `KB_SOURCE=beir|chroma`.
- Guard: forbids `_ground_truth/` paths and the clean `*_original.jsonl` during a
  watermarked run (diffing the corpora would leak the watermarks).
- Evaluation module (`kg_eval.evaluate`): exact/fuzzy tuple recall (raw vs
  clustered), injected-passage coverage, cluster-merge check; accepts both id spaces.
- Merge: `kg_gen_source_sha256` in stats; run files keyed by Slurm restart count.
- `check_kggen`: live SemHash call; pin `semhash<0.4` (kg-gen main uses
  `DeduplicationResult.duplicates`, removed in semhash 0.4).
- Slurm: setup, datasets, export, extract (array, self-requeue), merge, evaluate,
  report wrappers; offline end-to-end test with a fake OpenAI-compatible server.

## 0.1.0 (2026-11-23)
- Black-box guard, Chroma/BEIR export, per-passage KGGen extraction with shard
  checkpoints and parse-failure counting, aggregate + two-stage clustering with
  surface-form provenance, pilot report and estimate.
