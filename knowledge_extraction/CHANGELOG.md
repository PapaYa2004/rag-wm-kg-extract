# Changelog

Semantic versioning; the version lives in `kg_extract/__init__.py` and is written
into every output record, run file, `stats.json` and evaluation report.

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
