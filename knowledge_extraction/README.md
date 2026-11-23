# Knowledge Extraction stage (KGGen on a RAG-WM-watermarked KB)

Builds an entity-relation knowledge graph from a watermarked knowledge base with
[KGGen](https://github.com/stair-lab/kg-gen) (Mo et al., NeurIPS 2025). The output feeds
Step 2 (Relation Selection). Version: see `kg_extract/__init__.py` and `CHANGELOG.md`.

## Black-box design

The attacker doesn't know which passages are watermarked or what the tuples are.

| package | may read | must not read |
|---|---|---|
| `kg_extract` (export, extract, merge, report) | `doc_id`, passage `text` | RAG-WM tuple files, injection logs, hash keys, Chroma metadata (`change` flag), `datasets/_ground_truth/`, the clean `*_original.jsonl` (during a watermarked run) |
| `kg_eval` (build datasets, evaluate) | everything | runs only as its own process, after extraction |

How it's enforced in code (`kg_extract/guard.py`):
- Every extraction entrypoint installs a `sys.audit` hook. Opening a ground-truth path raises `BlackBoxViolation`.
- KB records may carry only `doc_id` and `text`.
- The Chroma export requests `include=["documents"]` only and checks that no metadata came back.
- `kg_eval` refuses to import inside an extraction process.
- Tests check all of this: `tests/test_blackbox.py`.

## Things that differ from the plan (read before running)

1. **kg-gen version.** `pip install kg-gen` gives 0.4.0 (PyPI), which is older than the checkout in `../kg-gen` (GitHub main).
   - In 0.4.0, `cluster()` puts *every entity of the graph* into one prompt as a `Literal`. On a corpus-level graph that overflows an 8B model's context.
   - We therefore use the vendored checkout via `PYTHONPATH` (`../kg-gen/src`). `check_kggen` verifies its API, and its source hash is recorded in `stats.json`.
2. **`context` in `generate()` does nothing** in that checkout: it isn't passed to the relation prompt, and `deduplicate()` ignores it (`# TODO`). We call the same step functions `generate()` calls (`get_entities`, `get_relations`) and pass `context=` to the relation step. Per-passage SemHash dedup is skipped, so surface forms stay raw.
3. **`kg.cluster()` is a deprecated alias** for `deduplicate(method=FULL)`, i.e. SemHash followed by KMeans + LLM dedup.
   - SemHash discards its surface→canonical map, so `merge.py` runs the two stages explicitly with kg-gen's own classes and composes the maps.
   - The composed result is checked against kg-gen's relations (`composition_consistent_with_kggen` in `stats.json`).
4. **semhash must be < 0.4.** The checkout reads `DeduplicationResult.duplicates`, which semhash 0.4 removed.
5. **LiteLLM prefix `hosted_vllm/`, not `openai/`.** With `openai/`, KGGen switches to the Responses API.
6. **DSPy 3.5 prompt formats.** The entity prompt uses XML tags. The relation prompt asks for nested `<item><subject>…</subject>…</item>`. If that fails to parse, DSPy retries with its JSON adapter, and then KGGen's relation fallback ("fix") prompt runs. All of these are counted: `relation_fallback`, `*_parse_fail` statuses.

## Layout

```
datasets/nfcorpus/nfcorpus_original.jsonl     clean BEIR corpus
datasets/nfcorpus/nfcorpus_watermarked.jsonl  built on OzSTAR (`submit.sh datasets`)
datasets/_ground_truth/nfcorpus/              eval only
knowledge_extraction/
  kg_extract/  guard, export_kb, extract, merge, report, check_kggen, config
  kg_eval/     build_corpus_variants, evaluate
  slurm/       config.env.example, setup.sh, submit.sh, report.sh, *.sbatch, common.sh
  tests/       test_blackbox.py (stdlib), test_end_to_end_fake.py (fake LLM server)
```

## Run on OzSTAR / Ngarrgu Tindebeek

All heavy work runs as Slurm jobs on NT A100 nodes (`--gres=gpu:1`, no `--partition`, per the OzSTAR docs).

**1. One-time setup (login node `tooarrana1`/`tooarrana2`)**
```bash
cd /fred/<project_id>/$USER
git clone https://github.com/PapaYa2004/rag-wm-kg-extract.git
cd rag-wm-kg-extract/knowledge_extraction
cp slurm/config.env.example slurm/config.env
module avail python                   # pick a >=3.10 module for PY_MODULE
nano slurm/config.env                 # EDIT: PROJECT_ID, SLURM_ACCOUNT, CHROMA_DIR, RAGWM_OUTPUT, PY_MODULE
export HF_TOKEN=hf_...                # accept the Llama-3.1 licence on huggingface.co first
bash slurm/setup.sh                   # venvs, vLLM image, model weights, kg-gen API check
```
The last line of the setup output must be `kg-gen API check OK`.

**2. Build the watermarked dataset file**
```bash
bash slurm/submit.sh datasets
tail -f /fred/<project_id>/$USER/kg_extract_outputs/logs/kgx-datasets_*.out
```
The log should show `verification against Chroma: 3633 passages, 0 mismatches`. Then commit `datasets/nfcorpus/nfcorpus_watermarked.jsonl`, `MANIFEST.json` and `datasets/_ground_truth/` (or copy them back with `scp`).

**3. Pilot (300 label-blind sampled passages, 1 A100)**
```bash
bash slurm/submit.sh pilot            # export -> extract -> merge (chained)
squeue -u $USER
bash slurm/report.sh                  # 20 sample triplets, stats.json, time/GPU-hour estimate
```

**4. Full run.** Only after you've approved the estimate:
```bash
bash slurm/submit.sh full --shards N --time HH:MM:SS
```

**5. Evaluate** (separate process, after merge):
```bash
bash slurm/submit.sh eval /fred/<project_id>/$USER/kg_extract_outputs/nfcorpus
```

**Resume.** Re-run the same `submit.sh` command. Every finished passage is `fsync`ed to `shards/part-XXXX.jsonl`, and doc_ids that already have a record are skipped.
- Jobs near their time limit requeue themselves (`--signal=B:USR1@600`).
- Transient server errors leave a passage unrecorded and the task exits with code 3. `--kill-on-invalid-dep` then cancels the merge, so re-run `submit.sh` to resume.
- Clustering LLM calls are cached in `$DSPY_CACHEDIR`, so a rerun of merge replays them from the cache.

## Outputs (`<OUT_ROOT>/<dataset>/`)

| file | content |
|---|---|
| `triples_raw.jsonl` | `{doc_id, subject, predicate, object}`, one row per passage × triple |
| `graph_raw.json` | `KGGen.aggregate` of the per-passage graphs, saved **before** clustering |
| `triples_clustered.jsonl` | canonical `subject/predicate/object` + `*_surface` original forms |
| `graph_clustered.json`, `entity_clusters.json`, `edge_clusters.json` | clustered graph; `{canonical: [raw surface forms]}` |
| `cluster_stages.json` | SemHash map, LLM clusters and cluster errors, for auditing |
| `stats.json` | #docs, #triples, #entities, #relations (predicates), triples/doc, parse-failure and fallback rates, wall time, GPU-hours, kg-gen source hash |
| `shards/`, `runs/` | per-passage checkpoints; per-task timing (makespan, GPU-hours) |
| `eval/eval_report.{json,md}` | evaluation (step 5) |

## Evaluation (`kg_eval.evaluate`)

- **Tuple recall**, raw vs clustered, globally and restricted to the injected passages.
  - *Exact*: NFKC, lower-case, `_`/punctuation → space, singularised tokens.
  - *Fuzzy*: `all-MiniLM-L6-v2` cosine, entities ≥ `tau_e` = 0.85 and predicate ≥ `tau_r` = 0.70.
  - Each comes as *pair* (E1/E2 match subject/object in either direction) and *full* (pair + predicate).
  - A threshold sweep is included so the choice of thresholds can be judged. RAG-WM tuples are `(E1, E2, RELATION)`, mapped accordingly.
- **Injected-passage coverage:** share of watermarked passages with ≥1 triplet, and with ≥1 triplet touching their watermark entities. The watermark text is appended to a real abstract, so this is measured per passage.
- **Cluster merges:** for each watermark entity, the clusters its surface forms ended up in, the non-watermark members, and its document frequency before vs after clustering (the frequency-anomaly signal).

## Tests

```bash
PYTHONPATH=knowledge_extraction python -m pytest knowledge_extraction/tests/test_blackbox.py
PYTHONPATH=knowledge_extraction:kg-gen/src python -m pytest knowledge_extraction/tests/test_end_to_end_fake.py
```
