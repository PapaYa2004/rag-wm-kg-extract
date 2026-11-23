# NFCorpus: original and RAG-WM-watermarked

| file | content |
|---|---|
| `nfcorpus_original.jsonl` | BEIR NFCorpus corpus, unchanged (3,633 docs; `_id`, `title`, `text`, `metadata`). Formerly `corpus.jsonl`. |
| `nfcorpus_watermarked.jsonl` | Same docs, same order, same fields. The passages RAG-WM injected into have the watermark text(s) appended to `text`, exactly as RAG-WM's `update_context(pos='end')` does. **No field marks which docs were changed.** |
| `queries.jsonl`, `qrels/` | Unchanged BEIR queries and relevance judgements. |
| `MANIFEST.json` | sha256 and doc counts of the two corpus files. |

Ground truth (the 50 watermark tuples, which passages were changed, and the
appended texts) lives in `../_ground_truth/nfcorpus/rag_wm_ground_truth.json`.
Only the evaluation module reads it. The extraction guard refuses to open it,
and also refuses to open `*_original.jsonl` during a watermarked run, because
diffing the two corpora would reveal the watermarks.

## How the watermarked file is built

It's reconstructed on OzSTAR from RAG-WM's injection log
(`wm_generate/nfcorpus/<mutual_times>/wmuint_inject.json`), optionally checked
passage by passage against the watermarked Chroma store. See
`knowledge_extraction/README.md`, section "Build the dataset files".

## Note for RAG-WM

RAG-WM's `load_beir_datasets` (BEIR `GenericDataLoader`) expects a file named
`corpus.jsonl`. To point RAG-WM at this folder again, copy
`nfcorpus_original.jsonl` to `corpus.jsonl` first.
