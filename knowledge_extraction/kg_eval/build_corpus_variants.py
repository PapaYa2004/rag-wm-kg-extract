"""Build the two NFCorpus dataset files plus a separate, evaluation-only ground truth.

  datasets/<ds>/<ds>_original.jsonl      BEIR corpus, unchanged
  datasets/<ds>/<ds>_watermarked.jsonl   same schema and order; RAG-WM watermark texts
                                         appended to the passages they were injected into
  datasets/_ground_truth/<ds>/rag_wm_ground_truth.json   (eval only; never read by extraction)

The watermarked file is reconstructed exactly as RAG-WM builds its KB:
  * rag-wm/rag/vectorstore.py stores sorted(corpus.keys())[n] as Chroma id "id_<n>";
  * src/main.py wm_inject_run -> Visiter.inject_wm -> VectorStore.update_context(id, WT, 'end')
    sets text = text + ' ' + WT, applied in the order of wmuint_inject.json, where the
    k-th id in an entry's loc list receives the k-th WT in its doc_info.
Each watermarked record carries the same fields as the original one; only `text`
changes. No label field is added.

Pass --verify-kb (an export_kb dump of the watermarked Chroma store) to check
the reconstruction against the real KB passage by passage.

  python -m kg_eval.build_corpus_variants --dataset nfcorpus \
      --beir-corpus datasets/nfcorpus/nfcorpus_original.jsonl \
      --inject-json <ragwm_out>/wm_generate/nfcorpus/10/wmuint_inject.json \
      --datasets-dir datasets [--verify-kb kb_from_chroma.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import kg_eval  # noqa: F401  (refuses to run inside an extraction process)
from kg_extract import __version__


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_corpus(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def reconstruct(records: list[dict], inject: list) -> tuple[dict, dict, list]:
    """Returns (beir_id -> watermarked text, beir_id -> injection info, tuples)."""
    chroma_to_beir = {f"id_{n}": k for n, k in enumerate(sorted(r["_id"] for r in records))}
    text = {r["_id"]: r["text"] for r in records}
    injected: dict[str, dict] = {}
    tuples = []
    for entry in inject:
        unit, locs, doc_info = entry[0], entry[1], entry[2]
        locs = locs if isinstance(locs, list) else [locs]
        e1, e2, rel = (str(x) for x in unit[:3])
        wts = [d[0] if isinstance(d, list) else d for d in doc_info]
        if len(wts) != len(locs):
            raise ValueError(f"tuple {unit}: {len(locs)} ids but {len(wts)} watermark texts")
        t = {"e1": e1, "e2": e2, "relation": rel, "docs": [], "chroma_ids": list(locs), "wt_texts": wts}
        for cid, wt in zip(locs, wts):
            bid = chroma_to_beir[cid]
            text[bid] = text[bid] + " " + wt
            t["docs"].append(bid)
            info = injected.setdefault(bid, {"chroma_id": cid, "appended": []})
            info["appended"].append({"tuple": [e1, e2, rel], "wt": wt})
        tuples.append(t)
    return text, injected, tuples


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--beir-corpus", required=True, help="original BEIR corpus.jsonl")
    ap.add_argument("--inject-json", required=True, help="RAG-WM wm_generate/<ds>/<mt>/wmuint_inject.json")
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--verify-kb", default=None, help="export_kb dump of the watermarked Chroma store")
    args = ap.parse_args(argv)

    ds_dir = os.path.join(args.datasets_dir, args.dataset)
    gt_dir = os.path.join(args.datasets_dir, "_ground_truth", args.dataset)
    os.makedirs(ds_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    records = load_corpus(args.beir_corpus)
    with open(args.inject_json, encoding="utf-8") as f:
        inject = json.load(f)
    wm_text, injected, tuples = reconstruct(records, inject)

    orig_path = os.path.join(ds_dir, f"{args.dataset}_original.jsonl")
    wm_path = os.path.join(ds_dir, f"{args.dataset}_watermarked.jsonl")
    if os.path.abspath(args.beir_corpus) != os.path.abspath(orig_path):
        shutil.copyfile(args.beir_corpus, orig_path)
    with open(wm_path + ".tmp", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({**r, "text": wm_text[r["_id"]]}, ensure_ascii=False) + "\n")
    os.replace(wm_path + ".tmp", wm_path)

    verify = None
    if args.verify_kb:
        chroma_to_beir = {f"id_{n}": k for n, k in enumerate(sorted(r["_id"] for r in records))}
        mism, n = [], 0
        with open(args.verify_kb, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                bid = chroma_to_beir.get(d["doc_id"])
                n += 1
                if bid is None or d["text"] != wm_text[bid]:
                    mism.append(d["doc_id"])
        verify = {"kb": os.path.abspath(args.verify_kb), "n_compared": n, "n_mismatch": len(mism),
                  "mismatch_examples": mism[:20]}
        print(f"[build] verification against Chroma: {n} passages, {len(mism)} mismatches")

    gt = {
        "kg_extract_version": __version__,
        "dataset": args.dataset,
        "doc_id_space": "beir _id (chroma id_<n> = sorted(_id)[n])",
        "source_inject_json": os.path.abspath(args.inject_json),
        "n_tuples": len(tuples),
        "n_injected_passages": len(injected),
        "tuples": tuples,
        "injected_passages": injected,
        "verification": verify,
    }
    gt_path = os.path.join(gt_dir, "rag_wm_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    manifest = {
        "kg_extract_version": __version__,
        "files": {os.path.basename(p): {"sha256": _sha256(p), "n_docs": len(records)} for p in (orig_path, wm_path)},
        "n_passages_modified": len(injected),
        "schema": "BEIR corpus (_id, title, text, metadata); watermarked differs only in `text`",
    }
    with open(os.path.join(ds_dir, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({"original": orig_path, "watermarked": wm_path, "ground_truth": gt_path,
                      "n_tuples": len(tuples), "n_passages_modified": len(injected)}, indent=2))
    if verify and verify["n_mismatch"]:
        print("[build] WARNING: reconstruction differs from the Chroma store; use the Chroma export instead",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
