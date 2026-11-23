"""Export the watermarked KB as black-box (doc_id, text) records.

Reads ONLY ids and documents from the RAG-WM Chroma collection. Metadata
(which holds RAG-WM's `change` flag, i.e. the watermark label) is never
requested, and the code checks that Chroma returned none.

Runs in a minimal venv with chromadb==0.5.20, the version RAG-WM used. A newer
chromadb may migrate the sqlite store on open, so always point --chroma-path at a
*copy* of the store (the Slurm script copies it to $JOBFS).

Usage:
  python -m kg_extract.export_kb --chroma-path DIR --collection nfcorpus_contriever_cosine --out kb.jsonl
  python -m kg_extract.export_kb --beir-corpus datasets/nfcorpus/corpus.jsonl --out kb_clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

from kg_extract import __version__
from kg_extract.guard import BlackBoxViolation, assert_no_label_fields, enforce_blackbox


def _sort_key(doc_id: str):
    m = re.fullmatch(r"id_(\d+)", doc_id)
    return (0, int(m.group(1)), "") if m else (1, 0, doc_id)


def iter_chroma(chroma_path: str, collection: str, batch: int = 5000):
    import chromadb  # only needed here

    client = chromadb.PersistentClient(path=chroma_path)
    col = client.get_collection(name=collection)
    total = col.count()
    offset = 0
    while offset < total:
        res = col.get(include=["documents"], limit=batch, offset=offset)
        if res.get("metadatas") is not None and any(m is not None for m in res["metadatas"]):
            raise BlackBoxViolation("Chroma returned metadata although only documents were requested")
        for doc_id, text in zip(res["ids"], res["documents"]):
            yield doc_id, text
        offset += len(res["ids"])
        if not res["ids"]:
            break


def iter_beir(corpus_path: str):
    # Mirrors RAG-WM's vectorstore, which stores only value['text'] (no title).
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield str(d["_id"]), d["text"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--chroma-path")
    src.add_argument("--beir-corpus")
    ap.add_argument("--collection", help="e.g. nfcorpus_contriever_cosine")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    enforce_blackbox(input_paths=[args.beir_corpus])

    if args.chroma_path:
        if not args.collection:
            ap.error("--collection is required with --chroma-path")
        records = list(iter_chroma(args.chroma_path, args.collection))
        source = {"type": "chroma", "collection": args.collection}
    else:
        records = list(iter_beir(args.beir_corpus))
        source = {"type": "beir", "path": os.path.abspath(args.beir_corpus)}

    records.sort(key=lambda r: _sort_key(r[0]))
    ids = [r[0] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate doc_ids in KB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h = hashlib.sha256()
    n_empty = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for doc_id, text in records:
            text = text or ""
            n_empty += not text.strip()
            rec = {"doc_id": doc_id, "text": text}
            assert_no_label_fields(rec)
            line = json.dumps(rec, ensure_ascii=False)
            h.update(line.encode("utf-8"))
            f.write(line + "\n")
    os.replace(tmp, args.out)

    manifest = {
        "kg_extract_version": __version__,
        "source": source,
        "n_docs": len(records),
        "n_empty_text": n_empty,
        "sha256": h.hexdigest(),
        "fields": ["doc_id", "text"],
    }
    with open(args.out + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    sys.exit(main())
