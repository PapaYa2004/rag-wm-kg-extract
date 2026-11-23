"""Post-extraction evaluation against RAG-WM ground truth.

Reads ground truth from RAG-WM's injection log
  <rag-wm output>/wm_generate/<dataset>/<mutual_times>/wmuint_inject.json
(entries: [[E1, E2, R], [chroma ids the watermark text was appended to], doc_info]).
Note RAG-WM's tuple order is (entity1, entity2, relation), NOT (s, p, o).

Reports, for the raw and the clustered graph:
  1. Recall of the RAG-WM tuples among the extracted triplets:
       exact  - normalised strings (NFKC, lower-case, punctuation/underscores -> space,
                singularised tokens)
       fuzzy  - embedding cosine similarity, entity threshold tau_e, predicate
                threshold tau_r (both reported; plus a threshold sweep)
     each as "pair" (E1/E2 match subject/object, either direction) and "full"
     (pair + predicate matches R), and each globally (any passage) and
     "in_injected" (only passages the tuple was injected into).
  2. Fraction of injected passages that produced >=1 triplet, and >=1 triplet
     touching one of the watermark entities injected there.
  3. Whether clustering merged a watermark entity into a cluster containing
     non-watermark surface forms, and how that changes its document frequency.

  python -m kg_eval.evaluate --out-dir OUT/nfcorpus --inject-json .../wmuint_inject.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np

import kg_eval  # noqa: F401  (import-time black-box assertion)
from kg_extract import __version__
from kg_extract.guard import EXTRACTION_PHASE_ENV

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------ helpers
_inflect = None


def _singular(tok: str) -> str:
    global _inflect
    if _inflect is None:
        try:
            import inflect
            _inflect = inflect.engine()
        except ImportError:
            _inflect = False
    if _inflect:
        s = _inflect.singular_noun(tok)
        return s if isinstance(s, str) and s else tok
    return tok[:-1] if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") else tok


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(_singular(t) for t in s.split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def load_ground_truth(path: str):
    """Either datasets/_ground_truth/<ds>/rag_wm_ground_truth.json (BEIR _id space,
    from build_corpus_variants) or RAG-WM's raw wmuint_inject.json (Chroma id_<n> space).
    Use the one whose id space matches the KB that was extracted."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tuples" in data:
        return ([(t["e1"], t["e2"], t["relation"]) for t in data["tuples"]],
                [set(t["docs"]) for t in data["tuples"]])
    tuples, tuple_docs = [], []
    for entry in data:
        unit, locs = entry[0], entry[1]
        e1, e2, r = (str(x) for x in unit[:3])
        tuples.append((e1, e2, r))
        tuple_docs.append(set(locs if isinstance(locs, list) else [locs]))
    return tuples, tuple_docs


class Embedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if todo:
            vecs = self.model.encode(todo, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
            self.cache.update(zip(todo, vecs))
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1))


# --------------------------------------------------------------- matching
class TripleIndex:
    """Column view of a triple list (subject/predicate/object as indices into vocabularies)."""

    def __init__(self, rows: list[dict], emb: Embedder):
        self.rows = rows
        self.doc = np.array([r["doc_id"] for r in rows], dtype=object)
        self.ents = sorted({r["subject"] for r in rows} | {r["object"] for r in rows})
        self.preds = sorted({r["predicate"] for r in rows})
        ei = {e: i for i, e in enumerate(self.ents)}
        pi = {p: i for i, p in enumerate(self.preds)}
        self.s = np.array([ei[r["subject"]] for r in rows], dtype=np.int64)
        self.o = np.array([ei[r["object"]] for r in rows], dtype=np.int64)
        self.p = np.array([pi[r["predicate"]] for r in rows], dtype=np.int64)
        self.ent_norm = np.array([norm(e) for e in self.ents], dtype=object)
        self.pred_norm = np.array([norm(p) for p in self.preds], dtype=object)
        self.ent_emb = emb.encode(self.ents)
        self.pred_emb = emb.encode(self.preds)

    def ent_sim(self, vec):
        return self.ent_emb @ vec if len(self.ents) else np.zeros(0)

    def pred_sim(self, vec):
        return self.pred_emb @ vec if len(self.preds) else np.zeros(0)


def tuple_match_masks(ix: TripleIndex, t, emb: Embedder, tau_e: float, tau_r: float):
    """Boolean masks over triples: exact_pair, exact_full, fuzzy_pair, fuzzy_full."""
    e1, e2, r = t
    n1, n2, nr = norm(e1), norm(e2), norm(r)
    ex_e1, ex_e2 = ix.ent_norm == n1, ix.ent_norm == n2
    ex_r = ix.pred_norm == nr
    v1, v2, vr = emb.encode([e1, e2, r])
    fz_e1 = (ix.ent_sim(v1) >= tau_e) | ex_e1
    fz_e2 = (ix.ent_sim(v2) >= tau_e) | ex_e2
    fz_r = (ix.pred_sim(vr) >= tau_r) | ex_r

    def pair(m1, m2):
        return (m1[ix.s] & m2[ix.o]) | (m2[ix.s] & m1[ix.o])

    exact_pair = pair(ex_e1, ex_e2)
    fuzzy_pair = pair(fz_e1, fz_e2)
    return {
        "exact_pair": exact_pair, "exact_full": exact_pair & ex_r[ix.p],
        "fuzzy_pair": fuzzy_pair, "fuzzy_full": fuzzy_pair & fz_r[ix.p],
    }


def recall_block(ix, tuples, tuple_docs, emb, tau_e, tau_r, keep_examples=True):
    keys = ["exact_pair", "exact_full", "fuzzy_pair", "fuzzy_full"]
    hit = {k: 0 for k in keys}
    hit_inj = {k: 0 for k in keys}
    per_tuple = []
    for t, docs in zip(tuples, tuple_docs):
        masks = tuple_match_masks(ix, t, emb, tau_e, tau_r)
        in_inj = np.isin(ix.doc, list(docs)) if docs else np.zeros(len(ix.rows), bool)
        row = {"tuple": list(t), "injected_docs": sorted(docs)}
        for k in keys:
            m = masks[k]
            hit[k] += bool(m.any())
            hit_inj[k] += bool((m & in_inj).any())
            row[k] = int(m.sum())
            row[k + "_in_injected"] = int((m & in_inj).sum())
        if keep_examples:
            idx = np.flatnonzero(masks["fuzzy_pair"])[:5]
            row["examples"] = [ix.rows[i] for i in idx]
        per_tuple.append(row)
    n = len(tuples)
    return {
        "recall": {k: hit[k] / n for k in keys},
        "recall_in_injected": {k: hit_inj[k] / n for k in keys},
        "per_tuple": per_tuple,
    }


def sweep(ix, tuples, tuple_docs, emb, taus):
    out = []
    for te in taus:
        for tr in taus:
            b = recall_block(ix, tuples, tuple_docs, emb, te, tr, keep_examples=False)
            out.append({"tau_e": te, "tau_r": tr, "fuzzy_pair": b["recall"]["fuzzy_pair"],
                        "fuzzy_full": b["recall"]["fuzzy_full"]})
    return out


# --------------------------------------------------- injected passage coverage
def injected_coverage(raw_rows, records_by_doc, tuples, tuple_docs, emb, tau_e):
    doc_tuples = defaultdict(list)
    for t, docs in zip(tuples, tuple_docs):
        for d in docs:
            doc_tuples[d].append(t)
    by_doc = defaultdict(list)
    for r in raw_rows:
        by_doc[r["doc_id"]].append(r)
    per_doc, any_n, wm_n = [], 0, 0
    for d, ts in sorted(doc_tuples.items()):
        rows = by_doc.get(d, [])
        wm_ents = sorted({x for t in ts for x in t[:2]})
        touches = 0
        if rows and wm_ents:
            surf = sorted({r["subject"] for r in rows} | {r["object"] for r in rows})
            sims = emb.encode(surf) @ emb.encode(wm_ents).T
            wn = {norm(w) for w in wm_ents}
            hit = {s for s, row in zip(surf, sims) if row.max() >= tau_e or norm(s) in wn}
            touches = sum(1 for r in rows if r["subject"] in hit or r["object"] in hit)
        any_n += bool(rows)
        wm_n += bool(touches)
        rec = records_by_doc.get(d, {})
        per_doc.append({"doc_id": d, "status": rec.get("status", "missing"), "n_triples": len(rows),
                        "n_triples_touching_wm_entity": touches, "wm_entities": wm_ents})
    n = len(doc_tuples)
    return {"n_injected_passages": n,
            "frac_with_any_triplet": any_n / n if n else None,
            "frac_with_wm_entity_triplet": wm_n / n if n else None,
            "per_doc": per_doc}


# ------------------------------------------------------------ cluster merges
def cluster_merge_check(raw_rows, cl_rows, entity_clusters, tuples, tuple_docs, emb, tau_e):
    injected = set().union(*tuple_docs) if tuple_docs else set()
    surf2canon = {s: c for c, members in entity_clusters.items() for s in members}
    surfaces = sorted(surf2canon)
    surf_emb = emb.encode(surfaces) if surfaces else np.zeros((0, 1))
    surf_norm = [norm(s) for s in surfaces]

    surf_docs = defaultdict(set)
    for r in raw_rows:
        surf_docs[r["subject"]].add(r["doc_id"])
        surf_docs[r["object"]].add(r["doc_id"])
    canon_docs = defaultdict(set)
    for r in cl_rows:
        canon_docs[r["subject"]].add(r["doc_id"])
        canon_docs[r["object"]].add(r["doc_id"])

    report, any_merge = [], False
    for w in sorted({x for t in tuples for x in t[:2]}):
        wn, wv = norm(w), emb.encode([w])[0]
        sims = surf_emb @ wv if surfaces else np.zeros(0)
        matched = {s for s, sn, sim in zip(surfaces, surf_norm, sims) if sn == wn or sim >= tau_e}
        clusters = []
        for c in sorted({surf2canon[s] for s in matched}):
            members = entity_clusters[c]
            foreign = [m for m in members if m not in matched]
            foreign_docs = set().union(*(surf_docs[m] for m in foreign)) if foreign else set()
            merged = bool(foreign)
            any_merge |= merged
            wm_docs = set().union(*(surf_docs[m] for m in members if m in matched))
            clusters.append({
                "canonical": c, "cluster_size": len(members),
                "wm_surface_forms": sorted(m for m in members if m in matched),
                "foreign_members": foreign[:50], "merged_with_non_wm_forms": merged,
                "foreign_docs_total": len(foreign_docs),
                "foreign_docs_non_injected": len(foreign_docs - injected),
                "docs_raw_wm_forms": len(wm_docs),
                "docs_raw_wm_forms_injected": len(wm_docs & injected),
                "docs_clustered_canonical": len(canon_docs.get(c, ())),
            })
        report.append({"wm_entity": w, "n_matching_surface_forms": len(matched), "clusters": clusters})
    return {"any_wm_entity_merged_with_non_wm_forms": any_merge,
            "n_wm_entities": len(report),
            "n_wm_entities_merged": sum(any(c["merged_with_non_wm_forms"] for c in e["clusters"]) for e in report),
            "per_entity": report}


# --------------------------------------------------------------------- main
def write_markdown(res, path):
    L = [f"# KG extraction evaluation ({res['dataset_dir']})", "",
         f"kg_extract v{res['kg_extract_version']} | embed model `{res['embed_model']}` | "
         f"tau_e={res['tau_e']} tau_r={res['tau_r']} | {res['n_tuples']} RAG-WM tuples", "",
         "## Tuple recall", "", "| graph | scope | exact pair | exact full | fuzzy pair | fuzzy full |",
         "|---|---|---|---|---|---|"]
    for g in ("raw", "clustered"):
        if g not in res:
            continue
        for scope in ("recall", "recall_in_injected"):
            r = res[g][scope]
            L.append(f"| {g} | {'any passage' if scope == 'recall' else 'injected passages'} | "
                     f"{r['exact_pair']:.2f} | {r['exact_full']:.2f} | {r['fuzzy_pair']:.2f} | {r['fuzzy_full']:.2f} |")
    c = res["injected_coverage"]
    L += ["", "## Injected passages", "",
          f"- passages: {c['n_injected_passages']}",
          f"- with >=1 triplet: {c['frac_with_any_triplet']:.2%}" if c["n_injected_passages"] else "- none",
          f"- with >=1 triplet touching a watermark entity: {c['frac_with_wm_entity_triplet']:.2%}"
          if c["n_injected_passages"] else ""]
    if "cluster_merge" in res:
        m = res["cluster_merge"]
        L += ["", "## Clustering merges", "",
              f"- watermark entities merged with non-watermark surface forms: "
              f"{m['n_wm_entities_merged']}/{m['n_wm_entities']}", "",
              "| wm entity | canonical | size | foreign members (first 5) | docs raw (inj) | docs clustered |",
              "|---|---|---|---|---|---|"]
        for e in m["per_entity"]:
            for cl in e["clusters"]:
                if cl["merged_with_non_wm_forms"]:
                    L.append(f"| {e['wm_entity']} | {cl['canonical']} | {cl['cluster_size']} | "
                             f"{'; '.join(cl['foreign_members'][:5])} | {cl['docs_raw_wm_forms']} "
                             f"({cl['docs_raw_wm_forms_injected']}) | {cl['docs_clustered_canonical']} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    if os.environ.get(EXTRACTION_PHASE_ENV) == "1":
        raise SystemExit("evaluation must not run inside an extraction process")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="extraction output dir for one dataset")
    ap.add_argument("--gt", "--inject-json", dest="inject_json", required=True,
                    help="rag_wm_ground_truth.json (BEIR ids) or RAG-WM wmuint_inject.json (Chroma ids)")
    ap.add_argument("--embed-model", default=os.environ.get("KG_EVAL_EMBED_MODEL", DEFAULT_EMBED_MODEL))
    ap.add_argument("--tau-e", type=float, default=0.85, help="entity cosine threshold")
    ap.add_argument("--tau-r", type=float, default=0.70, help="predicate cosine threshold")
    ap.add_argument("--sweep", type=float, nargs="*", default=[0.6, 0.7, 0.8, 0.85, 0.9, 0.95])
    args = ap.parse_args(argv)

    od = args.out_dir
    # Separation: evaluation only runs on finished extraction outputs.
    for req in ("stats.json", "triples_raw.jsonl"):
        if not os.path.exists(os.path.join(od, req)):
            raise SystemExit(f"{req} missing in {od}: run extraction + merge first")

    tuples, tuple_docs = load_ground_truth(args.inject_json)
    emb = Embedder(args.embed_model)
    raw_rows = read_jsonl(os.path.join(od, "triples_raw.jsonl"))
    from kg_extract.merge import load_records
    records = load_records(od)
    gt_ids = set().union(*tuple_docs) if tuple_docs else set()
    if gt_ids and not (gt_ids & set(records)):
        raise SystemExit("ground-truth doc ids do not occur in the extracted KB: wrong id space? "
                         "(Chroma-export KB -> wmuint_inject.json; BEIR-file KB -> rag_wm_ground_truth.json)")

    res = {"kg_extract_version": __version__, "dataset_dir": os.path.abspath(od),
           "inject_json": os.path.abspath(args.inject_json), "embed_model": args.embed_model,
           "tau_e": args.tau_e, "tau_r": args.tau_r, "n_tuples": len(tuples),
           "n_injected_passages_total": len(set().union(*tuple_docs)) if tuple_docs else 0}

    ix_raw = TripleIndex(raw_rows, emb)
    res["raw"] = recall_block(ix_raw, tuples, tuple_docs, emb, args.tau_e, args.tau_r)
    res["raw"]["sweep"] = sweep(ix_raw, tuples, tuple_docs, emb, args.sweep)

    cl_path = os.path.join(od, "triples_clustered.jsonl")
    if os.path.exists(cl_path):
        cl_rows = read_jsonl(cl_path)
        ix_cl = TripleIndex(cl_rows, emb)
        res["clustered"] = recall_block(ix_cl, tuples, tuple_docs, emb, args.tau_e, args.tau_r)
        res["clustered"]["sweep"] = sweep(ix_cl, tuples, tuple_docs, emb, args.sweep)
        with open(os.path.join(od, "entity_clusters.json"), encoding="utf-8") as f:
            entity_clusters = json.load(f)
        res["cluster_merge"] = cluster_merge_check(raw_rows, cl_rows, entity_clusters,
                                                   tuples, tuple_docs, emb, args.tau_e)

    res["injected_coverage"] = injected_coverage(raw_rows, records, tuples, tuple_docs, emb, args.tau_e)

    ev = os.path.join(od, "eval")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    write_markdown(res, os.path.join(ev, "eval_report.md"))
    with open(os.path.join(ev, "eval_report.md"), encoding="utf-8") as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
