"""Aggregate per-passage graphs, save the RAW graph, then cluster and save the
CLUSTERED graph separately. Writes everything under --out-dir:

  triples_raw.jsonl          {doc_id, subject, predicate, object}   (one row per doc x triple)
  graph_raw.json             KGGen.aggregate(...) of the per-passage graphs
  triples_clustered.jsonl    {doc_id, subject, predicate, object,
                              subject_surface, predicate_surface, object_surface}
  graph_clustered.json       clustered KGGen Graph
  entity_clusters.json       {canonical: [raw surface forms]}  (all stages composed)
  edge_clusters.json         {canonical: [raw surface forms]}
  cluster_stages.json        per-stage maps (SemHash map, LLM clusters) for auditing
  stats.json

Clustering = KGGen's `cluster()` / `deduplicate(method=FULL)`, i.e. SemHash
normalisation followed by KMeans + LLM deduplication. We run the two stages
explicitly with KGGen's own classes, because `run_semhash_deduplication` discards
its surface->canonical map and we need to keep the original surface forms.
KGGen's `context` argument to deduplicate is a no-op in the installed version
(`# TODO: implement context`), so it is not passed.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import socket
import statistics
import sys
import time

from kg_extract import __version__
from kg_extract.config import (
    DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_RETRIEVAL_MODEL, DEFAULT_SEMHASH_THRESHOLD,
    DEFAULT_TEMPERATURE, STATUS_CONTEXT_LENGTH, STATUS_ENTITY_PARSE_FAIL, STATUS_OK,
    STATUS_RELATION_PARSE_FAIL,
)
from kg_extract.guard import enforce_blackbox


# ---------------------------------------------------------------- load shards
def load_records(out_dir: str) -> dict[str, dict]:
    """One record per doc_id; a successful record wins over a failed one, later wins over earlier."""
    best: dict[str, dict] = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "shards", "part-*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prev = best.get(r["doc_id"])
                if prev is None or r["status"] == STATUS_OK or prev["status"] != STATUS_OK:
                    best[r["doc_id"]] = r
    return best


def load_runs(out_dir: str) -> list[dict]:
    runs = []
    for p in glob.glob(os.path.join(out_dir, "runs", "*.json")):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def write_jsonl(path: str, rows) -> int:
    n = 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(path + ".tmp", path)
    return n


def write_json(path: str, obj) -> None:
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=isinstance(obj, dict))
    os.replace(path + ".tmp", path)


def per_doc_graphs(records: dict[str, dict]):
    from kg_gen.models import Graph

    graphs = []
    for doc_id in sorted(records):
        r = records[doc_id]
        rels = {tuple(t) for t in r["triples"]}
        ents = set(r["entities"]) | {t[0] for t in rels} | {t[2] for t in rels}
        graphs.append(Graph(entities=ents, relations=rels, edges={t[1] for t in rels}))
    return graphs


# ---------------------------------------------------------------- clustering
class _ErrorCounter(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage()[:300])


def semhash_stage(graph, threshold: float):
    """Same transformation as kg_gen.utils.deduplicate.run_semhash_deduplication,
    but also returns the surface -> representative maps."""
    from kg_gen.models import Graph
    from kg_gen.utils.deduplicate import DeduplicateList

    def build_map(items):
        dd = DeduplicateList(threshold)
        dd.deduplicate(sorted(items))
        return {x: dd.items_map[dd.original_map[x]] if x in dd.original_map else x for x in items}

    ent_map = build_map(graph.entities)
    edge_map = build_map(graph.edges)
    rels = {(ent_map.get(s, s), edge_map.get(p, p), ent_map.get(o, o)) for s, p, o in graph.relations}
    g = Graph(entities=set(ent_map.values()), edges=set(edge_map.values()), relations=rels)
    return g, ent_map, edge_map


def _resolver(finals: set[str], clusters: dict[str, set[str]]):
    """Exactly KGGen LLMDeduplicate.deduplicate()'s lookup: keep x if it is a final
    name, else the first representative whose cluster contains x, else x."""
    first: dict[str, str] = {}
    for rep, members in clusters.items():
        for m in members:
            first.setdefault(m, rep)
    return lambda x: x if x in finals else first.get(x, x)


def llm_stage(kg, sem_graph, seed: int):
    import numpy as np
    from kg_gen.utils.llm_deduplicate import LLMDeduplicate

    counter = _ErrorCounter()
    logging.getLogger("kg_gen.utils.llm_deduplicate").addHandler(counter)
    np.random.seed(seed)  # KMeans(init="random") without random_state uses the global RNG
    dd = LLMDeduplicate(kg.retrieval_model, kg.lm, sem_graph)
    dd.cluster()
    out = dd.deduplicate()
    return out, counter.messages


def compose_clusters(surface_to_canon: dict[str, str]) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = {}
    for s, c in surface_to_canon.items():
        inv.setdefault(c, []).append(s)
    return {c: sorted(v) for c, v in sorted(inv.items())}


# --------------------------------------------------------------------- stats
def doc_stats(records: dict[str, dict], n_target: int, n_kb: int) -> dict:
    status = {}
    for r in records.values():
        status[r["status"]] = status.get(r["status"], 0) + 1
    n = len(records)
    tpd = [len(r["triples"]) for r in records.values()]
    with_ents = [r for r in records.values() if r["entities"]]
    fb = sum(1 for r in with_ents if r.get("relation_fallback", 0) > 0)
    parse_fail = status.get(STATUS_ENTITY_PARSE_FAIL, 0) + status.get(STATUS_RELATION_PARSE_FAIL, 0)
    lat = [r["latency_s"] for r in records.values() if r.get("latency_s")]
    return {
        "n_docs_kb": n_kb,
        "n_docs_target": n_target,  # = n_docs_kb, or the pilot sample size
        "n_docs_processed": n,
        "n_docs_missing": n_target - n,
        "status_counts": status,
        "parse_failure_rate": parse_fail / n if n else None,
        "entity_parse_failure_rate": status.get(STATUS_ENTITY_PARSE_FAIL, 0) / n if n else None,
        "relation_parse_failure_rate": status.get(STATUS_RELATION_PARSE_FAIL, 0) / n if n else None,
        "relation_fallback_rate": fb / len(with_ents) if with_ents else None,
        "context_length_failures": status.get(STATUS_CONTEXT_LENGTH, 0),
        "docs_with_zero_triples": sum(1 for x in tpd if x == 0),
        "triples_per_doc_mean": statistics.fmean(tpd) if tpd else None,
        "triples_per_doc_median": statistics.median(tpd) if tpd else None,
        "triples_per_doc_max": max(tpd) if tpd else None,
        "entities_per_doc_mean": statistics.fmean(len(r["entities"]) for r in records.values()) if n else None,
        "unlisted_entity_triples": sum(r.get("unlisted_entity_triples", 0) for r in records.values()),
        "latency_s_per_doc_mean": statistics.fmean(lat) if lat else None,
    }


def graph_stats(g, n_rows: int) -> dict:
    return {"n_triple_rows_with_provenance": n_rows, "n_triples_unique": len(g.relations),
            "n_entities": len(g.entities), "n_relations_predicates": len(g.edges)}


def time_stats(runs: list[dict]) -> dict:
    out = {}
    for kind in ("extract", "merge"):
        rs = [r for r in runs if r.get("kind") == kind and r.get("t_end")]
        if not rs:
            continue
        elapsed = [r["t_end"] - r["t_job_start"] for r in rs]
        out[kind] = {
            "n_tasks": len(rs),
            "task_seconds_total": sum(elapsed),
            "makespan_seconds": max(r["t_end"] for r in rs) - min(r["t_job_start"] for r in rs),
            "vllm_startup_seconds_mean": statistics.fmean(
                [r["vllm_ready_s"] for r in rs if r.get("vllm_ready_s") == r.get("vllm_ready_s")] or [0.0]),
            "gpu_hours": sum(e * r.get("n_gpus", 1) for e, r in zip(elapsed, rs)) / 3600,
            "gpu_names": sorted({str(r.get("gpu_name")) for r in rs}),
        }
    out["gpu_hours_total"] = sum(v["gpu_hours"] for v in out.values() if isinstance(v, dict))
    return out


def _kggen_fingerprint() -> str:
    import kg_gen
    from kg_extract.check_kggen import source_fingerprint
    return source_fingerprint(os.path.dirname(kg_gen.__file__))


# ---------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate + cluster KGGen outputs")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kb", required=True, help="kb.jsonl (for the doc count only)")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="merge even if some KB docs have no record")
    ap.add_argument("--model", default=os.environ.get("KG_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-base", default=os.environ.get("KG_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("KG_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--retrieval-model", default=os.environ.get("KG_RETRIEVAL_MODEL", DEFAULT_RETRIEVAL_MODEL))
    ap.add_argument("--semhash-threshold", type=float, default=DEFAULT_SEMHASH_THRESHOLD)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    enforce_blackbox(input_paths=[args.kb])
    job_t0 = float(os.environ.get("JOB_T0", time.time()))
    od = args.out_dir
    job_id = os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")
    run_path = os.path.join(od, "runs", f"merge_{job_id}_r{restart}.json")
    os.makedirs(os.path.dirname(run_path), exist_ok=True)
    run = {"kind": "merge", "kg_extract_version": __version__, "job_id": job_id, "host": socket.gethostname(),
           "gpu_name": os.environ.get("GPU_NAME"), "n_gpus": int(os.environ.get("N_GPUS", "1")),
           "t_job_start": job_t0, "vllm_ready_s": float(os.environ.get("VLLM_READY_S", "nan")), "t_end": None}

    with open(args.kb, encoding="utf-8") as f:
        kb_ids = {json.loads(l)["doc_id"] for l in f}
    records = load_records(od)
    records = {k: v for k, v in records.items() if k in kb_ids}
    missing = len(kb_ids) - len(records)
    # pilot runs use a sample: the denominator is the sampled set recorded in the run files
    sampled = {r.get("sample") for r in load_runs(od) if r.get("kind") == "extract"}
    n_target = min(s for s in sampled if s) if any(sampled) else len(kb_ids)
    if len(records) < n_target and not args.allow_missing:
        raise SystemExit(f"{n_target - len(records)} docs have no record yet; resubmit extraction "
                         f"or pass --allow-missing")
    print(f"[merge] v{__version__} records={len(records)} kb={len(kb_ids)} target={n_target}", flush=True)

    from kg_gen import KGGen

    # ---- RAW
    t_raw = time.time()
    raw_rows = [{"doc_id": d, "subject": s, "predicate": p, "object": o}
                for d in sorted(records) for s, p, o in records[d]["triples"]]
    n_raw = write_jsonl(os.path.join(od, "triples_raw.jsonl"), raw_rows)
    graphs = per_doc_graphs(records)
    kg_agg = KGGen.__new__(KGGen)  # aggregate() needs no LM; avoid building one
    raw_graph = KGGen.aggregate(kg_agg, graphs)
    KGGen.export_graph(raw_graph, os.path.join(od, "graph_raw.json"))
    stats = {
        "kg_extract_version": __version__,
        "dataset_dir": os.path.abspath(od),
        "docs": doc_stats(records, n_target, len(kb_ids)),
        "kg_gen_source_sha256": _kggen_fingerprint(),
        "raw": graph_stats(raw_graph, n_raw),
    }
    stats["raw"]["merge_seconds"] = time.time() - t_raw
    write_json(os.path.join(od, "stats.json"), stats)
    print(f"[merge] raw graph saved: {stats['raw']}", flush=True)

    # ---- CLUSTERED
    if not args.no_cluster and raw_graph.entities:
        t_cl = time.time()
        kg = KGGen(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                   api_key=args.api_key, api_base=args.api_base, retrieval_model=args.retrieval_model)
        sem_graph, sem_ent, sem_edge = semhash_stage(raw_graph, args.semhash_threshold)
        print(f"[merge] semhash: entities {len(raw_graph.entities)}->{len(sem_graph.entities)}, "
              f"edges {len(raw_graph.edges)}->{len(sem_graph.edges)}", flush=True)
        cl_graph, cluster_errors = llm_stage(kg, sem_graph, args.seed)

        r_ent = _resolver(cl_graph.entities, cl_graph.entity_clusters or {})
        r_edge = _resolver(cl_graph.edges, cl_graph.edge_clusters or {})
        ent_canon = {s: r_ent(sem_ent[s]) for s in raw_graph.entities}
        edge_canon = {p: r_edge(sem_edge[p]) for p in raw_graph.edges}

        cl_rows = [{"doc_id": r["doc_id"],
                    "subject": ent_canon[r["subject"]], "predicate": edge_canon[r["predicate"]],
                    "object": ent_canon[r["object"]],
                    "subject_surface": r["subject"], "predicate_surface": r["predicate"],
                    "object_surface": r["object"]} for r in raw_rows]
        n_cl = write_jsonl(os.path.join(od, "triples_clustered.jsonl"), cl_rows)
        recomposed = {(r["subject"], r["predicate"], r["object"]) for r in cl_rows}
        consistent = recomposed == set(cl_graph.relations)
        if not consistent:
            print(f"[merge] WARNING: composed map differs from KGGen relations "
                  f"({len(recomposed ^ set(cl_graph.relations))} triples)", flush=True)

        KGGen.export_graph(cl_graph, os.path.join(od, "graph_clustered.json"))
        write_json(os.path.join(od, "entity_clusters.json"), compose_clusters(ent_canon))
        write_json(os.path.join(od, "edge_clusters.json"), compose_clusters(edge_canon))
        write_json(os.path.join(od, "cluster_stages.json"), {
            "semhash_threshold": args.semhash_threshold,
            "semhash_entity_map": {k: v for k, v in sem_ent.items() if k != v},
            "semhash_edge_map": {k: v for k, v in sem_edge.items() if k != v},
            "llm_entity_clusters": {k: sorted(v) for k, v in (cl_graph.entity_clusters or {}).items() if len(v) > 1},
            "llm_edge_clusters": {k: sorted(v) for k, v in (cl_graph.edge_clusters or {}).items() if len(v) > 1},
            "cluster_errors": cluster_errors,
        })
        stats["clustered"] = graph_stats(cl_graph, n_cl)
        stats["clustered"].update({
            "n_entities_after_semhash": len(sem_graph.entities),
            "n_predicates_after_semhash": len(sem_graph.edges),
            "n_entity_clusters_merged": sum(1 for v in compose_clusters(ent_canon).values() if len(v) > 1),
            "n_edge_clusters_merged": sum(1 for v in compose_clusters(edge_canon).values() if len(v) > 1),
            "llm_cluster_errors": len(cluster_errors),
            "composition_consistent_with_kggen": consistent,
            "retrieval_model": args.retrieval_model,
            "cluster_seconds": time.time() - t_cl,
        })

    run["t_end"] = time.time()
    write_json(run_path, run)
    stats["time"] = time_stats(load_runs(od))
    write_json(os.path.join(od, "stats.json"), stats)
    print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
