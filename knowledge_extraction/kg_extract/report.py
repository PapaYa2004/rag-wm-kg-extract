"""Pilot reporting: sample triplets, stats, and a full-run time / GPU-hour estimate.

  python -m kg_extract.report samples  --out-dir OUT/nfcorpus_pilot --n 20
  python -m kg_extract.report estimate --out-dir OUT/nfcorpus_pilot --full-docs 3633
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys

from kg_extract.guard import enforce_blackbox
from kg_extract.merge import load_runs


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def cmd_samples(args):
    raw = _read_jsonl(os.path.join(args.out_dir, "triples_raw.jsonl"))
    cl_path = os.path.join(args.out_dir, "triples_clustered.jsonl")
    cl = _read_jsonl(cl_path) if os.path.exists(cl_path) else None
    idx = sorted(random.Random(args.seed).sample(range(len(raw)), min(args.n, len(raw))))
    print(f"## {len(idx)} sample triplets (seed={args.seed}) from {len(raw)} rows\n")
    print("| # | doc_id | subject | predicate | object | clustered (s / p / o) |")
    print("|---|---|---|---|---|---|")
    for k, i in enumerate(idx, 1):
        r = raw[i]
        c = f"{cl[i]['subject']} / {cl[i]['predicate']} / {cl[i]['object']}" if cl else "-"
        print(f"| {k} | {r['doc_id']} | {r['subject']} | {r['predicate']} | {r['object']} | {c} |")
    with open(os.path.join(args.out_dir, "stats.json")) as f:
        print("\n## stats.json\n```json\n" + f.read() + "\n```")


def cmd_estimate(args):
    runs = load_runs(args.out_dir)
    ex = [r for r in runs if r.get("kind") == "extract" and r.get("t_end") and r.get("n_written")]
    if not ex:
        raise SystemExit("no finished extract runs with written docs")
    work_s = sum(r["t_end"] - r["t_extract_start"] for r in ex)
    n_docs = sum(r["n_written"] for r in ex)
    rate = n_docs / work_s  # docs / s / GPU at the pilot concurrency
    startup = statistics.fmean(r["t_extract_start"] - r["t_job_start"] for r in ex)

    with open(os.path.join(args.out_dir, "stats.json")) as f:
        st = json.load(f)
    n_pilot = st["docs"]["n_docs_processed"]
    scale = args.full_docs / n_pilot
    cl = st.get("clustered", {})
    cl_items = cl.get("n_entities_after_semhash", 0) + cl.get("n_predicates_after_semhash", 0)
    cl_s_per_item = cl.get("cluster_seconds", 0) / cl_items if cl_items else 0.0
    # Upper bound: unique entities grow at most linearly with #docs.
    full_items = cl_items * scale
    merge_s = startup + full_items * cl_s_per_item + st["raw"].get("merge_seconds", 0) * scale

    print(f"## Estimate from pilot ({n_docs} docs, {len(ex)} task(s), concurrency={ex[0].get('concurrency')})")
    print(f"- extraction throughput: {rate:.3f} docs/s per GPU ({rate * 3600:.0f} docs/GPU-hour)")
    print(f"- per-task fixed cost (vLLM load + client init): {startup:.0f} s")
    print(f"- clustering: {cl_s_per_item:.3f} s/item on {cl_items} items "
          f"-> full run ~{full_items:.0f} items (linear upper bound), ~{merge_s / 3600:.2f} h on 1 GPU")
    print(f"\n### Extraction, {args.full_docs} docs (safety factor {args.safety}x on --time)\n")
    print("| shards | docs/shard | task wall (h) | suggested --time | makespan if all run (h) | GPU-hours |")
    print("|---|---|---|---|---|---|")
    rows = []
    for k in args.shards:
        per = math.ceil(args.full_docs / k)
        wall = startup + per / rate
        rows.append((k, per, wall))
        t = math.ceil(wall * args.safety / 900) * 900
        print(f"| {k} | {per} | {wall / 3600:.2f} | {t // 3600:02d}:{(t % 3600) // 60:02d}:00 "
              f"| {wall / 3600:.2f} | {k * wall / 3600:.2f} |")
    print(f"\nMerge + clustering (single GPU job): ~{merge_s / 3600:.2f} h, {merge_s / 3600:.2f} GPU-hours")
    good = [r for r in rows if r[2] <= args.target_task_hours * 3600]
    pick = good[0] if good else rows[-1]
    print(f"\nProposal: {pick[0]} shard(s) of ~{pick[1]} docs "
          f"(smallest count with task wall <= {args.target_task_hours} h; fixed cost is paid once per shard).")


def main(argv=None):
    enforce_blackbox()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("samples")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("estimate")
    e.add_argument("--out-dir", required=True)
    e.add_argument("--full-docs", type=int, required=True)
    e.add_argument("--shards", type=int, nargs="+", default=[1, 2, 4, 8])
    e.add_argument("--safety", type=float, default=1.5)
    e.add_argument("--target-task-hours", type=float, default=2.0)
    args = ap.parse_args(argv)
    {"samples": cmd_samples, "estimate": cmd_estimate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
