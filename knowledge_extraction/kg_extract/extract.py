"""Per-passage KGGen extraction worker (one Slurm array task = one shard).

For every passage, run KGGen's entity step and then its relation step on that
passage alone (no chunking across documents), so each triplet keeps its source
doc_id. Results are appended to <out-dir>/shards/part-XXXX.jsonl, one JSON line
per doc. When the job is restarted, any doc_id that already has a final record
in any shard file is skipped.

Why not `KGGen.generate()`: in the installed kg-gen, generate() does not pass
`context` to the relation prompt (it only forwards it to deduplicate(), which
ignores it). It also SemHash-deduplicates each passage by default, which would
rewrite surface forms before aggregation. We call the same step functions that
generate() calls (`get_entities`, `get_relations`) with the same KGGen-built
dspy LM, adding `context=` and leaving out the per-passage dedup.
generate() also catches no parse errors, so counting them happens here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from kg_extract import __version__
from kg_extract.config import (
    DATASET_CONTEXT, DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE,
    FINAL_STATUSES, STATUS_CONTEXT_LENGTH, STATUS_ENTITY_PARSE_FAIL, STATUS_OK,
    STATUS_RELATION_PARSE_FAIL,
)
from kg_extract.guard import assert_no_label_fields, enforce_blackbox

TRANSIENT_ERRORS = {
    "APIConnectionError", "Timeout", "APITimeoutError", "ServiceUnavailableError",
    "InternalServerError", "RateLimitError", "ConnectionError", "ReadTimeout",
    "RemoteDisconnected", "BadGatewayError",
}


class TransientError(RuntimeError):
    pass


# --------------------------------------------------------------------------- io
def load_kb(path: str) -> list[dict]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            assert_no_label_fields(rec)
            docs.append(rec)
    return docs


def select_docs(docs: list[dict], sample: int | None, seed: int) -> list[dict]:
    """Deterministic pseudo-random sample by hashed doc_id (label-blind)."""
    if not sample or sample >= len(docs):
        return docs
    key = lambda d: hashlib.sha1(f"{seed}:{d['doc_id']}".encode()).hexdigest()
    chosen = {d["doc_id"] for d in sorted(docs, key=key)[:sample]}
    return [d for d in docs if d["doc_id"] in chosen]


def read_done(shard_dir: str, retry_failed: bool, model: str | None = None) -> set[str]:
    """doc_ids with a final record. Refuses to resume into a dir written by another model,
    which would silently mix two extractors in one graph."""
    done = set()
    for p in glob.glob(os.path.join(shard_dir, "part-*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # truncated last line from a killed job
                if model and r.get("model") and r["model"] != model:
                    raise SystemExit(f"{shard_dir} holds records from model {r['model']!r}, not {model!r}: "
                                     f"use a separate --out-dir per model")
                st = r.get("status")
                if st == STATUS_OK or (st in FINAL_STATUSES and not retry_failed):
                    done.add(r["doc_id"])
    return done


def check_server(api_base: str, model: str, timeout: float = 10) -> None:
    served = model.split("/", 1)[1] if model.startswith("hosted_vllm/") else None
    with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=timeout) as r:
        ids = [m["id"] for m in json.load(r)["data"]]
    if served and served not in ids:
        raise SystemExit(f"Server at {api_base} serves {ids}, not {served!r}")
    print(f"[extract] server OK, models={ids}", flush=True)


# ----------------------------------------------------------------- extraction
def _is_transient(e: BaseException) -> bool:
    names = {c.__name__ for c in type(e).__mro__}
    return bool(names & TRANSIENT_ERRORS)


def _is_context_length(e: BaseException) -> bool:
    return "ContextWindowExceededError" in {c.__name__ for c in type(e).__mro__} or \
        "context length" in str(e).lower() or "maximum context" in str(e).lower()


class Extractor:
    def __init__(self, kg, context: str):
        import dspy
        import kg_gen.steps._2_get_relations as rel_mod
        from kg_gen.steps._1_get_entities import get_entities

        self.dspy = dspy
        self.kg = kg
        self.context = context
        self.get_entities = get_entities
        self.rel_mod = rel_mod
        self._tls = threading.local()
        # Count uses of KGGen's relation fallback (= primary structured parse failed)
        # by wrapping the module-level function that get_relations() calls.
        orig = rel_mod.fallback_extraction_sig
        if not getattr(orig, "_kg_extract_wrapped", False):
            def counting(*a, **k):
                self._tls.fallback = getattr(self._tls, "fallback", 0) + 1
                return orig(*a, **k)
            counting._kg_extract_wrapped = True
            rel_mod.fallback_extraction_sig = counting

    def _call(self, fn, *a, **k):
        delay = 5
        for attempt in range(4):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                if _is_transient(e) and attempt < 3:
                    time.sleep(delay)
                    delay *= 3
                    continue
                if _is_transient(e):
                    raise TransientError(repr(e)) from e
                raise

    def run(self, doc: dict) -> dict:
        t0 = time.time()
        self._tls.fallback = 0
        rec = {"doc_id": doc["doc_id"], "status": STATUS_OK, "entities": [], "triples": [],
               "relation_fallback": 0, "error_type": None, "error": None}
        text = doc["text"]
        if not text.strip():
            rec["empty_text"] = True
            rec["latency_s"] = 0.0
            return rec
        with self.dspy.context(lm=self.kg.lm):
            try:
                ents = self._call(self.get_entities, text, is_conversation=False)
                ents = list(dict.fromkeys(e for e in (ents or []) if isinstance(e, str) and e.strip()))
                rec["entities"] = ents
            except TransientError:
                raise
            except Exception as e:  # noqa: BLE001
                rec["status"] = STATUS_CONTEXT_LENGTH if _is_context_length(e) else STATUS_ENTITY_PARSE_FAIL
                rec["error_type"], rec["error"] = type(e).__name__, str(e)[:500]
                rec["latency_s"] = time.time() - t0
                return rec
            if ents:
                try:
                    rels = self._call(self.rel_mod.get_relations, text, ents,
                                      is_conversation=False, context=self.context)
                    rec["triples"] = [list(map(str, r)) for r in dict.fromkeys(tuple(r) for r in rels)]
                except TransientError:
                    raise
                except Exception as e:  # noqa: BLE001
                    rec["status"] = STATUS_CONTEXT_LENGTH if _is_context_length(e) else STATUS_RELATION_PARSE_FAIL
                    rec["error_type"], rec["error"] = type(e).__name__, str(e)[:500]
        ent_set = set(rec["entities"])
        rec["relation_fallback"] = self._tls.fallback
        rec["unlisted_entity_triples"] = sum(1 for s, _, o in rec["triples"] if s not in ent_set or o not in ent_set)
        rec["latency_s"] = round(time.time() - t0, 3)
        return rec


def build_kggen(args):
    from kg_gen import KGGen
    return KGGen(model=args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                 api_key=args.api_key, api_base=args.api_base)


# ----------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="KGGen per-passage extraction (one shard)")
    ap.add_argument("--kb", required=True, help="kb.jsonl from export_kb (doc_id, text only)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", required=True, help="<output dir>/<dataset>[_pilot]")
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None, help="pilot: label-blind sample of N docs")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--model", default=os.environ.get("KG_MODEL", DEFAULT_MODEL))
    ap.add_argument("--api-base", default=os.environ.get("KG_API_BASE", "http://127.0.0.1:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("KG_API_KEY", "EMPTY"))
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--context", default=None, help="override DATASET_CONTEXT[dataset]")
    ap.add_argument("--retry-failed", action="store_true", help="re-run docs whose final status was a failure")
    ap.add_argument("--max-docs", type=int, default=None, help="stop after N docs (debug)")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    enforce_blackbox(input_paths=[args.kb])
    t_proc = time.time()
    job_t0 = float(os.environ.get("JOB_T0", t_proc))
    context = args.context if args.context is not None else DATASET_CONTEXT.get(args.dataset, "")

    shard_dir = os.path.join(args.out_dir, "shards")
    run_dir = os.path.join(args.out_dir, "runs")
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    docs = select_docs(load_kb(args.kb), args.sample, args.sample_seed)
    mine = docs[args.shard_index::args.num_shards]
    done = read_done(shard_dir, args.retry_failed, args.model)
    todo = [d for d in mine if d["doc_id"] not in done]
    if args.max_docs:
        todo = todo[: args.max_docs]
    print(f"[extract] v{__version__} dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
          f"selected={len(docs)} in_shard={len(mine)} done={len(mine) - len(todo)} todo={len(todo)}", flush=True)

    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}"
    restart = os.environ.get("SLURM_RESTART_COUNT", "0")  # requeued jobs keep their id
    run_path = os.path.join(run_dir, f"extract_{job_id}_{args.shard_index:04d}_r{restart}.json")
    run = {
        "kind": "extract", "kg_extract_version": __version__, "job_id": job_id,
        "shard_index": args.shard_index, "num_shards": args.num_shards, "host": socket.gethostname(),
        "gpu_name": os.environ.get("GPU_NAME"), "n_gpus": int(os.environ.get("N_GPUS", "1")),
        "model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "concurrency": args.concurrency, "context": context, "sample": args.sample,
        "sample_seed": args.sample_seed, "kb": os.path.abspath(args.kb),
        "t_job_start": job_t0, "vllm_ready_s": float(os.environ.get("VLLM_READY_S", "nan")),
        "t_extract_start": None, "t_end": None, "n_todo": len(todo), "n_written": 0,
        "n_transient_unfinished": 0, "status_counts": {},
    }

    def save_run():
        with open(run_path + ".tmp", "w") as f:
            json.dump(run, f, indent=2)
        os.replace(run_path + ".tmp", run_path)

    save_run()
    if not todo:
        run["t_extract_start"] = run["t_end"] = time.time()
        save_run()
        print("[extract] nothing to do", flush=True)
        return 0

    check_server(args.api_base, args.model)
    kg = build_kggen(args)
    ex = Extractor(kg, context)
    run["t_extract_start"] = time.time()

    out_path = os.path.join(shard_dir, f"part-{args.shard_index:04d}.jsonl")
    counts: dict[str, int] = {}
    n_written = n_transient = 0
    t_last = time.time()
    with open(out_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.concurrency) as pool:
        futs = {pool.submit(ex.run, d): d["doc_id"] for d in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except TransientError as e:
                n_transient += 1
                print(f"[extract] transient failure on {futs[fut]} (left for resume): {e}", flush=True)
                continue
            rec.update(model=args.model, temperature=args.temperature, kg_extract_version=__version__)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
            n_written += 1
            counts[rec["status"]] = counts.get(rec["status"], 0) + 1
            if time.time() - t_last > 60 or n_written == len(todo):
                el = time.time() - run["t_extract_start"]
                print(f"[extract] {n_written}/{len(todo)} docs, {n_written / el:.2f} doc/s, {counts}", flush=True)
                run.update(n_written=n_written, status_counts=counts, n_transient_unfinished=n_transient)
                save_run()
                t_last = time.time()

    run.update(t_end=time.time(), n_written=n_written, status_counts=counts, n_transient_unfinished=n_transient)
    save_run()
    print(f"[extract] finished: {json.dumps(run['status_counts'])}, transient={n_transient}", flush=True)
    return 3 if n_transient else 0  # non-zero => afterok dependants wait for a resubmit


if __name__ == "__main__":
    sys.exit(main())
