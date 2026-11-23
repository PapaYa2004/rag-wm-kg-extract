"""Verify that the kg-gen on sys.path has exactly the API this stage relies on,
and print a fingerprint of its source for the record.

The stage targets the kg-gen *source checkout* vendored at <repo>/kg-gen/src
(GitHub main, with deduplicate()/LLMDeduplicate). PyPI kg-gen 0.4.0 has a
different, older API: its cluster() puts all entities of the graph into one
prompt, which does not fit an 8B model's context on a corpus-level graph.

  python -m kg_extract.check_kggen
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys


def source_fingerprint(pkg_dir: str) -> str:
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(pkg_dir)):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if fn.endswith((".py", ".txt")):
                p = os.path.join(root, fn)
                h.update(os.path.relpath(p, pkg_dir).replace("\\", "/").encode())
                with open(p, "rb") as f:
                    h.update(f.read())
    return h.hexdigest()


def check() -> dict:
    import kg_gen
    from kg_gen import KGGen
    from kg_gen.models import Graph
    from kg_gen.steps import _2_get_relations as rel_mod
    from kg_gen.steps._1_get_entities import get_entities
    from kg_gen.utils.deduplicate import DeduplicateList
    from kg_gen.utils.llm_deduplicate import LLMDeduplicate

    problems = []
    fields = set(Graph.model_fields)
    for f in ("entities", "edges", "relations", "entity_clusters", "edge_clusters"):
        if f not in fields:
            problems.append(f"Graph lacks field {f}")
    for name in ("aggregate", "deduplicate", "cluster", "export_graph"):
        if not hasattr(KGGen, name):
            problems.append(f"KGGen.{name} missing")
    if "context" not in inspect.signature(rel_mod.get_relations).parameters:
        problems.append("get_relations has no context parameter")
    if not hasattr(rel_mod, "fallback_extraction_sig"):
        problems.append("get_relations fallback hook missing")
    if "is_conversation" not in inspect.signature(get_entities).parameters:
        problems.append("get_entities signature changed")
    for attr in ("original_map", "items_map", "deduplicate"):
        if not hasattr(DeduplicateList(), attr):
            problems.append(f"DeduplicateList.{attr} missing")
    try:  # live call: catches semhash API drift (needs the potion model in the HF cache)
        dl = DeduplicateList(0.95)
        dl.deduplicate(["vitamin D", "vitamin d", "heart disease"])
        if dl.items_map[dl.original_map["vitamin d"]] != dl.items_map[dl.original_map["vitamin D"]]:
            problems.append("SemHash did not merge 'vitamin D'/'vitamin d'")
    except Exception as e:  # noqa: BLE001
        problems.append(f"DeduplicateList failed (semhash version?): {type(e).__name__}: {e}")
    for attr in ("cluster", "deduplicate"):
        if not hasattr(LLMDeduplicate, attr):
            problems.append(f"LLMDeduplicate.{attr} missing")
    init_params = inspect.signature(KGGen.__init__).parameters
    for p in ("model", "temperature", "max_tokens", "api_key", "api_base", "retrieval_model"):
        if p not in init_params:
            problems.append(f"KGGen.__init__ lacks {p}")

    pkg_dir = os.path.dirname(kg_gen.__file__)
    import dspy
    import litellm
    info = {
        "kg_gen_path": pkg_dir,
        "kg_gen_source_sha256": source_fingerprint(pkg_dir),
        "graph_fields": sorted(fields),
        "dspy": getattr(dspy, "__version__", "?"),
        "litellm": getattr(litellm, "version", None) or _dist_version("litellm"),
        "python": sys.version.split()[0],
        "problems": problems,
    }
    return info


def _dist_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return "?"


def main():
    info = check()
    print(json.dumps(info, indent=2))
    if info["problems"]:
        print("kg-gen API check FAILED", file=sys.stderr)
        return 1
    print("kg-gen API check OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
