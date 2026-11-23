"""Offline end-to-end test: export -> extract (with resume) -> merge/cluster -> evaluate,
against a fake OpenAI-compatible server. Needs the client requirements + the
vendored kg-gen on PYTHONPATH, but no GPU and no real LLM.

  PYTHONPATH=knowledge_extraction:kg-gen/src python -m pytest knowledge_extraction/tests/test_end_to_end_fake.py
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

KE = pathlib.Path(__file__).resolve().parents[1]
REPO = KE.parent
sys.path.insert(0, str(KE / "tests"))

pytest.importorskip("dspy")
import fake_openai_server  # noqa: E402

DOCS = {
    "id_0": "Statin use and breast cancer mortality in Finland.",
    "id_1": "Cholesterol and heart disease are linked; statin lowers cholesterol.",
    "id_2": "GARBAGE passage the fake model refuses to parse.",
    "id_3": "BADREL statin and mortality in heart disease.",
    "id_4": "Omega-3 and fish oil intake.",
    "id_5": "Statin use was studied. vitamin D reduces the risk of breast cancer.",   # injected (wm 1)
    "id_6": "Mortality data. Omega-3 rich fish oil reduces inflammation markers.",    # injected (wm 2)
    "id_7": "Vitamin D supplements and heart disease.",
}
INJECT = [[["Vitamin D", "Breast Cancer", "REDUCES_RISK_OF"], ["id_5"], []],
          [["Fish Oil", "Inflammation", "TREATS"], ["id_6"], []]]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    srv, base = fake_openai_server.start()
    pp = [str(KE), str(REPO / "kg-gen" / "src")]
    if importlib.util.find_spec("sentence_transformers") is None:
        pp.insert(0, str(KE / "tests" / "stubs"))
    e = {**os.environ, "PYTHONPATH": os.pathsep.join(pp), "PYTHONHASHSEED": "0",
         "DSPY_CACHEDIR": str(tmp / "dspy_cache"), "KG_API_BASE": base,
         "KG_MODEL": "hosted_vllm/fake-model", "PYTHONIOENCODING": "utf-8"}
    yield tmp, e
    srv.shutdown()


def run(e, *args):
    r = subprocess.run([sys.executable, "-m", *args], env=e, capture_output=True, text=True,
                       encoding="utf-8", timeout=900)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}"
    return r.stdout


def test_pipeline(env):
    tmp, e = env
    corpus = tmp / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"_id": k, "title": "", "text": v}) for k, v in DOCS.items()))
    kb = tmp / "kb.jsonl"
    out = tmp / "out" / "toy"
    run(e, "kg_extract.export_kb", "--beir-corpus", str(corpus), "--out", str(kb))

    common = ["--kb", str(kb), "--dataset", "nfcorpus", "--out-dir", str(out), "--concurrency", "4"]
    run(e, "kg_extract.extract", *common, "--max-docs", "3")
    first = (out / "shards" / "part-0000.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first) == 3
    run(e, "kg_extract.extract", *common)  # resume
    recs = [json.loads(l) for l in (out / "shards" / "part-0000.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["doc_id"] for r in recs) == sorted(DOCS), "resume must not duplicate or skip docs"
    by = {r["doc_id"]: r for r in recs}
    assert by["id_2"]["status"] == "entity_parse_fail"
    assert by["id_3"]["status"] == "ok" and by["id_3"]["relation_fallback"] >= 1
    assert any(t == ["vitamin D", "reduces risk of", "breast cancer"] for t in by["id_5"]["triples"])

    run(e, "kg_extract.merge", "--out-dir", str(out), "--kb", str(kb),
        "--retrieval-model", "stub-model")
    for f in ("triples_raw.jsonl", "graph_raw.json", "triples_clustered.jsonl", "graph_clustered.json",
              "entity_clusters.json", "edge_clusters.json", "stats.json"):
        assert (out / f).exists(), f
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["docs"]["n_docs_processed"] == 8
    assert abs(stats["docs"]["entity_parse_failure_rate"] - 1 / 8) < 1e-9
    assert stats["clustered"]["composition_consistent_with_kggen"]
    cl = [json.loads(l) for l in (out / "triples_clustered.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all({"subject_surface", "predicate_surface", "object_surface"} <= set(r) for r in cl)
    clusters = json.loads((out / "entity_clusters.json").read_text(encoding="utf-8"))
    assert any({"vitamin D", "Vitamin D"} <= set(m) for m in clusters.values()), clusters

    print(run(e, "kg_extract.report", "samples", "--out-dir", str(out), "--n", "5"))
    print(run(e, "kg_extract.report", "estimate", "--out-dir", str(out), "--full-docs", "3633"))

    gt = tmp / "ragwm_out" / "wm_generate" / "nfcorpus" / "10" / "wmuint_inject.json"
    gt.parent.mkdir(parents=True)
    gt.write_text(json.dumps(INJECT))
    run(e, "kg_eval.evaluate", "--out-dir", str(out), "--inject-json", str(gt), "--embed-model", "stub-model")
    rep = json.loads((out / "eval" / "eval_report.json").read_text(encoding="utf-8"))
    assert rep["raw"]["recall"]["exact_pair"] == 1.0
    assert rep["raw"]["recall"]["exact_full"] == 0.5
    assert rep["injected_coverage"]["frac_with_any_triplet"] == 1.0
    assert "any_wm_entity_merged_with_non_wm_forms" in rep["cluster_merge"]


def test_extract_cannot_read_ground_truth(env):
    tmp, e = env
    gt = tmp / "x" / "wm_generate" / "nfcorpus" / "10" / "wmuint_inject.json"
    gt.parent.mkdir(parents=True)
    gt.write_text("[]")
    r = subprocess.run([sys.executable, "-m", "kg_extract.extract", "--kb", str(gt), "--dataset", "nfcorpus",
                        "--out-dir", str(tmp / "o2")], env=e, capture_output=True, text=True)
    assert r.returncode != 0 and "Black-box violation" in r.stderr
