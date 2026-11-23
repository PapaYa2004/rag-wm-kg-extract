"""Black-box separation tests (stdlib only; no kg-gen / GPU needed).

  python -m pytest knowledge_extraction/tests/test_blackbox.py
"""

import ast
import json
import os
import pathlib
import subprocess
import sys

KE = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "PYTHONPATH": str(KE)}


def _run(code: str):
    return subprocess.run([sys.executable, "-c", code], env=ENV, capture_output=True, text=True)


def test_extraction_modules_never_import_eval():
    for py in (KE / "kg_extract").glob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any(n == "kg_eval" or n.startswith("kg_eval.") for n in names), py


def test_extraction_modules_do_not_name_ground_truth_files():
    from kg_extract.guard import FORBIDDEN_PATH_PATTERNS  # noqa: F401
    import re
    pat = re.compile(r"wmuint_|wmunit\.json|init_hash|entity_hash|wm_generate|wm_prepare")
    for py in (KE / "kg_extract").glob("*.py"):
        if py.name == "guard.py":
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and not _is_docstring(node, py):
                assert not pat.search(node.value), f"{py.name}: {node.value!r}"


def _is_docstring(node, py):
    tree = ast.parse(py.read_text(encoding="utf-8"))
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body and \
                isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
            docs.add(n.body[0].value.value)
    return node.value in docs


def test_guard_blocks_ground_truth_open(tmp_path):
    gt = tmp_path / "wm_generate" / "nfcorpus" / "10" / "wmuint_inject.json"
    gt.parent.mkdir(parents=True)
    gt.write_text("[]")
    r = _run(f"from kg_extract.guard import enforce_blackbox; enforce_blackbox(); open({str(gt)!r})")
    assert r.returncode != 0 and "Black-box violation" in r.stderr


def test_guard_allows_normal_files(tmp_path):
    f = tmp_path / "kb.jsonl"
    f.write_text("{}")
    r = _run(f"from kg_extract.guard import enforce_blackbox; enforce_blackbox(); open({str(f)!r}).read()")
    assert r.returncode == 0, r.stderr


def test_eval_refuses_to_import_in_extraction_process():
    r = _run("from kg_extract.guard import enforce_blackbox; enforce_blackbox(); import kg_eval")
    assert r.returncode != 0 and "must not be imported" in r.stderr


def test_guard_refuses_if_eval_already_imported():
    r = _run("import kg_eval; from kg_extract.guard import enforce_blackbox; enforce_blackbox()")
    assert r.returncode != 0 and "Evaluation modules imported" in r.stderr


def test_record_with_label_fields_rejected():
    from kg_extract.guard import BlackBoxViolation, assert_no_label_fields
    assert_no_label_fields({"doc_id": "id_1", "text": "x"})
    try:
        assert_no_label_fields({"doc_id": "id_1", "text": "x", "change": True})
    except BlackBoxViolation:
        return
    raise AssertionError("label field accepted")


def _toy_corpus(tmp_path):
    ds = tmp_path / "datasets" / "toy"
    ds.mkdir(parents=True)
    ids = ["MED-10", "MED-2", "MED-33"]           # sorted -> id_0=MED-10, id_1=MED-2, id_2=MED-33
    orig = ds / "toy_original.jsonl"
    orig.write_text("\n".join(json.dumps({"_id": i, "title": "t", "text": f"text {i}", "metadata": {}})
                              for i in ids) + "\n")
    inject = tmp_path / "ragwm" / "wm_generate" / "toy" / "10" / "wmuint_inject.json"
    inject.parent.mkdir(parents=True)
    inject.write_text(json.dumps([
        [["Vitamin D", "Breast Cancer", "AFFECTS"], ["id_1"], [["WT one.", 1, 1]]],
        [["Statin", "Mortality", "REDUCES"], ["id_1", "id_2"], [["WT two.", 1, 1], ["WT three.", 1, 1]]],
    ]))
    return ds, orig, inject


def test_build_corpus_variants(tmp_path):
    ds, orig, inject = _toy_corpus(tmp_path)
    r = subprocess.run([sys.executable, "-m", "kg_eval.build_corpus_variants", "--dataset", "toy",
                        "--beir-corpus", str(orig), "--inject-json", str(inject),
                        "--datasets-dir", str(tmp_path / "datasets")], env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    wm = [json.loads(l) for l in (ds / "toy_watermarked.jsonl").read_text().splitlines()]
    og = [json.loads(l) for l in orig.read_text().splitlines()]
    assert [w["_id"] for w in wm] == [o["_id"] for o in og]
    assert all(set(w) == set(o) for w, o in zip(wm, og)), "no label field may be added"
    by = {w["_id"]: w["text"] for w in wm}
    assert by["MED-10"] == "text MED-10"
    assert by["MED-2"] == "text MED-2 WT one. WT two."      # appended in log order
    assert by["MED-33"] == "text MED-33 WT three."
    gt = json.loads((tmp_path / "datasets" / "_ground_truth" / "toy" / "rag_wm_ground_truth.json").read_text())
    assert gt["n_tuples"] == 2 and gt["tuples"][1]["docs"] == ["MED-2", "MED-33"]


def test_guard_blocks_original_corpus_during_watermarked_run(tmp_path):
    ds, orig, _ = _toy_corpus(tmp_path)
    wm = ds / "toy_watermarked.jsonl"
    wm.write_text(orig.read_text())
    code = ("from kg_extract.guard import enforce_blackbox; "
            f"enforce_blackbox(input_paths=[{str(wm)!r}]); open({str(orig)!r})")
    r = _run(code)
    assert r.returncode != 0 and "Black-box violation" in r.stderr
    # clean-baseline run: the original IS the declared input -> allowed
    r = _run(f"from kg_extract.guard import enforce_blackbox; enforce_blackbox(input_paths=[{str(orig)!r}]); "
             f"open({str(orig)!r}).read()")
    assert r.returncode == 0, r.stderr


def test_guard_rejects_ground_truth_as_declared_input(tmp_path):
    gt = tmp_path / "datasets" / "_ground_truth" / "toy" / "rag_wm_ground_truth.json"
    gt.parent.mkdir(parents=True)
    gt.write_text("{}")
    r = _run(f"from kg_extract.guard import enforce_blackbox; enforce_blackbox(input_paths=[{str(gt)!r}])")
    assert r.returncode != 0 and "ground-truth artefact" in r.stderr


def test_export_beir_has_only_doc_id_and_text(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"_id": f"MED-{i}", "title": "t", "text": f"text {i}",
                                            "metadata": {}}) for i in range(3)))
    out = tmp_path / "kb.jsonl"
    r = subprocess.run([sys.executable, "-m", "kg_extract.export_kb", "--beir-corpus", str(corpus),
                        "--out", str(out)], env=ENV, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 3 and all(set(x) == {"doc_id", "text"} for x in rows)
