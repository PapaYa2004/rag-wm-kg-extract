"""The RAG-WM launcher redirects the hard-coded Chroma path without touching files (stdlib only)."""

import os
import pathlib
import subprocess
import sys

KE = pathlib.Path(__file__).resolve().parents[1]
LAUNCH = KE / "ragwm_run" / "launch.py"
REAL_VS = KE.parent / "rag-wm" / "rag" / "vectorstore.py"
HARD = "ChromadbPath = '/workspace/ragwm/ragwm/chromadb_db'"


def _fake_ragwm(tmp):
    root = tmp / "rag-wm"
    (root / "rag").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "rag" / "__init__.py").write_text("")
    (root / "rag" / "vectorstore.py").write_text(
        f"{HARD}\n"
        "def check_collection():\n    return ChromadbPath\n"
        "if __name__ == '__main__':\n    import sys; print('VS', ChromadbPath, sys.argv[1:])\n")
    (root / "src" / "main.py").write_text(
        "import os, sys\n"
        "from rag.vectorstore import check_collection\n"
        "import fastchat.model\n"
        "import google.generativeai as palm\n"
        "import google.ai.generativelanguage as gl\n"
        "from helper import X\n"            # sibling import: script dir must be on sys.path
        "print('MAIN', check_collection(), os.getcwd() == os.environ['RAGWM_DIR'], X, sys.argv[1:])\n")
    (root / "src" / "helper.py").write_text("X = 'helper-ok'\n")
    return root


def _run(root, chroma, *args):
    env = {**os.environ, "RAGWM_DIR": str(root), "RAGWM_CHROMA_PATH": str(chroma)}
    return subprocess.run([sys.executable, str(LAUNCH), *args], env=env, capture_output=True, text=True,
                          cwd=str(root.parent))


def test_vectorstore_script_path_redirected(tmp_path):
    root = _fake_ragwm(tmp_path)
    chroma = tmp_path / "fred" / "chromadb_db"
    r = _run(root, chroma, "rag/vectorstore.py", "--eval_dataset", "nfcorpus")
    assert r.returncode == 0, r.stderr
    assert f"VS {chroma} ['--eval_dataset', 'nfcorpus']" in r.stdout
    assert HARD in (root / "rag" / "vectorstore.py").read_text(), "file on disk must be unchanged"


def test_imported_vectorstore_redirected_and_providers_stubbed(tmp_path):
    root = _fake_ragwm(tmp_path)
    chroma = tmp_path / "fred" / "chromadb_db"
    r = _run(root, chroma, "src/main.py", "--doc", "1")
    assert r.returncode == 0, r.stderr
    assert f"MAIN {chroma} True helper-ok ['--doc', '1']" in r.stdout


def test_real_ragwm_still_has_exactly_one_hard_coded_path():
    assert REAL_VS.read_text(encoding="utf-8").count(HARD) == 1
