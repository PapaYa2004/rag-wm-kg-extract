"""Run an unmodified RAG-WM script on OzSTAR.

RAG-WM hard-codes one path with no CLI flag:
    rag/vectorstore.py:  ChromadbPath = '/workspace/ragwm/ragwm/chromadb_db'
This launcher redirects it to $RAGWM_CHROMA_PATH in memory, both for scripts that
import `rag.vectorstore` and for running `rag/vectorstore.py` itself. The file on
disk is never changed. Every other path is passed via RAG-WM's own --basepath flags.

It also stubs the optional provider packages that `src/models/__init__.py` imports
unconditionally (fastchat for Vicuna, google.generativeai for PaLM2). Only the
OpenAI-compatible `GPT` provider is used, pointed at a local vLLM server.

Usage (cwd is irrelevant):
  python launch.py <script relative to $RAGWM_DIR> [script args...]
"""

from __future__ import annotations

import importlib
import os
import sys
import types

HARD_CODED = "ChromadbPath = '/workspace/ragwm/ragwm/chromadb_db'"


class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def _unavailable(*a, **k):
            raise RuntimeError(f"{self.__name__}.{name} is a stub: this provider is not installed on OzSTAR")
        return _unavailable


def stub_if_missing(dotted: str) -> bool:
    try:
        importlib.import_module(dotted)
        return False
    except Exception:  # noqa: BLE001
        pass
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except Exception:  # noqa: BLE001
            mod = _Stub(name)
            mod.__path__ = []  # behave like a package
            sys.modules[name] = mod
            if i > 1:
                setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
    return True


def patched_source(path: str, chroma: str) -> str:
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if os.path.normpath(path).endswith(os.path.join("rag", "vectorstore.py")):
        n = src.count(HARD_CODED)
        if n != 1:
            raise SystemExit(f"expected exactly one `{HARD_CODED}` in {path}, found {n}: RAG-WM changed?")
        src = src.replace(HARD_CODED, f"ChromadbPath = {chroma!r}")
    return src


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    ragwm = os.path.abspath(os.environ["RAGWM_DIR"])
    chroma = os.path.abspath(os.environ["RAGWM_CHROMA_PATH"])
    script = os.path.join(ragwm, sys.argv[1])
    args = sys.argv[2:]

    os.chdir(ragwm)  # RAG-WM reads model_configs/... relative to its root
    sys.path[:0] = [os.path.dirname(script), ragwm]  # as `python <script>` would, + project root
    stubbed = [m for m in ("fastchat.model", "google.generativeai", "google.ai.generativelanguage")
               if stub_if_missing(m)]
    if stubbed:
        print(f"[launch] stubbed unused providers: {stubbed}", flush=True)

    import rag.vectorstore as vs  # the module every other RAG-WM file imports
    vs.ChromadbPath = chroma
    print(f"[launch] ChromadbPath -> {chroma}", flush=True)

    sys.argv = [script] + args
    code = compile(patched_source(script, chroma), script, "exec")
    exec(code, {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__})


if __name__ == "__main__":
    main()
