"""Black-box enforcement for the extraction phase.

The attacker does not know which passages are watermarked or what the RAG-WM
tuples are. Every extraction entrypoint calls `enforce_blackbox()` first, which:

1. Marks the process as an extraction process (env var), so the evaluation
   package refuses to import inside it.
2. Fails if the evaluation package was already imported.
3. Installs a `sys.audit` hook that raises on any attempt to open a RAG-WM
   ground-truth artefact (tuple files, injection logs, hash keys, ...).

Audit hooks cannot be removed once installed, so the guard holds for the
whole process lifetime. Stdlib-only on purpose.
"""

from __future__ import annotations

import os
import re
import sys

EXTRACTION_PHASE_ENV = "KG_EXTRACT_BLACKBOX"
EVAL_PACKAGE = "kg_eval"

# File-name / path fragments that identify RAG-WM ground truth or secret keys.
# Written by rag-wm/entity_generate/generate_hash_entity.py and rag-wm/src/main.py.
FORBIDDEN_PATH_PATTERNS = [
    r"wmuint_inject",      # injected tuples + chroma ids (main.py, note RAG-WM's typo)
    r"wmuint_doc",         # generated watermark texts
    r"wmuint_verify",
    r"wmuint_stat",
    r"wmunit\.json",       # all watermark units
    r"init_hash\.json",    # secret key
    r"entity_hash\.json",
    r"relation_top\.json",
    r"entity_top(_list)?\.json",
    r"[\\/]wm_prepare[\\/]",
    r"[\\/]wm_generate[\\/]",
    r"Statistic of Watermark",
    r"[\\/]_ground_truth[\\/]",       # datasets/_ground_truth/<ds>/ (kg_eval.build_corpus_variants)
    r"rag_wm_ground_truth",
    # The clean corpus next to the watermarked one: diffing the two reveals every
    # watermark. Only allowed when it is itself the KB being extracted (baseline).
    r"_original\.jsonl$",
]
_FORBIDDEN_RE = re.compile("|".join(FORBIDDEN_PATH_PATTERNS))
_allowed: set[str] = set()

# Chroma metadata keys that label watermark passages (rag-wm/rag/vectorstore.py).
FORBIDDEN_METADATA_KEYS = ("change", "id", "title")

_installed = False


class BlackBoxViolation(RuntimeError):
    pass


def is_forbidden_path(path) -> bool:
    try:
        p = os.fspath(path)
    except TypeError:
        return False
    if isinstance(p, bytes):
        p = p.decode(errors="ignore")
    if _allowed and os.path.normcase(os.path.abspath(p)) in _allowed:
        return False
    return bool(_FORBIDDEN_RE.search(p))


def _audit(event: str, args) -> None:
    if event in ("open", "os.listdir", "os.scandir") and args:
        if is_forbidden_path(args[0]):
            raise BlackBoxViolation(
                f"Black-box violation: extraction tried to access ground-truth path {args[0]!r}"
            )


def enforce_blackbox(input_paths=()) -> None:
    """`input_paths`: the KB file(s) this process was explicitly asked to read.
    They may match `_original.jsonl` (clean-baseline run) but never a ground-truth pattern."""
    global _installed
    for p in input_paths:
        if p is None:
            continue
        if _FORBIDDEN_RE.search(os.fspath(p)) and not re.search(r"_original\.jsonl$", os.fspath(p)):
            raise BlackBoxViolation(f"Black-box violation: input {p!r} is a ground-truth artefact")
        _allowed.add(os.path.normcase(os.path.abspath(p)))
    os.environ[EXTRACTION_PHASE_ENV] = "1"
    loaded = [m for m in sys.modules if m == EVAL_PACKAGE or m.startswith(EVAL_PACKAGE + ".")]
    if loaded:
        raise BlackBoxViolation(f"Evaluation modules imported in extraction process: {loaded}")
    if not _installed:
        sys.addaudithook(_audit)
        _installed = True


def assert_no_label_fields(record: dict) -> None:
    """A KB record handed to the extractor may contain only doc_id and text."""
    extra = set(record) - {"doc_id", "text"}
    if extra:
        raise BlackBoxViolation(f"KB record carries non-text fields {sorted(extra)}")
