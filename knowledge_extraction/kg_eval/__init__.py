"""Evaluation of the Knowledge Extraction stage against RAG-WM ground truth.

This is the ONLY package allowed to read RAG-WM labels (tuples, injected ids).
It refuses to import inside an extraction process, and it runs only after
extraction has produced its outputs.
"""

import os

from kg_extract.guard import EXTRACTION_PHASE_ENV

if os.environ.get(EXTRACTION_PHASE_ENV) == "1":
    raise ImportError("kg_eval must not be imported in an extraction (black-box) process")
