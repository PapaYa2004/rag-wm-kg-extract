"""Shared defaults. Stdlib only."""

from __future__ import annotations

# Domain context handed to KGGen's relation-extraction prompt. It describes the
# public dataset only and says nothing about watermarks.
DATASET_CONTEXT = {
    "nfcorpus": "The text is a biomedical / nutrition research abstract (NFCorpus, "
                "PubMed-style articles on diet, nutrition, disease and medicine).",
    "trec-covid": "The text is a biomedical research abstract about COVID-19 and coronaviruses (TREC-COVID / CORD-19).",
    "nq": "The text is a Wikipedia passage (Natural Questions).",
    "hotpotqa": "The text is a Wikipedia passage (HotpotQA).",
    "msmarco": "The text is a web passage returned by a search engine (MS MARCO).",
}

DEFAULT_MODEL = "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_RETRIEVAL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SEMHASH_THRESHOLD = 0.95  # KGGen's recommended value

# Record status values in the per-shard checkpoint files.
STATUS_OK = "ok"
STATUS_ENTITY_PARSE_FAIL = "entity_parse_fail"
STATUS_RELATION_PARSE_FAIL = "relation_parse_fail"
STATUS_CONTEXT_LENGTH = "context_length"
FINAL_STATUSES = {STATUS_OK, STATUS_ENTITY_PARSE_FAIL, STATUS_RELATION_PARSE_FAIL, STATUS_CONTEXT_LENGTH}
