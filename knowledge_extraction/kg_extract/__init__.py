"""Knowledge Extraction stage: builds an entity-relation KG from a (possibly
watermarked) RAG knowledge base with KGGen, under a strict black-box constraint.

Keep this module import-light (stdlib only): `export_kb` runs in a minimal
environment that has chromadb==0.5.20 but not kg-gen.
"""

__version__ = "0.3.0"
