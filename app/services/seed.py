"""Auto-seed the demo knowledge base.

Cloud Run's filesystem is ephemeral, so the ChromaDB collection starts empty on
every cold start. This module loads a bundled, curated compliance document into
the vector store the first time it's needed, making the demo self-contained.
"""

import os
import threading

from app.services.ingest import get_collection, ingest_text
from app.services.retriever import invalidate_bm25_cache

_DEMO_DOC = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "gdpr_excerpts.txt"
)
_lock = threading.Lock()
_seeded = False


def ensure_seeded() -> None:
    """Ingest the bundled demo document if the collection is empty. Idempotent."""
    global _seeded
    if _seeded:
        return
    with _lock:
        if _seeded:
            return
        try:
            collection = get_collection()
            if collection.count() > 0:
                _seeded = True
                return
            if not os.path.exists(_DEMO_DOC):
                return
            with open(_DEMO_DOC, "r", encoding="utf-8") as fh:
                text = fh.read()
            ingest_text(text, os.path.basename(_DEMO_DOC))
            invalidate_bm25_cache()
            _seeded = True
        except Exception:
            # No OpenAI key yet, or ChromaDB unreachable -- leave unseeded and retry later.
            pass
