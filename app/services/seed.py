"""Auto-seed the demo knowledge base from bundled default documents.

Drop any .pdf / .txt / .md / .csv files into  app/data/  and they are ingested on
first use as the always-on "default" corpus (shown read-only in the UI). Cloud
Run's filesystem is ephemeral, so this re-runs on every cold start, keeping the
defaults present even though uploads do not persist.
"""

import os
import threading

from app.services.ingest import get_collection, ingest_pdf, ingest_text
from app.services.retriever import invalidate_bm25_cache

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_EXTS = (".pdf", ".txt", ".md", ".csv")

_lock = threading.Lock()
_seeded = False


def default_sources() -> list[str]:
    """Filenames bundled in app/data/ — the always-on, non-removable default corpus."""
    if not os.path.isdir(_DATA_DIR):
        return []
    return sorted(f for f in os.listdir(_DATA_DIR) if f.lower().endswith(_EXTS))


def ensure_seeded() -> None:
    """Ingest every bundled default document if the collection is empty. Idempotent."""
    global _seeded
    if _seeded:
        return
    with _lock:
        if _seeded:
            return
        try:
            if get_collection().count() > 0:
                _seeded = True
                return

            seeded_any = False
            for name in default_sources():
                path = os.path.join(_DATA_DIR, name)
                try:
                    if name.lower().endswith(".pdf"):
                        with open(path, "rb") as fh:
                            ingest_pdf(fh.read(), name)
                    else:
                        with open(path, "r", encoding="utf-8", errors="replace") as fh:
                            ingest_text(fh.read(), name)
                    seeded_any = True
                except Exception:
                    continue  # skip a bad file, keep going

            if seeded_any:
                invalidate_bm25_cache()
                _seeded = True
        except Exception:
            # No OpenAI key yet, or ChromaDB unreachable -- leave unseeded and retry later.
            pass
