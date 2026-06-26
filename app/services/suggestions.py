"""Dynamic suggested questions, generated from the current knowledge base.

The LLM is asked to propose a few natural questions about whatever documents are
currently indexed, so the chips update when a user uploads a new PDF. The result
is cached and only regenerated when the corpus size changes, keeping cost to
roughly one cheap call per change. Falls back to defaults if anything fails.
"""

import threading
from typing import List, Optional

from app.config import settings
from app.services.ingest import get_collection
from app.services.seed import default_sources

_DEFAULTS = [
    "What is the right to erasure?",
    "What are the penalties for non-compliance?",
    "When is a Data Protection Officer required?",
    "What rights do data subjects have?",
]

_lock = threading.Lock()
_cache_sig: Optional[int] = None
_cache_qs: List[str] = list(_DEFAULTS)


def _sample_corpus(collection, max_sources: int = 6, per_source: int = 4, char_limit: int = 6000) -> str:
    """Pull a few chunks from EACH indexed document so suggestions reflect the whole
    corpus (especially freshly uploaded files) instead of just the first chunks."""
    try:
        meta = collection.get(include=["metadatas"])
    except Exception:
        meta = {"metadatas": []}

    sources: List[str] = []
    for m in meta.get("metadatas") or []:
        src = (m or {}).get("source")
        if src and src not in sources:
            sources.append(src)
    # Uploaded docs first, bundled default docs last, so uploads drive the questions.
    defaults = set(default_sources())
    sources.sort(key=lambda s: s in defaults)

    if not sources:
        data = collection.get(include=["documents"], limit=12)
        return "\n\n".join(data.get("documents") or [])[:char_limit]

    parts: List[str] = []
    for src in sources[:max_sources]:
        try:
            d = collection.get(where={"source": src}, include=["documents"], limit=per_source)
            parts.extend(d.get("documents") or [])
        except Exception:
            continue
    return "\n\n".join(parts)[:char_limit]


def _generate(collection, api_key: Optional[str]) -> Optional[List[str]]:
    key = api_key or settings.openai_api_key
    if not key:
        return None
    try:
        sample = _sample_corpus(collection)
        if not sample.strip():
            return None

        from openai import OpenAI

        client = OpenAI(api_key=key)
        prompt = (
            "Below are excerpts from documents in a knowledge base. Write 4 short, "
            "natural questions a user might ask to explore them. Return ONLY the "
            "questions, one per line, no numbering or bullets, each under 12 words.\n\n"
            + sample
        )
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.4,
        )
        text = resp.choices[0].message.content or ""
        questions = [
            line.strip().lstrip("-•0123456789. ").strip().strip('"')
            for line in text.splitlines()
            if line.strip()
        ]
        questions = [q for q in questions if len(q) > 5][:4]
        return questions or None
    except Exception:
        return None


def get_suggestions(api_key: Optional[str] = None) -> List[str]:
    """Suggested questions for the current corpus (cached until its size changes)."""
    global _cache_sig, _cache_qs
    try:
        collection = get_collection()
        count = collection.count()
    except Exception:
        return list(_DEFAULTS)

    if count == 0:
        return list(_DEFAULTS)

    with _lock:
        if count == _cache_sig:
            return list(_cache_qs)
        generated = _generate(collection, api_key)
        if generated:
            _cache_sig = count
            _cache_qs = generated
            return list(generated)
    # Generation unavailable (e.g. no key yet) -- serve defaults without caching,
    # so it retries once a key is available.
    return list(_DEFAULTS)
