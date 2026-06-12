import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from app.agent.state import RetrievedChunk
from app.config import settings
from app.services.ingest import embed_texts, get_collection

_bm25_cache: Optional[Tuple[BM25Okapi, list, list, list]] = None


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def _build_bm25_index():
    global _bm25_cache
    collection = get_collection()
    all_docs = collection.get(include=["documents", "metadatas"])

    if not all_docs["documents"]:
        _bm25_cache = None
        return

    tokenized = [_tokenize(doc) for doc in all_docs["documents"]]
    bm25 = BM25Okapi(tokenized)
    _bm25_cache = (
        bm25,
        all_docs["ids"],
        all_docs["documents"],
        all_docs["metadatas"],
    )


def invalidate_bm25_cache():
    global _bm25_cache
    _bm25_cache = None


def hybrid_retrieve(
    query: str, top_k: Optional[int] = None, api_key: Optional[str] = None
) -> List[RetrievedChunk]:
    if top_k is None:
        top_k = settings.final_top_k
    retrieval_k = settings.retrieval_top_k

    collection = get_collection()
    doc_count = collection.count()
    if doc_count == 0:
        return []

    # --- vector search ---
    query_embedding = embed_texts([query], api_key=api_key)[0]
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(retrieval_k, doc_count),
        include=["documents", "metadatas", "distances"],
    )

    # --- BM25 search ---
    global _bm25_cache
    if _bm25_cache is None:
        _build_bm25_index()

    bm25_ranked: list[dict] = []
    if _bm25_cache is not None:
        bm25, ids, docs, metas = _bm25_cache
        scores = bm25.get_scores(_tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:retrieval_k]
        bm25_ranked = [
            {
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
            }
            for i in top_indices
            if scores[i] > 0
        ]

    # --- Reciprocal Rank Fusion ---
    K = 60
    rrf_scores: dict[str, float] = {}
    doc_lookup: dict[str, dict] = {}

    if vector_results["ids"] and vector_results["ids"][0]:
        for rank, doc_id in enumerate(vector_results["ids"][0]):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
            doc_lookup[doc_id] = {
                "document": vector_results["documents"][0][rank],
                "metadata": vector_results["metadatas"][0][rank],
            }

    for rank, item in enumerate(bm25_ranked):
        doc_id = item["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (K + rank + 1)
        if doc_id not in doc_lookup:
            doc_lookup[doc_id] = {
                "document": item["document"],
                "metadata": item["metadata"],
            }

    sorted_ids = sorted(
        rrf_scores, key=lambda x: rrf_scores[x], reverse=True
    )[:top_k]

    return [
        RetrievedChunk(
            text=doc_lookup[doc_id]["document"],
            source=doc_lookup[doc_id]["metadata"]["source"],
            page=doc_lookup[doc_id]["metadata"]["page"],
            chunk_index=doc_lookup[doc_id]["metadata"]["chunk_index"],
            score=rrf_scores[doc_id],
        )
        for doc_id in sorted_ids
    ]
