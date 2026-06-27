"""Prometheus metrics for the RAG pipeline.

Exposes the two things you actually want to watch on a production RAG service:

  * latency  -- broken out per stage (retrieval vs generation) and end-to-end, so
                you can tell *where* a slow request spent its time.
  * relevance -- the top fused (RRF) retrieval score per query, as a cheap, no-LLM
                proxy for "did we retrieve anything good?". A drop in this
                distribution is an early warning that retrieval quality is drifting
                (e.g. the corpus changed, embeddings went stale).

Scrape with any Prometheus-compatible collector at  GET /metrics .

Metrics are recorded at the API boundary (app/api/routes.py), not inside the
graph nodes, so direct graph invocations (the Ragas eval, tests) don't pollute the
serving metrics.
"""

from prometheus_client import Counter, Histogram

# Seconds. Retrieval is typically sub-second; an LLM generation is a few seconds.
_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

# RRF scores are small: a rank-0 hit in one ranked list is ~1/(60+1)=0.016, and a
# doc that tops *both* the vector and BM25 lists is ~0.033. Buckets reflect that.
_RRF_BUCKETS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05)

queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries processed, labelled by pipeline outcome.",
    ["outcome"],  # "success" | "error"
)

retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Time spent in hybrid (vector + BM25) retrieval.",
    buckets=_LATENCY_BUCKETS,
)

generation_latency_seconds = Histogram(
    "rag_generation_latency_seconds",
    "Time spent in the LLM generation call.",
    buckets=_LATENCY_BUCKETS,
)

query_latency_seconds = Histogram(
    "rag_query_latency_seconds",
    "End-to-end latency of a /query request (validate -> format).",
    buckets=_LATENCY_BUCKETS,
)

top_relevance_score = Histogram(
    "rag_top_relevance_score",
    "Top retrieved chunk's fused RRF score per query (retrieval-quality proxy).",
    buckets=_RRF_BUCKETS,
)

chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned by retrieval for a query.",
    buckets=(0, 1, 2, 3, 4, 5, 6, 8, 10),
)
