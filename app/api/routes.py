from typing import Optional

import filetype
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile

from app.agent.graph import qa_graph
from app.config import settings
from app.schemas.responses import (
    CitationResponse,
    CollectionStatsResponse,
    IngestResponse,
    QueryResponse,
    RetrievedRef,
)
from app.services.ingest import get_collection, ingest_pdf, ingest_text
from app.services.ratelimit import (
    consume_quota,
    consume_upload,
    remaining_quota,
    remaining_uploads,
)
from app.services.retriever import invalidate_bm25_cache
from app.services.seed import ensure_seeded
from app.services.suggestions import get_suggestions


def _client_ip(request: Request) -> str:
    """Real visitor IP. On Cloud Run the client sits behind Google's proxy, so the
    true IP is the first entry of X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_owner(x_api_key: Optional[str]) -> bool:
    """True when the caller presents the configured owner key (unlimited access)."""
    return bool(settings.api_key) and x_api_key == settings.api_key


router = APIRouter()


@router.get("/api/suggestions")
async def suggestions():
    """Suggested questions generated from the current knowledge base."""
    ensure_seeded()
    return {"suggestions": get_suggestions()}


@router.get("/api/limits")
async def get_limits(request: Request):
    """Free questions/uploads this visitor has left today (for the UI counters)."""
    ip = _client_ip(request)
    return {
        "free_remaining": remaining_quota(ip),
        "free_per_day": settings.free_queries_per_day,
        "uploads_remaining": remaining_uploads(ip),
        "uploads_per_day": settings.free_uploads_per_day,
    }


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: Request,
    question: str = Form(...),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    own_key = (x_openai_key or "").strip()
    used_own_key = bool(own_key)
    free_remaining: Optional[int] = None

    if used_own_key:
        # Bring-your-own-key: used for this request only, never stored.
        if not own_key.startswith("sk-"):
            raise HTTPException(
                status_code=400,
                detail="That doesn't look like an OpenAI key (it should start with 'sk-').",
            )
        request_key: Optional[str] = own_key
    else:
        allowed, free_remaining = consume_quota(_client_ip(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    "You've used up the free demo questions for now. Add your own "
                    "OpenAI key to keep going, or come back later."
                ),
            )
        request_key = None  # fall back to the server's key

    ensure_seeded()

    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "error": None,
        "api_key": request_key,
    }

    result = qa_graph.invoke(initial_state)

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    try:
        total_vectors = get_collection().count()
    except Exception:
        total_vectors = None

    return QueryResponse(
        answer=result["answer"],
        citations=[
            CitationResponse(source=c["source"], page=c["page"], text=c["text"])
            for c in result["citations"]
        ],
        model=settings.openai_model,
        chunks_used=len(result["retrieved_chunks"]),
        free_remaining=free_remaining,
        used_own_key=used_own_key,
        total_vectors=total_vectors,
        retrieved=[
            RetrievedRef(source=c["source"], page=c["page"], text=c["text"][:200])
            for c in result["retrieved_chunks"]
        ],
    )


@router.get("/collection/stats", response_model=CollectionStatsResponse)
async def collection_stats():
    """Public, read-only: lets the UI show the chunk/document count."""
    ensure_seeded()
    collection = get_collection()
    count = collection.count()

    sources: list[str] = []
    if count > 0:
        all_meta = collection.get(include=["metadatas"])
        sources = sorted({m["source"] for m in all_meta["metadatas"]})

    return CollectionStatsResponse(total_chunks=count, sources=sources)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_openai_key: Optional[str] = Header(default=None, alias="X-OpenAI-Key"),
):
    """Public upload, but cost-protected: per-visitor daily limit and a small size
    cap. The owner key or a bring-your-own OpenAI key bypasses the rate limit."""
    owner = _is_owner(x_api_key)
    own_key = (x_openai_key or "").strip()
    if own_key and not own_key.startswith("sk-"):
        raise HTTPException(
            status_code=400,
            detail="That doesn't look like an OpenAI key (it should start with 'sk-').",
        )

    file_bytes = await file.read()

    # Size cap (owner gets the larger limit).
    limit_mb = settings.max_file_size_mb if owner else settings.max_upload_mb
    if len(file_bytes) > limit_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {limit_mb} MB limit")

    # Detect type (cheap) before spending any quota.
    kind = filetype.guess(file_bytes)
    detected_mime = kind.mime if kind else None
    if detected_mime is None:
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        detected_mime = "text/plain" if ext in ("txt", "md", "csv") else "application/octet-stream"
    if detected_mime not in ("application/pdf", "text/plain"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {detected_mime}. Upload PDF or text files.",
        )

    # Rate limit only public visitors (owner and bring-your-own-key are exempt).
    if not owner and not own_key:
        allowed, _ = consume_upload(_client_ip(request))
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=(
                    "You've reached today's upload limit. Add your own OpenAI key to "
                    "keep uploading, or come back later."
                ),
            )

    request_key: Optional[str] = own_key or None
    max_pages = None if owner else settings.max_demo_pdf_pages

    try:
        if detected_mime == "application/pdf":
            result = ingest_pdf(
                file_bytes, file.filename or "document.pdf",
                api_key=request_key, max_pages=max_pages,
            )
        else:
            text = file_bytes.decode("utf-8", errors="replace")
            result = ingest_text(text, file.filename or "document.txt", api_key=request_key)
    except ValueError:
        # require_api_key(): no OpenAI key on the server and none supplied by the user.
        raise HTTPException(
            status_code=503,
            detail=(
                "This demo has no OpenAI key configured. Set OPENAI_API_KEY on the "
                "server, or click '🔑 Use your own key' and upload again."
            ),
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail=(
                "Couldn't embed the document — the OpenAI request failed. Check the "
                "API key and that the account has credit."
            ),
        )

    invalidate_bm25_cache()

    return IngestResponse(
        filename=result["filename"],
        chunks_added=result["chunks_added"],
        message=f"Ingested {result['chunks_added']} chunks from {result['filename']}",
    )
