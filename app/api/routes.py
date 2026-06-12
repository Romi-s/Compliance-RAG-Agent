from typing import Optional

import filetype
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.security import APIKeyHeader

from app.agent.graph import qa_graph
from app.config import settings
from app.schemas.responses import (
    CitationResponse,
    CollectionStatsResponse,
    IngestResponse,
    QueryResponse,
)
from app.services.ingest import get_collection, ingest_pdf, ingest_text
from app.services.ratelimit import consume_quota, remaining_quota
from app.services.retriever import invalidate_bm25_cache
from app.services.seed import ensure_seeded

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Depends(_api_key_header)):
    """Owner-only gate (used for /ingest). If no key is configured, allow (local dev)."""
    if not settings.api_key:
        return
    if key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _client_ip(request: Request) -> str:
    """Real visitor IP. On Cloud Run the client sits behind Google's proxy, so the
    true IP is the first entry of X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


router = APIRouter()


@router.get("/api/limits")
async def get_limits(request: Request):
    """How many free questions this visitor has left today (for the UI counter)."""
    return {
        "free_remaining": remaining_quota(_client_ip(request)),
        "free_per_day": settings.free_queries_per_day,
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
        # Bring-your-own-key: validated lightly, used for this request only, never stored.
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


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
async def ingest_document(file: UploadFile = File(...)):
    """Owner-only: requires the X-API-Key. Public demo visitors cannot upload."""
    file_bytes = await file.read()

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {settings.max_file_size_mb} MB limit",
        )

    kind = filetype.guess(file_bytes)
    detected_mime = kind.mime if kind else None

    if detected_mime is None:
        ext = (file.filename or "").rsplit(".", 1)[-1].lower()
        if ext in ("txt", "md", "csv"):
            detected_mime = "text/plain"
        else:
            detected_mime = "application/octet-stream"

    if detected_mime == "application/pdf":
        result = ingest_pdf(file_bytes, file.filename or "document.pdf")
    elif detected_mime == "text/plain":
        text = file_bytes.decode("utf-8", errors="replace")
        result = ingest_text(text, file.filename or "document.txt")
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {detected_mime}. Upload PDF or text files.",
        )

    invalidate_bm25_cache()

    return IngestResponse(
        filename=result["filename"],
        chunks_added=result["chunks_added"],
        message=f"Ingested {result['chunks_added']} chunks from {result['filename']}",
    )
