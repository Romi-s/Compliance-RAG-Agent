import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.services.seed import ensure_seeded

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Seed the demo knowledge base in a background thread so startup and the
    # /health probe stay fast even while embeddings are being computed.
    threading.Thread(target=ensure_seeded, daemon=True).start()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Compliance RAG Agent",
        description=(
            "Retrieval-Augmented Generation for compliance and regulatory documents. "
            "Hybrid search (vector + BM25) with source citations."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_code": "INTERNAL_ERROR"},
        )

    # Serve the demo UI. Mounted last so the API routes above take precedence;
    # html=True makes "/" return index.html.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

    return app


app = create_app()
