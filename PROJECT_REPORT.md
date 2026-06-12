# Compliance RAG Agent -- Project Report


## Project Summary

**Project name:** Compliance RAG Agent
**Role:** Sole developer (design, implementation, testing)
**Type:** Personal portfolio project
**Status:** Complete, functional

A production-style backend service that indexes compliance and regulatory documents and answers natural-language questions about them using Retrieval-Augmented Generation. The system uses hybrid search -- combining vector similarity with BM25 keyword matching via Reciprocal Rank Fusion -- to retrieve the most relevant document passages, then generates grounded answers with source citations through a LangGraph agentic pipeline.

---

## Technical Decisions and Why They Matter

### 1. Hybrid search with Reciprocal Rank Fusion (not just vector similarity)

Implemented a two-stage retrieval system that runs both **cosine-similarity search** (via ChromaDB embeddings) and **BM25 keyword search** (via rank-bm25) in parallel, then merges results using **Reciprocal Rank Fusion (RRF)**. This matters because pure vector search misses exact keyword matches (e.g., specific regulation article numbers like "Article 5(1)(a)"), while pure keyword search misses semantic paraphrases. RRF combines both ranked lists without requiring score normalization, producing better recall than either method alone. This demonstrates understanding of **information retrieval beyond basic embedding search** -- a distinction that separates production RAG systems from tutorial-level implementations.

### 2. Source citation tracking in generated answers

Designed the generation prompt to instruct the LLM to cite numbered references (`[1]`, `[2]`, etc.) corresponding to retrieved chunks. The format node then parses the answer to extract which sources were actually cited and returns structured citation metadata (source document, page number, excerpt). This is critical for compliance use cases where **traceability to source material** is a regulatory requirement, not just a nice-to-have. It demonstrates understanding of how to make LLM outputs auditable and trustworthy.

### 3. Agentic orchestration with LangGraph (consistent with Visual QA Agent)

Built the query pipeline as a LangGraph `StateGraph` with 4 nodes (validate, retrieve, generate, format) and conditional error-abort edges at every stage. This is the same architectural pattern used in the companion Visual QA Agent project, demonstrating **fluency with the LangGraph framework across multiple projects** rather than one-time usage. Each node is a pure function returning a partial state update, making individual stages testable without mocking the entire pipeline.

### 4. Document ingestion pipeline with configurable chunking

Built a separate ingestion pipeline that extracts text from PDFs (via PyMuPDF), splits into chunks using LangChain's `RecursiveCharacterTextSplitter` with configurable size and overlap, embeds via OpenAI `text-embedding-3-small`, and upserts into ChromaDB with metadata (source filename, page number, chunk index). Content-addressed chunk IDs (SHA-256 based) enable idempotent re-ingestion without duplicates. The BM25 index is lazily built from ChromaDB contents and cache-invalidated on new ingestion.

### 5. Production-grade API design

- Magic-byte MIME detection (via `filetype` library) with extension-based fallback for plain text
- Configurable file-size limits enforced before processing
- Pydantic v2 response models for type-safe API contracts
- API key authentication via `X-API-Key` header (gracefully disabled when unconfigured for local dev)
- Global exception handler to prevent stack traces leaking to clients
- FastAPI with auto-generated OpenAPI docs and `/health` endpoint for Cloud Run liveness checks
- Four endpoints covering the full lifecycle: ingest, query, collection inspection, and health

### 6. Dockerized cloud deployment (Google Cloud Run)

Containerized the application with Docker and deployed it as a serverless service on Google Cloud Run. This demonstrates understanding of **cloud-native deployment patterns** beyond just writing application code:

- **Multi-service architecture**: the FastAPI app and ChromaDB run as separate Cloud Run services, communicating via ChromaDB's HTTP client with bearer-token authentication. The app auto-detects whether to use a local `PersistentClient` (dev) or remote `HttpClient` (prod) based on the `CHROMA_HOST` environment variable.
- **CI/CD pipeline**: Cloud Build (`cloudbuild.yaml`) automates the build-push-deploy cycle -- builds the Docker image, pushes to Artifact Registry, and deploys to Cloud Run with Secret Manager references for API keys.
- **Secret management**: all credentials (OpenAI API key, app API key, ChromaDB token) stored in GCP Secret Manager and injected at deploy time, never baked into the image or checked into source control.
- **Docker Compose for local dev**: `docker-compose.yml` runs both services locally with a single command, mirroring the production topology.

### 7. Test architecture

- **Unit tests** for each graph node function in isolation (validation edge cases, citation extraction, whitespace handling, error propagation, duplicate citation deduplication, text truncation)
- **Integration tests** for the HTTP layer with mocked services, verifying status codes for valid requests, oversized files, unsupported MIME types, pipeline errors, and empty collections
- **Retriever tests** for tokenization logic (case normalization, punctuation stripping)
- **Auth tests** for API key enforcement (missing key, valid key, wrong key)
- Test suite runs without API keys or external dependencies (25 tests)

---

## Skills Demonstrated

| Category | Specifics |
|---|---|
| **RAG Systems** | Document chunking strategies (recursive character splitting with overlap), embedding generation (OpenAI text-embedding-3-small), vector storage (ChromaDB with cosine similarity), hybrid retrieval (vector + BM25), Reciprocal Rank Fusion for result merging, citation tracking and source attribution |
| **Information Retrieval** | BM25 keyword scoring (Okapi BM25), vector similarity search, rank fusion algorithms, two-stage retrieval pipelines, lazy index construction with cache invalidation |
| **Agent/LLM Systems** | LangGraph StateGraph design, conditional edge routing, stateful pipeline with error propagation, pure-function node architecture, grounded generation with citation prompting |
| **Cloud & DevOps** | Docker containerization, Docker Compose multi-service setup, Google Cloud Run (serverless), Cloud Build CI/CD pipeline, Artifact Registry, GCP Secret Manager, health check endpoints |
| **Backend Development** | FastAPI REST API, async file upload handling, Pydantic v2 validation, pydantic-settings for env config, API key authentication, Uvicorn ASGI server |
| **Document Processing** | PDF text extraction with PyMuPDF, configurable page limits, content-addressed deduplication (SHA-256 chunk IDs), batch upsert to vector store |
| **Software Design** | Clean separation of concerns (agent/api/services/schemas layers), TypedDict state schema, persistent vector storage, lazy initialization pattern for BM25 index |
| **Testing** | pytest unit and integration tests, mock-based isolation of external dependencies, edge-case coverage for validation, citation, and auth logic |

---

## Tech Stack (for keyword matching)

Python, FastAPI, LangGraph, LangChain, OpenAI API, GPT-4o, text-embedding-3-small, ChromaDB, BM25, Okapi BM25, rank-bm25, Reciprocal Rank Fusion (RRF), Retrieval-Augmented Generation (RAG), hybrid search, vector search, keyword search, PyMuPDF, Pydantic v2, pydantic-settings, Uvicorn, REST API, pytest, httpx, PDF processing, document chunking, text embeddings, vector database, agentic AI, source citation, compliance AI, regulatory document analysis, Docker, Docker Compose, Google Cloud Run, Google Cloud Build, Google Artifact Registry, GCP Secret Manager, CI/CD, serverless, containerization, microservices

---

## Quantifiable Highlights

- 4-node agentic pipeline with conditional error routing at every stage
- 2 parallel retrieval methods (vector + BM25) merged with Reciprocal Rank Fusion
- Configurable chunking (default 512 characters, 64 overlap) with 5 separator levels
- Handles PDF documents up to 100 pages with page-level metadata tracking
- 25 automated tests covering unit, integration, auth, and retriever layers
- Zero external dependencies required to run the test suite
- 4 API endpoints covering full document lifecycle (ingest, query, inspect, health)
- Configurable via 13 environment variables with sensible defaults
- Dockerized with multi-service Docker Compose for local dev
- CI/CD pipeline via Cloud Build (build, push, deploy in 3 steps)
- Secrets managed via GCP Secret Manager (zero credentials in source control)

---

## Suggested CV Bullet Points

For a resume or CV, adapt these to match the target job description:

- **Designed and built** a Compliance RAG backend service using FastAPI and LangGraph that indexes regulatory documents and answers natural-language questions with source citations via GPT-4o
- **Implemented** hybrid retrieval combining vector similarity (ChromaDB) with BM25 keyword search, merged using Reciprocal Rank Fusion for higher recall than either method alone
- **Engineered** an end-to-end document ingestion pipeline with configurable chunking (LangChain RecursiveCharacterTextSplitter), OpenAI embeddings (text-embedding-3-small), and content-addressed deduplication
- **Built** source citation tracking that parses LLM-generated references and maps them back to specific documents and pages, enabling auditable compliance answers
- **Architected** a LangGraph StateGraph with conditional error-abort edges at every stage, matching the same agentic pattern used across multiple portfolio projects
- **Dockerized** the application and deployed to Google Cloud Run as a serverless service with CI/CD via Cloud Build, Artifact Registry for image storage, and GCP Secret Manager for credential management
- **Wrote** 25 automated tests (unit + integration + auth + retriever) achieving full coverage of validation, citation extraction, API key enforcement, and HTTP endpoints without requiring external API keys

---

## How This Relates to the Visual QA Agent

This project and the Visual QA Agent share the same architectural pattern (LangGraph StateGraph, conditional error edges, pure-function nodes, FastAPI + Pydantic) but solve different problems:

| Aspect | Visual QA Agent | Compliance RAG Agent |
|---|---|---|
| **Input** | Images and PDFs (visual) | PDFs and text (textual) |
| **Processing** | PDF-to-image rendering, VLM inference | Text extraction, chunking, embedding, vector storage |
| **Retrieval** | None (direct model query) | Hybrid search (vector + BM25 + RRF) |
| **Model** | Vision-language models (Claude, GPT-4o) | Text LLM (GPT-4o) with retrieved context |
| **Output** | Direct visual answer | Grounded answer with source citations |
| **Deployment** | Local only | Dockerized, Cloud Run serverless, CI/CD |

Together they demonstrate: multi-modal AI, RAG pipelines, agentic orchestration, information retrieval, cloud-native deployment, and production API design -- using a consistent, testable architecture.

---

## How to Use This Document

Feed this document to an LLM with a prompt like:

```
Using the attached project report, write a CV entry / cover letter paragraph / LinkedIn summary
for an AI/ML Engineer targeting [specific job description]. Emphasize the skills most relevant
to the role and use the quantifiable highlights where they strengthen the narrative.
```

The structured format (skills table, tech stack keywords, bullet points) is designed to give the LLM specific, factual material to work with rather than generating generic claims.
