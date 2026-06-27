# Compliance RAG Agent

A full-stack, production-style **Retrieval-Augmented Generation** app that answers natural-language questions about compliance and regulatory documents — with **source citations**. It ships with a polished web UI, a **LangGraph** agentic pipeline, **hybrid search** (vector similarity + BM25, merged with Reciprocal Rank Fusion), and is deployed to **Google Cloud Run** with **keyless GitHub Actions CI/CD**.

> **Live demo:** `https://compliance-rag-agent-kwxekadtlq-ew.a.run.app/` &nbsp;·&nbsp;

The demo is **public and cost-safe**: every visitor gets a few free questions on a server key (rate-limited, with a hard global cap), or can paste their **own OpenAI key** for unlimited use — and the server key never reaches the browser.

---

## Features

- **Hybrid retrieval** — semantic (ChromaDB cosine) + lexical (BM25) search merged with **Reciprocal Rank Fusion** for better recall than either alone.
- **Grounded answers with citations** — every answer references the exact document and page it drew from (e.g. `[1] gdpr.pdf, Page 12`).
- **Agentic pipeline** — a LangGraph `StateGraph` (validate → retrieve → generate → format) with conditional error-abort edges at each stage.
- **Built-in web UI** (served by FastAPI, same-origin) — ask box, **dynamic suggested questions generated from the indexed corpus**, a **live "under the hood" pipeline view** (vector/BM25 bars, RRF, model, citation chips), document upload, and a Google Cloud request-flow panel.
- **Public-demo cost controls** — per-visitor rate limits (questions + uploads), a global daily cap, small upload size/page caps, and optional **bring-your-own-key**. See [Cost & security](#cost--security).
- **Self-seeding corpus** — a curated GDPR knowledge base auto-loads on startup, so the demo always works out of the box.
- **Observability & evaluation** — **LangSmith** tracing on the LangGraph pipeline (per-node latency + token/cost), a Prometheus **`/metrics`** endpoint (retrieval/generation latency, retrieval-quality score), and a **Ragas** evaluation over a held-out QA set (faithfulness, relevancy, context precision/recall). See [Observability & evaluation](#observability--evaluation).
- **Cloud-native** — Dockerized, serverless on Cloud Run, secrets in Secret Manager, **CI/CD via GitHub Actions with Workload Identity Federation (no stored keys)**.

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              Ingest Pipeline                 │
  PDF / Text  ───────────►│  Extract Text ─► Chunk ─► Embed ─► ChromaDB │
                          └──────────────────────────────────────────────┘

                          ┌──────────────────────────────────────────────┐
                          │         Query Pipeline (LangGraph)           │
  Question    ───────────►│  Validate ─► Retrieve ─► Generate ─► Format │
                          │               (hybrid)   (gpt-4o-mini) (cite)│
                          └──────────────────────────────────────────────┘
                                   ┌─────────────┐
                                   │  Hybrid     │  Vector (ChromaDB cosine)
                                   │  Retrieval  │  BM25   (rank-bm25)
                                   │  Merge (RRF)│  Reciprocal Rank Fusion
                                   └─────────────┘
```

### Deployment & CI/CD

```
  git push main ─► GitHub Actions ─► gcloud run deploy --source
                   (keyless / WIF)          │
                                            ▼
                   ┌──────────────────────────────────────────┐
                   │            Cloud Run service             │
                   │   FastAPI + LangGraph + UI + ChromaDB    │
                   └───────────────┬──────────────────────────┘
                                   │ reads at runtime
                   ┌───────────────┴───────────┐   ┌──────────────┐
                   │  Secret Manager           │   │  OpenAI API  │
                   │  OPENAI_API_KEY           │   │  LLM + embed │
                   └───────────────────────────┘   └──────────────┘
```

ChromaDB runs **embedded** in the app (auto-seeded on boot). This keeps the demo to a single service. *(Vector data is per-instance and resets on cold start — fine for a demo; a separate persistent ChromaDB service is the next step for shared/durable storage.)*

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/` | None | The web UI |
| `POST` | `/query` | Public (rate-limited) | Ask a question; returns answer + citations + pipeline telemetry |
| `POST` | `/ingest` | Public (rate-limited) | Upload a small PDF/text file to index |
| `GET`  | `/collection/stats` | None | Chunk count + indexed document names |
| `GET`  | `/api/suggestions` | None | Suggested questions generated from the current corpus |
| `GET`  | `/api/limits` | None | Free questions/uploads remaining for this visitor |
| `GET`  | `/health` | None | Liveness check for Cloud Run |
| `GET`  | `/metrics` | None | Prometheus metrics (latency + retrieval-quality) |
| `GET`  | `/docs` | None | Auto-generated OpenAPI docs |

Send `X-OpenAI-Key: sk-...` on `/query` or `/ingest` to use your own key (bypasses the rate limit). An optional owner `X-API-Key` bypasses all limits.

---

## Cost & security

The demo is public, so the OpenAI key is protected on **five layers**:

1. **Key stays server-side** — injected from Secret Manager into the container; never sent to the browser.
2. **OpenAI monthly hard cap** — set in the OpenAI dashboard; the ultimate ceiling on spend.
3. **Per-visitor rate limits** — free questions/uploads per IP per day (`free_queries_per_day`, `free_uploads_per_day`).
4. **Global daily cap** — a hard ceiling across all visitors (`global_daily_cap`).
5. **Upload caps + bring-your-own-key** — small size/page limits; visitors can supply their own key for unlimited use.

Generation uses **gpt-4o-mini** by default (cheap) with **text-embedding-3-small** for vectors.

---

## Observability & evaluation

The pipeline is instrumented for both live monitoring and offline quality evaluation.

### LangSmith tracing
Every LangGraph run is traced to **LangSmith**. The `generate` node's OpenAI client is wrapped, so each trace shows the full `validate → retrieve → generate` tree with **per-node latency and token/cost**; runs carry a name, tags, and metadata so they're filterable. Enable by setting the env vars below — tracing is a **no-op when they're unset**, so it never changes production behaviour:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=RAG-chatbot
```

### Prometheus metrics — `GET /metrics`
Exposes retrieval / generation / end-to-end **latency histograms**, the **top retrieval (RRF) relevance score** (a no-LLM proxy for retrieval quality), and chunks-retrieved per query. Scrape with any Prometheus-compatible collector to watch latency SLOs and catch retrieval-quality drift.

### Ragas evaluation (`eval/`)
A held-out GDPR QA set (`eval/qa_set.json`) is scored on **faithfulness, answer relevancy, context precision, and context recall**, plus a refusal check on an out-of-scope question. It runs in **two steps with separate virtualenvs** — ragas needs an older LangChain stack that conflicts with the app's `langchain-core` 1.x, so eval deps stay isolated and never ship in the Cloud Run image:

```bash
# 1. Generate predictions with the real pipeline (serving venv)
python -m eval.generate_predictions

# 2. Score them with ragas (isolated eval venv)
python -m venv eval/.venv
eval/.venv/Scripts/python.exe -m pip install -r eval/requirements-eval.txt
eval/.venv/Scripts/python.exe eval/score_ragas.py
```

The steps hand off via `eval/predictions.json`; per-question results are written to `eval/ragas_results.csv`. Generation runs are tagged `eval` in LangSmith.

---

## Quick start (local)

```bash
cd compliance-rag-agent
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

copy .env.example .env          # then put your OpenAI key in .env
uvicorn app.main:app --reload --port 8080
```

Open **http://localhost:8080**. On first load it embeds the bundled GDPR corpus (a few seconds, shown as "warming up…"), then you can ask questions and upload documents.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI key (from `.env` locally, Secret Manager in prod) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Where the embedded vector store writes |
| `CHROMA_HOST` | — | Set to use a remote ChromaDB HTTP server instead of embedded |
| `API_KEY` | — | Optional owner key; bypasses demo limits |
| `FREE_QUERIES_PER_DAY` | `5` | Free questions per visitor (per IP) per day |
| `FREE_UPLOADS_PER_DAY` | `3` | Free uploads per visitor (per IP) per day |
| `GLOBAL_DAILY_CAP` | `300` | Hard ceiling on questions+uploads across all visitors per day |
| `MAX_UPLOAD_MB` | `5` | Size cap for public demo uploads |
| `MAX_DEMO_PDF_PAGES` | `30` | Page cap for public demo PDF uploads |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` | Chunking parameters |
| `RETRIEVAL_TOP_K` / `FINAL_TOP_K` | `10` / `5` | Candidates per method / chunks sent to the LLM |
| `LANGSMITH_TRACING` | — | Set `true` to send LangGraph traces to LangSmith |
| `LANGSMITH_API_KEY` | — | LangSmith API key (enables tracing) |
| `LANGSMITH_PROJECT` | — | LangSmith project name traces are grouped under |

---

## Deployment (Google Cloud Run)

### One-command deploy
```bash
echo -n "sk-your-key" | gcloud secrets create openai-api-key --data-file=-

gcloud run deploy compliance-rag-agent \
  --source . --region europe-west1 --allow-unauthenticated \
  --set-secrets=OPENAI_API_KEY=openai-api-key:latest \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 3 --timeout 300
```

### Continuous deployment (GitHub Actions, keyless)
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) deploys on every push to `main`. It authenticates to GCP via **Workload Identity Federation** — GitHub uses a short-lived OIDC token to impersonate a deploy service account, so **no service-account key is ever stored** in the repo.

> **Note:** `cloudbuild.yaml` in this repo is an *alternative* GCP-native pipeline (Cloud Build) kept for reference; the active CI/CD is the GitHub Actions workflow above.

---

## Project structure

```
compliance-rag-agent/
├── app/
│   ├── main.py                 # FastAPI app factory, UI mount, startup seeding
│   ├── config.py               # pydantic-settings (model, ChromaDB, demo limits)
│   ├── agent/                  # LangGraph: state, nodes, graph
│   ├── api/routes.py           # endpoints, rate limiting, bring-your-own-key
│   ├── services/
│   │   ├── ingest.py           # chunk, embed, ChromaDB upsert
│   │   ├── retriever.py        # hybrid search (vector + BM25 + RRF)
│   │   ├── ratelimit.py        # per-visitor + global cost controls
│   │   ├── seed.py             # auto-seed the demo corpus
│   │   ├── suggestions.py      # corpus-aware suggested questions
│   │   ├── metrics.py          # Prometheus metrics (latency + relevance)
│   │   └── text_extractor.py   # PDF text extraction (PyMuPDF)
│   ├── schemas/responses.py    # Pydantic v2 response models
│   ├── static/index.html       # the web UI
│   └── data/gdpr_excerpts.txt  # bundled demo knowledge base
├── eval/                       # offline evaluation (isolated deps)
│   ├── qa_set.json             # held-out GDPR QA set
│   ├── generate_predictions.py # step 1: run the pipeline -> predictions.json
│   ├── score_ragas.py          # step 2: grade with ragas -> ragas_results.csv
│   └── requirements-eval.txt   # eval-only deps (separate venv)
├── tests/                      # 25 tests (hermetic — no network/keys)
├── .github/workflows/deploy.yml# GitHub Actions CI/CD (WIF)
├── Dockerfile
└── requirements.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

25 hermetic tests (node logic, API endpoints, rate-limit/upload access, citation extraction, tokenization) — run with no API keys or external services.

## Tech stack

Python · FastAPI · LangGraph · LangChain · OpenAI (gpt-4o-mini, text-embedding-3-small) · ChromaDB · rank-bm25 · Reciprocal Rank Fusion · PyMuPDF · Pydantic v2 · Uvicorn · HTML/CSS/JS · Docker · Google Cloud Run · Secret Manager · GitHub Actions · Workload Identity Federation (OIDC) · LangSmith · Prometheus · Ragas · pytest
