# H3Chat Agent Playbook

This document briefs an AI agent on how to operate the H3Chat service (FastAPI + pgvector RAG for cardiology device manuals).

## What this app does
- Serves a RAG API for pacemaker/defibrillator manuals. Users ask device questions; the API retrieves embedded chunks from PostgreSQL (pgvector) and can generate answers via Ollama.
- Two main endpoints: semantic search and question answering with source citations.

## Key components
- FastAPI app in `main.py`.
- PDF ingestion pipeline in `ingest_pdfs.py` to extract, chunk, embed, and store manuals.
- PostgreSQL with pgvector extension. Tables expected: `reference_documents` (metadata) and `document_chunks` (content, embeddings, is_table flag).
- Embeddings via SentenceTransformers (default `Qwen/Qwen3-Embedding-0.6B`). LLM generation via Ollama (default model `MedAIBase/MedGemma1.5:4b`).

## Runtime configuration (env vars)
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASS`: PostgreSQL connection (pgvector enabled).
- `EMBEDDING_MODEL` (optional): SentenceTransformers model id. Defaults to `Qwen/Qwen3-Embedding-0.6B`.
- `OLLAMA_URL`: Full URL to Ollama chat/completions endpoint (e.g., `http://localhost:11434/api/generate`).
- `LLM_MODEL_NAME` (optional): Ollama model name. Defaults to `MedAIBase/MedGemma1.5:4b`.

## Start the API
- Command: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
- Docs: `http://localhost:8000/docs` (served locally with bundled Swagger assets).

## API contracts
- `POST /api/search`
  - Body: `{ "query": str, "limit": int=5 }`
  - Returns list of search hits with metadata, chunk content, page number, and similarity.
- `POST /api/ask`
  - Body: same as search.
  - Behavior: retrieves top-N chunks, builds strict system prompt to avoid hallucination, calls Ollama, returns `{ answer, sources }` where `sources` mirrors search results.
  - If no chunks found, responds with message and empty sources.
- `POST /api/upload_pdf`
  - Multipart form fields: `file` (pdf), `title`, `manufacturer`, `device_model`, `doc_type`, `year`.
  - Returns `job_id` and `status_url`; ingestion runs in background and leverages the same pipeline as `ingest_pdfs.py`.
- `GET /api/upload_pdf/{job_id}/status`
  - Poll to see progress: status, percent, message, and chunk counters.
- `POST /api/upload_md`
  - Multipart form fields: `file` (markdown), `title`, `manufacturer`, `device_model`, `doc_type`, `year`.
  - Returns `job_id` and `status_url`; ingestion runs in background and leverages the Markdown ingestion pipeline.
- `GET /api/upload_md/{job_id}/status`
  - Poll to see progress: status, percent, message, and chunk counters.

## Ingestion workflow (run before serving queries)
1) Ensure PostgreSQL has pgvector and tables `reference_documents` and `document_chunks` with matching embedding dimension (384 for Qwen3 small embedding; adjust schema if model changes).
2) Place PDFs under a known path (e.g., `manuals/`).
3) Run `ingest_pdfs.py` (edit the `process_pdf` call or import and call it):
   - Converts PDF to Markdown via `pymupdf4llm`.
   - Splits into 1000-char chunks with 100 overlap using `MarkdownTextSplitter`.
   - Embeds chunks locally; inserts document metadata then chunks + embeddings into PostgreSQL. Flags table-like chunks via a simple heuristic.

## External dependencies/services
- PostgreSQL with pgvector extension and correct schema.
- Ollama running and serving the configured model at `OLLAMA_URL`.
- SentenceTransformers model available locally (download on first use).

## Operational notes
- App loads embedding model at startup in `lifespan` to keep it in memory; if not loaded, endpoints return 503.
- Similarity uses cosine distance operator `<=>`; higher `similarity` (1 - distance) is better.
- Generation prompt forces citation-only answers; instructs model to abstain if context lacks answer.
- Timeout to Ollama: 300s total, 10s connect. HTTP errors propagated with 502/504 codes.

## Quick test payloads
- Search: `curl -X POST http://localhost:8000/api/search -H "Content-Type: application/json" -d '{"query":"How to initiate MRI mode on Azure XT?","limit":3}'`
- Ask: `curl -X POST http://localhost:8000/api/ask -H "Content-Type: application/json" -d '{"query":"Battery replacement indicators for Azure XT","limit":3}'`

## Safety and domain scope
- Domain is technical guidance for implanted cardiology devices. The prompt explicitly forbids guessing; answers must come from retrieved context or state inability.
- No PHI handling assumed; data are manual texts. Ensure deployments comply with org policies.
