import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
import psycopg2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg2 import register_vector
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ingest_markdown import process_markdown
from ingest_pdfs import process_pdf

# Load .env if present (no-op in production where env vars are injected)
load_dotenv()

# 1. Configuration (read from environment with sensible defaults)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Global variable to hold the model in memory (string name until loaded)
embedding_model = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

# Default Ollama port is 11434.
OLLAMA_URL = os.getenv("OLLAMA_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "MedAIBase/MedGemma1.5:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# Simple in-memory progress store for uploads
UPLOAD_PROGRESS = {}


# 2. Define Request and Response Models using Pydantic
class SearchRequest(BaseModel):
    query: str
    limit: int = 5


class SearchResult(BaseModel):
    title: str
    manufacturer: str
    device_model: str
    content: str
    page_number: Optional[int]
    similarity: float


class GenerationResponse(BaseModel):
    answer: str
    sources: List[SearchResult]


# 3. Lifespan context manager to load the model on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedding_model
    print("Loading Qwen3 embedding model into memory...")
    # Update this to match the exact Qwen3 model you chose during ingestion
    # If `embedding_model` is a string name, load the model into memory
    if isinstance(embedding_model, str):
        embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True)
    print("Model loaded successfully. API is ready.")
    yield
    # Clean up resources on shutdown if necessary
    print("Shutting down API...")


# 4. Initialize FastAPI
app = FastAPI(
    title="H3 Cardiology Expert System API",
    description="RAG Pipeline for querying pacemaker and defibrillator technical manuals & resources.",
    version="1.0.0",
    docs_url=None,  # Disable default CDN-based docs
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=app.title,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )
    register_vector(conn)
    return conn


def _set_progress(
    job_id: str,
    status: str,
    percent: float,
    message: str = "",
    extra: dict | None = None,
):
    payload = {
        "status": status,
        "percent": percent,
        "message": message,
    }
    if extra:
        payload.update(extra)
    UPLOAD_PROGRESS[job_id] = payload


def _run_ingest_job(
    job_id: str,
    temp_path: str,
    title: str,
    manufacturer: str,
    device_model: str,
    doc_type: str,
    year: int,
):
    def progress(stage, payload):
        if stage == "start":
            _set_progress(
                job_id, "running", 5, "Upload received. Starting extraction..."
            )
        elif stage == "markdown_extracted":
            _set_progress(job_id, "running", 25, "Markdown extracted from PDF.")
        elif stage == "chunked":
            total = payload.get("total_chunks", 0)
            _set_progress(
                job_id,
                "running",
                40,
                f"Chunked document into {total} parts.",
                {"total_chunks": total},
            )
        elif stage == "chunk_progress":
            current = payload.get("current", 0)
            total = payload.get("total", 1)
            percent = 40 + (50 * (current / total))
            _set_progress(
                job_id,
                "running",
                min(percent, 95),
                f"Embedding chunks ({current}/{total})...",
                {"current": current, "total": total},
            )
        elif stage == "completed":
            _set_progress(job_id, "completed", 100, "Ingestion complete.", payload)
        elif stage == "error":
            _set_progress(
                job_id, "error", 100, f"Ingestion failed: {payload.get('message', '')}"
            )

    try:
        _set_progress(job_id, "running", 1, "Preparing ingestion job...")
        process_pdf(
            file_path=temp_path,
            title=title,
            manufacturer=manufacturer,
            device_model=device_model,
            doc_type=doc_type,
            year=year,
            progress_callback=progress,
        )
        if UPLOAD_PROGRESS.get(job_id, {}).get("status") not in {"completed", "error"}:
            _set_progress(job_id, "completed", 100, "Ingestion complete.")
    except Exception as e:
        _set_progress(job_id, "error", 100, f"Unhandled error: {e}")
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


def _run_ingest_markdown_job(
    job_id: str,
    temp_path: str,
    title: str,
    manufacturer: str,
    device_model: str,
    doc_type: str,
    year: int,
):
    def progress(stage, payload):
        if stage == "start":
            _set_progress(
                job_id, "running", 5, "Upload received. Starting ingestion..."
            )
        elif stage == "markdown_loaded":
            _set_progress(job_id, "running", 20, "Markdown loaded successfully.")
        elif stage == "chunked":
            total = payload.get("total_chunks", 0)
            _set_progress(
                job_id,
                "running",
                40,
                f"Chunked document into {total} parts.",
                {"total_chunks": total},
            )
        elif stage == "chunk_progress":
            current = payload.get("current", 0)
            total = payload.get("total", 1)
            percent = 40 + (50 * (current / total))
            _set_progress(
                job_id,
                "running",
                min(percent, 95),
                f"Embedding chunks ({current}/{total})...",
                {"current": current, "total": total},
            )
        elif stage == "completed":
            _set_progress(job_id, "completed", 100, "Ingestion complete.", payload)
        elif stage == "error":
            _set_progress(
                job_id, "error", 100, f"Ingestion failed: {payload.get('message', '')}"
            )

    try:
        _set_progress(job_id, "running", 1, "Preparing ingestion job...")
        process_markdown(
            file_path=temp_path,
            title=title,
            manufacturer=manufacturer,
            device_model=device_model,
            doc_type=doc_type,
            year=year,
            progress_callback=progress,
        )
        if UPLOAD_PROGRESS.get(job_id, {}).get("status") not in {"completed", "error"}:
            _set_progress(job_id, "completed", 100, "Ingestion complete.")
    except Exception as e:
        _set_progress(job_id, "error", 100, f"Unhandled error: {e}")
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


@app.post("/api/upload_pdf")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    manufacturer: str = Form(...),
    device_model: str = Form(...),
    doc_type: str = Form(...),
    year: int = Form(...),
):
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    job_id = str(uuid.uuid4())
    _set_progress(job_id, "queued", 0, "File queued for ingestion.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
    except Exception as e:
        _set_progress(job_id, "error", 100, f"Failed to store upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.")

    background_tasks.add_task(
        _run_ingest_job,
        job_id,
        temp_path,
        title,
        manufacturer,
        device_model,
        doc_type,
        year,
    )

    return {
        "job_id": job_id,
        "status_url": f"/api/upload_pdf/{job_id}/status",
        "message": "Upload accepted. Poll status_url for progress.",
    }


@app.post("/api/upload_md")
async def upload_markdown(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(...),
    manufacturer: str = Form(...),
    device_model: str = Form(...),
    doc_type: str = Form(...),
    year: int = Form(...),
):
    if file.content_type not in {
        "text/markdown",
        "text/plain",
        "application/octet-stream",
    }:
        raise HTTPException(status_code=400, detail="Uploaded file must be Markdown.")

    job_id = str(uuid.uuid4())
    _set_progress(job_id, "queued", 0, "File queued for ingestion.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md") as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name
    except Exception as e:
        _set_progress(job_id, "error", 100, f"Failed to store upload: {e}")
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.")

    background_tasks.add_task(
        _run_ingest_markdown_job,
        job_id,
        temp_path,
        title,
        manufacturer,
        device_model,
        doc_type,
        year,
    )

    return {
        "job_id": job_id,
        "status_url": f"/api/upload_md/{job_id}/status",
        "message": "Upload accepted. Poll status_url for progress.",
    }


@app.get("/api/upload_md/{job_id}/status")
async def upload_markdown_status(job_id: str):
    progress = UPLOAD_PROGRESS.get(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found.")
    return progress


@app.get("/api/upload_pdf/{job_id}/status")
async def upload_status(job_id: str):
    progress = UPLOAD_PROGRESS.get(job_id)
    if not progress:
        raise HTTPException(status_code=404, detail="Job not found.")
    return progress


# 5. The Retrieval Endpoint
@app.post("/api/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet.")

    try:
        # Step A: Vectorize the user's query locally in real-time
        query_vector = embedding_model.encode(request.query).tolist()

        # Step B: Connect to PostgreSQL and execute the similarity search
        conn = get_db_connection()
        cur = conn.cursor()

        # Using the cosine distance operator (<=>)
        # Note: In pgvector, lower distance means higher similarity.
        sql = """
            SELECT
                r.title, r.manufacturer, r.device_model,
                d.content, d.page_number,
                1 - (d.embedding <=> %s::vector) AS similarity
            FROM document_chunks d
            JOIN reference_documents r ON d.document_id = r.id
            ORDER BY d.embedding <=> %s::vector
            LIMIT %s;
        """

        cur.execute(sql, (query_vector, query_vector, request.limit))
        rows = cur.fetchall()

        # Step C: Format and return the results
        results = []
        for row in rows:
            results.append(
                SearchResult(
                    title=row[0],
                    manufacturer=row[1],
                    device_model=row[2],
                    content=row[3],
                    page_number=row[4],
                    similarity=float(row[5]),
                )
            )

        cur.close()
        conn.close()

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database search failed: {str(e)}")


@app.post("/api/ask", response_model=GenerationResponse)
async def ask_documents(request: SearchRequest):
    if not embedding_model:
        raise HTTPException(status_code=503, detail="Embedding model not loaded yet.")

    try:
        # 1. Vectorize the Question
        query_vector = embedding_model.encode(request.query).tolist()

        # 2. Retrieve the Chunks from PostgreSQL
        conn = get_db_connection()
        cur = conn.cursor()

        sql = """
            SELECT r.title, r.manufacturer, r.device_model, d.content, d.page_number, 1 - (d.embedding <=> %s::vector) AS similarity
            FROM document_chunks d
            JOIN reference_documents r ON d.document_id = r.id
            ORDER BY d.embedding <=> %s::vector
            LIMIT %s;
        """
        cur.execute(sql, (query_vector, query_vector, request.limit))
        rows = cur.fetchall()

        # Format the retrieved database rows
        retrieved_chunks = []
        for row in rows:
            retrieved_chunks.append(
                SearchResult(
                    title=row[0],
                    manufacturer=row[1],
                    device_model=row[2],
                    content=row[3],
                    page_number=row[4],
                    similarity=float(row[5]),
                )
            )

        cur.close()
        conn.close()

        # 3. Generate the Final Answer using Ollama
        # If no chunks were found, skip generation and tell the user
        if not retrieved_chunks:
            return GenerationResponse(
                answer="No relevant documents found in the database.", sources=[]
            )

        final_answer = await generate_rag_answer(request.query, retrieved_chunks)

        # 4. Return the LLM's answer AND the raw chunks so your frontend can display citations
        return GenerationResponse(answer=final_answer, sources=retrieved_chunks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process ask request: {str(e)}"
        )


async def generate_rag_answer(query: str, retrieved_chunks: List[SearchResult]) -> str:
    """
    Takes the retrieved pgvector chunks, formats them into a strict prompt,
    and asks Ollama to generate an answer.
    """

    # 1. Compile the retrieved chunks into a single formatted text block
    context_text = ""
    for i, chunk in enumerate(retrieved_chunks):
        page_ref = chunk.page_number if chunk.page_number is not None else "N/A"
        # Keep prompt size bounded so local inference remains responsive.
        chunk_excerpt = chunk.content[:1500]
        context_text += f"\n--- Source {i + 1}: {chunk.manufacturer} {chunk.device_model} (Page {page_ref}) ---\n"
        context_text += f"{chunk_excerpt}\n"

    # 2. Construct the System Prompt
    # This strict prompting prevents the LLM from hallucinating medical data.
    prompt = f"""You are an expert technical assistant for implanted cardiology devices.
Your job is to answer the user's question using ONLY the provided technical context.
If the answer is not explicitly contained within the context, you must state: "I cannot find the answer in the provided documents."
Do not rely on outside knowledge or guess. Cite your sources using the source numbers provided.

CONTEXT:
{context_text}

USER QUESTION:
{query}

EXPERT ANSWER:"""

    # 3. Call the Ollama API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                OLLAMA_URL,
                json={
                    "model": LLM_MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,  # Set to True later if you want a ChatGPT-style typing effect
                },
                timeout=httpx.Timeout(
                    OLLAMA_TIMEOUT_SECONDS, connect=10.0
                ),  # Allow slower local-model inference while failing fast on bad connectivity
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "Error: No response generated.")
        except httpx.TimeoutException as e:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Ollama request timed out. "
                    f"url={OLLAMA_URL}, model={LLM_MODEL_NAME}, error={type(e).__name__}"
                ),
            )
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body = e.response.text if e.response is not None else ""
            raise HTTPException(
                status_code=502,
                detail=f"Ollama request failed with status {status} at {OLLAMA_URL}. Body: {body[:300]}",
            )

        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Error communicating with Ollama: "
                    f"{type(e).__name__}: {str(e) or repr(e)}"
                ),
            )
