"""ingest.py – Universal document ingestion via the Docling conversion service.

Supported formats (anything Docling can handle):
  .pdf  .docx  .pptx  .xlsx  .html  .htm  .md
  .png  .jpg  .jpeg  .tiff  .tif  .bmp
"""

import mimetypes
import os

import psycopg2
import requests
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load .env for local development
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Docling service URL (running in Docker on the local server)
DOCLING_URL = os.getenv("DOCLING_URL", "http://10.0.0.32:5001").rstrip("/")
if DOCLING_URL.startswith("https://"):
    raise RuntimeError(
        f"DOCLING_URL '{DOCLING_URL}' uses HTTPS but the Docling service runs plain HTTP. "
        "Update DOCLING_URL to use http://"
    )

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

# File extensions accepted by Docling's /v1alpha/convert/file endpoint
DOCLING_SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    register_vector(conn)
    return conn


def _convert_to_markdown(file_path: str) -> str:
    """Send a document to the Docling service and return the Markdown output."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in DOCLING_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(DOCLING_SUPPORTED_EXTENSIONS))}"
        )

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{DOCLING_URL}/v1/convert/file",
                files={"files": (os.path.basename(file_path), f, mime_type)},
                timeout=300,
            )
        response.raise_for_status()
    except requests.exceptions.SSLError as e:
        raise ConnectionError(
            f"SSL error connecting to Docling at {DOCLING_URL}. "
            "The service is likely plain HTTP — make sure DOCLING_URL starts with http://, not https://. "
            f"Detail: {e}"
        ) from e
    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            f"Could not reach Docling service at {DOCLING_URL}. "
            f"Check that the container is running and the URL/port are correct. Detail: {e}"
        ) from e
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"Docling conversion timed out after 300 s for '{os.path.basename(file_path)}'. "
            "The document may be too large or the service is overloaded."
        )

    return response.json()["document"]["md_content"]


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------
def process_document(
    file_path: str,
    title: str,
    manufacturer: str,
    device_model: str,
    doc_type: str,
    year: int,
    progress_callback=None,
):
    def notify(stage, payload=None):
        if progress_callback:
            progress_callback(stage, payload or {})

    ext = os.path.splitext(file_path)[1].lower()
    notify("start", {"title": title})
    print(f"\n--- Starting Processing: {title} ({ext}) ---")

    conn = None
    cur = None
    try:
        # Phase 1: Convert to Markdown via Docling
        print("Phase 1: Sending document to Docling for conversion...")
        notify("converting", {"filename": os.path.basename(file_path)})
        md_text = _convert_to_markdown(file_path)
        print("Phase 1 Complete: Markdown extracted successfully.")
        notify("markdown_extracted")

        # Phase 2: Chunk the Markdown
        print("Phase 2: Chunking document structure...")
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([md_text])
        print(f"Phase 2 Complete: Created {len(chunks)} chunks.")
        notify("chunked", {"total_chunks": len(chunks)})

        conn = get_db_connection()
        cur = conn.cursor()

        # Insert document metadata
        cur.execute(
            """
            INSERT INTO reference_documents
            (title, manufacturer, device_model, document_type, publication_year)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """,
            (title, manufacturer, device_model, doc_type, year),
        )
        document_id = cur.fetchone()[0]

        # Phase 3: Embed and insert chunks
        print("Phase 3: Generating embeddings and saving to PostgreSQL...")
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks", unit="chunk")):
            content = chunk.page_content
            embedding = EMBEDDING_MODEL.encode(content).tolist()
            is_table = "|" in content and "---" in content

            cur.execute(
                """
                INSERT INTO document_chunks
                (document_id, content, is_table, embedding)
                VALUES (%s, %s, %s, %s);
                """,
                (document_id, content, is_table, embedding),
            )

            if progress_callback and (i % 25 == 0 or i == len(chunks) - 1):
                progress_callback(
                    "chunk_progress",
                    {"current": i + 1, "total": len(chunks)},
                )

        conn.commit()
        print(f"\nSUCCESS: Fully ingested {len(chunks)} chunks for '{title}'.\n")
        notify("completed", {"total_chunks": len(chunks)})

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\nERROR: Failed processing '{title}': {e}\n")
        notify("error", {"message": str(e)})

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    import sys

    file = sys.argv[1] if len(sys.argv) > 1 else "manuals/example.pdf"
    if os.path.exists(file):
        process_document(
            file_path=file,
            title="Example Document",
            manufacturer="ExampleCo",
            device_model="ExampleModel",
            doc_type="Manual",
            year=2024,
        )
    else:
        print(f"File not found: {file}")
