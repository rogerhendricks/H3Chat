"""ingest-v2.py – Single-table document ingestion via the Docling conversion service.

Supported formats (anything Docling can handle):
.pdf .docx .pptx .xlsx .html .htm .md
.png .jpg .jpeg .tiff .tif .bmp
"""

import hashlib
import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import psycopg2
import requests
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from pgvector.psycopg2 import register_vector
from psycopg2.extras import Json, execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_TABLE = os.getenv("DB_TABLE", "documents")

DOCLING_URL = os.getenv("DOCLING_URL", "http://10.0.0.32:5001").rstrip("/")
if DOCLING_URL.startswith("https://"):
    raise RuntimeError(
        f"DOCLING_URL '{DOCLING_URL}' uses HTTPS but the Docling service runs plain HTTP. "
        "Update DOCLING_URL to use http://"
    )

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

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

ProgressCallback = Optional[Callable[[str, dict], None]]


def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    register_vector(conn)
    return conn


def ensure_required_columns(conn):
    required_columns = {
        "id",
        "document_id",
        "chunk_index",
        "text",
        "embedding",
        "metadata",
        "file_name",
        "file_path",
        "source_uri",
        "document_title",
        "author",
        "version",
        "description",
        "document_type",
        "mime_type",
        "file_extension",
        "file_size_bytes",
        "publication_date",
        "created_date",
        "modified_date",
        "ingested_at",
        "checksum_sha256",
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = %s
            """,
            (DB_TABLE,),
        )
        present = {row[0] for row in cur.fetchall()}

    missing = sorted(required_columns - present)
    if missing:
        raise RuntimeError(
            f"Table '{DB_TABLE}' is missing required columns: {', '.join(missing)}"
        )


def sha256sum(file_path: str) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_document_title(file_path: str, explicit_title: Optional[str]) -> str:
    if explicit_title and explicit_title.strip():
        return explicit_title.strip()
    return Path(file_path).stem.replace("_", " ").replace("-", " ").strip()


def file_timestamps(file_path: str):
    stat = os.stat(file_path)
    created = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc)
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return created, modified


def _convert_to_markdown(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in DOCLING_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {', '.join(sorted(DOCLING_SUPPORTED_EXTENSIONS))}"
        )

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{DOCLING_URL}/v1/convert/file",
                files={"files": (os.path.basename(file_path), f, mime_type)},
                data={"image_export_mode": "placeholder"},
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
    except requests.exceptions.Timeout as e:
        raise TimeoutError(
            f"Docling conversion timed out after 300 s for '{os.path.basename(file_path)}'. "
            "The document may be too large or the service is overloaded."
        ) from e

    payload = response.json()
    try:
        return payload["document"]["md_content"]
    except KeyError as e:
        raise ValueError(
            f"Unexpected Docling response format: {json.dumps(payload)[:1000]}"
        ) from e


def delete_existing_document(cur, checksum_sha256_value: str) -> int:
    cur.execute(
        f"DELETE FROM {DB_TABLE} WHERE checksum_sha256 = %s", (checksum_sha256_value,)
    )
    return cur.rowcount


def process_document(
    file_path: str,
    title: Optional[str] = None,
    author: Optional[str] = None,
    version: Optional[str] = None,
    description: Optional[str] = None,
    document_type: Optional[str] = None,
    publication_date: Optional[str] = None,
    source_uri: Optional[str] = None,
    progress_callback: ProgressCallback = None,
):
    def notify(stage: str, payload: Optional[dict] = None):
        if progress_callback:
            progress_callback(stage, payload or {})

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    path_obj = Path(file_path).resolve()
    ext = path_obj.suffix.lower()
    checksum = sha256sum(str(path_obj))
    document_id = uuid.uuid4()
    ingested_at = datetime.now(timezone.utc)
    mime_type, _ = mimetypes.guess_type(str(path_obj))
    mime_type = mime_type or "application/octet-stream"
    created_date, modified_date = file_timestamps(str(path_obj))
    file_size_bytes = path_obj.stat().st_size
    file_name = path_obj.name
    file_extension = ext.lstrip(".")
    document_title = infer_document_title(str(path_obj), title)
    resolved_source_uri = source_uri or path_obj.as_uri()

    notify("start", {"title": document_title, "file_name": file_name})
    print(f"\n--- Starting Processing: {document_title} ({ext}) ---")

    conn = None
    cur = None
    try:
        print("Phase 1: Sending document to Docling for conversion...")
        notify("converting", {"filename": file_name})
        md_text = _convert_to_markdown(str(path_obj))
        print("Phase 1 Complete: Markdown extracted successfully.")
        notify("markdown_extracted")

        print("Phase 2: Chunking document structure...")
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([md_text])
        if not chunks:
            raise ValueError("No chunks were produced from the converted markdown.")
        print(f"Phase 2 Complete: Created {len(chunks)} chunks.")
        notify("chunked", {"total_chunks": len(chunks)})

        conn = get_db_connection()
        cur = conn.cursor()
        ensure_required_columns(conn)

        deleted = delete_existing_document(cur, checksum)
        if deleted:
            print(f"Removed {deleted} existing chunk(s) for checksum {checksum}.")
            notify(
                "deduplicated", {"deleted_chunks": deleted, "checksum_sha256": checksum}
            )

        print("Phase 3: Generating embeddings and saving to PostgreSQL...")
        total_chunks = len(chunks)
        for batch_start in tqdm(
            range(0, total_chunks, EMBEDDING_BATCH_SIZE),
            desc="Embedding Chunks",
            unit="batch",
        ):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_chunks)
            batch_chunks = chunks[batch_start:batch_end]
            batch_texts = [chunk.page_content for chunk in batch_chunks]

            embeddings = EMBEDDING_MODEL.encode(
                batch_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )

            rows = []
            for idx_offset, (chunk_text, embedding) in enumerate(
                zip(batch_texts, embeddings)
            ):
                embedding_list = (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                )
                if len(embedding_list) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding_list)}. "
                        f"Model={EMBEDDING_MODEL_NAME}"
                    )

                chunk_index = batch_start + idx_offset
                rows.append(
                    (
                        str(document_id),
                        chunk_index,
                        chunk_text,
                        embedding_list,
                        Json({}),
                        file_name,
                        str(path_obj),
                        resolved_source_uri,
                        document_title,
                        author,
                        version,
                        description,
                        document_type,
                        mime_type,
                        file_extension,
                        file_size_bytes,
                        publication_date,
                        created_date,
                        modified_date,
                        ingested_at,
                        checksum,
                    )
                )

            execute_values(
                cur,
                f"""
                INSERT INTO {DB_TABLE} (
                    document_id,
                    chunk_index,
                    text,
                    embedding,
                    metadata,
                    file_name,
                    file_path,
                    source_uri,
                    document_title,
                    author,
                    version,
                    description,
                    document_type,
                    mime_type,
                    file_extension,
                    file_size_bytes,
                    publication_date,
                    created_date,
                    modified_date,
                    ingested_at,
                    checksum_sha256
                )
                VALUES %s
                """,
                rows,
            )

            notify("chunk_progress", {"current": batch_end, "total": total_chunks})

        conn.commit()
        print(
            f"\nSUCCESS: Fully ingested {len(chunks)} chunks for '{document_title}'.\n"
        )
        notify(
            "completed",
            {
                "document_id": str(document_id),
                "total_chunks": len(chunks),
                "checksum_sha256": checksum,
            },
        )
        return str(document_id)

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"\nERROR: Failed processing '{document_title}': {e}\n")
        notify("error", {"message": str(e)})
        raise

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a document into PostgreSQL pgvector."
    )
    parser.add_argument("file_path", help="Path to the file to ingest")
    parser.add_argument("--title", default=None)
    parser.add_argument("--author", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--publication-date", dest="publication_date", default=None)
    parser.add_argument("--description", default=None)
    parser.add_argument("--document-type", dest="document_type", default=None)
    parser.add_argument("--source-uri", dest="source_uri", default=None)
    args = parser.parse_args()

    process_document(
        file_path=args.file_path,
        title=args.title,
        author=args.author,
        version=args.version,
        description=args.description,
        document_type=args.document_type,
        source_uri=args.source_uri,
    )
