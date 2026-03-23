import os

import psycopg2
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownTextSplitter
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load .env for local development
load_dotenv()

# 1. Configuration and Database Setup (read from env with defaults)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Using a strong, lightweight local model for embeddings (identifier from env)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    register_vector(conn)
    return conn


def _read_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def process_markdown(
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

    notify("start", {"title": title})
    print(f"\n--- Starting Markdown Processing: {title} ---")

    # 2. Read Markdown
    print("Phase 1: Reading Markdown content...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    md_text = _read_markdown(file_path)
    print("Phase 1 Complete: Markdown loaded successfully.")
    notify("markdown_loaded")

    # 3. Chunk the Markdown
    print("Phase 2: Chunking document structure...")
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([md_text])
    print(f"Phase 2 Complete: Created {len(chunks)} chunks.")
    notify("chunked", {"total_chunks": len(chunks)})

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 4. Insert Document Metadata
        cur.execute(
            """
            INSERT INTO reference_documents
            (title, manufacturer, device_model, document_type, publication_year)
            VALUES (%s, %s, %s, %s, %s) RETURNING id;
            """,
            (title, manufacturer, device_model, doc_type, year),
        )
        document_id = cur.fetchone()[0]

        # 5. Embed and Insert Chunks (WITH PROGRESS BAR)
        print("Phase 3: Generating embeddings and saving to PostgreSQL...")

        for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks", unit="chunk")):
            content = chunk.page_content

            # Generate the vector embedding locally
            embedding = EMBEDDING_MODEL.encode(content).tolist()

            # Flag if the chunk contains a Markdown table
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
        print(f"\nSUCCESS: Fully ingested {len(chunks)} chunks for {title}.\n")
        notify("completed", {"total_chunks": len(chunks)})

    except Exception as e:
        conn.rollback()
        print(f"\nERROR: Failed processing {title}: {e}\n")
        notify("error", {"message": str(e)})

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    # Example Usage:
    md_file = "manuals/example_manual.md"

    if os.path.exists(md_file):
        process_markdown(
            file_path=md_file,
            title="Example Manual (Markdown)",
            manufacturer="ExampleCo",
            device_model="ExampleModel",
            doc_type="Manual",
            year=2024,
        )
    else:
        print(f"File not found: {md_file}")
