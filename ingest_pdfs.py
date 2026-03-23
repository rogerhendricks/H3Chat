import os
from dotenv import load_dotenv
import pymupdf4llm
import psycopg2
from pgvector.psycopg2 import register_vector
from langchain_text_splitters import MarkdownTextSplitter
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
# Note: You'll need to update your PostgreSQL schema's VECTOR(1536) to VECTOR(384) for this model.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)

def get_db_connection():
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    register_vector(conn)
    return conn

def process_pdf(file_path, title, manufacturer, device_model, doc_type, year, progress_callback=None):
    def notify(stage, payload=None):
        if progress_callback:
            progress_callback(stage, payload or {})

    notify("start", {"title": title})
    print(f"\n--- Starting Processing: {title} ---")
    
    # 2. Extract PDF to Markdown
    # Note: For massive textbooks, this step might take a minute or two before the progress bar appears.
    print("Phase 1: Extracting layout and tables to Markdown (This may take a moment on large files)...")
    md_text = pymupdf4llm.to_markdown(file_path)
    print("Phase 1 Complete: Markdown extracted successfully.")
    notify("markdown_extracted")
    
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
            (title, manufacturer, device_model, doc_type, year)
        )
        document_id = cur.fetchone()[0]
        
        # 5. Embed and Insert Chunks (WITH PROGRESS BAR)
        print("Phase 3: Generating embeddings and saving to PostgreSQL...")
        
        # Wrap the chunks list in tqdm() to generate the progress bar
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
                (document_id, content, is_table, embedding)
            )

            if progress_callback and (i % 25 == 0 or i == len(chunks) - 1):
                progress_callback(
                    "chunk_progress",
                    {"current": i + 1, "total": len(chunks)}
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
    pdf_file = "manuals/medtronic_azure_xt.pdf"
    
    if os.path.exists(pdf_file):
        process_pdf(
            file_path=pdf_file,
            title="Azure XT DR Technical Manual",
            manufacturer="Medtronic",
            device_model="Azure XT DR",
            doc_type="Manual",
            year=2023
        )
    else:
        print(f"File not found: {pdf_file}")
