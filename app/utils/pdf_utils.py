import os
import shutil
import sqlite3
import gc
from typing import List
import pdfplumber
from langchain.schema import Document
from fastapi import HTTPException
import logging

BASE_DB_DIR = os.getenv("BASE_DB_DIR", "./project_data")
CHROMA_DIR = os.path.join(BASE_DB_DIR, "chroma")
SQLITE_PATH = os.path.join(BASE_DB_DIR, "documents_meta.sqlite")

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(BASE_DB_DIR, exist_ok=True)

# PDF -> Document
def load_and_split_pdf(file_path: str) -> List[Document]:
    docs = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(Document(page_content=text, metadata={"page": i}))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF 파싱 실패: {e}")
    return docs

# SQLite 초기화
def init_sqlite():
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        chunk_id TEXT NOT NULL,
        source_url TEXT,
        text_snippet TEXT,
        extra_meta TEXT
    )
    """)
    conn.commit()
    conn.close()

init_sqlite()

# Metadata 저장
def save_meta(project_id: str, chunk_id: str, source_url: str, text_snippet: str):
    conn = sqlite3.connect(SQLITE_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents (project_id, chunk_id, source_url, text_snippet, extra_meta) VALUES (?, ?, ?, ?, ?)",
        (project_id, chunk_id, source_url, text_snippet[:1000], "source:S3")
    )
    conn.commit()
    conn.close()

# DB 초기화
def clear_project_documents(project_id: str):
    logging.info(f"[clear_project_documents] Clearing project: {project_id}")

    # 1) SQLite meta 삭제
    if os.path.exists(SQLITE_PATH):
        try:
            conn = sqlite3.connect(SQLITE_PATH)
            c = conn.cursor()
            c.execute("DELETE FROM documents WHERE project_id = ?", (project_id,))
            conn.commit()
            conn.close()
            logging.info(f"[clear_project_documents] SQLite meta cleared for {project_id}")
        except Exception as e:
            logging.error(f"[clear_project_documents] SQLite deletion failed: {e}")

    # 2) Chroma DB 삭제
    persist_dir = os.path.join(CHROMA_DIR, project_id)
    if os.path.isdir(persist_dir):
        try:
            shutil.rmtree(persist_dir, ignore_errors=True)
            gc.collect()
            logging.info(f"[clear_project_documents] Chroma DB cleared for {project_id}")
        except Exception as e:
            logging.error(f"[clear_project_documents] Chroma deletion failed: {e}")
