import os
from dotenv import load_dotenv

# .env 파일 불러오기
load_dotenv()

# 공통 경로 및 API Key
BASE_DB_DIR = os.getenv("BASE_DB_DIR", "./project_data")
CHROMA_DIR = os.path.join(BASE_DB_DIR, "chroma")
SQLITE_PATH = os.path.join(BASE_DB_DIR, "documents_meta.sqlite")

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/KoSimCSE-bert")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 폴더 생성 보장
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(BASE_DB_DIR, exist_ok=True)
