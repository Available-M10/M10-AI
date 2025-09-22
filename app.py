from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import logging
import shutil
import time

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import google.generativeai as genai


# 환경 설정
BASE_DIR = "/app/vector_store"
os.makedirs(BASE_DIR, exist_ok=True)
EMBEDDING_MODEL_NAME = "./models--BM-K--KoSimCSE-bert-multitask"
VECTOR_DB_NAME = "chroma"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# 임베딩 모델 로드
embedding_model = SentenceTransformer("/app/models/bert")

class MyEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embedding_model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return embedding_model.encode([text], convert_to_numpy=True).tolist()[0]

embedding_instance = MyEmbedding()

# Gemini API 설정
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# -------------------------------
# Request Schemas
# -------------------------------
class DocumentNodeRequest(BaseModel):
    chunk_size: int
    embedding_model: str
    vector_db: str
    object_key: str  # S3 URL
    projectId: str

class LLMNodeRequest(BaseModel):
    llm: str
    prompt: Optional[str] = None
    message: str
    clear_after_answer: bool = True

@app.get("/")
def root():
    return {"message": "Server is running"}


# 문서 노드
@app.post("/node/{projectId}/document")
def document_node(projectId: str, req: DocumentNodeRequest):
    try:
        logging.info(f"[문서 노드] PDF 다운로드 시작: {req.object_key}")
        response = requests.get(req.object_key)
        if response.status_code != 200:
            raise Exception("PDF 다운로드 실패")

        tmp_path = f"./tmp_{time.time_ns()}.pdf"
        with open(tmp_path, "wb") as f:
            f.write(response.content)

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=req.chunk_size,
            chunk_overlap=50
        )
        split_docs = splitter.split_documents(documents)

        db_dir = os.path.join(BASE_DIR, f"{req.projectId}_{VECTOR_DB_NAME}")
        if os.path.exists(db_dir):
            shutil.rmtree(db_dir)

        vectorstore = Chroma.from_documents(
            split_docs, embedding_instance, persist_directory=db_dir
        )
        vectorstore.persist()
        logging.info(f"[문서 노드] Vector DB 저장 완료: {db_dir}")

        os.remove(tmp_path)
        return {"status": 200}

    except Exception as e:
        logging.error(f"[문서 노드] 처리 실패: {e}")
        raise HTTPException(status_code=500, detail="문서 처리 실패")


# LLM 노드
@app.post("/node/{projectId}/llm")
def llm_node(projectId: str, req: LLMNodeRequest):
    db_dir = os.path.join(BASE_DIR, f"{projectId}_{VECTOR_DB_NAME}")
    try:
        logging.info(f"[LLM 노드] 호출: projectId={projectId}, 질문='{req.message}'")

        context_text = "문서 없음"
        if os.path.exists(db_dir):
            vectorstore = Chroma(
                persist_directory=db_dir,
                embedding_function=embedding_instance
            )
            related_docs = vectorstore.similarity_search(req.message, k=3)
            if related_docs:
                context_text = "\n".join([doc.page_content for doc in related_docs])

        prompt_to_use = req.prompt or (
            "너는 문서를 기반으로 답해줘야 해. "
            "만약 문서가 없으면 질문에 적합한 답변만 해줘."
        )

        full_prompt = f"{prompt_to_use}\n\n문서 참고 내용:\n{context_text}\n\n질문: {req.message}"

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        logging.info("[LLM 노드] Gemini 응답 완료")

        return {"answer": response.text}

    except Exception as e:
        logging.error(f"[LLM 노드] 오류: {e}")
        raise HTTPException(status_code=500, detail="LLM 처리 실패")