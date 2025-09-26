from fastapi import APIRouter, HTTPException
from app.utils.file_utils import download_pdf_from_s3
from app.utils.pdf_utils import load_and_split_pdf, save_meta
from app.models.embedding_model import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from app.schemas.document_schema import DocumentRequest
from app.config import CHROMA_DIR, EMBEDDING_MODEL_PATH
import os

router = APIRouter()

@router.post("/node/{project_id}/document")
def add_document(project_id: str, body: DocumentRequest):
    pdf_path = None
    try:
        # 1) PDF 다운로드
        pdf_path = download_pdf_from_s3(body.object_key)

        # 2) PDF 로드 + chunking
        chunks = load_and_split_pdf(pdf_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없음")

        # 3) SQLite에 메타데이터 저장
        for idx, doc in enumerate(chunks):
            chunk_id = f"{project_id}__{idx}"
            save_meta(project_id, chunk_id, body.object_key, doc.page_content)

        # 4) 임베딩 모델 로드
        embedding_fn = SentenceTransformerEmbeddings(model_path=EMBEDDING_MODEL_PATH)

        # 5) Chroma persist
        persist_dir = os.path.join(CHROMA_DIR, project_id)
        os.makedirs(persist_dir, exist_ok=True)

        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_fn)
        db.add_documents(chunks)

        # 문서 노드는 응답 바디 없이 200만 반환
        return {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if pdf_path and os.path.exists(pdf_path):
            try:
                os.remove(pdf_path)
            except Exception:
                pass
