import os
from langchain_community.vectorstores import Chroma  # 수정된 패키지 경로
from app.utils.pdf_utils import clear_project_documents

class DocumentService:
    def __init__(self, embedding_fn, chroma_dir=None):
        self.embedding_fn = embedding_fn
        self.CHROMA_DIR = chroma_dir or "./project_data/chroma"
        os.makedirs(self.CHROMA_DIR, exist_ok=True)

    def upload_document(self, project_id: str, object_key: str, chunk_size: int = 500):
        """
        PDF 문서를 Chroma DB에 업로드
        """
        # 프로젝트별 persist 디렉토리
        persist_dir = os.path.join(self.CHROMA_DIR, project_id)
        os.makedirs(persist_dir, exist_ok=True)

        print(f"[INFO] Persist directory: {persist_dir}")
        print(f"[INFO] Uploading PDF: {object_key}")

        try:
            # 기존 DB 초기화 (선택적)
            clear_project_documents(project_id)

            # Chroma DB 연결
            db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding_fn)

            # 실제 PDF 업로드 로직 (문서 분할, 임베딩 등)
            # 여기서는 예시로 문서 하나 추가
            db.add_documents([object_key], chunk_size=chunk_size)  # 필요에 따라 PDF를 로컬로 다운로드 후 경로 전달

            print("[INFO] 문서 업로드 성공")
            return True

        except Exception as e:
            print("[ERROR] 문서 업로드 실패:", e)
            return False
