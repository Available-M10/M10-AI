import os
import shutil
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from app.models.embedding_model import SentenceTransformerEmbeddings
from app.utils.pdf_utils import clear_project_documents

CHROMA_DIR = os.getenv("CHROMA_DIR", "./project_data/chroma")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

conversation_memory = {}  # 프로젝트별 대화 기록

class LLMService:
    def __init__(self, model_name="gemini-2.0-flash-lite"):
        self.model = genai.GenerativeModel(model_name)
        embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/KoSimCSE-bert")
        self.embedding_fn = SentenceTransformerEmbeddings(model_path=embedding_model_path)

    def query(self, project_id: str, user_message: str, prompt: str, top_k: int = 5, clear_after: bool = True):
        # 대화 메모리 초기화
        conversation_memory.setdefault(project_id, [])
        conversation_memory[project_id].append({"role": "user", "content": user_message})

        # Chroma DB에서 유사문서 검색
        retrieved_docs = self._retrieve_docs(project_id, user_message, top_k)
        context_text = "\n".join([d.page_content for d in retrieved_docs]) or "(문서 없음)"
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in conversation_memory[project_id][-10:]])

        # 최종 프롬프트 구성
        final_prompt = f"{prompt}\n\n문서 컨텍스트:\n{context_text}\n\n대화 히스토리:\n{history_text}\n\n사용자 질문:\n{user_message}"

        # LLM 호출
        response = self.model.generate_content(final_prompt)
        answer_text = self._extract_text(response)
        conversation_memory[project_id].append({"role": "assistant", "content": answer_text})

        # 문서/DB 초기화
        if clear_after:
            self._clear_project(project_id)

        return answer_text

    def _retrieve_docs(self, project_id: str, query: str, top_k: int):
        persist_dir = os.path.join(CHROMA_DIR, project_id)
        if os.path.isdir(persist_dir) and os.listdir(persist_dir):
            db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding_fn)
            return db.similarity_search(query, k=top_k)
        return []

    def _extract_text(self, response):
        if hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", [])
            return "".join([p.text for p in parts if hasattr(p, "text")])
        return getattr(response, "text", str(response))

    def _clear_project(self, project_id: str):
        try:
            clear_project_documents(project_id)
            project_dir = os.path.join(CHROMA_DIR, project_id)
            if os.path.exists(project_dir):
                shutil.rmtree(project_dir, ignore_errors=True)
        except Exception as e:
            print(f"[LLMService] clear_project failed: {e}")
