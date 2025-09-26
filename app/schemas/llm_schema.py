from pydantic import BaseModel

class LLMRequest(BaseModel):
    llm: str
    prompt: str
    message: str
    top_k: int = 5  # 문서 검색 시 가져올 개수
