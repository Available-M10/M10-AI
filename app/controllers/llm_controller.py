from fastapi import APIRouter, HTTPException
from app.services.llm_service import LLMService
from app.schemas.llm_schema import LLMRequest

router = APIRouter()
llm_service = LLMService()

@router.post("/node/{project_id}/llm")
def call_llm(project_id: str, body: LLMRequest):
    if body.llm.lower() != "gemini":
        raise HTTPException(status_code=400, detail="Only 'gemini' supported")
    try:
        answer = llm_service.query(
            project_id=project_id,
            user_message=body.message,
            prompt=body.prompt,
            clear_after=getattr(body, "clear_after_answer", True),
            top_k=getattr(body, "top_k", 5)
        )
        return {"answer": answer}  # JSON 형태로 반환
    except HTTPException:
        raise
    except Exception as e:
        # 예외 발생 시에도 JSON 반환
        raise HTTPException(status_code=500, detail=str(e))
