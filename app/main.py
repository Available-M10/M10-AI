from fastapi import FastAPI
from app.controllers import document_controller, llm_controller

app = FastAPI()

app.include_router(document_controller.router)
app.include_router(llm_controller.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
