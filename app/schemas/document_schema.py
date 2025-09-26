from pydantic import BaseModel

class DocumentRequest(BaseModel):
    chunk_size: int = 500
    embedding_model: str = "BERT"
    vector_db: str = "CHROMA"
    object_key: str
