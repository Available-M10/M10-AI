from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_path: str = "/app/app/models/KoSimCSE-bert"):
        self.model = SentenceTransformer(model_path)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        return embeddings.cpu().numpy().tolist()

    def embed_query(self, text):
        embedding = self.model.encode([text], convert_to_tensor=True)[0]
        return embedding.cpu().numpy().tolist()
