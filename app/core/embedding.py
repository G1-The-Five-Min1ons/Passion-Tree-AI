from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_vector(self, text: str) -> List[float]:
        """Converts text to a vector locally on your CPU/GPU."""
        embedding = self.model.encode(text)
        return embedding.tolist()

    def prepare_learning_path_text(self, title: str, description: str) -> str:
        return f"Learning Path Title: {title}. Content Summary: {description}"

    def get_path_vector(self, title: str, description: str) -> List[float]:
        combined_text = self.prepare_learning_path_text(title, description)
        return self.generate_vector(combined_text)
