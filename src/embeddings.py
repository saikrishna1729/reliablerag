import numpy as np
import requests
from typing import Union, List
from config import ExperimentConfig

class MockEmbedder:
    """Mock embedder that returns random vectors without dependencies."""
    is_mock = True

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate stable mock vectors by hashing string contents
        vectors = []
        for text in texts:
            # Seed generator with hash value to get deterministic random vectors
            hash_val = hash(text) % (2**32)
            rng = np.random.default_rng(hash_val)
            vectors.append(rng.normal(0, 1, self.dimension))
            
        return np.array(vectors)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode([text])[0].tolist()


class SentenceTransformerEmbedder:
    """SentenceTransformer wrapper for local embedding generation."""
    is_mock = False

    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import SentenceTransformer
        import torch
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode([text])[0].tolist()


class OllamaEmbedder:
    """Ollama wrapper that calls local Ollama server embeddings API."""
    is_mock = False

    def __init__(self, model_name: str, host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            response = requests.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model_name, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(data["embedding"])

        return np.array(embeddings)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.encode([text])[0].tolist()


def get_embedder(config: ExperimentConfig):
    if config.embedder == "mock":
        return MockEmbedder()
    elif config.embedder == "minilm":
        return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
    elif config.embedder == "bge-small":
        return SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    elif config.embedder == "bge-large":
        return SentenceTransformerEmbedder("BAAI/bge-large-en-v1.5")
    elif config.embedder == "ollama":
        return OllamaEmbedder("embeddinggemma:300m-qat-q4_0")
    else:
        raise ValueError(f"Unknown embedder: {config.embedder}")
