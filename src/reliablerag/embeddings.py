from langchain_ollama import OllamaEmbeddings


def get_embeddings(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(model=model)
