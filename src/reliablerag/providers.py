import importlib

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

_PROVIDERS: dict[str, dict[str, tuple[str, str, str]]] = {
    "ollama": {
        "embeddings": ("langchain_ollama", "OllamaEmbeddings", "model"),
        "llm":        ("langchain_ollama", "ChatOllama",        "model"),
    },
    "huggingface": {
        "embeddings": ("langchain_huggingface", "HuggingFaceEmbeddings", "model_name"),
        "llm":        ("langchain_huggingface", "ChatHuggingFace",        "model"),
    },
}


def _resolve_class(provider: str, kind: str) -> tuple:
    """Dynamically import and return (class, model_param_name) for the given provider and kind."""
    if provider not in _PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider!r}. Supported: {list(_PROVIDERS)}")
    module_name, class_name, model_param = _PROVIDERS[provider][kind]
    module = importlib.import_module(module_name)
    return getattr(module, class_name), model_param


def create_embeddings(provider: str, model: str, **kwargs) -> Embeddings:
    cls, model_param = _resolve_class(provider, "embeddings")
    return cls(**{model_param: model}, **kwargs)


def create_llm(provider: str, model: str, **kwargs) -> BaseChatModel:
    cls, model_param = _resolve_class(provider, "llm")
    return cls(**{model_param: model}, **kwargs)