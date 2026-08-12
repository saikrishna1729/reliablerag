"""Factory functions for the embedding / generation / judge models.

All notebooks talk to models through OpenRouter. The three "standard" models
(embedding, generator, judge) are configured centrally via environment
variables so changing one doesn't mean hunting through every notebook:

    RAGBENCH_EMBEDDING_MODEL   (default: openai/text-embedding-3-small)
    RAGBENCH_GENERATOR_MODEL   (default: meta-llama/llama-3.1-8b-instruct)
    RAGBENCH_JUDGE_MODEL       (default: meta-llama/llama-3.3-70b-instruct)

Set these in delucion_dataset/.env (see .env.example). A notebook can still
pass `model=...` explicitly to any of these factories when it's deliberately
testing a *different* model as its independent variable (e.g. comparing
generator choices) -- an explicit argument always wins over the env var.
"""

import os
import time

import requests
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_GENERATOR_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_JUDGE_MODEL = "meta-llama/llama-3.3-70b-instruct"


class OpenRouterEmbeddings(Embeddings):
    """Calls OpenRouter's dedicated /embeddings endpoint directly via `requests`.

    langchain_openai.OpenAIEmbeddings (used here previously) locally tokenizes
    input with tiktoken keyed off the model name before batching -- fine for
    `openai/*` models, but tiktoken has no encoding for third-party embedding
    models (e.g. `qwen/qwen3-embedding-8b`, `baai/bge-m3`), so requests for
    those models never even reach OpenRouter correctly. This class skips local
    tokenization entirely and POSTs raw text, matching OpenRouter's documented
    request shape -- works uniformly across every embedding model on its
    catalog, OpenAI's own included.
    """

    def __init__(self, model: str, api_key: str, batch_size: int = 96):
        self.model = model
        self.api_key = api_key
        self.batch_size = batch_size
        self.endpoint = f"{OPENROUTER_BASE_URL}/embeddings"

    def _embed_batch(self, texts: list[str], max_retries: int = 8) -> list[list[float]]:
        for attempt in range(max_retries):
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model, "input": texts, "encoding_format": "float"},
                timeout=60,
            )
            if response.status_code == 429 and attempt < max_retries - 1:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d.get("index", 0))
            return [d["embedding"] for d in data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            if i > 0:
                time.sleep(1.0)
            embeddings.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]


def get_embedding_model(
    openrouter_token: str,
    model: str | None = None,
) -> OpenRouterEmbeddings:
    model = model or os.environ.get("RAGBENCH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return OpenRouterEmbeddings(model=model, api_key=openrouter_token)


def get_generation_llm(
    openrouter_token: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> ChatOpenAI:
    model = model or os.environ.get("RAGBENCH_GENERATOR_MODEL", DEFAULT_GENERATOR_MODEL)
    return ChatOpenAI(
        api_key=openrouter_token,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=OPENROUTER_BASE_URL,
    )


def get_judge_llm(
    openrouter_token: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """LLM used to annotate responses for TRACe metrics (relevance/utilization/adherence)."""
    model = model or os.environ.get("RAGBENCH_JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    return ChatOpenAI(
        model=model,
        api_key=openrouter_token,
        temperature=temperature,
        base_url=OPENROUTER_BASE_URL,
    )
