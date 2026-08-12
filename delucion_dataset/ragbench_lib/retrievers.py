"""Retriever implementations shared verbatim across multiple experiment notebooks.

Retrievers that are each notebook's actual independent variable (BM25Retriever
in 05, OptimizedRetriever in 10_full, LiveRetriever in 10_live, the rerankers
in 08) are intentionally left notebook-local -- unifying those would hide the
thing each experiment is testing.

DenseMMRRetriever, HybridRetriever, and OpenRouterReranker, by contrast, are *fixed
foundations* in every notebook downstream of the experiments that established them
as winners (11_chunking_goldilocks.ipynb's chunking config,
12_retriever_goldilocks.ipynb's hybrid alpha=0.3 retriever config,
14_reranking_goldilocks.ipynb's Cohere 4-Pro reranker) -- shared here so later
notebooks (e.g. 13_query_transformation_goldilocks.ipynb, 15_document_repacking_
goldilocks.ipynb, 16_holdout_validation.ipynb) build on the exact same
implementation instead of a redefinition that could silently drift.
"""

import numpy as np
import requests
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from .vector_store import get_persist_dir, vector_store_names


class DenseMMRRetriever:
    """Dense retrieval with Maximal Marginal Relevance."""

    def __init__(self, documents, embedding_model, k=8, dataset_name: str = "delucionqa"):
        self.k = k
        prefix, collection_name = vector_store_names(dataset_name, "mmr")
        persist_dir = get_persist_dir(prefix)
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

    def retrieve(self, query):
        return self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "lambda_mult": 0.5},
        ).invoke(query)


class HybridRetriever:
    """BM25 + dense similarity, linearly combined after min-max normalizing each
    side. `alpha` is the weight on BM25; `(1 - alpha)` is the weight on the
    normalized dense score. Winning config from 12_retriever_goldilocks.ipynb:
    k=4, alpha=0.3.
    """

    def __init__(self, documents, embedding_model, k=4, alpha=0.3, dataset_name: str = "delucionqa",
                 config_name: str = "hybrid", persist_directory: str | None = None):
        """`persist_directory`, when given, points at an existing on-disk Chroma store
        (e.g. one a prior notebook run already embedded) to reuse instead of always
        creating a fresh directory and re-embedding every document via the API. If the
        collection there already holds exactly `len(documents)` vectors it's loaded
        as-is; otherwise it falls back to the normal from_documents build (e.g. first
        run, or the store is stale/partial)."""
        self.k = k
        self.alpha = alpha
        self.documents = documents
        tokenized = [d.page_content.split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)

        prefix, collection_name = vector_store_names(dataset_name, config_name)
        persist_dir = persist_directory or get_persist_dir(prefix)

        reuse = False
        if persist_directory is not None:
            try:
                existing = Chroma(
                    collection_name=collection_name, embedding_function=embedding_model,
                    persist_directory=persist_dir,
                )
                reuse = existing._collection.count() == len(documents)
            except Exception:
                reuse = False

        if reuse:
            self.vector_store = existing
        else:
            self.vector_store = Chroma.from_documents(
                documents=documents, embedding=embedding_model,
                collection_name=collection_name, persist_directory=persist_dir,
            )

    def retrieve(self, query):
        bm25_scores = np.array(self.bm25.get_scores(query.split()), dtype=float)
        if np.max(bm25_scores) > np.min(bm25_scores):
            bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
        else:
            bm25_scores = np.ones_like(bm25_scores) / max(len(bm25_scores), 1)

        dense_results = self.vector_store.similarity_search_with_score(query, k=len(self.documents))
        dense_score_by_content = {}
        if dense_results:
            raw = np.array([score for _, score in dense_results], dtype=float)
            if np.max(raw) > np.min(raw):
                norm = 1.0 - (raw - np.min(raw)) / (np.max(raw) - np.min(raw))
            else:
                norm = np.ones_like(raw)
            for (doc, _), score in zip(dense_results, norm):
                dense_score_by_content[doc.page_content] = score

        combined = np.array([
            self.alpha * bm25_scores[i] + (1 - self.alpha) * dense_score_by_content.get(d.page_content, 0.0)
            for i, d in enumerate(self.documents)
        ])
        top_indices = np.argsort(combined)[-self.k:][::-1]
        return [self.documents[i] for i in top_indices]


class OpenRouterReranker:
    """Cross-encoder reranking via OpenRouter's dedicated /api/v1/rerank endpoint.
    Winning model from 14_reranking_goldilocks.ipynb: cohere/rerank-4-pro.
    """

    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.api_key = api_key
        self.endpoint = "https://openrouter.ai/api/v1/rerank"

    def rerank(self, query, documents, top_k):
        if not documents:
            return documents
        doc_texts = [d.page_content for d in documents]
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "query": query, "documents": doc_texts, "top_n": top_k}
        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            if "results" in result:
                ranked = sorted(result["results"], key=lambda r: r.get("relevance_score", 0), reverse=True)
                indices = [r["index"] for r in ranked[:top_k]]
                return [documents[i] for i in indices]
            return documents[:top_k]
        except Exception:
            return documents[:top_k]
