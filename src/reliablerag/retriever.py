from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field, PrivateAttr
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


_DEFAULT_COLLECTION_METADATA = {"hnsw:space": "cosine"}


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str | None = None,
    collection_name: str = "reliablerag",
    collection_metadata: dict | None = None,
) -> Chroma:
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata=collection_metadata or _DEFAULT_COLLECTION_METADATA,
    )


def load_vector_store(
    embeddings: Embeddings,
    persist_directory: str,
    collection_name: str = "reliablerag",
    collection_metadata: dict | None = None,
) -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
        collection_metadata=collection_metadata or _DEFAULT_COLLECTION_METADATA,
    )


def get_or_build_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str,
    collection_name: str,
    collection_metadata: dict | None = None,
) -> tuple[Chroma, bool]:
    """Load an existing persisted collection or build and persist a new one.

    Returns (vector_store, cache_hit) so callers can log whether we skipped embedding.

    Chroma 1.x stores collections inside chroma.sqlite3 + UUID-named index dirs, NOT a directory
    named after the collection — so we detect a cache hit by loading the collection and checking
    whether it already holds vectors, rather than by testing for a directory on disk. Loading a
    non-existent collection yields an empty one (count 0), which we then populate; this also avoids
    re-adding documents to an already-built collection (which duplicated chunks previously).
    """
    vs = load_vector_store(
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata=collection_metadata,
    )
    if vs._collection.count() > 0:
        return vs, True
    vs.add_documents(documents)
    return vs, False


def get_retriever(
    vector_store: Chroma,
    top_k: int = 2,
) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> CrossEncoder:
    """Load a cross-encoder reranker. The first call downloads weights (~280MB for the base model)."""
    return CrossEncoder(model_name)


def rerank_documents(
    reranker: CrossEncoder,
    query: str,
    documents: list[Document],
    top_n: int = 8,
) -> list[Document]:
    """Score every (query, doc) pair with the cross-encoder and return the top_n by score."""
    if not documents:
        return []
    pairs = [(query, d.page_content) for d in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [d for d, _ in ranked[:top_n]]


def get_reranked_retriever(
    vector_store: Chroma,
    reranker: CrossEncoder,
    fetch_k: int = 50,
    top_n: int = 8,
) -> Runnable:
    """Bi-encoder retrieves fetch_k candidates from Chroma, cross-encoder reranks to top_n."""
    base = vector_store.as_retriever(search_kwargs={"k": fetch_k})

    def _retrieve_and_rerank(query: str) -> list[Document]:
        return rerank_documents(reranker, query, base.invoke(query), top_n=top_n)

    return RunnableLambda(_retrieve_and_rerank)


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """Keyword retriever backed by rank-bm25 (BM25Okapi)."""

    documents: list[Document] = Field(repr=False)
    k: int = 4
    _bm25: BM25Okapi = PrivateAttr()

    def model_post_init(self, __context) -> None:
        corpus = [_tokenize(d.page_content) for d in self.documents]
        self._bm25 = BM25Okapi(corpus)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return [self.documents[i] for i in top_indices]

    @classmethod
    def from_documents(cls, documents: list[Document], k: int = 4) -> "BM25Retriever":
        return cls(documents=documents, k=k)


def get_hybrid_reranked_retriever(
    vector_store: Chroma,
    documents: list[Document],
    reranker: CrossEncoder,
    fetch_k: int = 40,
    top_n: int = 20,
    rrf_k: int = 60,
    bm25_weight: float = 0.5,
) -> Runnable:
    """Equal-weight hybrid (BM25 + cosine, RRF) over-fetches fetch_k candidates,
    then a cross-encoder reranks to top_n.  Keeps BM25's recall while filtering noise."""
    hybrid = get_hybrid_retriever(
        vector_store, documents, top_k=fetch_k, rrf_k=rrf_k, bm25_weight=bm25_weight
    )

    def _retrieve_and_rerank(query: str) -> list[Document]:
        candidates = hybrid.invoke(query)
        return rerank_documents(reranker, query, candidates, top_n=top_n)

    return RunnableLambda(_retrieve_and_rerank)


def get_wide_hybrid_reranked_retriever(
    vector_store: Chroma,
    documents: list[Document],
    reranker: CrossEncoder,
    fetch_k: int = 100,
    top_n: int = 20,
) -> Runnable:
    """Dense and BM25 each independently fetch fetch_k candidates (no RRF fusion beforehand),
    unioned and deduped, then a cross-encoder reranks the union down to top_n.

    Diagnosed against 56 CUAD date-extraction questions: with fetch_k=20 (the RRF-fused
    get_hybrid_reranked_retriever's effective candidate depth), the correct chunk landed in
    either retriever's own top-20 for only 39% of questions, capping recall regardless of the
    fusion/rerank step. Widening each retriever's independent fetch to top-100 got the correct
    chunk into the candidate pool 84% of the time, and cross-encoder reranking down to top_n=20
    then recovered 77% overall (vs 30% for the current top_k=20 hybrid RRF retriever) — RRF's
    rank-fusion math was truncating away single-source-only hits before the reranker ever saw
    them, so the union is taken directly instead of pre-fusing via RRF.
    """
    dense = vector_store.as_retriever(search_kwargs={"k": fetch_k})
    bm25 = BM25Retriever.from_documents(documents, k=fetch_k)

    def _retrieve_and_rerank(query: str) -> list[Document]:
        seen: dict[str, Document] = {}
        for doc in dense.invoke(query) + bm25.invoke(query):
            seen[doc.page_content] = doc
        return rerank_documents(reranker, query, list(seen.values()), top_n=top_n)

    return RunnableLambda(_retrieve_and_rerank)


_HYDE_PROMPT = (
    "You are a contract analyst. Write a short hypothetical contract clause (2-4 sentences) "
    "that directly answers the following question. Write only the clause text, no preamble.\n\n"
    "Question: {question}"
)


def get_hyde_retriever(
    vector_store: Chroma,
    embeddings: Embeddings,
    llm: Runnable[LanguageModelInput, AIMessage],
    top_k: int = 20,
) -> Runnable:
    """HyDE retriever: generates a hypothetical answer clause, embeds it, then retrieves by vector.

    Addresses vocabulary mismatch — the hypothetical uses contract terminology that better
    matches how clauses are phrased in the document, rather than matching the query wording.
    """
    def _hyde_retrieve(query: str) -> list[Document]:
        prompt = _HYDE_PROMPT.format(question=query)
        hypothetical = str(llm.invoke([HumanMessage(content=prompt)]).content)
        hyp_vector = embeddings.embed_query(hypothetical)
        return vector_store.similarity_search_by_vector(hyp_vector, k=top_k)

    return RunnableLambda(_hyde_retrieve)


def get_hybrid_retriever(
    vector_store: Chroma,
    documents: list[Document],
    top_k: int = 20,
    rrf_k: int = 60,
    bm25_weight: float = 0.5,
) -> Runnable:
    """Combine dense cosine similarity (Chroma) and sparse keyword (BM25) retrieval via Reciprocal Rank Fusion.

    Each retriever fetches top_k candidates. Weighted RRF score:
      score += dense_weight / (rrf_k + rank + 1)  for cosine results
      score += bm25_weight  / (rrf_k + rank + 1)  for BM25 results
    where dense_weight = 1 - bm25_weight. Top top_k unique docs by score are returned.
    """
    dense_weight = 1.0 - bm25_weight
    cosine_retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    bm25_retriever = BM25Retriever.from_documents(documents, k=top_k)

    def _hybrid_retrieve(query: str) -> list[Document]:
        dense_cosine_results = cosine_retriever.invoke(query)
        sparse_bm25_results = bm25_retriever.invoke(query)

        rrf_scores: dict[str, float] = {}
        content_to_doc: dict[str, Document] = {}

        for rank, doc in enumerate(dense_cosine_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + dense_weight / (rrf_k + rank + 1)
            content_to_doc[key] = doc

        for rank, doc in enumerate(sparse_bm25_results):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + bm25_weight / (rrf_k + rank + 1)
            content_to_doc[key] = doc

        ranked = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        return [content_to_doc[k] for k in ranked[:top_k]]

    return RunnableLambda(_hybrid_retrieve)
