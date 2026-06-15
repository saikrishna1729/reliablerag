import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever
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
    The collection is considered cached if its subdirectory already exists on disk.
    """
    collection_dir = os.path.join(persist_directory, collection_name)
    if os.path.isdir(collection_dir):
        vs = load_vector_store(
            embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
            collection_metadata=collection_metadata,
        )
        return vs, True
    vs = build_vector_store(
        documents,
        embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata=collection_metadata,
    )
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
