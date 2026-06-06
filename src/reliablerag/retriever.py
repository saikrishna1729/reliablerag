from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from reliablerag.embeddings import get_embeddings


def build_vector_store(
    documents: list[Document],
    embedding_model: str,
    persist_directory: str,
    collection_name: str = "reliablerag",
) -> Chroma:
    embeddings = get_embeddings(embedding_model)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


def load_vector_store(
    embedding_model: str,
    persist_directory: str,
    collection_name: str = "reliablerag",
) -> Chroma:
    embeddings = get_embeddings(embedding_model)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )


def get_retriever(
    vector_store: Chroma,
    top_k: int = 2,
) -> VectorStoreRetriever:
    return vector_store.as_retriever(search_kwargs={"k": top_k})
