import re
import chromadb
from typing import List, Dict, Any
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from config import ExperimentConfig, CHROMA_DIR

def clean_tokenize(text: str) -> List[str]:
    """Simple tokenizer that splits words and lowercases them."""
    return re.findall(r'\w+', text.lower())

class DenseRetriever:
    def __init__(self, collection, embedder, top_k: int):
        self.collection = collection
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, query: str) -> List[Document]:
        query_vector = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.top_k
        )
        
        docs = []
        if results and results["documents"] and len(results["documents"]) > 0:
            documents = results["documents"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(documents)
            ids = results["ids"][0]
            
            for doc_text, meta, doc_id in zip(documents, metadatas, ids):
                meta_copy = meta.copy() if meta else {}
                meta_copy["id"] = doc_id
                docs.append(Document(page_content=doc_text, metadata=meta_copy))
        return docs


class SparseRetriever:
    def __init__(self, chunks: List[Document], top_k: int):
        self.chunks = chunks
        self.top_k = top_k
        self.corpus_tokens = [clean_tokenize(chunk.page_content) for chunk in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str) -> List[Document]:
        query_tokens = clean_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort chunks by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        docs = []
        for idx in ranked_indices[:self.top_k]:
            if scores[idx] > 0: # Only return matching documents
                docs.append(self.chunks[idx])
                
        # If no documents match, fallback to returning top_k documents
        if not docs:
            docs = self.chunks[:self.top_k]
            
        return docs


class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: SparseRetriever, top_k: int, rrf_k: int = 60):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> List[Document]:
        # Get rankings from both retrievers
        dense_results = self.dense_retriever.retrieve(query=query)
        sparse_results = self.sparse_retriever.retrieve(query=query)
        
        # Apply Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_map = {}
        
        # Score Dense results
        for rank, doc in enumerate(dense_results):
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Score Sparse results
        for rank, doc in enumerate(sparse_results):
            doc_key = doc.page_content
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            
        # Sort docs by RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        
        return [doc_map[key] for key in sorted_keys[:self.top_k]]


def initialize_chroma_collection(config: ExperimentConfig, chunks: List[Document], embedder: Any):
    """
    Initializes a persistent ChromaDB collection. 
    Builds the collection if empty, otherwise loads the existing index.
    """
    if config.profile == "local-mock" or (hasattr(embedder, "is_mock") and embedder.is_mock):
        # Use an ephemeral client for mocking to avoid cluttering disk
        client = chromadb.EphemeralClient()
    else:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
    collection = client.get_or_create_collection(config.chroma_collection_name)
    
    if collection.count() == 0 and chunks:
        print(f"Building persistent index '{config.chroma_collection_name}' with {len(chunks)} chunks...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedder.encode(texts).tolist()
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Add to Chroma in batches to prevent payload issues
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            collection.add(
                embeddings=embeddings[i:i+batch_size],
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print("Indexing completed.")
    else:
        print(f"Loaded existing index '{config.chroma_collection_name}': {collection.count()} chunks. Skipping embedding.")
        
    return collection


def get_retriever(config: ExperimentConfig, chunks: List[Document], embedder: Any):
    """
    Returns the configured retriever (dense, sparse, hybrid) pre-populated with chunks.
    """
    # Initialize Dense Index
    collection = initialize_chroma_collection(config, chunks, embedder)
    dense = DenseRetriever(collection, embedder, config.top_k)
    
    if config.retriever == "dense":
        return dense
    elif config.retriever == "sparse":
        return SparseRetriever(chunks, config.top_k)
    elif config.retriever == "hybrid":
        sparse = SparseRetriever(chunks, config.top_k)
        return HybridRetriever(dense, sparse, config.top_k)
    else:
        raise ValueError(f"Unknown retriever: {config.retriever}")
