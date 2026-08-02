import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(
    texts: List[str],
    metadatas: Optional[List[Dict[str, Any]]] = None,
    chunk_strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    embedder: Optional[Any] = None,
    dataset_name: str = "unknown"
) -> List[Document]:
    """
    Splits a list of raw texts into a list of LangChain Document chunks.
    
    Args:
        texts: List of strings (documents) to be split.
        metadatas: Optional list of dictionaries containing metadata for each text.
        chunk_strategy: Strategy to use ('recursive', 'semantic', 'metadata').
        chunk_size: Target size of each chunk (characters).
        chunk_overlap: Target overlap between chunks (characters).
        embedder: Embedding model instance (required for semantic chunking).
        dataset_name: Name of the dataset/domain (used in metadata chunking).
        
    Returns:
        List of Document objects representing the chunks.
    """
    if metadatas is None:
        metadatas = [{} for _ in range(len(texts))]

    docs = []
    for text, meta in zip(texts, metadatas):
        docs.append(Document(page_content=text, metadata=meta.copy()))

    if chunk_strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(docs)

    elif chunk_strategy == "metadata":
        # Metadata-aware chunking: prepend metadata context to each chunk
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        sub_docs = splitter.split_documents(docs)
        for d in sub_docs:
            meta_str = f"[Domain: {dataset_name.upper()}] "
            if "source" in d.metadata:
                meta_str += f"[Source: {d.metadata['source']}] "
            d.page_content = meta_str + d.page_content
        return sub_docs

    elif chunk_strategy == "semantic":
        # Semantic chunking: split into sentences, embed them, group based on similarity
        chunks = []
        for doc in docs:
            text = doc.page_content
            # Split into sentences
            sentences = re.split(r'(?<=[.?!])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                continue

            if embedder is None or hasattr(embedder, "is_mock") and embedder.is_mock:
                # If mock embedder, fallback to character-based grouping up to chunk_size
                current_chunk = []
                current_len = 0
                for sent in sentences:
                    if current_len + len(sent) > chunk_size and current_chunk:
                        chunks.append(Document(
                            page_content=" ".join(current_chunk),
                            metadata=doc.metadata.copy()
                        ))
                        current_chunk = []
                        current_len = 0
                    current_chunk.append(sent)
                    current_len += len(sent)
                if current_chunk:
                    chunks.append(Document(
                        page_content=" ".join(current_chunk),
                        metadata=doc.metadata.copy()
                    ))
            else:
                # Semantic distance splitting using real embeddings
                try:
                    embeddings = embedder.encode(sentences)
                    # Compute similarities between adjacent sentences
                    import numpy as np
                    similarities = []
                    for i in range(len(sentences) - 1):
                        vec1 = embeddings[i]
                        vec2 = embeddings[i + 1]
                        norm1 = np.linalg.norm(vec1)
                        norm2 = np.linalg.norm(vec2)
                        if norm1 > 0 and norm2 > 0:
                            sim = np.dot(vec1, vec2) / (norm1 * norm2)
                        else:
                            sim = 0.0
                        similarities.append(sim)

                    # Determine splits based on percentile threshold
                    if similarities:
                        threshold = np.percentile(similarities, 40) # Split at bottom 40% similarity
                    else:
                        threshold = 0.5

                    current_chunk = [sentences[0]]
                    for i, sim in enumerate(similarities):
                        if sim < threshold:
                            # Start a new chunk
                            chunks.append(Document(
                                page_content=" ".join(current_chunk),
                                metadata=doc.metadata.copy()
                            ))
                            current_chunk = [sentences[i + 1]]
                        else:
                            current_chunk.append(sentences[i + 1])
                    if current_chunk:
                        chunks.append(Document(
                            page_content=" ".join(current_chunk),
                            metadata=doc.metadata.copy()
                        ))
                except Exception as e:
                    # Fallback to length-based grouping on exception
                    current_chunk = []
                    current_len = 0
                    for sent in sentences:
                        if current_len + len(sent) > chunk_size and current_chunk:
                            chunks.append(Document(
                                page_content=" ".join(current_chunk),
                                metadata=doc.metadata.copy()
                            ))
                            current_chunk = []
                            current_len = 0
                        current_chunk.append(sent)
                        current_len += len(sent)
                    if current_chunk:
                        chunks.append(Document(
                            page_content=" ".join(current_chunk),
                            metadata=doc.metadata.copy()
                        ))

        return chunks

    else:
        raise ValueError(f"Unknown chunk strategy: {chunk_strategy}")
