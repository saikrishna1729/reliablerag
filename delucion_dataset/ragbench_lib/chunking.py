"""Tokenization and semantic (paragraph-grouping) chunking helpers."""

import tiktoken
from langchain_core.documents import Document
from nltk.tokenize import sent_tokenize

encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def get_sentences(text: str) -> list[str]:
    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def create_semantic_chunks(
    docs_df,
    target_tokens: int = 192,
    output: str = "documents",
    id_infix: str = "c",
):
    """Group paragraphs into ~target_tokens-sized chunks.

    output="documents" (default) returns list[Document], as used by the
    retriever/reranker/repacking notebooks.
    output="dicts" returns list[dict] with chunk_id/text/tokens/sentences/row_id,
    as used by the chunk-size sweep in 03_chunking_optimization.ipynb.
    id_infix controls the chunk_id suffix, e.g. "{doc_id}_{id_infix}{chunk_idx}".
    """
    chunks = []

    for _, doc in docs_df.iterrows():
        text = doc["text"]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            paragraphs = [text]

        chunk_idx = 0
        i = 0
        while i < len(paragraphs):
            chunk_parts = []
            current_tokens = 0
            start_idx = i

            while i < len(paragraphs) and current_tokens < target_tokens:
                para = paragraphs[i]
                para_tokens = count_tokens(para)

                if current_tokens + para_tokens <= target_tokens or not chunk_parts:
                    chunk_parts.append(para)
                    current_tokens += para_tokens
                    i += 1
                else:
                    break

            if chunk_parts:
                chunk_text = "\n\n".join(chunk_parts)
                chunk_id = f"{doc['doc_id']}_{id_infix}{chunk_idx}"

                if output == "dicts":
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                        "tokens": count_tokens(chunk_text),
                        "sentences": len(get_sentences(chunk_text)),
                        "row_id": doc["row_id"],
                    })
                else:
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "row_id": doc["row_id"],
                            "tokens": count_tokens(chunk_text),
                        },
                    ))
                chunk_idx += 1

            if i == start_idx:
                i += 1

    return chunks
