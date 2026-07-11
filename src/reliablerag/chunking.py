import re

from langchain_text_splitters import TextSplitter

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


class SentenceTextSplitter(TextSplitter):
    """Merges sentences into chunks up to chunk_size, never splitting a sentence across chunks.

    Unlike RecursiveCharacterTextSplitter, chunk boundaries always fall between sentences —
    addresses CUAD clauses (typically one sentence) being cut mid-sentence by character-based
    splitting. No overlap: sentence boundaries make character-offset overlap not meaningful here.
    """

    def __init__(self, chunk_overlap: int = 0, **kwargs):
        super().__init__(chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> list[str]:
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
        chunks = []
        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > self._chunk_size and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks
