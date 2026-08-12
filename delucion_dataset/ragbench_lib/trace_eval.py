"""TRACe (uTilization, Relevance, Adherence, Completeness) evaluation.

Shared by every experiment notebook so the annotation prompt and metric
formulas can't drift between them again.

Bug fixes applied here vs. the code this was extracted from:
  1. Sentence keys used to be generated with `chr(97 + sent_idx % 26)`, which
     wraps every 26 sentences and silently collides two different sentences
     onto the same key (e.g. sentence 0 and sentence 26 both become "0a").
     Long documents (CUAD contracts, TechQA notes) regularly exceed 26
     sentences, so this was corrupting relevance/utilization attribution.
     Fixed with an unbounded spreadsheet-style suffix (a, b, ..., z, aa, ab, ...).
  2. The judge model's returned relevant/utilized keys were never checked
     against the keys that actually exist in the documents, so a hallucinated,
     malformed, or duplicated key could silently inflate a score above 1.0.
     Fixed by intersecting returned keys with the real key set before scoring.
"""

import json
import re

from .chunking import get_sentences

ANNOTATION_PROMPT_TEMPLATE = """I asked someone to answer a question based on documents.
Your task is to identify:
1. Which SPECIFIC sentences in the documents are relevant to the question
2. Which of those sentences are actually USED in the response
3. Is the response supported by the context (true/false)

Here are the documents with sentence keys:
'''{documents}'''

Question:
'''{question}'''

Response (with sentence keys):
'''{response}'''

RESPOND WITH ONLY VALID JSON, no markdown:
{{
  "relevant_sentence_keys": ["0a", "0b", "1c"],
  "utilized_sentence_keys": ["0a", "1c"],
  "is_supported": true,
  "brief_explanation": "The response uses 0a and 1c which are relevant."
}}"""


def _key_suffix(idx: int) -> str:
    """Unbounded spreadsheet-column-style letter suffix: 0->a, 25->z, 26->aa, ..."""
    idx += 1
    letters = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        letters = chr(97 + rem) + letters
    return letters


def _normalize_key(key: str) -> str:
    return key.strip().rstrip(".")


def format_documents_with_keys(documents: list[str]) -> str:
    formatted = []
    for doc_idx, doc in enumerate(documents):
        formatted.append(f"\n--- Document {doc_idx} ---")
        sentences = get_sentences(doc)
        for sent_idx, sent in enumerate(sentences):
            key = f"{doc_idx}{_key_suffix(sent_idx)}"
            formatted.append(f"{key}. {sent}")
    return "\n".join(formatted)


def get_valid_document_keys(documents: list[str]) -> set[str]:
    """The full set of sentence keys that actually exist in `documents`."""
    keys = set()
    for doc_idx, doc in enumerate(documents):
        for sent_idx in range(len(get_sentences(doc))):
            keys.add(f"{doc_idx}{_key_suffix(sent_idx)}")
    return keys


def annotate_response_for_metrics(judge_llm, documents: list[str], question: str, response: str) -> dict:
    """Annotate a RAG response to extract relevant/utilized sentence keys for TRACe.

    `judge_llm` is any LangChain chat model (e.g. ragbench_lib.models.get_judge_llm(...)).
    """
    formatted_docs = format_documents_with_keys(documents)
    valid_keys = get_valid_document_keys(documents)

    response_sentences = get_sentences(response)
    response_with_keys = "\n".join(
        f"{_key_suffix(i)}. {s}" for i, s in enumerate(response_sentences)
    )

    prompt_text = ANNOTATION_PROMPT_TEMPLATE.format(
        documents=formatted_docs,
        question=question,
        response=response_with_keys,
    )

    try:
        llm_response = judge_llm.invoke(prompt_text)
        raw_json = llm_response.content

        try:
            output = json.loads(raw_json)
        except json.JSONDecodeError:
            json_match = re.search(r"\{.*\}", raw_json, re.DOTALL)
            if not json_match:
                return {
                    "success": False,
                    "error": "JSON parse failed",
                    "relevant_keys": [],
                    "utilized_keys": [],
                    "is_supported": False,
                }
            output = json.loads(json_match.group())

        raw_relevant = {_normalize_key(k) for k in output.get("relevant_sentence_keys", [])}
        raw_utilized = {_normalize_key(k) for k in output.get("utilized_sentence_keys", [])}

        # Drop hallucinated/malformed keys that don't correspond to a real sentence.
        relevant_keys = sorted(raw_relevant & valid_keys)
        utilized_keys = sorted(raw_utilized & valid_keys)

        return {
            "success": True,
            "relevant_keys": relevant_keys,
            "utilized_keys": utilized_keys,
            "is_supported": bool(output.get("is_supported", False)),
            "dropped_relevant_keys": sorted(raw_relevant - valid_keys),
            "dropped_utilized_keys": sorted(raw_utilized - valid_keys),
            "num_response_sentences": len(response_sentences),
            "num_documents": len(documents),
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "relevant_keys": [],
            "utilized_keys": [],
            "is_supported": False,
        }


def annotate_response_for_metrics_with_retry(
    judge_llm, documents: list[str], question: str, response: str, max_attempts: int = 3
) -> dict:
    """Same as annotate_response_for_metrics, but retries on transient judge failures
    (malformed JSON, dropped connections, etc.) instead of silently discarding the sample.

    Comparing chunking strategies on tiny sample sizes makes every dropped
    annotation disproportionately noisy -- a config that happens to lose 1 of
    5 samples to a JSON hiccup looks artificially different from one that
    doesn't. Retrying up to `max_attempts` times before giving up cuts that
    noise down without changing what "success" means.
    """
    last_result = None
    for _ in range(max_attempts):
        last_result = annotate_response_for_metrics(judge_llm, documents, question, response)
        if last_result["success"]:
            return last_result
    return last_result


def compute_context_relevance(documents: list[str], annotation: dict) -> float:
    """Context Relevance = |relevant keys| / total sentences. Paper Eq. (2)."""
    total_sentences = sum(len(get_sentences(doc)) for doc in documents)
    if total_sentences == 0:
        return 0.0
    return round(len(set(annotation.get("relevant_keys", []))) / total_sentences, 4)


def compute_utilization(documents: list[str], annotation: dict) -> float:
    """Context Utilization = |utilized keys| / total sentences. Paper Eq. (3)."""
    total_sentences = sum(len(get_sentences(doc)) for doc in documents)
    if total_sentences == 0:
        return 0.0
    return round(len(set(annotation.get("utilized_keys", []))) / total_sentences, 4)


def compute_completeness(annotation: dict) -> float:
    """Completeness = |relevant ∩ utilized| / |relevant|. Paper Eq. (4)."""
    relevant_keys = set(annotation.get("relevant_keys", []))
    utilized_keys = set(annotation.get("utilized_keys", []))
    if len(relevant_keys) == 0:
        return 0.0
    return round(len(relevant_keys & utilized_keys) / len(relevant_keys), 4)


def compute_adherence(annotation: dict) -> float:
    """Adherence = 1.0 if the response is fully supported by the context, else 0.0."""
    return 1.0 if annotation.get("is_supported", False) else 0.0
