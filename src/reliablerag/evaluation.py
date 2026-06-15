import re
import string
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

_ANNOTATION_PROMPT = """\
IMPORTANT: You must respond with a valid JSON object ONLY. The very first character of your response must be '{{' and the last must be '}}'. Do not write anything before or after the JSON. Do not wrap it in markdown fences or backticks.

I asked someone to answer a question based on one or more documents.
Your task is to review their response and assess whether or not each sentence
in that response is supported by text in the documents. And if so, which
sentences in the documents provide that support. You will also tell me which
of the documents contain useful information for answering the question, and
which of the documents the answer was sourced from.
Here are the documents, each of which is split into sentences. Alongside each
sentence is associated key, such as '0a.' or '0b.' that you can use to refer
to it:
'''
{documents}
'''
The question was:
'''
{question}
'''
Here is their response, split into sentences. Alongside each sentence is
associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
that these keys are unique to the response, and are not related to the keys
in the documents:
'''
{answer}
'''
You must respond with a JSON object matching this schema:
'''
{{
"relevance_explanation": string,
"all_relevant_sentence_keys": [string],
"overall_supported_explanation": string,
"overall_supported": boolean,
"sentence_support_information": [
{{
"response_sentence_key": string,
"explanation": string,
"supporting_sentence_keys": [string],
"fully_supported": boolean
}},
],
"all_utilized_sentence_keys": [string]
}}
'''
The relevance_explanation field is a string explaining which documents
contain useful information for answering the question. Provide a step-by-step
breakdown of information provided in the documents and how it is useful for
answering the question.
The all_relevant_sentence_keys field is a list of all document sentences keys
(e.g. '0a') that are relevant to the question. Include every sentence that is
useful and relevant to the question, even if it was not used in the response,
or if only parts of the sentence are useful. Ignore the provided response when
making this judgement and base your judgement solely on the provided documents
and question. Omit sentences that, if removed from the document, would not
impact someone's ability to answer the question.
The overall_supported_explanation field is a string explaining why the response
*as a whole* is or is not supported by the documents. In this field, provide a
step-by-step breakdown of the claims made in the response and the support (or
lack thereof) for those claims in the documents. Begin by assessing each claim
separately, one by one; don't make any remarks about the response as a whole
until you have assessed all the claims in isolation.
The overall_supported field is a boolean indicating whether the response as a
whole is supported by the documents. This value should reflect the conclusion
you drew at the end of your step-by-step breakdown in overall_supported_explanation.
In the sentence_support_information field, provide information about the support
*for each sentence* in the response.
The sentence_support_information field is a list of objects, one for each sentence
in the response. Each object MUST have the following fields:
- response_sentence_key: a string identifying the sentence in the response.
This key is the same as the one used in the response above.
- explanation: a string explaining why the sentence is or is not supported by the
documents.
- supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
support the response sentence. If the sentence is not supported, this list MUST
be empty. If the sentence is supported, this list MUST contain one or more keys.
In special cases where the sentence is supported, but not by any specific sentence,
you can use the string "supported_without_sentence" to indicate that the sentence
is generally supported by the documents. Consider cases where the sentence is
expressing inability to answer the question due to lack of relevant information in
the provided context as "supported_without_sentence". In cases where the sentence
is making a general statement (e.g. outlining the steps to produce an answer, or
summarizing previously stated sentences, or a transition sentence), use the
string "general". In cases where the sentence is correctly stating a well-known fact,
like a mathematical formula, use the string "well_known_fact". In cases where the
sentence is performing numerical reasoning (e.g. addition, multiplication), use
the string "numerical_reasoning".
- fully_supported: a boolean indicating whether the sentence is fully supported by
the documents.
- This value should reflect the conclusion you drew at the end of your step-by-step
breakdown in explanation.
- If supporting_sentence_keys is an empty list, then fully_supported must be false.
- Otherwise, use fully_supported to clarify whether everything in the response
sentence is fully supported by the document text indicated in supporting_sentence_keys
(fully_supported = true), or whether the sentence is only partially or incompletely
supported by that document text (fully_supported = false).
The all_utilized_sentence_keys field is a list of all sentences keys (e.g. '0a') that
were used to construct the answer. Include every sentence that either directly supported
the answer, or was implicitly used to construct the answer, even if it was not used
in its entirety. Omit sentences that were not used, and could have been removed from
the documents without affecting the answer.
You must respond with a valid JSON string. Use escapes for quotes, e.g. '\\"', and
newlines, e.g. '\\n'. Do not write anything before or after the JSON string. Do not
wrap the JSON string in backticks like ''' or '''json.
As a reminder: your task is to review the response and assess which documents contain
useful information pertaining to the question, and how each sentence in the response
is supported by the text in the documents.
REMINDER: Your entire response must be a single raw JSON object. First character: '{{'. Last character: '}}'. Nothing else.\
"""

_CORRECTIVE_RETRY_MSG = """\
Your previous response was not valid JSON. Here is what you returned:

{bad_output}

Return ONLY the raw JSON object — first character must be '{{', last must be '}}'. \
No markdown fences, no explanation, no text before or after the JSON.\
"""


@dataclass
class TRACeScores:
    adherence: bool
    relevance: float
    utilization: float
    completeness: float
    adherence_explanation: str
    relevance_explanation: str
    annotation: dict | None = None


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _index_to_key(i: int) -> str:
    """Convert 0-based index to letter key: 0→a, 1→b, ..., 25→z, 26→aa, ..."""
    letters = string.ascii_lowercase
    if i < 26:
        return letters[i]
    return letters[i // 26 - 1] + letters[i % 26]


def _label_context_sentences(chunks: list[Document]) -> tuple[str, dict[str, str]]:
    """
    Split each chunk into sentences and assign keys like '0a', '0b', '1a' ...
    Returns the labeled text block for the prompt and a {key: sentence} mapping.
    """
    labeled_lines = []
    sentence_by_key: dict[str, str] = {}
    for chunk_idx, chunk in enumerate(chunks):
        sentences = _split_sentences(chunk.page_content)
        for sent_idx, sentence in enumerate(sentences):
            key = f"{chunk_idx}{_index_to_key(sent_idx)}"
            sentence_by_key[key] = sentence
            labeled_lines.append(f"{key}. {sentence}")
    return "\n".join(labeled_lines), sentence_by_key


def _label_response_sentences(response: str) -> tuple[str, dict[str, str]]:
    """
    Split response into sentences and assign keys like 'a', 'b' ...
    Returns the labeled text block for the prompt and a {key: sentence} mapping.
    """
    sentences = _split_sentences(response)
    labeled_lines = []
    sentence_by_key: dict[str, str] = {}
    for i, sentence in enumerate(sentences):
        key = _index_to_key(i)
        sentence_by_key[key] = sentence
        labeled_lines.append(f"{key}. {sentence}")
    return "\n".join(labeled_lines), sentence_by_key


def _token_len(s: str) -> int:
    # RAGBench TRACe uses sentence-length measured in tokens (whitespace-delimited).
    return len(s.split())


def _compute_scores(annotation: dict, context_sentences: dict[str, str]) -> TRACeScores:
    """Compute all four TRACe scores from GPT-4-style span annotations using length-ratio formulas."""
    total_len = sum(_token_len(s) for s in context_sentences.values())

    relevant_keys = set(annotation.get("all_relevant_sentence_keys", []))
    utilized_keys = set(annotation.get("all_utilized_sentence_keys", []))

    relevant_len = sum(_token_len(context_sentences[k]) for k in relevant_keys if k in context_sentences)
    utilized_len = sum(_token_len(context_sentences[k]) for k in utilized_keys if k in context_sentences)
    overlap_len = sum(_token_len(context_sentences[k]) for k in relevant_keys & utilized_keys if k in context_sentences)

    # Relevance: fraction of retrieved context (by token length) that is relevant to the question.
    relevance = relevant_len / total_len if total_len > 0 else 0.0
    # Utilization: fraction of retrieved context (by token length) actually used in the answer.
    utilization = utilized_len / total_len if total_len > 0 else 0.0
    # Completeness: fraction of relevant context (by token length) that was actually utilized.
    completeness = overlap_len / relevant_len if relevant_len > 0 else 0.0

    support_info = annotation.get("sentence_support_information", [])
    # Adherence: True only if every response sentence is fully supported by the context (no hallucinations).
    adherence = all(s.get("fully_supported", False) for s in support_info) if support_info else False

    return TRACeScores(
        adherence=adherence,
        relevance=relevance,
        utilization=utilization,
        completeness=completeness,
        adherence_explanation=annotation.get("overall_supported_explanation", ""),
        relevance_explanation=annotation.get("relevance_explanation", ""),
        annotation=annotation,
    )


def _evaluate_once(llm, question: str, chunks: list[Document], response: str) -> TRACeScores:
    labeled_context, context_sentences = _label_context_sentences(chunks)
    labeled_response, _                = _label_response_sentences(response)

    prompt = _ANNOTATION_PROMPT.format(
        documents=labeled_context,
        question=question,
        answer=labeled_response,
    )

    parser = JsonOutputParser()

    # First attempt.
    raw = llm.invoke(prompt)
    try:
        annotation = parser.parse(raw.content)
        return _compute_scores(annotation, context_sentences)
    except OutputParserException:
        pass

    # Corrective retry — show the model its bad output and ask it to fix it.
    retry_messages = [
        HumanMessage(content=prompt),
        AIMessage(content=raw.content),
        HumanMessage(content=_CORRECTIVE_RETRY_MSG.format(bad_output=raw.content[:500])),
    ]
    raw2 = llm.invoke(retry_messages)
    try:
        annotation = parser.parse(raw2.content)
        return _compute_scores(annotation, context_sentences)
    except OutputParserException as e:
        return TRACeScores(
            adherence=False,
            relevance=0.0,
            utilization=0.0,
            completeness=0.0,
            adherence_explanation=f"parse error: {str(e)[:120]}",
            relevance_explanation="",
        )


def _is_parse_error(run: TRACeScores) -> bool:
    return run.adherence_explanation.startswith("parse error")


def evaluate(llm, question: str, chunks: list[Document], response: str, n_runs: int = 1) -> TRACeScores:
    """
    LLM-as-judge TRACe scoring. With n_runs > 1, calls the judge n times and averages
    numeric scores (majority vote for adherence) to reduce judge variance.
    Parse-error runs are excluded from the average — a failed run is not a measurement.
    If all runs fail, returns the last error sentinel.
    Use a deterministic judge LLM (temperature=0) for best results.
    """
    if n_runs <= 1:
        return _evaluate_once(llm, question, chunks, response)

    all_runs = [_evaluate_once(llm, question, chunks, response) for _ in range(n_runs)]
    good = [r for r in all_runs if not _is_parse_error(r)]

    if not good:
        return all_runs[-1]

    n = len(good)
    return TRACeScores(
        adherence=sum(r.adherence for r in good) * 2 > n,  # strict majority over successful runs only
        relevance=sum(r.relevance for r in good) / n,
        utilization=sum(r.utilization for r in good) / n,
        completeness=sum(r.completeness for r in good) / n,
        adherence_explanation=good[-1].adherence_explanation,
        relevance_explanation=good[-1].relevance_explanation,
        annotation={"runs": [r.annotation for r in good]},
    )
