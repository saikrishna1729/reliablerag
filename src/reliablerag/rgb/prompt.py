"""The RGB Figure-3 prompt (English), verbatim, plus single-turn ``ask``/``ask_direct`` helpers.

The two control sentences in the system prompt are exact — the rejection-rate and error-detection
metrics string-match against them, so do not paraphrase.
"""

import time

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

SYSTEM_PROMPT = (
    "You are an accurate and reliable AI assistant that can answer questions with the help of "
    "external documents. Please note that external documents may contain noisy or factually "
    "incorrect information. If the information in the document contains the correct answer, you "
    "will give an accurate answer. If the information in the document does not contain the answer, "
    "you will generate 'I can not answer the question because of the insufficient information in "
    "documents.' If there are inconsistencies with the facts in some of the documents, please "
    "generate the response 'There are factual errors in the provided documents.' and provide the "
    "correct answer."
)

USER_TEMPLATE = "Document:\n{DOCS} \n\nQuestion:\n{QUERY}"


MAX_RETRIES = 1
RETRY_BACKOFF_SECONDS = 2.0  # multiplied by attempt number for a simple linear backoff


def format_docs(docs: list[str]) -> str:
    return "\n".join(docs)


def _invoke_with_retries(llm: BaseChatModel, messages: list[BaseMessage], max_retries: int) -> str:
    """Shared retry loop for a single chat turn. See ask()'s docstring for the retry rationale."""
    if max_retries < 1:
        max_retries = 1

    last_error: Exception = RuntimeError("unreachable")
    for attempt in range(1, max_retries + 1):
        try:
            resp = llm.invoke(messages)
            if not isinstance(resp.content, str):
                raise TypeError(f"Expected str content from LLM, got {type(resp.content)}")
            return resp.content
        except Exception as e:
            last_error = e
            if attempt == max_retries:
                break
            wait_seconds = RETRY_BACKOFF_SECONDS * attempt
            print(f"[ask] attempt {attempt}/{max_retries} failed ({e!r}); retrying in {wait_seconds:.1f}s")
            time.sleep(wait_seconds)
    raise last_error


def ask(llm: BaseChatModel, query: str, docs: list[str], max_retries: int = MAX_RETRIES) -> str:
    """Send one question to the model with the given documents as context, using the RGB
    document-grounded system prompt (Figure 3).

    Pass docs=[] to send an empty Document section — NOT the same as asking with no RAG framing
    at all. The system prompt still tells the model it's a document-grounded assistant that should
    refuse when "the document" lacks the answer, so an empty doc list makes a well-aligned model
    refuse regardless of what it actually knows. Use ask_direct() to test raw internal knowledge
    instead (the counterfactual-robustness ACC baseline).

    By default this makes a single attempt (max_retries=1) — the OpenAI client already retries
    transient errors internally (see create_llm's max_retries). Pass a higher max_retries to also
    retry the whole call here, e.g. on the non-str-content check in _invoke_with_retries.
    """
    user = USER_TEMPLATE.format(DOCS=format_docs(docs), QUERY=query)
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    return _invoke_with_retries(llm, messages, max_retries)


def ask_direct(llm: BaseChatModel, query: str, max_retries: int = MAX_RETRIES) -> str:
    """Ask a question with no system prompt and no document-grounding framing at all — just the
    bare question. Used for the counterfactual-robustness ACC baseline, which is meant to measure
    the model's own internal knowledge (matches the paper's own methodology: "we assess their
    performance by directly asking them questions"). ask() cannot be reused for this with docs=[]
    since its system prompt still instructs the model to refuse when "the document" (here, empty)
    lacks the answer — that measures instruction-following under an empty-context RAG persona, not
    raw knowledge.
    """
    messages = [HumanMessage(content=query)]
    return _invoke_with_retries(llm, messages, max_retries)