import time
from collections.abc import Callable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough

PROMPT_V1: str = """\
You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_V2: str = """\
You are a contract analysis assistant. The context below contains retrieved chunks from a legal contract.

Answer directly based on the contract text in the context.
- If the question asks whether a clause exists:
  - If YES: quote the exact text from the context that answers the question.
  - If NO: state it is absent. Do not describe what the context does contain.
- If the question asks for a specific value (date, amount, period): state it directly and quote where it appears.
- If the value is not present: state that it is absent.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_V3: str = """\
You are a contract analysis assistant. The context below contains retrieved chunks from a legal contract.

Answer the question directly based on the contract text in the context.
- If you can answer: quote the exact text from the context that supports the answer.
- If the requested clause or value is not present: state that it is absent, and briefly note
  which section(s) or topic areas in the context you checked that would normally cover this
  (e.g. "No such clause; the Termination and Confidentiality sections do not address this.").

Keep answers concise (1-3 sentences), but always ground the answer in specific context — a quote,
a section reference, or both. Never answer with a bare "absent"/"no" alone.

Context:
{context}

Question: {question}

Answer:"""

_RAG_PROMPT_TEMPLATE: str = PROMPT_V2


class TimingCallbackHandler(BaseCallbackHandler):
    def __init__(self, provider_resolver: Callable[[str], str | None] | None = None):
        self._llm_t0 = None
        self._llm_label = "llm"
        self._retriever_t0 = None
        # Optional hook: given a provider-side generation id (from response_metadata["id"]),
        # returns a human-readable provider name to print alongside the timing line. Lets callers
        # verify which upstream provider actually served a call (e.g. after pinning one on
        # OpenRouter) without coupling this handler to any specific API.
        self._provider_resolver = provider_resolver

    def on_retriever_start(self, serialized, query, **kwargs):
        self._retriever_t0 = time.perf_counter()

    def on_retriever_end(self, documents, **kwargs):
        print(f"[timing] retriever: {time.perf_counter() - self._retriever_t0:.3f}s")

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self._llm_t0 = time.perf_counter()
        # Label the timing line by the tag bound to the model (see notebook: .with_config(tags=...)).
        tags = kwargs.get("tags") or []
        if "hyde" in tags:
            self._llm_label = "hyde llm"
        elif "generator" in tags:
            self._llm_label = "generator llm"
        elif "judge" in tags:
            self._llm_label = "judge llm"
        else:
            self._llm_label = "llm"

    def on_llm_end(self, response, **kwargs):
        elapsed = time.perf_counter() - self._llm_t0
        provider_note = ""
        if self._provider_resolver is not None:
            gen_id = None
            try:
                generation = response.generations[0][0]
                if isinstance(generation, ChatGeneration):
                    gen_id = generation.message.response_metadata.get("id")
            except IndexError:
                gen_id = None
            if isinstance(gen_id, str):
                provider = self._provider_resolver(gen_id)
                if provider:
                    provider_note = f"  [{provider}]"
        print(f"[timing] {self._llm_label:<14}: {elapsed:.3f}s{provider_note}")


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: Runnable,
    llm: Runnable[LanguageModelInput, AIMessage],
    prompt_template: str = _RAG_PROMPT_TEMPLATE,
) -> Runnable:
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        RunnableParallel({"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()})
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def build_generation_chain(
    llm: Runnable[LanguageModelInput, AIMessage],
    prompt_template: str = _RAG_PROMPT_TEMPLATE,
) -> Runnable:
    """Generation-only chain — no retriever, so it never re-retrieves.

    Input is a dict {"context": list[Document], "question": str} of *already-retrieved* chunks.
    Use this when the caller has retrieved once and wants to reuse those chunks for generation,
    instead of build_rag_chain which re-runs the retriever internally.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        RunnableParallel({
            "context":  RunnableLambda(lambda x: _format_docs(x["context"])),
            "question": RunnableLambda(lambda x: x["question"]),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
