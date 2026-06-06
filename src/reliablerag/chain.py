from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

_RAG_PROMPT_TEMPLATE: str = """\
You are a helpful assistant. Use the following pieces of retrieved context to answer the question.
If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: list) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(
    retriever: VectorStoreRetriever,
    model: str,
    temperature: float = 0.0,
    prompt_template: str = _RAG_PROMPT_TEMPLATE,
) -> Runnable:
    llm = ChatOllama(model=model, temperature=temperature)
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain