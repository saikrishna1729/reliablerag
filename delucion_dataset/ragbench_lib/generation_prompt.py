"""RAG answer-generation prompt shared by all experiment notebooks.

Note: this wording is a project-specific variant, not the flat template from
RAGBench paper Section 7.3 ("Use the following pieces of context to answer
the question. {documents} Question: {question}"). Numbers produced with this
prompt are therefore not directly comparable to the paper's reported TRACe
results -- keep that in mind when writing up findings.
"""

from langchain_core.prompts import ChatPromptTemplate

RAG_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Answer the question based only on the context.
Use the complete context and infer the answer, don't conclude individually.

Context:
{context}"""),
    ("human", "{question}"),
])
