# RGB Prompt Configuration

# System prompt directing the model to answer correctly or report lack of information or factual errors.
RGB_SYSTEM_PROMPT = (
    "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. "
    "Please note that external documents may contain noisy or factually incorrect information. "
    "If the information in the document contains the correct answer, you will give an accurate answer. "
    "If the information in the document does not contain the answer, you will generate "
    "'I can not answer the question because of the insufficient information in documents.'. "
    "If there are inconsistencies with the facts in some of the documents, please generate the response "
    "'There are factual errors in the provided documents.' and provide the correct answer."
)

# Instruction template mapping documents and query.
RGB_INSTRUCTION_TEMPLATE = "Document:\n{DOCS}\n\nQuestion:\n{QUERY}"

# Prompt for the LLM judge to classify whether the model rejected correctly when no information was available.
REJECTION_JUDGE_TEMPLATE = """I will give you a question and an answer generated through document retrieval. Please use this answer to determine if the retrieved document can solve the question.
Demonstrations:
Question: 2023年澳网女单冠军是谁
Answer:文档信息不足，因此我无法基于提供的文档回答该问题。
No, the question is not addressed by the documents.

Question: Who is the champion of Australian Open 2023 Women's Singles?
Answer: Serena Williams
Yes, the question is addressed by the documents.

Question: Where is ACL2023 held?
Answer: Location of ACL2023 has not been confirmed.
No, the question is not addressed by the documents.

Question:  2023年中国GDP是多少?
Answer: I can not answer this question。
No, the question is not addressed by the documents.

Begin to generate:
Question: {question}
Answer: {answer}"""

# Prompt for the LLM judge to classify whether the model successfully identified factual errors.
FACTUAL_JUDGE_TEMPLATE = """Please determine whether the model can identify factual errors in the document. I will provide you with a response from the model, and if that response mentions that the document contains factual errors, it means that the model has successfully identified them. Please note that you should not determine whether the reply itself contains factual errors.
Demonstrations:
Response:文档信息有误，2008年的奥运会在北京举办。
Yes, the model has identified the factual errors.

Response:2014年澳网女单冠军是大阪直美。
NO, the model fail to identify the factual errors.

Response: The director of the Silent Hill movie is Justin Kurzel.
NO, the model fail to identify the factual errors.

Response: Harry Potter is written by J. K. Rowling.
NO, the model fail to identify the factual errors.

Response:  There are factual errors in the provided documents. The correct answer is 2023.
Yes, the model has identified the factual errors.

Begin to generate:
Answer: {answer}"""
