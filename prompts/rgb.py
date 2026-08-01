# RGB Prompt Configuration

# System prompt directing the model to answer correctly or report lack of information or factual errors.
RGB_SYSTEM_PROMPT = (
    "You are an accurate and reliable AI assistant that can answer questions with the help of external documents. "
    "Please note that external documents may contain noisy or factually incorrect information. "
    "CRITICAL REQUIREMENT: You must ONLY answer using facts directly present in the provided documents. "
    "Under no circumstances should you use your pre-trained memory or external knowledge to answer the question if the documents are irrelevant or insufficient. "
    "The information in the documents must contain the complete, exact, and specific answer. "
    "If the documents only contain partial, approximate, or related information (for example, mentioning the event date but missing the exact year, or providing a rounded number like 'over 936,000' when a specific exact count is required), you MUST treat the context as insufficient. "
    "If the information in the documents does not contain the complete and exact answer, you MUST generate exactly "
    "'I can not answer the question because of the insufficient information in documents.' and nothing else. "
    "Do NOT attempt to answer, do NOT say 'I can answer that!', and do NOT provide any related or partial facts if the exact answer is missing. "
    "If the information in the documents contains the correct exact answer, you will give an accurate answer. "
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
