EVALUATOR_JUDGE_TEMPLATE = (
    "You are an expert evaluator. Rate the system on {criteria}.\n\n"
    "{data_str}\n\n"
    "Please respond exactly in this format:\n"
    "Score: [Insert a single number from 1 to 5]\n"
    "Reasoning: [One line explanation]\n"
)

# Criteria templates
CRITERIA_CONTEXT_RELEVANCE = "Context Relevance: Is the context pertinent to answering the question?"
CRITERIA_CONTEXT_UTILIZATION = "Context Utilization: Does the answer utilize facts from the context?"
CRITERIA_COMPLETENESS = "Completeness: Does the answer fully address the user question?"
CRITERIA_ADHERENCE = "Adherence: Is the answer fully supported by the context without hallucinating information not present in the context?"

# Data format templates
DATA_CONTEXT_RELEVANCE = "Question: {question}\nContext: {context}"
DATA_CONTEXT_UTILIZATION = "Question: {question}\nContext: {context}\nAnswer: {answer}"
DATA_COMPLETENESS = "Question: {question}\nAnswer: {answer}"
DATA_ADHERENCE = "Context: {context}\nAnswer: {answer}"
