import re
from typing import Dict, Any, List, Optional
from config import ExperimentConfig
from prompts.evaluation import (
    EVALUATOR_JUDGE_TEMPLATE,
    CRITERIA_CONTEXT_RELEVANCE,
    CRITERIA_CONTEXT_UTILIZATION,
    CRITERIA_COMPLETENESS,
    CRITERIA_ADHERENCE,
    DATA_CONTEXT_RELEVANCE,
    DATA_CONTEXT_UTILIZATION,
    DATA_COMPLETENESS,
    DATA_ADHERENCE,
)

def clean_tokenize_set(text: str) -> set:
    """Returns a set of lowercase words from text, ignoring short tokens."""
    words = re.findall(r'\w+', text.lower())
    return {w for w in words if len(w) > 2}

class MockEvaluator:
    """Mock evaluator that returns fixed 0.5 scores."""
    def score(self, question: str, context: str, answer: str) -> Dict[str, float]:
        return {
            "context_relevance": 0.5,
            "context_utilization": 0.5,
            "completeness": 0.5,
            "adherence": 0.5
        }


class HeuristicEvaluator:
    """Heuristic-based evaluator using token overlaps (no LLM required)."""
    def score(self, question: str, context: str, answer: str) -> Dict[str, float]:
        q_words = clean_tokenize_set(question)
        c_words = clean_tokenize_set(context)
        a_words = clean_tokenize_set(answer)
        
        # 1. Context Relevance (Jaccard-like overlap between question and context)
        if q_words:
            relevance = len(q_words.intersection(c_words)) / len(q_words)
        else:
            relevance = 1.0
            
        # 2. Context Utilization (what fraction of the answer's words are in the context)
        if a_words:
            utilization = len(a_words.intersection(c_words)) / len(a_words)
        else:
            utilization = 1.0
            
        # 3. Completeness (heuristic: answer length + answer contains question words)
        ans_length_score = min(len(a_words) / 30.0, 1.0)
        q_overlap_score = len(a_words.intersection(q_words)) / len(q_words) if q_words else 1.0
        completeness = 0.5 * ans_length_score + 0.5 * q_overlap_score
        
        # 4. Adherence (fraction of answer words present in the context)
        if a_words:
            adherence = len(a_words.intersection(c_words)) / len(a_words)
        else:
            adherence = 1.0
            
        return {
            "context_relevance": min(max(relevance, 0.0), 1.0),
            "context_utilization": min(max(utilization, 0.0), 1.0),
            "completeness": min(max(completeness, 0.0), 1.0),
            "adherence": min(max(adherence, 0.0), 1.0)
        }

def parse_judge_score(response: str) -> Optional[float]:
    """Parses the judge's response to extract a score, mapping it to a float in [0.2, 1.0]."""
    if not response:
        return None

    response_clean = response.strip()
    
    # 1. Search for a standard pattern like "Score: [1-5]" or "Rating: [1-5]" (handles decimals too, e.g. 4.5)
    match = re.search(r"(?:score|rating|rate|value):\s*([1-5](?:\.\d+)?)", response_clean, re.IGNORECASE)
    if match:
        return min(max(float(match.group(1)) / 5.0, 0.2), 1.0)

    # 2. Search for score format like X/5
    match = re.search(r"\b([1-5](?:\.\d+)?)\s*/\s*5\b", response_clean, re.IGNORECASE)
    if match:
        return min(max(float(match.group(1)) / 5.0, 0.2), 1.0)

    # 3. Search for a standalone number from 1 to 5
    match = re.search(r"\b([1-5](?:\.\d+)?)\b", response_clean)
    if match:
        return min(max(float(match.group(1)) / 5.0, 0.2), 1.0)

    # 4. Search for textual representations of numbers 1-5 (case-insensitive)
    text_nums = {
        "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
        "first": 1.0, "second": 2.0, "third": 3.0, "fourth": 4.0, "fifth": 5.0
    }
    response_lower = response_clean.lower()
    for word, val in text_nums.items():
        if re.search(r"\b" + word + r"\b", response_lower):
            return val / 5.0

    # 5. Semantic classification for binary questions:
    # All evaluation criteria are Yes/No questions, so yes-like means 5/5, no-like means 1/5.
    # Note: negative check comes first to handle cases like "not relevant".
    neg_pattern = r"\b(?:no|incorrect|false|irrelevant|unsupported|hallucinated|hallucinating|hallucination|incomplete|not)\b"
    pos_pattern = r"\b(?:yes|correct|true|relevant|pertinent|supported|adherent|utilize|utilized|completely|fully|complete)\b"

    if re.search(neg_pattern, response_lower):
        return 0.2
    if re.search(pos_pattern, response_lower):
        return 1.0

    return None


class LLMEvaluator:
    """LLM-as-a-judge evaluator using prompt templates to score responses 1-5."""
    def __init__(self, generator: Any):
        self.generator = generator
        self.heuristic = HeuristicEvaluator()

    def _ask_judge(self, criteria: str, data_str: str) -> float:
        prompt = EVALUATOR_JUDGE_TEMPLATE.format(criteria=criteria, data_str=data_str)
        
        try:
            # We call the generator's generate method directly
            judge_response = self.generator.generate(prompt=prompt, context="")
            
            # Extract score using the robust parsing helper
            score_val = parse_judge_score(judge_response)
            if score_val is not None:
                return score_val
            else:
                print(f"Warning: Judge response was unparseable. Response was:\n{judge_response}")
                return 0.6  # Default fallback (3/5)
        except Exception as e:
            print(f"Warning: Judge LLM call failed: {e}. Falling back to heuristic.")
            return -1.0 # Sentinel value for fallback

    def score(self, question: str, context: str, answer: str) -> Dict[str, float]:
        # If generator is mock, fall back to mock score
        if hasattr(self.generator, "__class__") and self.generator.__class__.__name__ == "MockGenerator":
            return {
                "context_relevance": 0.5,
                "context_utilization": 0.5,
                "completeness": 0.5,
                "adherence": 0.5
            }
            
        heuristics = self.heuristic.score(question, context, answer)
        
        # 1. Context Relevance
        relevance_data = DATA_CONTEXT_RELEVANCE.format(question=question, context=context)
        relevance = self._ask_judge(CRITERIA_CONTEXT_RELEVANCE, relevance_data)
        if relevance < 0:
            relevance = heuristics["context_relevance"]
            
        # 2. Context Utilization
        utilization_data = DATA_CONTEXT_UTILIZATION.format(question=question, context=context, answer=answer)
        utilization = self._ask_judge(CRITERIA_CONTEXT_UTILIZATION, utilization_data)
        if utilization < 0:
            utilization = heuristics["context_utilization"]
            
        # 3. Completeness
        completeness_data = DATA_COMPLETENESS.format(question=question, answer=answer)
        completeness = self._ask_judge(CRITERIA_COMPLETENESS, completeness_data)
        if completeness < 0:
            completeness = heuristics["completeness"]
            
        # 4. Adherence
        adherence_data = DATA_ADHERENCE.format(context=context, answer=answer)
        adherence = self._ask_judge(CRITERIA_ADHERENCE, adherence_data)
        if adherence < 0:
            adherence = heuristics["adherence"]
            
        return {
            "context_relevance": relevance,
            "context_utilization": utilization,
            "completeness": completeness,
            "adherence": adherence
        }


def get_evaluator(config: ExperimentConfig, generator: Optional[Any] = None):
    if config.evaluator == "mock":
        return MockEvaluator()
    elif config.evaluator == "heuristic":
        return HeuristicEvaluator()
    elif config.evaluator == "llm":
        return LLMEvaluator(generator)
    else:
        raise ValueError(f"Unknown evaluator: {config.evaluator}")
