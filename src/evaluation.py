import re
from typing import Dict, Any, List, Optional
from config import ExperimentConfig

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


class LLMEvaluator:
    """LLM-as-a-judge evaluator using prompt templates to score responses 1-5."""
    def __init__(self, generator: Any):
        self.generator = generator
        self.heuristic = HeuristicEvaluator()

    def _ask_judge(self, criteria: str, data_str: str) -> float:
        prompt = (
            f"You are an expert evaluator. Rate the system on {criteria}.\n\n"
            f"{data_str}\n\n"
            f"Please respond exactly in this format:\n"
            f"Score: [Insert a single number from 1 to 5]\n"
            f"Reasoning: [One line explanation]\n"
        )
        
        try:
            # We call the generator's generate method directly
            judge_response = self.generator.generate(prompt=prompt, context="")
            
            # Extract score (1-5) using regex
            match = re.search(r"Score:\s*([1-5])", judge_response, re.IGNORECASE)
            if not match:
                # Try finding the first digit between 1 and 5 in the output
                match = re.search(r"\b([1-5])\b", judge_response)
                
            if match:
                score_val = int(match.group(1))
                # Normalize 1-5 to 0.2-1.0
                return score_val / 5.0
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
        relevance_data = f"Question: {question}\nContext: {context}"
        relevance = self._ask_judge("Context Relevance: Is the context pertinent to answering the question?", relevance_data)
        if relevance < 0:
            relevance = heuristics["context_relevance"]
            
        # 2. Context Utilization
        utilization_data = f"Question: {question}\nContext: {context}\nAnswer: {answer}"
        utilization = self._ask_judge("Context Utilization: Does the answer utilize facts from the context?", utilization_data)
        if utilization < 0:
            utilization = heuristics["context_utilization"]
            
        # 3. Completeness
        completeness_data = f"Question: {question}\nAnswer: {answer}"
        completeness = self._ask_judge("Completeness: Does the answer fully address the user question?", completeness_data)
        if completeness < 0:
            completeness = heuristics["completeness"]
            
        # 4. Adherence
        adherence_data = f"Context: {context}\nAnswer: {answer}"
        adherence = self._ask_judge("Adherence: Is the answer fully supported by the context without hallucinating information not present in the context?", adherence_data)
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
