import sys
import os

# Add workspace root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation import parse_judge_score

def run_tests():
    # Test cases mapping raw response strings to expected normalized score values
    test_cases = [
        # Standard format
        ("Score: 5\nReasoning: great job", 1.0),
        ("Score: 1\nReasoning: bad job", 0.2),
        ("score: 3", 0.6),
        ("Rating: 4", 0.8),
        ("Rate: 2", 0.4),
        
        # Standalone digits / decimal digits
        ("5", 1.0),
        ("4.0", 0.8),
        ("3.5", 0.7),
        ("Score: 4.5", 0.9),
        ("I rate this system 3/5 stars.", 0.6),
        ("The score is 2.", 0.4),
        
        # Textual numbers
        ("Score is three", 0.6),
        ("four", 0.8),
        ("second option", 0.4),
        
        # Yes/No and semantic classifications
        ("Yes, the answer is fully supported.", 1.0),
        ("No, it does not utilize the context.", 0.2),
        ("The context is relevant to the question.", 1.0),
        ("It is irrelevant.", 0.2),
        ("The answer is unsupported.", 0.2),
        ("There is a hallucination in the response.", 0.2),
        ("The answer is incomplete.", 0.2),
        ("yes", 1.0),
        ("no", 0.2),
        
        # Mixed/difficult cases
        ("The context is not relevant.", 0.2),  # negative should override positive
        ("not supported at all", 0.2),
        
        # Fallback case
        ("Random text without any indicators.", None),
    ]

    print("Running parser unit tests...")
    failed = 0
    for idx, (response, expected) in enumerate(test_cases):
        actual = parse_judge_score(response)
        if actual != expected and (actual is None or expected is None or abs(actual - expected) > 1e-5):
            print(f"Test case {idx} FAILED!")
            print(f"  Response: {repr(response)}")
            print(f"  Expected: {expected}, Got: {actual}")
            failed += 1
        else:
            print(f"Test case {idx} PASSED (expected: {expected}, got: {actual})")
            
    if failed == 0:
        print("\nAll parser tests PASSED successfully!")
    else:
        print(f"\n{failed} parser tests FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
