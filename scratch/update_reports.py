import os

def update_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replacements for Noise Robustness Accuracy Table
    old_table_acc = """| Model | Noise=0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `llama3:8b` | *Pending* | *Pending* | **94.00%** | *Pending* | *Pending* |
| `qwen2.5:7b` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |"""

    new_table_acc = """| Model | Noise=0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `llama3:8b` | **95.33%** | **93.00%** | **94.00%** | **88.33%** | **80.00%** |
| `qwen2.5:7b` | **93.00%** | **89.00%** | *Pending* | **82.33%** | **70.00%** |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |"""

    content = content.replace(old_table_acc, new_table_acc)

    # Replacements for Negative Rejection Table
    old_table_rej = """| Model | Baseline Rejection (%) | Chat API + Parser Optimization (%) | Prompt Exactness Refinement (%) |
|---|---|---|---|
| `llama3:8b` | 27.33% | 30.00% | **34.67%** |
| `qwen2.5:7b` | *Pending* | *Pending* | *Pending* |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* |"""

    new_table_rej = """| Model | Baseline Rejection (%) | Chat API + Parser Optimization (%) | Prompt Exactness Refinement (%) |
|---|---|---|---|
| `llama3:8b` | 27.33% | 30.00% | **34.67%** |
| `qwen2.5:7b` | N/A | N/A | **57.33%** |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* |"""

    content = content.replace(old_table_rej, new_table_rej)
    
    # Update Phase 5 Summary Table in experiments_report.md if present
    old_summary_table = """| Phase | Noise Rate | Key Refinements | Measured Rejection | Primary Failure Point |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | 0.4 | Default prompt | 2.33% (full run) | High distraction sensitivity |
| **Phase 3** | 1.0 | Added system constraints | 27.33% (full run) | Flat-string prompt ignore; Rigid parser |
| **Phase 4** | 1.0 | `/api/chat` mapping + parsed rules | 30.00% (full run) | Approximate answer leakage in distractor docs |
| **Phase 5** | 1.0 | Exactness prompt parameters | **34.67% (300 runs)** | Approximate context leakage vs strict heuristics |"""

    new_summary_table = """| Phase | Noise Rate | Key Refinements | Measured Rejection | Primary Failure Point |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 2** | 0.4 | Default prompt | 2.33% (full run) | High distraction sensitivity |
| **Phase 3** | 1.0 | Added system constraints | 27.33% (full run) | Flat-string prompt ignore; Rigid parser |
| **Phase 4** | 1.0 | `/api/chat` mapping + parsed rules | 30.00% (full run) | Approximate answer leakage in distractor docs |
| **Phase 5** | 1.0 | Exactness prompt parameters | **34.67% (Llama3-8b)** / **57.33% (Qwen2.5-7b)** | Approximate context leakage vs strict heuristics |"""

    content = content.replace(old_summary_table, new_summary_table)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated: {os.path.basename(filepath)}")
    return True

def main():
    brain_dir = "/Users/aishwaryashilpi/.gemini/antigravity-ide/brain/339dfc73-6752-46c4-b087-c71e5710d379"
    update_file(os.path.join(brain_dir, "experiments_report.md"))
    update_file(os.path.join(brain_dir, "rag_evaluation_report.md"))

if __name__ == '__main__':
    main()
