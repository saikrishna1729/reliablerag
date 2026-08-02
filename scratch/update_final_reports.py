import os

def update_file(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 1. Update Noise Robustness Accuracy Table
    old_table_acc = """| Model | Noise=0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `llama3:8b` | **95.33%** | **93.00%** | **94.00%** | **88.33%** | **80.00%** |
| `qwen2.5:7b` | **93.00%** | **89.00%** | *Pending* | **82.33%** | **70.00%** |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* | *Pending* | *Pending* |"""

    new_table_acc = """| Model | Noise=0 | 0.2 | 0.4 | 0.6 | 0.8 |
|---|---|---|---|---|---|
| `llama3:8b` | **95.33%** | **93.00%** | **94.00%** | **88.33%** | **80.00%** |
| `qwen2.5:7b` | **93.00%** | **89.00%** | **87.67%** | **82.33%** | **70.00%** |
| `qwen2.5:14b` | **94.33%** | **94.67%** | **93.33%** | **88.33%** | *Pending* |"""

    content = content.replace(old_table_acc, new_table_acc)

    # 2. Update Negative Rejection Table
    old_table_rej = """| Model | Baseline Rejection (%) | Chat API + Parser Optimization (%) | Prompt Exactness Refinement (%) |
|---|---|---|---|
| `llama3:8b` | 27.33% | 30.00% | **34.67%** |
| `qwen2.5:7b` | N/A | N/A | **57.33%** |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* |"""

    new_table_rej = """| Model | Baseline Rejection (%) | Chat API + Parser Optimization (%) | Prompt Exactness Refinement (%) |
|---|---|---|---|
| `llama3:8b` | 27.33% | 30.00% | **34.67%** |
| `qwen2.5:7b` | N/A | N/A | **57.33%** |
| `qwen2.5:14b` | *Pending* | *Pending* | *Pending* |"""

    content = content.replace(old_table_rej, new_table_rej)

    # 3. Update Overall Comparison Table in Section 6
    old_overall_table = """| Metric | `llama3:8b` | `qwen2.5:7b` | `qwen2.5:14b` | Best |
|---|---|---|---|---|
| Noise Robustness (Acc at 0.4) | **94.00%** | *Pending* | *Pending* | `llama3:8b` |
| Negative Rejection (Rej at 1.0) | **34.67%** | *Pending* | *Pending* | `llama3:8b` |
| Information Integration | *Pending* | *Pending* | *Pending* | TBD |
| Counterfactual (Correction at 1.0) | **13.64%** | *Pending* | *Pending* | `llama3:8b` |"""

    new_overall_table = """| Metric | `llama3:8b` | `qwen2.5:7b` | `qwen2.5:14b` | Best |
|---|---|---|---|---|
| Noise Robustness (Acc at 0.4) | **94.00%** | **87.67%** | **93.33%** | `llama3:8b` (94.00%) |
| Negative Rejection (Rej at 1.0) | **34.67%** | **57.33%** | *Pending* | `qwen2.5:7b` (57.33%) |
| Information Integration | *Pending* | *Pending* | *Pending* | TBD |
| Counterfactual (Correction at 1.0) | **13.64%** | *Pending* | *Pending* | `llama3:8b` (13.64%) |"""

    content = content.replace(old_overall_table, new_overall_table)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print(f"Updated: {os.path.basename(filepath)}")
    return True

def main():
    workspace_report = "/Users/aishwaryashilpi/workspace/ragstack/reports/rag_evaluation_report.md"
    brain_report = "/Users/aishwaryashilpi/.gemini/antigravity-ide/brain/339dfc73-6752-46c4-b087-c71e5710d379/rag_evaluation_report.md"
    
    update_file(workspace_report)
    update_file(brain_report)

if __name__ == '__main__':
    main()
