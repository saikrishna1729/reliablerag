import json
import os

def main():
    filepath = "/Users/aishwaryashilpi/workspace/ragstack/notebooks/run_rgb_eval_colab.ipynb"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
        
    # Check if Step 6 is already added to prevent duplicate additions
    step_6_exists = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = ''.join(cell.get('source', []))
            if 'Step 6: Run Automated Batch Experiments' in source:
                step_6_exists = True
                break
                
    if step_6_exists:
        print("Step 6 cells already exist in the notebook. Skipping...")
        return
        
    # Define Step 6 Markdown Cell
    markdown_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🤖 Step 6: Run Automated Batch Experiments (All 26 Scheduled Runs)\n",
            "\n",
            "Use the code cell below to trigger the sequential automated batch run of all 26 scheduled experiments (covering sweeps for Qwen and Llama across Noise Robustness, Negative Rejection, Information Integration, and Counterfactual Robustness).\n",
            "\n",
            "### Features:\n",
            "1. **Resumable**: If your Colab tab disconnects, simply reconnect and run this cell. It will auto-detect completed files on your Google Drive and skip them.\n",
            "2. **Auto-Pull**: Missing models (`qwen2.5:7b`, `qwen2.5:14b`) will be fetched automatically in the background.\n",
            "3. **Persistent Logging**: Real-time logs are saved to `eval/results/rgb/batch_experiments.log` on Google Drive."
        ]
    }
    
    # Define Step 6 Code Cell
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run all 26 experiments back-to-back\n",
            "!python3 scripts/auto_run_experiments.py\n",
            "\n",
            "# Optional: Un-comment the line below to run a quick 3-run mock validation test (completes in seconds)\n",
            "# !python3 scripts/auto_run_experiments.py --mock"
        ]
    }
    
    # Append cells to notebook
    notebook['cells'].append(markdown_cell)
    notebook['cells'].append(code_cell)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write('\n')
        
    print("Successfully added Step 6 cells to run_rgb_eval_colab.ipynb.")

if __name__ == '__main__':
    main()
