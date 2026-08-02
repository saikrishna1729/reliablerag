import subprocess
import sys
import os
import glob
import json
import time
import argparse

class TeeLogger:
    """
    Custom logger that redirects stdout to both the console and a file.
    Robust to OSError when Google Drive disconnects.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.filepath = filepath
        self.log = None
        self.last_retry_time = 0
        self._open_log()
        
    def _open_log(self):
        now = time.time()
        if now - self.last_retry_time < 10:
            return
        self.last_retry_time = now
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            self.log = open(self.filepath, "a", encoding="utf-8")
        except Exception as e:
            self.terminal.write(f"\n⚠️ TeeLogger: Failed to open/create log file at {self.filepath}: {e}\n")
            self.log = None

    def write(self, message):
        self.terminal.write(message)
        if self.log is None:
            self._open_log()
        if self.log is not None:
            try:
                self.log.write(message)
                self.log.flush()
            except Exception as e:
                self.terminal.write(f"\n⚠️ TeeLogger Error writing log: {e}\n")
                try:
                    self.log.close()
                except Exception:
                    pass
                self.log = None
        
    def flush(self):
        self.terminal.flush()
        if self.log is not None:
            try:
                self.log.flush()
            except Exception as e:
                self.terminal.write(f"\n⚠️ TeeLogger Error flushing log: {e}\n")
                self.log = None

def check_and_pull_model(model_name):
    """
    Checks if an Ollama model is available locally, and pulls it if it's missing.
    """
    if model_name in ["mock", "hf-small", "hf-large"]:
        return
        
    print(f"Checking if Ollama model '{model_name}' is loaded...")
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            if model_name in result.stdout:
                print(f"  Model '{model_name}' is already pulled.")
                return
        
        print(f"  Model '{model_name}' not found. Pulling model (this might take a few minutes)...")
        pull_result = subprocess.run(["ollama", "pull", model_name], check=True)
        if pull_result.returncode == 0:
            print(f"  Successfully pulled model '{model_name}'.")
    except FileNotFoundError:
        print("  Warning: 'ollama' command not found. Skipping auto-pull check.")
    except Exception as e:
        print(f"  Error checking/pulling model: {e}")

def run_experiment(idx, total, gen, dataset, noise, is_mock=False):
    """
    Runs a single experiment configuration.
    """
    display_gen = "mock" if is_mock else gen
    sanitized_gen = "mock" if is_mock else gen.replace(":", "_")
    
    # 1. Resumability Check: Check if run was already completed
    summary_pattern = f"eval/results/rgb/prediction_{dataset}_{sanitized_gen}_noise{noise}_passage5_correct0.0*_summary.json"
    matching_files = glob.glob(summary_pattern)
    if matching_files:
        latest_summary = max(matching_files, key=os.path.getmtime)
        print(f"⏭️  [Run {idx}/{total}] Skipping: Model={display_gen}, Dataset={dataset}, Noise={noise}")
        print(f"    Reason: Summary file already exists at {latest_summary}\n")
        return

    print("=" * 70)
    print(f"🚀 RUN {idx}/{total} | Model: {display_gen} | Dataset: {dataset} | Noise Rate: {noise}")
    print(f"Remaining Runs: {total - idx}")
    print("=" * 70)
    
    if not is_mock:
        # Auto-pull the generator if needed
        check_and_pull_model(gen)
    
    # Determine correct Python interpreter (prefer virtual env)
    venv_python = os.path.join(os.getcwd(), ".venv/bin/python")
    if not os.path.exists(venv_python):
        venv_python = os.path.join(os.getcwd(), ".venv/bin/python3")
    python_exe = venv_python if os.path.exists(venv_python) else sys.executable

    cmd = [
        python_exe,
        "scripts/run_rgb_eval.py",
        "--generator", "mock" if is_mock else gen,
        "--dataset", dataset,
        "--noise_rate", str(noise),
        "--passage_num", "5",
        "--correct_rate", "0.0",
        "--use_llm_judge"
    ]
    
    if is_mock:
        cmd.extend(["--n_records", "2", "--judge_generator", "mock"])
        
    start_time = time.time()
    try:
        # Run evaluation script and stream outputs to log file in real-time
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
            
        elapsed = time.time() - start_time
        print(f"✅ Run completed in {elapsed/60:.2f} minutes.")
        
        # Locate the summary file generated by this run to display metrics
        matching_files = glob.glob(summary_pattern)
        if matching_files:
            latest_summary = max(matching_files, key=os.path.getmtime)
            with open(latest_summary, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            
            # Inject execution duration into the printed output
            metrics["execution_time_minutes"] = round(elapsed / 60, 2)
            
            print("\n📈 SUMMARY METRICS:")
            print(json.dumps(metrics, indent=4))
        else:
            print("⚠️ Warning: Summary file not found matching pattern:", summary_pattern)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during execution: {e}")
        print("Continuing with the next scheduled run...")
    print("\n")

def main():
    parser = argparse.ArgumentParser(description="Batch RAG Evaluation experiments runner.")
    parser.add_argument("--mock", action="store_true", help="Run in mock validation mode with 2 records per dataset.")
    args = parser.parse_args()

    # Enable Tee Logger to duplicate all prints to the Google Drive log file
    log_path = "eval/results/rgb/batch_experiments.log"
    sys.stdout = TeeLogger(log_path)
    
    print(f"📝 Logging session output to: {log_path}")

    if args.mock:
        print("🔧 Mock Validation mode enabled.")
        experiments = [
            ("mock", "en", 0.4),
            ("mock", "en_int", 0.2),
            ("mock", "en_fact", 1.0)
        ]
    else:
        # Complete list of 26 runs back to back (excluding the ongoing qwen2.5:7b, en, noise=0.4 run)
        experiments = [
            # 1. Llama 3 (8B) Noise Robustness sweep
            ("llama3:8b", "en", 0.0),
            ("llama3:8b", "en", 0.2),
            ("llama3:8b", "en", 0.6),
            ("llama3:8b", "en", 0.8),
            
            # 2. Qwen 2.5 (7B) Noise Robustness & Rejection sweep
            ("qwen2.5:7b", "en", 0.0),
            ("qwen2.5:7b", "en", 0.2),
            ("qwen2.5:7b", "en", 0.6),
            ("qwen2.5:7b", "en", 0.8),
            ("qwen2.5:7b", "en", 1.0),
            
            # 3. Qwen 2.5 (14B) Noise Robustness & Rejection sweep
            ("qwen2.5:14b", "en", 0.0),
            ("qwen2.5:14b", "en", 0.2),
            ("qwen2.5:14b", "en", 0.4),
            ("qwen2.5:14b", "en", 0.6),
            ("qwen2.5:14b", "en", 0.8),
            ("qwen2.5:14b", "en", 1.0),
            
            # 4. Llama 3 (8B) Information Integration sweep
            ("llama3:8b", "en_int", 0.0),
            ("llama3:8b", "en_int", 0.2),
            ("llama3:8b", "en_int", 0.4),
            
            # 5. Qwen 2.5 (7B) Information Integration sweep
            ("qwen2.5:7b", "en_int", 0.0),
            ("qwen2.5:7b", "en_int", 0.2),
            ("qwen2.5:7b", "en_int", 0.4),
            
            # 6. Qwen 2.5 (14B) Information Integration sweep
            ("qwen2.5:14b", "en_int", 0.0),
            ("qwen2.5:14b", "en_int", 0.2),
            ("qwen2.5:14b", "en_int", 0.4),
            
            # 7. Qwen 2.5 (7B & 14B) Counterfactual sweep (noise_rate=1.0)
            ("qwen2.5:7b", "en_fact", 1.0),
            ("qwen2.5:14b", "en_fact", 1.0)
        ]
    
    total_runs = len(experiments)
    print(f"📋 Starting automated batch pipeline of {total_runs} scheduled runs.")
    print("This will execute all scheduled runs sequentially.\n")
    
    for i, (gen, dataset, noise) in enumerate(experiments, 1):
        run_experiment(i, total_runs, gen, dataset, noise, is_mock=args.mock)
        
    print("🎉 All scheduled experiments completed!")

if __name__ == '__main__':
    main()
