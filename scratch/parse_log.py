import os
import re
import json

log_path = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/batch_experiments.log"

with open(log_path, "r", encoding="utf-8") as f:
    log_content = f.read()

# We want to find each RUN block, e.g.:
# 🚀 RUN X/Y | Model: <model> | Dataset: <dataset> | Noise Rate: <noise>
# ...
# 📈 SUMMARY METRICS:
# <JSON block>

# Let's split or scan the log content for "🚀 RUN " patterns or "SUMMARY METRICS:" patterns.
# Let's find all matches of the "🚀 RUN" start.
run_matches = list(re.finditer(r"🚀 RUN (\d+)/(\d+) \| Model: ([^\s|]+) \| Dataset: ([^\s|]+) \| Noise Rate: ([^\s\n]+)", log_content))

print(f"Found {len(run_matches)} run starts in log.")

# For each run start, let's find the subsequent "SUMMARY METRICS:" json block before the next run start.
runs_data = []
for i, match in enumerate(run_matches):
    start_pos = match.start()
    end_pos = run_matches[i+1].start() if i + 1 < len(run_matches) else len(log_content)
    
    run_segment = log_content[start_pos:end_pos]
    
    run_idx = match.group(1)
    total_runs = match.group(2)
    model = match.group(3)
    dataset = match.group(4)
    noise = match.group(5)
    
    # Try to find SUMMARY METRICS json block inside this segment
    metrics_match = re.search(r"📈 SUMMARY METRICS:\s*(\{.*?\})", run_segment, re.DOTALL)
    metrics = None
    if metrics_match:
        try:
            metrics = json.loads(metrics_match.group(1))
        except Exception as e:
            print(f"Error parsing metrics json for run {run_idx}: {e}")
    else:
        # Check if it was skipped
        skip_match = re.search(r"Skip|already exists", run_segment)
        # Or check if it says "Run completed"
        pass
        
    runs_data.append({
        "run_idx": int(run_idx),
        "total_runs": int(total_runs),
        "model": model,
        "dataset": dataset,
        "noise": float(noise),
        "metrics": metrics,
        "has_metrics": metrics is not None
    })

print(f"Processed {len(runs_data)} runs. Detailed count:")
metrics_count = sum(1 for r in runs_data if r["has_metrics"])
print(f"Runs with metrics: {metrics_count}")

# Print runs with metrics:
for r in runs_data:
    if r["has_metrics"]:
        m = r["metrics"]
        print(f"Run {r['run_idx']}: {r['model']} | {r['dataset']} | {r['noise']} | Acc: {m.get('accuracy')} | Rej: {m.get('rejection_rate')} | ErrDet: {m.get('error_detection_rate')} | ErrCorr: {m.get('error_correction_rate')}")
    else:
        # Search the run_segment for skipping or other messages
        print(f"Run {r['run_idx']}: {r['model']} | {r['dataset']} | {r['noise']} - No direct metrics block in log segment.")
