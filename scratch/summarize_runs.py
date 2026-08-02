import os
import glob
import json
import datetime

downloads_dir = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/"
summary_files = glob.glob(os.path.join(downloads_dir, "*_summary.json"))

print(f"{'Filename':<70} | {'Model':<12} | {'Dataset':<8} | {'Noise':<5} | {'Acc':<6} | {'Rej':<6} | {'Time':<20}")
print("-" * 145)

for filepath in sorted(summary_files):
    filename = os.path.basename(filepath)
    mtime = os.path.getmtime(filepath)
    mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            model = data.get("generator")
            dataset = data.get("dataset")
            noise = str(data.get("noise_rate"))
            acc = f"{data.get('accuracy') * 100:.2f}%" if data.get('accuracy') is not None else "N/A"
            rej = f"{data.get('rejection_rate') * 100:.2f}%" if data.get('rejection_rate') is not None else "N/A"
            print(f"{filename:<70} | {model:<12} | {dataset:<8} | {noise:<5} | {acc:<6} | {rej:<6} | {mtime_str}")
        except Exception as e:
            print(f"{filename:<70} | Error: {e}")
