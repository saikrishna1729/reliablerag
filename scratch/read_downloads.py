import os
import glob
import json

downloads_dir = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/"
summary_files = glob.glob(os.path.join(downloads_dir, "*_summary.json"))

print(f"Found {len(summary_files)} summary files.")

results = []
for filepath in sorted(summary_files):
    filename = os.path.basename(filepath)
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            results.append({
                "file": filename,
                "dataset": data.get("dataset"),
                "generator": data.get("generator"),
                "noise_rate": data.get("noise_rate"),
                "total_records": data.get("total_records"),
                "accuracy": data.get("accuracy"),
                "rejection_rate": data.get("rejection_rate"),
                "error_detection_rate": data.get("error_detection_rate"),
                "error_correction_rate": data.get("error_correction_rate"),
                "execution_time_minutes": data.get("execution_time_minutes")
            })
        except Exception as e:
            print(f"Error parsing {filename}: {e}")

# Group and display results
for r in results:
    print(f"Gen: {r['generator']} | Dataset: {r['dataset']} | Noise: {r['noise_rate']} | Acc: {r['accuracy']} | Rej: {r['rejection_rate']} | ErrDet: {r['error_detection_rate']} | ErrCorr: {r['error_correction_rate']}")
