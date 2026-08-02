log_path = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/batch_experiments.log"

with open(log_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        if "🚀 RUN" in line:
            print(f"Line {line_num}: {line.strip()}")
