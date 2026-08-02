log_path = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/batch_experiments.log"

with open(log_path, "r", encoding="utf-8") as f:
    for i in range(150):
        line = f.readline()
        if not line:
            break
        print(f"{i+1}: {line.strip()}")
