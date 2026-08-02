log_path = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/batch_experiments.log"

with open(log_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")
print("Last 50 lines:")
for line in lines[-50:]:
    print(line.strip())
