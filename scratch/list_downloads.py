import os

downloads_dir = "/Users/aishwaryashilpi/Downloads/drive-download-20260802T031937Z-1-001/"
files = sorted(os.listdir(downloads_dir))

print(f"Total files in downloads directory: {len(files)}")
for f in files:
    print(f)
